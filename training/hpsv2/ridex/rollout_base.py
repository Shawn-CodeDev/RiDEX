from dataclasses import dataclass
import math
import torch
from flow_grpo.diffusers_patch.solver import run_sampling


@dataclass
class Center:
    state: torch.Tensor
    velocity: torch.Tensor
    direction: torch.Tensor
    scale: torch.Tensor
    time: torch.Tensor
    credit: float
    fold: int
    diagnostics: dict


class TerminalCollector:
    def __init__(self, pipeline, gradient_reward, canonical_reward, *, step=8, radius=.005):
        self.pipe, self.gradient_reward, self.canonical_reward = pipeline, gradient_reward, canonical_reward
        self.step, self.radius = int(step), float(radius)
        self.metadata = {}
        self.reset_counts()

    def reset_counts(self):
        self.counts = dict(terminal_forwards=0, decoded_images=0, transformer_calls=0,
                          action_backwards=0, canonical_images=0, gradient_scorer_calls=0,
                          canonical_scorer_images=0)

    def velocity(self, x, sigma, embeds, pooled):
        self.counts["transformer_calls"] += 1
        time = torch.full((len(x),), float(sigma)*1000, device=x.device, dtype=torch.long)
        return self.pipe.transformer(hidden_states=x, timestep=time,
            encoder_hidden_states=embeds.expand(len(x), -1, -1),
            pooled_projections=pooled.expand(len(x), -1), return_dict=False)[0]

    def terminal(self, initial, sigmas, embeds, pooled, action=None, *, reference=None):
        index, saved = 0, {}
        def prediction(x, sigma):
            nonlocal index
            current = index
            index += 1
            value = self.velocity(x, sigma, embeds, pooled)
            if current == self.step:
                saved.update(state=x.detach().clone(), velocity=value.detach().clone(),
                    scale=value.detach().float().norm(),
                    time=torch.tensor([float(sigma)*1000], device=x.device, dtype=torch.long))
                if reference is not None:
                    if not torch.equal(x.detach(), reference["state"]):
                        raise RuntimeError("Probe prefix differs from clean state")
                    if not torch.equal(value.detach(), reference["velocity"]):
                        raise RuntimeError("Probe behavior differs from clean velocity")
                if action is not None:
                    scale = saved["scale"] if reference is None else reference["scale"]
                    value = value + scale*action
            return value
        endpoint, _, _ = run_sampling(prediction, initial, sigmas, "dpm2", True)
        self.counts["terminal_forwards"] += len(initial)
        if action is not None and not saved:
            raise ValueError("Injection index was not reached")
        return endpoint, saved

    def decode(self, latent):
        value = self.pipe.vae.decode(latent/self.pipe.vae.config.scaling_factor + self.pipe.vae.config.shift_factor,
                                     return_dict=False)[0]
        self.counts["decoded_images"] += len(latent)
        return (value/2+.5).clamp(0,1)

    def canonical(self, images, prompts):
        scores, info = self.canonical_reward(images.detach(), prompts, [self.metadata for _ in prompts])
        if len(scores) != len(prompts) or not all(math.isfinite(float(v)) for v in scores):
            raise RuntimeError("Invalid canonical reward response")
        self.counts["canonical_images"] += len(prompts)
        for name, values in info["components"].items():
            if len(values) != len(prompts) or not all(math.isfinite(float(v)) for v in values):
                raise RuntimeError(f"Invalid {name} component scores")
        self.counts["canonical_scorer_images"] += len(prompts)*len(info["components"])
        self.last_info = info
        return [float(v) for v in scores]

    def continuous(self, image, prompt):
        self.counts["gradient_scorer_calls"] += len(image)
        return self.gradient_reward.score(image, [prompt])

    def clean(self, initial, sigmas, embeds, pooled, prompt):
        with torch.no_grad():
            latent, _ = self.terminal(initial, sigmas, embeds, pooled)
            image = self.decode(latent)
        return latent.detach(), image.detach(), self.canonical(image, [prompt])[0]
