import gc
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
from ridex.nft import nft_loss

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path, data):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(data, allow_nan=False) + "\n")


def gather_objects(value, world_size):
    if world_size == 1:
        return [value]
    values = [None] * world_size
    dist.all_gather_object(values, value)
    return values


def reduce_gradients(gradients, world_size):
    for gradient in gradients:
        if world_size > 1:
            dist.all_reduce(gradient)
            gradient.div_(world_size)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite distributed gradient")
    return gradients


def select_policy(pipe, name, training=False):
    pipe.transformer.set_adapter(name)
    pipe.transformer.requires_grad_(False)
    pipe.transformer.train(training)
    if training:
        for key, parameter in pipe.transformer.named_parameters():
            if ".default." in key and "lora_" in key:
                parameter.requires_grad_(True)


def adapter_parameters(pipe, name):
    return [p for key, p in pipe.transformer.named_parameters()
            if f".{name}." in key and "lora_" in key]


class Conditioning:

    def __init__(self, pipe, scorer, config, device):
        self.pipe, self.scorer, self.config, self.device = pipe, scorer, config, device
        self.encoders = [pipe.text_encoder, pipe.text_encoder_2, pipe.text_encoder_3]
        self.tokenizers = [pipe.tokenizer, pipe.tokenizer_2, pipe.tokenizer_3]
        pipe.text_encoder = pipe.text_encoder_2 = pipe.text_encoder_3 = None
        for model in self.encoders:
            model.eval().requires_grad_(False).to("cpu", dtype=torch.float16)
        self.cache = {}

    def prepare(self, prompts):
        prompts = list(dict.fromkeys([""] + list(prompts)))
        missing = [p for p in prompts if p not in self.cache]
        if missing:
            self.pipe.transformer.to("cpu")
            self.pipe.vae.to("cpu")
            if self.scorer is not None:
                self.scorer.model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            for model in self.encoders:
                model.to(self.device)
            with torch.no_grad():
                for start in range(0, len(missing), 2):
                    subset = missing[start:start + 2]
                    embeds, pooled = encode_prompt(self.encoders, self.tokenizers, subset,
                        self.config.sequence_length, device=self.device)
                    for index, prompt in enumerate(subset):
                        self.cache[prompt] = (embeds[index:index + 1].detach().float().cpu(),
                                              pooled[index:index + 1].detach().float().cpu())
            for model in self.encoders:
                model.to("cpu")
            gc.collect()
            torch.cuda.empty_cache()
            self.pipe.transformer.to(self.device)
            self.pipe.vae.to(self.device)
            if self.scorer is not None:
                self.scorer.model.to(self.device)
        self.cache = {prompt: self.cache[prompt] for prompt in prompts}

    def get(self, prompt):
        return tuple(value.to(self.device) for value in self.cache[prompt])


def latent_from_seed(pipe, config, seed, device):
    return torch.randn((1, pipe.transformer.config.in_channels,
                        config.resolution // pipe.vae_scale_factor,
                        config.resolution // pipe.vae_scale_factor),
                       generator=torch.Generator().manual_seed(seed), dtype=torch.float32).to(device)


def clean_gradient(pipe, conditioning, config, samples, advantages, sigmas, parameters,
                   epoch, device, world_size, *, global_sample_count=None):
    for parameter in parameters:
        parameter.grad = None
    stats = {"policy_loss": 0., "kl_div": 0., "label_mean": 0.}
    count = len(samples)
    denominator = count if global_sample_count is None else global_sample_count / world_size
    generator = torch.Generator().manual_seed(config.seed + 2000000 + epoch)
    order = torch.randperm(count, generator=generator).tolist()
    time_values = (sigmas[:-1] * 1000).detach()
    for offset, index in enumerate(order):
        sample = samples[index]
        embeds, pooled = conditioning.get(sample["prompt"])
        x0 = sample["latent"].to(device)
        view_generator = torch.Generator().manual_seed(
            config.seed + 10000000 + epoch * 100000 + sample["group"] * 100 + sample["slot"])
        time_order = torch.randperm(config.steps, generator=view_generator).tolist()
        for view in time_order:
            noise_generator = torch.Generator().manual_seed(
                config.seed + 30000000 + epoch * 1000000
                + (sample["group"] * config.clean_per_prompt + sample["slot"]) * config.steps + view)
            noise = torch.randn(x0.shape, generator=noise_generator, dtype=torch.float32).to(device)
            timestep = time_values[view:view + 1]
            t = timestep / 1000.
            xt = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * noise
            select_policy(pipe, "old")
            with torch.no_grad():
                old = pipe.transformer(hidden_states=xt, timestep=timestep,
                    encoder_hidden_states=embeds, pooled_projections=pooled, return_dict=False)[0]
                with pipe.transformer.disable_adapter():
                    reference = pipe.transformer(hidden_states=xt, timestep=timestep,
                        encoder_hidden_states=embeds, pooled_projections=pooled, return_dict=False)[0]
            select_policy(pipe, "default", training=True)
            prediction = pipe.transformer(hidden_states=xt, timestep=timestep,
                encoder_hidden_states=embeds, pooled_projections=pooled, return_dict=False)[0]
            loss, metrics = nft_loss(x0, xt, t, prediction, old, reference,
                                    torch.tensor([advantages[index]], device=device, dtype=torch.float32))
            (loss.sum() / (denominator * config.steps)).backward()
            for key, value in metrics.items():
                stats[key] += float(value) / (denominator * config.steps)
        if offset % 16 == 0:
            print(json.dumps({"phase": "clean_backward", "epoch": epoch,
                              "rank": int(os.environ.get("RANK", 0)), "samples": offset + 1,
                              "total": count}), flush=True)
    gradients = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in parameters]
    for parameter in parameters:
        parameter.grad = None
    return reduce_gradients(gradients, world_size), stats
