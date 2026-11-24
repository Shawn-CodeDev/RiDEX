import math
import torch
from ridex.rollout_base import TerminalCollector, Center
from ridex.scoring import component_at

def pair_credit(positive, negative, radius):
    if radius <= 0 or not all(math.isfinite(x) for x in (positive, negative, radius)):
        raise ValueError("Nonfinite pair or invalid radius")
    return (positive-negative)/(2*radius)


def directional_loss(prediction, behavior, direction, scale, credit):
    if not torch.isfinite(scale).all() or not bool(scale > 0):
        raise ValueError("Invalid action coordinate scale")
    return -credit * ((prediction-behavior.detach()) * direction.detach()).sum() / scale.detach()


class ActionCreditCollector(TerminalCollector):
    def center(self, initial, sigmas, embeds, pooled, prompt):
        if any(p.requires_grad for p in self.pipe.transformer.parameters()):
            raise RuntimeError("Freeze behavior parameters before proposal differentiation")
        with torch.enable_grad():
            action = torch.zeros_like(initial, requires_grad=True)
            latent, saved = self.terminal(initial, sigmas, embeds, pooled, action)
            image = self.decode(latent)
            objective = self.continuous(image, prompt).sum()
            gradient, = torch.autograd.grad(objective, action)
        self.counts["action_backwards"] += 1
        continuous_score = float(objective.detach())
        del action, objective
        latent, image, gradient = latent.detach(), image.detach(), gradient.detach()
        gradient_norm = float(gradient.norm())
        if not math.isfinite(gradient_norm) or not bool(saved["scale"] > 0):
            raise FloatingPointError("Invalid terminal action derivative")
        direction = gradient/gradient_norm if gradient_norm else torch.zeros_like(gradient)
        baseline = self.canonical(image, [prompt])[0]
        baseline_components = component_at(self.last_info, 0)
        endpoints, pictures = [], []
        with torch.no_grad():
            for sign in (1, -1):
                endpoint, _ = self.terminal(initial, sigmas, embeds, pooled,
                    sign*self.radius*direction, reference=saved)
                endpoints.append(endpoint.detach().cpu())
                pictures.append(self.decode(endpoint).detach())
        scores = self.canonical(torch.cat(pictures), [prompt, prompt])
        positive_components, negative_components = [component_at(self.last_info, index) for index in (0, 1)]
        credit = pair_credit(*scores, self.radius)
        diagnostics = dict(positive=scores[0], negative=scores[1], baseline=baseline,
            odd=(scores[0]-scores[1])/2, even=(scores[0]+scores[1])/2-baseline,
            gradient_norm=gradient_norm, continuous_score=continuous_score,
            continuous_training_score=continuous_score/100.,
            canonical_gap=baseline-continuous_score/100.,
            proposal="terminal_gradient", proposal_reward="hpsv2",
            baseline_components=baseline_components, positive_components=positive_components,
            negative_components=negative_components,
            component_credits={name: pair_credit(positive_components[name], negative_components[name], self.radius)
                for name in baseline_components},
            credit=credit, radius=self.radius)
        center = Center(saved["state"].cpu(), saved["velocity"].cpu(), direction.cpu(),
            saved["scale"].cpu(), saved["time"].cpu(), credit, 0, diagnostics)
        return latent.cpu(), image.cpu(), baseline, center, endpoints, scores
