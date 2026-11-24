import numpy as np
import torch


def clean_advantages(prompts, rewards):
    prompts = np.asarray(prompts)
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or rewards.shape != prompts.shape or not np.isfinite(rewards).all():
        raise ValueError("Expected aligned finite clean rewards and prompts")
    advantages = np.empty_like(rewards)
    std = rewards.std() + 1e-4
    for prompt in np.unique(prompts):
        rows = prompts == prompt
        advantages[rows] = (rewards[rows] - rewards[rows].mean()) / std
    return advantages


def nft_loss(x0, xt, t, prediction, old_prediction, reference_prediction,
             advantage, *, beta=1., kl_weight=1e-4, advantage_clip=5.):
    dims = tuple(range(1, x0.ndim))
    t = t.view(-1, *([1] * (x0.ndim - 1)))
    r = .5 + .5 * advantage.clamp(-advantage_clip, advantage_clip) / advantage_clip
    positive = beta * prediction + (1 - beta) * old_prediction.detach()
    negative = (1 + beta) * old_prediction.detach() - beta * prediction
    positive_x0, negative_x0 = xt - t * positive, xt - t * negative
    with torch.no_grad():
        positive_weight = (positive_x0.double() - x0.double()).abs().mean(dims, keepdim=True).clamp_min(1e-5)
        negative_weight = (negative_x0.double() - x0.double()).abs().mean(dims, keepdim=True).clamp_min(1e-5)
    positive_loss = ((positive_x0 - x0).square() / positive_weight).mean(dims)
    negative_loss = ((negative_x0 - x0).square() / negative_weight).mean(dims)
    policy = (r * positive_loss + (1 - r) * negative_loss) * advantage_clip / beta
    kl = (prediction - reference_prediction.detach()).square().mean(dims)
    return policy + kl_weight * kl, {"policy_loss": policy.detach().mean(),
        "kl_div": kl.detach().mean(), "label_mean": r.detach().mean()}


def old_policy_decay(completed_updates):
    return min(float(completed_updates) * .001, .5)
