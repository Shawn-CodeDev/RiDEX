from pathlib import Path
import torch
import torch.distributed as dist
from diffusers import StableDiffusion3Pipeline
from peft import LoraConfig, get_peft_model
from ridex.gradient_rewards import GradientReward
from ridex.training import Conditioning, adapter_parameters
from ridex.scoring import canonical_scorer


def create_model(config, device, world_size):
    pipe = StableDiffusion3Pipeline.from_pretrained(config.model, torch_dtype=torch.float32, local_files_only=True)
    torch.manual_seed(config.seed)
    lora = LoraConfig(r=32, lora_alpha=64, init_lora_weights="gaussian", target_modules=[
        "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
        "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v"])
    pipe.transformer = get_peft_model(pipe.transformer, lora)
    pipe.transformer.add_adapter("old", lora)
    params, old = adapter_parameters(pipe, "default"), adapter_parameters(pipe, "old")
    if not params or len(params) != len(old):
        raise RuntimeError("Incomplete adapter mapping")
    with torch.no_grad():
        for p, o in zip(params, old, strict=True):
            if world_size > 1:
                value = p.detach().to(device)
                dist.broadcast(value, src=0)
                p.copy_(value.cpu())
            o.copy_(p)
    pipe.transformer.enable_gradient_checkpointing()
    pipe.vae.requires_grad_(False).eval().enable_gradient_checkpointing()
    pipe.set_progress_bar_config(disable=True)
    reward = GradientReward(Path(config.reward_models), device=device) if config.proposal else None
    remote = canonical_scorer()
    conditioning = Conditioning(pipe, reward, config, device)
    counts = {}
    def model_call(module, inputs):
        counts["all_transformer_forwards"] = counts.get("all_transformer_forwards", 0) + 1
    def block_call(module, inputs):
        key = "transformer_block_forwards_including_recompute"
        counts[key] = counts.get(key, 0) + 1
    pipe.transformer.register_forward_pre_hook(model_call)
    for block in pipe.transformer.transformer_blocks:
        block.register_forward_pre_hook(block_call)
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, betas=(.9,.999), eps=1e-8, weight_decay=1e-4)
    return pipe, reward, conditioning, remote, counts, params, old, optimizer


def joint_adam_step(params, optimizer, gradients):
    before = [p.detach().cpu().clone() for p in params]
    optimizer.zero_grad(set_to_none=True)
    for p, g in zip(params, gradients, strict=True):
        if not torch.isfinite(g).all():
            raise FloatingPointError("Nonfinite joint gradient")
        p.grad = g
    norm = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
    optimizer.step()
    with torch.no_grad():
        delta = sum(float((p.detach().cpu().double()-b.double()).square().sum()) for p,b in zip(params,before))**.5
        if not all(bool(torch.isfinite(p).all()) for p in params):
            raise FloatingPointError("Nonfinite updated parameter")
    optimizer.zero_grad(set_to_none=True)
    return dict(gradient_norm_before_clip=norm, actual_fp32_stored_delta_norm=delta)
