import numpy as np
import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from flow_grpo.diffusers_patch.solver import run_sampling
from ridex.training import gather_objects, select_policy, write_json
from ridex.scoring import component_at, component_metrics


@torch.no_grad()
def image_for_row(pipe, conditioning, cfg, prompt, seed, device):
    embeds, pooled = conditioning.get(prompt)
    negative, negative_pooled = conditioning.get("")
    latent = pipe.prepare_latents(1, pipe.transformer.config.in_channels, cfg.resolution, cfg.resolution,
        embeds.dtype, device, [torch.Generator().manual_seed(seed)], None)
    retrieve_timesteps(pipe.scheduler, cfg.eval_steps, device, sigmas=None)
    sigmas = pipe.scheduler.sigmas.float()
    def prediction(x, sigma):
        time = torch.full((2,), float(sigma)*1000, device=device, dtype=torch.long)
        v = pipe.transformer(hidden_states=torch.cat([x,x]), timestep=time,
            encoder_hidden_states=torch.cat([negative,embeds]),
            pooled_projections=torch.cat([negative_pooled,pooled]),
            joint_attention_kwargs=None, return_dict=False)[0].to(embeds.dtype)
        vu, vc = v.chunk(2)
        return vu + cfg.eval_guidance*(vc-vu)
    latent, _, _ = run_sampling(prediction, latent, sigmas, "flow", True, .7)
    image = pipe.vae.decode((latent/pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor).to(pipe.vae.dtype),
                           return_dict=False)[0]
    return pipe.image_processor.postprocess(image, output_type="pt")


def evaluate(pipe, conditioning, collector, cfg, records, output, completed, rank, world):
    ids = list(range(rank, len(records), world))
    rows = []
    select_policy(pipe, "default")
    device = next(pipe.transformer.parameters()).device
    for offset in range(0, len(ids), 16):
        block = ids[offset:offset+16]
        conditioning.prepare([records[i]["prompt"] for i in block])
        for index in block:
            record, seed = records[index], cfg.seed+1000000+index
            image = image_for_row(pipe, conditioning, cfg, record["prompt"], seed, device)
            collector.metadata = record
            score = collector.canonical(image, [record["prompt"]])[0]
            row = dict(row=index, seed=seed, score=score, prompt=record["prompt"],
                components=component_at(collector.last_info, 0))
            rows.append(row)
        print(dict(phase="eval", checkpoint=completed, rank=rank, done=len(rows), total=len(ids)), flush=True)
    rows = sorted([r for part in gather_objects(rows, world) for r in part], key=lambda x:x["row"])
    if [r["row"] for r in rows] != list(range(len(records))):
        raise RuntimeError("Missing or duplicate evaluation rows")
    metrics = dict(eval_reward_avg=float(np.mean([r["score"] for r in rows]))*cfg.display_scale)
    metrics[f"eval_reward_{cfg.reward}"] = metrics["eval_reward_avg"]
    metrics.update(component_metrics(rows, "eval_reward"))
    if rank == 0:
        write_json(output/f"eval_{completed}.json", dict(checkpoint_epoch=completed,
            protocol="fixed_row_seed_arithmetic_v27", weights="raw", rows=rows, metrics=metrics))
    return metrics
