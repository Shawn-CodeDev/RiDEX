import argparse
import datetime
import os
import time
import traceback
import json
import fcntl
import importlib.metadata
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from ridex.config import Config
from ridex.runtime import create_model, joint_adam_step
from ridex.data import epoch_plan, load_dataset
from ridex.checkpoint import save_checkpoint, load_checkpoint
from ridex.nft import clean_advantages, old_policy_decay
from ridex.rollout import ActionCreditCollector, directional_loss
from ridex.rollout_base import TerminalCollector
from ridex.training import (select_policy, latent_from_seed, clean_gradient,
    reduce_gradients, gather_objects, write_json, append_json)
from ridex.evaluation import evaluate
from ridex.scoring import component_at, component_metrics
from ridex.continuation import (prepare as prepare_continuation, replay_differences, publish_epoch,
    TRAINING_HISTORY_KEYS, training_rows)


def policy_update(pipe, conditioning, params, optimizer, nft, centers, diagnostics, count, device, world):
    if count == 0:
        return joint_adam_step(params, optimizer, nft), 0., {}
    for p in params:
        p.grad = None
    select_policy(pipe, "default", training=True)
    local_loss = 0.
    before_projections = []
    for prompt, center in centers:
        embeds, pooled = conditioning.get(prompt)
        predicted = pipe.transformer(hidden_states=center.state.to(device), timestep=center.time.to(device),
            encoder_hidden_states=embeds, pooled_projections=pooled, return_dict=False)[0]
        loss = directional_loss(predicted, center.velocity.to(device), center.direction.to(device),
            center.scale.to(device), center.credit)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite directional loss")
        before_projections.append(float(((predicted.detach()-center.velocity.to(device)) *
            center.direction.to(device)).sum()/center.scale.to(device)))
        (loss * world/count).backward()
        local_loss += float(loss.detach()) * world/count
    local_grad = reduce_gradients([p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params], world)
    for p in params:
        p.grad = None
    dot = sum(float((a.double()*b.double()).sum()) for a,b in zip(nft,local_grad))
    norm_n = sum(float(a.double().square().sum()) for a in nft)**.5
    norm_l = sum(float(a.double().square().sum()) for a in local_grad)**.5
    step = joint_adam_step(params, optimizer, [a+b for a,b in zip(nft,local_grad)])
    select_policy(pipe, "default")
    with torch.no_grad():
        for index, (prompt, center) in enumerate(centers):
            embeds, pooled = conditioning.get(prompt)
            predicted = pipe.transformer(hidden_states=center.state.to(device), timestep=center.time.to(device),
                encoder_hidden_states=embeds, pooled_projections=pooled, return_dict=False)[0]
            after = float(((predicted-center.velocity.to(device))*center.direction.to(device)).sum()/center.scale.to(device))
            diagnostics[index].update(projection_before=before_projections[index], projection_after=after,
                credit_times_projection_change=center.credit*(after-before_projections[index]))
    return step, local_loss, {"gradient/nft_norm":norm_n, "gradient/local_norm":norm_l,
        "gradient/cosine":dot/max(norm_n*norm_l,1e-30)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reward-models", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-run", type=Path, help="Continue the original online W&B run from its latest complete checkpoint")
    parser.add_argument("--stop-after-updates", type=int,
                        help="Save a complete checkpoint and stop this segment at this update count")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--wandb-project", default="ridex-hpsv2")
    parser.add_argument("--dataset", help="Directory containing train.txt and test.txt")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use a scheduler allocation")
    if args.resume_run and (args.resume or args.wandb_mode != "online"):
        parser.error("--resume-run requires online mode and chooses its own checkpoint")
    rank, world, local = [int(os.environ[x]) for x in ("RANK", "WORLD_SIZE", "LOCAL_RANK")]
    if world != 16 or not torch.cuda.is_available() or "H200" not in torch.cuda.get_device_name(local).upper():
        raise RuntimeError("The released profiles require 16 NVIDIA H200 training workers")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings = json.loads(args.config.read_text())
    settings.update(model=args.model, reward_models=args.reward_models)
    if args.dataset is not None:
        settings["dataset"] = args.dataset
    if args.seed is not None:
        settings["seed"] = args.seed
    cfg = Config(**settings)
    cfg.validate(world)
    end = cfg.phase_end_update(args.stop_after_updates)
    torch.manual_seed(cfg.seed)
    scale = cfg.display_scale
    run = None
    continuation = None
    writer_lock = None
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=4))
    try:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output/"config.json", cfg.to_dict())
            if args.wandb_mode != "disabled":
                import wandb
                name = f"sd3_hpsv2_{cfg.method.lower()}_img{cfg.budget()['total_images']}_seed{cfg.seed}_{world}h200_" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y.%m.%d_%H.%M.%S")
                if args.resume_run:
                    source = args.resume_run.resolve()
                    identity = json.loads((source/"wandb_run.json").read_text())
                    run_path = identity["url"].split("wandb.ai/",1)[1].replace("/runs/","/").rstrip("/")
                    entity, project, run_id = run_path.split("/")
                    if project != args.wandb_project or run_id != identity["id"]:
                        raise ValueError("W&B identity/project mismatch")
                    lock_root = Path(__file__).resolve().parents[1]/"runs"/"writer_locks"
                    lock_root.mkdir(parents=True, exist_ok=True)
                    writer_lock = (lock_root/f"{entity}-{project}-{run_id}.lock").open("a")
                    fcntl.flock(writer_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
                    remote = wandb.Api(timeout=120).run(run_path)
                    remote_rows = training_rows(remote.scan_history(keys=list(TRAINING_HISTORY_KEYS),page_size=100))
                    continuation = prepare_continuation(source,remote_rows)
                    remote_epochs = {int(r["epoch"]) for r in remote_rows}
                    if remote_epochs != set(range(continuation["highwater_epoch"]+1)):
                        raise RuntimeError("Cloud training history has gaps; sync old data before continuing")
                    continuation["remote_last_step"] = remote.lastHistoryStep
                    write_json(args.output/"resume_context.json",continuation)
                    run = wandb.init(entity=entity,project=project,id=run_id,resume="must",
                                     dir=str(args.output),mode="online")
                    if not run.resumed or run.step <= continuation["remote_last_step"]:
                        raise RuntimeError("W&B did not resume after the last existing history step")
                    print(dict(phase="append_only_resume",run_id=run_id,**continuation),flush=True)
                else:
                    run = wandb.init(project=args.wandb_project, name=name, config=dict(cfg.to_dict(), world_size=world,
                        gpu_name=torch.cuda.get_device_name(device), resumed_from=str(args.resume) if args.resume else None,
                        phase_end_updates=end),dir=str(args.output), mode=args.wandb_mode)
                run.define_metric("epoch")
                run.define_metric("*", step_metric="epoch")
                write_json(args.output/"wandb_run.json", dict(id=run.id, url=run.url))
            write_json(args.output/"status.json", dict(phase="model_loading"))
        dist.barrier()
        contexts = [continuation]
        dist.broadcast_object_list(contexts,src=0)
        continuation = contexts[0]
        if continuation:
            args.resume = Path(continuation["checkpoint"])
        pipe, reward, conditioning, remote, model_counts, params, old, optimizer = create_model(cfg, device, world)
        collector_type = ActionCreditCollector if cfg.proposal else TerminalCollector
        collector = collector_type(pipe, reward, remote, step=cfg.action_step,
            radius=cfg.action_radius)
        prompts = load_dataset(cfg, "train")
        eval_prompts = load_dataset(cfg, "test")
        conditioning.prepare([g["prompt"] for g in epoch_plan(cfg, prompts, 0)[rank::world]])
        start = load_checkpoint(args.resume, pipe, optimizer, params, old, cfg, rank, world, device) if args.resume else 0
        if start >= end:
            raise ValueError("Requested phase end has already been reached by this checkpoint")
        if rank == 0:
            versions = {}
            for name in ("torch", "torchvision", "diffusers", "transformers", "peft", "wandb", "numpy", "hpsv2"):
                try:
                    versions[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    versions[name] = "not_in_distribution_metadata"
            write_json(args.output/"environment.json", dict(packages=versions, world_size=world,
                gpu=torch.cuda.get_device_name(device), torch_cuda=torch.version.cuda))
        gpu_seconds = continuation["prior_training_gpu_seconds"] if continuation else 0.
        for epoch in range(start, end):
            started = time.monotonic()
            collector.reset_counts()
            model_counts.clear()
            if reward is not None:
                reward._text_cache.clear()
            plan = epoch_plan(cfg, prompts, epoch)
            conditioning.prepare([g["prompt"] for g in plan[rank::world]])
            select_policy(pipe, "old")
            pipe.scheduler.set_timesteps(cfg.steps, device=device)
            sigmas = pipe.scheduler.sigmas.detach().float().clone()
            samples, centers, diagnostics, images = [], [], [], []
            if rank == 0:
                write_json(args.output/"status.json", dict(phase="collecting", epoch=epoch))
                write_json(args.output/f"plan_{epoch}.json", plan)
            for group in plan[rank::world]:
                prompt = group["prompt"]
                collector.metadata = group["metadata"]
                embeds, pooled = conditioning.get(prompt)
                for slot, seed in enumerate(group["seeds"]):
                    initial = latent_from_seed(pipe, cfg, seed, device)
                    if slot < cfg.centers_per_prompt:
                        latent, image, score, center, endpoints, scores = collector.center(
                            initial, sigmas, embeds, pooled, prompt)
                        clean_components = center.diagnostics["baseline_components"]
                        centers.append((prompt, center))
                        diagnostics.append(dict(center.diagnostics, group=group["group"], seed=seed))
                        for sign, (endpoint, value) in enumerate(zip(endpoints, scores)):
                            samples.append(dict(prompt=prompt, group=group["group"],
                                slot=cfg.clean_per_prompt+slot*2+sign, latent=endpoint,
                                reward=value, clean=False, seed=seed,
                                components=center.diagnostics[("positive_components", "negative_components")[sign]]))
                    else:
                        latent, image, score = collector.clean(initial, sigmas, embeds, pooled, prompt)
                        clean_components = component_at(collector.last_info, 0)
                    samples.append(dict(prompt=prompt, group=group["group"], slot=slot,
                        latent=latent.cpu(), reward=score, clean=True, seed=seed, components=clean_components))
                    if len(images) < 4:
                        images.append((image.cpu(), prompt, score))
                print(f"collection epoch={epoch} rank={rank} samples={len(samples)}", flush=True)
            shards = gather_objects([{k:v for k,v in s.items() if k != "latent"} for s in samples], world)
            records = [row for shard in shards for row in shard]
            advantages = clean_advantages([r["group"] for r in records], [r["reward"] for r in records])
            offset = sum(len(s) for s in shards[:rank])
            if rank == 0:
                write_json(args.output/"status.json", dict(phase="training", epoch=epoch))
            train_cfg = replace(cfg, clean_per_prompt=cfg.clean_per_prompt+2*cfg.centers_per_prompt)
            nft, losses = clean_gradient(pipe, conditioning, train_cfg, samples,
                advantages[offset:offset+len(samples)], sigmas, params, epoch, device, world,
                global_sample_count=len(records))
            count = cfg.prompts_per_epoch * cfg.centers_per_prompt
            step, local_loss, gradient_stats = policy_update(pipe, conditioning, params, optimizer,
                nft, centers, diagnostics, count, device, world)
            del nft, samples
            completed = epoch+1
            decay = old_policy_decay(completed)
            with torch.no_grad():
                for p,o in zip(params,old):
                    o.mul_(decay).add_(p, alpha=1-decay)
            elapsed = gather_objects(time.monotonic()-started, world)
            gpu_seconds += sum(elapsed)
            loss_rows = gather_objects(dict(losses, directional_loss=local_loss), world)
            pair_rows = [r for rows in gather_objects(diagnostics, world) for r in rows] if count else []
            if len(pair_rows) != count:
                raise RuntimeError("Directional credit count mismatch")
            costs = gather_objects(collector.counts, world)
            model_costs = gather_objects(model_counts, world)
            budget = cfg.budget()
            actual = sum(c["decoded_images"] for c in costs)
            if actual != budget["total_images"] or len(records) != actual:
                raise RuntimeError("Collection budget mismatch")
            if sum(c["canonical_scorer_images"] for c in costs) != budget["canonical_scorer_images"]:
                raise RuntimeError("Canonical scorer cost mismatch")
            if sum(c["action_backwards"] for c in costs) != budget["direction_backward_calls"]:
                raise RuntimeError("Proposal backward budget mismatch")
            metrics = dict(epoch=epoch, sample_epoch=epoch, completed_updates=completed, optimizer_step=completed,
                reward_avg=np.mean([r["reward"] for r in records if r["clean"]])*scale,
                learning_rate=cfg.learning_rate, train_keep_rate=1., main_loss_denominator=len(records),
                old_policy_decay=decay,
                **{k:float(np.mean([r[k] for r in loss_rows])) for k in loss_rows[0]})
            metrics[f"reward_{cfg.reward}"] = metrics["reward_avg"]
            metrics.update(component_metrics([r for r in records if r["clean"]], "reward"))
            metrics.update(component_metrics(records, "reward/all_train"))
            metrics.update({f"cost/{k}":v for k,v in budget.items()})
            metrics.update({f"work/{k}":sum(c[k] for c in costs) for k in costs[0]})
            metrics.update({f"work/{k}":sum(c.get(k,0) for c in model_costs)
                for k in ("all_transformer_forwards","transformer_block_forwards_including_recompute")})
            metrics.update({"work/nft_loss_backwards":len(records)*cfg.steps,"work/local_loss_backwards":count})
            metrics.update({f"joint_step/{k}":v for k,v in step.items()})
            metrics.update({"cost/epoch_wall_seconds":max(elapsed), "cost/cumulative_training_gpu_seconds":gpu_seconds,
                "cost/training_gpu_seconds":sum(elapsed), "reward/clean_train":metrics["reward_avg"],
                "reward/all_train":np.mean([r["reward"] for r in records])*scale, **gradient_stats})
            if pair_rows:
                nonzero = sum(r["credit"] != 0 for r in pair_rows)
                positive_gain = float(np.mean([r["positive"]-r["baseline"] for r in pair_rows]))
                credit_abs = float(np.mean([abs(r["credit"]) for r in pair_rows]))
                realized = float(np.mean([r["credit_times_projection_change"] for r in pair_rows]))
                metrics.update({
                    "direction/nonzero_credit_rate": nonzero/count,
                    "direction/positive_gain_mean": positive_gain*scale,
                    "direction/negative_gain_mean": float(np.mean([r["negative"]-r["baseline"] for r in pair_rows]))*scale,
                    "direction/credit_abs_mean": credit_abs,
                    "direction/hpsv2_credit_abs_mean": credit_abs,
                    "direction/credit_times_projection_change": realized,
                    "direction/realized_sign_agreement": sum(r["credit_times_projection_change"]>0 for r in pair_rows)/max(1,nonzero),
                    "direction/terminal_gradient/centers": count,
                    "direction/terminal_gradient/credit_mean": float(np.mean([r["credit"] for r in pair_rows])),
                    "direction/terminal_gradient/credit_abs_mean": credit_abs,
                    "direction/terminal_gradient/positive_gain_mean": positive_gain,
                    "direction/terminal_gradient/credit_times_projection_change": realized,
                })
            if continuation:
                save_checkpoint(pipe, optimizer, params, old, cfg, args.output, completed, rank, world, device=device)
            if rank == 0:
                write_json(args.output/f"round_{epoch}.json", dict(records=records,pairs=pair_rows,metrics=metrics))
                if not publish_epoch(epoch,continuation):
                    differences = replay_differences(metrics,continuation["replay_expected"][str(epoch)])
                    append_json(args.output/"replay_metrics.jsonl",dict(metrics,replay_differences=differences))
                else:
                    append_json(args.output/"metrics.jsonl", metrics)
                print(metrics, flush=True)
                if run and publish_epoch(epoch,continuation):
                    run.log(metrics)
                    if completed == 1 or completed%cfg.save_freq == 0:
                        run.log({"epoch":epoch,"images":[wandb.Image(im[0],caption=f"{p} | {s*scale:.4f}") for im,p,s in images]})
            if not continuation and (completed%cfg.save_freq == 0 or completed == end):
                save_checkpoint(pipe, optimizer, params, old, cfg, args.output, completed, rank, world, device=device)
            if centers and (completed == 1 or completed%cfg.save_freq == 0):
                torch.save([dict(prompt=p,state=c.state,velocity=c.velocity,direction=c.direction,
                    scale=c.scale,time=c.time,credit=c.credit,diagnostics=c.diagnostics) for p,c in centers],
                    args.output/f"centers_epoch{epoch}_rank{rank}.pt")
            if completed%cfg.eval_freq == 0 and publish_epoch(epoch,continuation):
                eval_started = time.monotonic()
                evaluation = eval_prompts
                eval_metrics = evaluate(pipe, conditioning, collector, cfg, evaluation, args.output, completed, rank, world)
                if rank == 0:
                    values = dict(epoch=epoch, optimizer_step=completed, eval_checkpoint_epoch=completed,
                        **eval_metrics, eval_weights="raw",eval_images_count=len(evaluation),
                        eval_seconds=time.monotonic()-eval_started,eval_protocol="fixed_row_seed_arithmetic_v27")
                    append_json(args.output/"metrics.jsonl",values)
                    if run:
                        run.log(values)
        if rank == 0:
            completion = dict(completed_updates=end, planned_updates=cfg.epochs,
                              training_complete=end == cfg.epochs)
            marker = "completed.json" if completion["training_complete"] else "phase_complete.json"
            write_json(args.output/marker,completion)
            write_json(args.output/"status.json", dict(completion,
                phase="completed" if completion["training_complete"] else "phase_complete"))
            if run:
                run.summary.update(completion)
    except BaseException as error:
        if args.output.exists():
            write_json(args.output/f"failure_rank{rank}.json",dict(error=repr(error),traceback=traceback.format_exc()))
        traceback.print_exc()
        os._exit(1)
    finally:
        if run:
            run.finish()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
