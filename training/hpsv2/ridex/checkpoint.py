from pathlib import Path
import torch
import torch.distributed as dist

from ridex.training import gather_objects, write_json


def parameter_names(pipe, parameters):
    names = {id(value): name for name, value in pipe.transformer.named_parameters()}
    return [names[id(value)] for value in parameters]


def save_checkpoint(pipe, optimizer, parameters, old_parameters, config, output, completed,
                    rank, world_size, *, device):
    rng = {"cpu": torch.get_rng_state(),
           "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None}
    states = gather_objects(rng, world_size)
    path = Path(output) / "checkpoints" / f"checkpoint-{completed}"
    if rank == 0:
        path.mkdir(parents=True, exist_ok=False)
        pipe.transformer.save_pretrained(path / "lora", selected_adapters=["default"], safe_serialization=True)
        torch.save({"parameters": [p.detach().cpu() for p in parameters],
                    "old_parameters": [p.detach().cpu() for p in old_parameters],
                    "optimizer": optimizer.state_dict(), "completed_updates": completed,
                    "parameter_names": parameter_names(pipe, parameters), "config": config.to_dict(),
                    "world_size": world_size, "rank_rng": states}, path / "training_state.pt")
        write_json(path / "complete.json", {"version": f"ifanft_{config.method}",
            "completed_updates": completed, "weight_kind": "raw", "world_size": world_size})
    if world_size > 1:
        dist.barrier()
    return path


def load_checkpoint(path, pipe, optimizer, parameters, old_parameters, config, rank, world_size, device):
    path = Path(path)
    if not (path / "complete.json").is_file():
        raise ValueError("Incomplete checkpoint")
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=True)
    expected = config.to_dict()
    for key, value in expected.items():
        if key != "epochs" and state["config"].get(key) != value:
            raise ValueError(f"Checkpoint configuration differs at {key}")
    if state["world_size"] != world_size or state["parameter_names"] != parameter_names(pipe, parameters):
        raise ValueError("Checkpoint world size or parameter identities differ")
    if not 0 <= state["completed_updates"] < config.epochs:
        raise ValueError("Checkpoint has no remaining requested updates")
    with torch.no_grad():
        for targets, values in ((parameters, state["parameters"]), (old_parameters, state["old_parameters"])):
            if len(targets) != len(values):
                raise ValueError("Checkpoint parameter count differs")
            for target, value in zip(targets, values, strict=True):
                if target.shape != value.shape or target.dtype != value.dtype or not torch.isfinite(value).all():
                    raise ValueError("Checkpoint parameter shape, dtype or value is invalid")
                target.copy_(value.to(device))
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["rank_rng"][rank]["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["rank_rng"][rank]["cuda"], device)
    return int(state["completed_updates"])
