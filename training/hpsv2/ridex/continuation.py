import json
import math
from pathlib import Path

TRAINING_HISTORY_KEYS = ("epoch", "policy_loss", "reward_avg", "reward_hpsv2", "kl_div")


def training_rows(rows):
    return [row for row in rows if row.get("epoch") is not None and row.get("policy_loss") is not None]


def history(source):
    source = Path(source)
    context = source / "resume_context.json"
    rows = history(json.loads(context.read_text())["source"]) if context.exists() else []
    metrics = source / "metrics.jsonl"
    if metrics.exists() or not context.exists():
        with metrics.open() as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def prepare(source, remote_rows):
    source = Path(source).resolve()
    remote_rows = training_rows(remote_rows)
    rows = history(source)
    local_train = {int(r["epoch"]):r for r in rows if "policy_loss" in r}
    if not local_train:
        raise ValueError("No completed training history to continue")
    remote_epochs = [int(r["epoch"]) for r in remote_rows if r.get("epoch") is not None]
    highwater = max(max(local_train), max(remote_epochs, default=-1))
    if highwater != max(local_train):
        raise ValueError("Remote history is ahead of the supplied source; refusing ambiguous continuation")
    if len(remote_epochs) != len(set(remote_epochs)):
        raise ValueError("Duplicate remote training epochs")
    for row in remote_rows:
        epoch = int(row["epoch"])
        for key in TRAINING_HISTORY_KEYS[1:]:
            if key in row and not math.isclose(float(row[key]), float(local_train[epoch][key]),
                                              rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError(f"Cloud/local conflict at epoch {epoch}, metric {key}")
    checkpoints = []
    for path in (source / "checkpoints").glob("checkpoint-*"):
        marker = path / "complete.json"
        if marker.exists() and (path / "training_state.pt").exists():
            count = int(json.loads(marker.read_text())["completed_updates"])
            if count <= highwater+1:
                checkpoints.append((count,path))
    if not checkpoints:
        raise ValueError("No complete checkpoint compatible with history")
    completed, checkpoint = max(checkpoints)
    expected = {str(e):local_train[e] for e in range(completed,highwater+1)}
    prior_cost = float(local_train[highwater]["cost/cumulative_training_gpu_seconds"])
    restored_round = source / f"round_{completed-1}.json"
    if restored_round.exists():
        prior_cost = max(prior_cost, float(json.loads(restored_round.read_text())["metrics"]
            ["cost/cumulative_training_gpu_seconds"]))
    return dict(source=str(source), checkpoint=str(checkpoint), checkpoint_updates=completed,
        highwater_epoch=highwater, replay_expected=expected,
        prior_training_gpu_seconds=prior_cost,
        logged_eval_updates=sorted({int(r["eval_checkpoint_epoch"]) for r in rows if "eval_checkpoint_epoch" in r}))


def replay_differences(metrics, expected):
    keys = [key for key,value in expected.items() if isinstance(value,(int,float))
            and isinstance(metrics.get(key),(int,float))
            and not (key.startswith("cost/") and "seconds" in key)]
    return {key:float(metrics[key])-float(expected[key]) for key in keys}


def publish_epoch(epoch, context):
    return context is None or int(epoch) > context["highwater_epoch"]
