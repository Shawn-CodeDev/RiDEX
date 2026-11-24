from pathlib import Path
import torch


def load_dataset(cfg, split):
    path = Path(cfg.dataset) / f"{split}.txt"
    records = [dict(prompt=line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Empty {split} split")
    return records


def epoch_plan(config, records, epoch):
    if len(records) < config.prompts_per_epoch:
        raise ValueError("Too few dataset rows")
    generator = torch.Generator().manual_seed(config.seed + epoch)
    ids = torch.randperm(len(records), generator=generator)[:config.prompts_per_epoch].tolist()
    seeds = torch.randint(0, 2**31, (len(ids), config.clean_per_prompt), generator=generator).tolist()
    return [dict(group=group, prompt_id=row, prompt=records[row]["prompt"],
                 metadata=records[row], seeds=seeds[group]) for group, row in enumerate(ids)]
