#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Scheduler allocation required}"
cd "$RIDEX_ROOT"
output="${1:?Shared output directory}"
config="${2:?Recipe configuration}"
shift 2
"$PYTHON" scripts/wait_services.py external
exec "$PYTHON" -m torch.distributed.run --nnodes="$SLURM_NNODES" \
  --node_rank="$SLURM_NODEID" --nproc_per_node="$GPUS_PER_NODE" \
  --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" --max_restarts=0 \
  -m ridex.train --config "$config" --model "$SD3_MODEL" \
  --reward-models "$REWARD_MODEL_ROOT" --dataset "${PROMPT_DATASET:?Set PROMPT_DATASET in site.env}" --output "$output" \
  --wandb-project "$WANDB_PROJECT" --wandb-mode "${WANDB_MODE:-online}" "$@"
