#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Scheduler allocation required}"
source "$RIDEX_ROOT/site.env"
exec "$PYTHON" -m ridex.reward_server \
  --host 0.0.0.0 --port "${1:?Port}" --device cuda \
  --openclip-checkpoint "$REWARD_MODEL_ROOT/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin" \
  --hps-checkpoint "$REWARD_MODEL_ROOT/HPSv2/HPS_v2.1_compressed.pt"
