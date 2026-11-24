#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
usage() { echo 'Usage: bash submit.sh {--dry-run|--submit} {ridex|diffusionnft} [--stop-after-updates N | --resume-run OUTPUT]'; }
mode=${1:---dry-run}
profile=${2:-ridex}
case "$profile" in
  ridex) config=configs/hpsv2.json; budget='16 prompts x (32 clean + 4 paired endpoints) = 576 imgs/epoch' ;;
  diffusionnft) config=configs/diffusionnft_hpsv2.json; budget='48 prompts x 24 clean = 1152 imgs/epoch' ;;
  *) usage; exit 2 ;;
esac
case "$mode" in
  --dry-run)
    printf 'Profile: %s\nConfig: %s\nTraining: 16 H200 workers\nBudget: %s\nReward: HPSv2.1\n' "$profile" "$config" "$budget"
    usage
    exit 0 ;;
  --submit) [[ $# -ge 2 ]] || { usage; exit 2; }; shift 2 ;;
  *) usage; exit 2 ;;
esac
source site.env
: "${RIDEX_ROOT:?}" "${TRAIN_NODES:?}" "${GPUS_PER_NODE:?}" "${TRAIN_PARTITION:?}" "${H200_GRES_TYPE:?}"
: "${PROMPT_DATASET:?}" "${SD3_MODEL:?}" "${REWARD_MODEL_ROOT:?}"
[[ -x "$PYTHON" ]] || { echo 'Set PYTHON to the CUDA environment interpreter' >&2; exit 2; }
[[ -d "$SD3_MODEL" && -s "$PROMPT_DATASET/train.txt" && -s "$PROMPT_DATASET/test.txt" ]] || {
  echo 'Set valid shared SD3_MODEL and PROMPT_DATASET paths before submitting' >&2
  exit 2
}
[[ -s "$REWARD_MODEL_ROOT/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin" &&
   -s "$REWARD_MODEL_ROOT/HPSv2/HPS_v2.1_compressed.pt" ]] || {
  echo 'Missing HPSv2 or OpenCLIP checkpoint under REWARD_MODEL_ROOT' >&2
  exit 2
}
[[ "$TRAIN_NODES" =~ ^[1-9][0-9]*$ && "$GPUS_PER_NODE" =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $((TRAIN_NODES*GPUS_PER_NODE)) == 16 ]] || { echo 'Expected 16 training H200 GPUs'; exit 2; }
extra=()
if (($#)); then
  [[ $# == 2 ]] || { usage; exit 2; }
  case "$1" in
    --resume-run)
      [[ -d "$2" && "${WANDB_MODE:-online}" == online ]] || { echo 'Resume requires an existing output and online W&B'; exit 2; }
      extra=(--resume-run "$(cd -- "$2" && pwd)") ;;
    --stop-after-updates)
      [[ "$2" =~ ^[1-9][0-9]*$ && "$2" -le 500 ]] || { echo 'Expected 1 to 500 updates'; exit 2; }
      extra=("$@") ;;
    *) usage; exit 2 ;;
  esac
fi
site_options=()
[[ -z "${ACCOUNT:-}" ]] || site_options+=(--account="$ACCOUNT")
[[ -z "${QOS:-}" ]] || site_options+=(--qos="$QOS")
if [[ -z "${HPSV2_REWARD_URLS:-}" ]]; then
  : "${SERVICE_PARTITION:?Set SERVICE_PARTITION or HPSV2_REWARD_URLS}"
fi
launch="$RIDEX_ROOT/runs/${profile}_$(date -u +%Y%m%dT%H%M%S)_$$"
mkdir -p "$launch"
output="$launch/output"
ready="$launch/ready.json"
[[ -z "${HPSV2_REWARD_URLS:-}" ]] || ready=external
submitted=()
rollback() { ((${#submitted[@]} == 0)) || scancel "${submitted[@]}" || true; }
trap rollback ERR
train=$(sbatch --parsable --partition="$TRAIN_PARTITION" "${site_options[@]}" --job-name="${profile}-hpsv2-h200" \
  --nodes="$TRAIN_NODES" --ntasks-per-node=1 --cpus-per-task="$GPUS_PER_NODE" --gres="gpu:$H200_GRES_TYPE:$GPUS_PER_NODE" \
  --mem="${TRAIN_MEM_PER_NODE:-512G}" --time="${TRAIN_TIME_LIMIT:-2-00:00:00}" \
  --export=ALL --output="$launch/train_%j.log" run_train.sbatch "$ready" "$output" "$config" "${extra[@]}")
train="${train%%;*}"
submitted+=("$train")
server=external
if [[ "$ready" != external ]]; then
  server=$(sbatch --parsable --partition="$SERVICE_PARTITION" "${site_options[@]}" \
    --time="${SERVICE_TIME_LIMIT:-2-02:00:00}" \
    --export=ALL --dependency="after:$train" --kill-on-invalid-dep=yes \
    --output="$launch/services_%j.log" run_services.sbatch "$ready")
  server="${server%%;*}"
  submitted+=("$server")
  cleanup=$(sbatch --parsable --partition="$SERVICE_PARTITION" "${site_options[@]}" \
    --dependency="afterany:$train" --kill-on-invalid-dep=yes \
    --output="$launch/cleanup_%j.log" cleanup.sbatch "$server")
  submitted+=("${cleanup%%;*}")
fi
printf 'train=%s\nserver=%s\noutput=%s\n' "$train" "$server" "$output" | tee "$launch/jobs.txt"
trap - ERR
