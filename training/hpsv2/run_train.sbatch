#!/bin/bash
#SBATCH --no-requeue
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch}"
cd "${RIDEX_ROOT:?Source site.env}"
source site.env
export PYTHONPATH="$RIDEX_ROOT" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HPSV2_REWARD_BATCH_SIZE=8
[[ $((SLURM_NNODES*GPUS_PER_NODE)) == 16 ]] || { echo 'Expected 16 training H200 workers'; exit 2; }
ready="${1:?Fresh readiness JSON or external}"
output="${2:?Shared output directory}"
config="${3:?Recipe configuration}"
shift 3
"$PYTHON" scripts/wait_services.py "$ready"
if [[ "$ready" != external ]]; then
    export HPSV2_REWARD_URLS
    HPSV2_REWARD_URLS=$("$PYTHON" -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["urls"]["hpsv2"]))' "$ready")
fi
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((20000+SLURM_JOB_ID%20000))
srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 --kill-on-bad-exit=1 \
    bash scripts/node_train.sh "$output" "$config" "$@"
