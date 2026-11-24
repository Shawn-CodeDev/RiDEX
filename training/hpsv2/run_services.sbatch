#!/bin/bash
#SBATCH --job-name=ridex-hps-reward
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=2-02:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Use sbatch}"
cd "${RIDEX_ROOT:?Source site.env}"
source site.env
export PYTHONPATH="$RIDEX_ROOT" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec "$PYTHON" scripts/manage_services.py "${1:?Fresh readiness path}"
