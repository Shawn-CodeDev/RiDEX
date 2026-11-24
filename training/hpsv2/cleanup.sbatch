#!/bin/bash
#SBATCH --job-name=ridex-service-cleanup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
set -euo pipefail
[[ "${1:-}" =~ ^[0-9]+$ ]] || exit 2
scancel "$1"
