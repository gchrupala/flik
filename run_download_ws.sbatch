#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --job-name=flik-qc-download
#SBATCH --output=logdir/qc-download-%j.out
#SBATCH --error=logdir/qc-download-%j.err

# Download the 20 QC pilot films from Internet Archive (network job,
# low CPU so it can share the node with other jobs).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logdir

uv run --extra cpu python -m scripts.download_qc_films \
    --plan data/qc_pilot_downloads.json

echo "=== Downloads complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
