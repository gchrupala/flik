#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --job-name=flik-qc-ws
#SBATCH --output=logdir/qc-ws-%j.out
#SBATCH --error=logdir/qc-ws-%j.err

# ====================================================================
# QC screening pilot on the local cuda-workstation (CPU-only).
#
# PREREQUISITE: data/qc_pilot_manifest.json — a subset of the real
# 307k-window expanded manifest restricted to films downloaded under
# data/out/cinedantan/ (built by the orchestrator-side prep step after
# the cluster manifest is synced). Video paths must resolve locally.
# ====================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logdir

export OMP_NUM_THREADS=1
export OPENCV_FFMPEG_THREADS=1

echo "=== QC screening pilot (workstation) ==="
echo "Job ID:     $SLURM_JOB_ID"
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

uv run --extra cpu python -m scripts.qc_segments \
    --manifest data/qc_pilot_manifest.json \
    --total-samples 4000 \
    --workers 10 \
    --seed 42 \
    --out-dir qc_results_ws/

echo "=== QC complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
