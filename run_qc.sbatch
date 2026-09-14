#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=genoa
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --job-name=flik-qc
#SBATCH --output=logdir/qc-%j.out
#SBATCH --error=logdir/qc-%j.err

# CPU-only data-quality screening of the full training manifest.
# Phase 1 reads every segment. Phase 2 decodes a stratified sample.
# No GPU is needed: decode uses OpenCV (video) and PyAV (audio).

set -euo pipefail
module purge && module load 2025
module load FFmpeg/7.1.1-GCCcore-14.2.0
cd "${SLURM_SUBMIT_DIR}"

source .venv/bin/activate
export FLIK_DATA_ROOT="${SLURM_SUBMIT_DIR}/data"

# Per-thread caps: 32 worker threads decode in parallel. Keep BLAS/OpenCV
# single-threaded so 72 cores are not oversubscribed.
export OMP_NUM_THREADS=1
export OPENCV_FFMPEG_THREADS=1
ulimit -u 65536 2>/dev/null || true

echo "=== Data-quality screening ==="
echo "Job ID:     $SLURM_JOB_ID"
echo "Partition:  $SLURM_JOB_PARTITION"
echo "CPUs:       ${SLURM_CPUS_PER_TASK:-72}"
echo "Work dir:   $SLURM_SUBMIT_DIR"
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

uv run --extra cu128 python -m scripts.qc_segments \
    --manifest data/expanded_manifest_segments.json \
    --total-samples 5000 \
    --workers 32 \
    --seed 42 \
    --out-dir qc_results/

echo ""
echo "=== QC complete ==="
echo "End time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
