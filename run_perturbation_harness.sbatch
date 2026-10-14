#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --mem=30G
#SBATCH --job-name=flik-perturb
#SBATCH --chdir=$HOME/work_dir/flik
#SBATCH --output=logs/perturb-%j.out
#SBATCH --error=logs/perturb-%j.err

# ====================================================================
# Real perturbation-ladder run: validate the image-text alignment
# instrument on matched vs shuffled WINDOW pairs.
#
# PREREQUISITES
#   - data/qc_pilot_manifest.json (or another window manifest) exists and
#     its video_path entries resolve on this machine.
#   - The HF model is cached locally (the node may have no internet):
#       openai/clip-vit-base-patch32
#   - logs/ must exist before sbatch opens the output file (tracked via
#     logs/.gitkeep).  perturbation_results/ is created by the script.
#
# ACCELERATOR NOTE (GTX 1080 Ti / Pascal, sm_61)
#   The default ACCEL=cu128 wheel is built for sm_70+ and has no sm_60
#   kernels, so on the workstation it fails with "no kernel image".  If
#   this job targets that GPU, run with ACCEL=cpu (slower but correct):
#       sbatch --export=ALL,ACCEL=cpu run_perturbation_harness.sbatch
#   On a non-Pascal GPU leave ACCEL=cu128.
# ====================================================================

set -euo pipefail

ACCEL="${ACCEL:-cu128}"
MANIFEST="${MANIFEST:-data/qc_pilot_manifest.json}"
OUT="${OUT:-perturbation_results/clip_openai__clip-vit-base-patch32.json}"

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs perturbation_results

# Thread control: inference is GPU-bound; keep CPU decode/OpenCV modest.
export OMP_NUM_THREADS=1
export OPENCV_FFMPEG_THREADS=1
ulimit -u 65536 2>/dev/null || true

echo "=== Perturbation harness (real run) ==="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Partition:  ${SLURM_JOB_PARTITION}"
echo "CPUs:       ${SLURM_CPUS_PER_TASK:-12}"
echo "Accel:      ${ACCEL}"
echo "Manifest:   ${MANIFEST}"
echo "Work dir:   ${SLURM_SUBMIT_DIR}"
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

uv run --extra "${ACCEL}" python -m scripts.perturbation_harness \
    --manifest "${MANIFEST}" \
    --model openai/clip-vit-base-patch32 \
    --instrument clip \
    --n-windows 2000 \
    --n-films 200 \
    --frames 8 \
    --batch-size 256 \
    --device cuda \
    --seed 42 \
    --workers 12 \
    --out "${OUT}"

echo ""
echo "=== Perturbation harness complete ==="
echo "Report:     ${OUT} (and matching .md)"
echo "End time:   $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
