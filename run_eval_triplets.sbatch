#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --job-name=flik-eval-triplets
#SBATCH --output=logdir/eval-triplets-%j.out
#SBATCH --error=logdir/eval-triplets-%j.err

# ====================================================================
# Real 2AFC triplet eval (Peppa Pig design) where the films live.
# Do NOT submit from a machine without the corpus and the warm HF cache.
#
# PREREQUISITES (login node, before submitting):
#   - data/expanded_manifest_segments.json exists (segment-level manifest).
#   - The films live under data/out/ (the manifest's video_path values).
#   - HuggingFace encoders (wav2vec2-base, videomae-base) are in <repo>/cache;
#     compute nodes have no internet, so HF_HUB_OFFLINE=1 is set below.
#   - Checkpoint(s) exist. Adjust CKPT_* below to the run you want to probe.
#
# Held-out films: with no --holdout-films the script reuses
# _split_manifest_by_video (scripts/train_hydra.py) at val-ratio 0.1 / seed 42,
# so triples come only from films the training split holds out. Pass
# --holdout-films FILE to pin an explicit list instead.
#
# Conditions: cross_film, same_film, and speaker_matched. The corpus has no
# speaker labels, so speaker_matched is OMITTED and reported as omitted. To
# enable it, generate a --speaker-map first (pyannote diarization or MFCC
# clustering) and add the flag.
#
# SBU estimate (gpu_a100, 128 SBU/GPU-hr): 1 GPU x 4h = 512 SBU.
# ====================================================================

set -euo pipefail

source snellius_modules

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logdir

# Per-run settings (edit these).
CKPT_NAME="aug13_best"
CKPT_PATH="checkpoints/checkpoint_best.pth"
MANIFEST="data/expanded_manifest_segments.json"
N_TRIALS=4000

# HF cache: must match scripts/train_hydra.py (<repo>/cache).
export HF_HOME="${SLURM_SUBMIT_DIR}/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Checkpoint loading: the .pth dicts contain config/RNG state, not just tensors.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

# CUDA libraries (cuDNN, cuBLAS) from pip-installed nvidia packages.
export LD_LIBRARY_PATH=$(uv run --extra cu128 python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cudnn/lib')"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(uv run --extra cu128 python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cublas/lib')"):$LD_LIBRARY_PATH

# Eval is GPU-bound; keep decode processes from over-subscribing the cores.
export OMP_NUM_THREADS=1
export OPENCV_FFMPEG_THREADS=1
ulimit -u 65536 2>/dev/null || true

echo "=== Flik triplet (2AFC) eval ==="
echo "Job ID:      $SLURM_JOB_ID"
echo "Manifest:    $MANIFEST"
echo "Checkpoint:  $CKPT_NAME:$CKPT_PATH"
echo "n trials:    $N_TRIALS"
echo "Start time:  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

uv run --extra cu128 python -m scripts.eval_triplets \
    --checkpoint "${CKPT_NAME}:${CKPT_PATH}" \
    --random-init \
    --manifest "${MANIFEST}" \
    --n-trials "${N_TRIALS}" \
    --duration-bins 10 \
    --bootstrap-iterations 2000 \
    --val-ratio 0.1 --split-seed 42 \
    --batch-size 32 --workers 8 \
    --device cuda \
    --out triplet_eval_results.json \
    --markdown-out triplet_eval_results.md

echo ""
echo "=== Eval complete ==="
echo "End time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Results:  triplet_eval_results.json / .md"
