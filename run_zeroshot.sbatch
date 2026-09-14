#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=gpu_a100
#SBATCH --time=01:00:00
#SBATCH --job-name=flik-zeroshot-vggsound
#SBATCH --output=logdir/zeroshot-vggsound-%j.out
#SBATCH --error=logdir/zeroshot-vggsound-%j.err

# ====================================================================
# Zero-shot VGGSound retrieval eval on Snellius (1x A100).
#
# PREREQUISITE (login node, internet required):
#   uv run --extra cpu python -m scripts.download_vggsound_subset \
#       --classes 40 --per-class 15 --out-dir data/vggsound_subset
#   -> writes data/vggsound_subset/clips_manifest.json used below.
#
# HuggingFace encoders (wav2vec2-base, videomae-base) must be present in
# <repo>/cache. They are normally warmed by the training job; compute nodes
# have no internet, so HF_HUB_OFFLINE=1 is set below.
#
# SBU estimate (gpu_a100, 128 SBU/GPU-hr): 1 GPU x 1h = 128 SBU.
# ====================================================================

set -euo pipefail

source snellius_modules

cd "${SLURM_SUBMIT_DIR}"

# HF cache: must match scripts/train_hydra.py (<repo>/cache).
export HF_HOME="${SLURM_SUBMIT_DIR}/cache"
# Compute nodes have no outbound internet; use the warm shared cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Checkpoint loading: the .pth dicts contain config/RNG state, not just tensors.
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true

# CUDA libraries (cuDNN, cuBLAS) from pip-installed nvidia packages.
export LD_LIBRARY_PATH=$(uv run --extra cu128 python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cudnn/lib')"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(uv run --extra cu128 python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cublas/lib')"):$LD_LIBRARY_PATH

# Thread control: eval is GPU-bound; keep CPU decode single-threaded.
export OMP_NUM_THREADS=1
export OPENCV_FFMPEG_THREADS=1
ulimit -u 65536 2>/dev/null || true

echo "=== Zero-shot VGGSound eval ==="
echo "Job ID:      $SLURM_JOB_ID"
echo "GPU:         1"
echo "Working dir: $SLURM_SUBMIT_DIR"
echo "Start time:  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

uv run --extra cu128 python -m scripts.eval_zeroshot_vggsound \
    --checkpoint aug11_final:checkpoints/checkpoint_epoch50.pth \
    --checkpoint aug13_best:checkpoints/checkpoint_best.pth \
    --random-init \
    --clips-manifest data/vggsound_subset/clips_manifest.json \
    --out zeroshot_results.json

echo ""
echo "=== Eval complete ==="
echo "End time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
