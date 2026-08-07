#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=gpu_a100
#SBATCH --time=24:00:00
#SBATCH --job-name=flik-train
#SBATCH --output=logdir/train-%j.out

# ====================================================================
# Flik training on Snellius (A100, multi-node DDP)
#
# Configurations:
#   2 nodes × 4 GPUs = 8 GPUs (default):  --nodes=2
#   1 node  × 4 GPUs = 4 GPUs:           change to --nodes=1, --time=72:00:00
#
# SBU estimate (gpu_a100, 128 SBU/GPU-hr):
#   8 GPUs × 72h = 73,728 SBUs  (2 nodes)
#   4 GPUs × 72h = 36,864 SBUs  (1 node)
#
# Estimated training time: ~64h for 50 epochs (17,800 steps @ ~13s/step)
# Walltime 72h gives ~8h buffer for eval + checkpointing + startup.
# ====================================================================

set -euo pipefail

source snellius_modules

cd "${SLURM_SUBMIT_DIR}"

# Activate venv (must run `uv sync --extra cu128` before first submission)
source .venv/bin/activate

# CUDA libraries (cuDNN, cuBLAS) from pip-installed nvidia packages
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cudnn/lib')"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cublas/lib')"):$LD_LIBRARY_PATH

# Thread control: GPU-bound training, prevent oversubscription
export OMP_NUM_THREADS=1
export OPENCV_FFMPEG_THREADS=1
ulimit -u 65536 2>/dev/null || true

# Reduce PyTorch CUDA memory fragmentation over long runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Job Info ==="
echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $SLURM_JOB_NUM_NODES"
echo "GPUs/node:    $SLURM_GPUS_ON_NODE"
echo "Total GPUs:   $((SLURM_JOB_NUM_NODES * SLURM_GPUS_ON_NODE))"
echo "Working dir:  $SLURM_SUBMIT_DIR"
echo "Start time:   $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# Pre-download HuggingFace models BEFORE launching torchrun.
# This avoids the GPFS cache race condition where multiple ranks call
# from_pretrained() simultaneously on the same cache directory.
# (The training script also has a rank-0-first barrier, but pre-downloading
# here is even safer - the models are cached before any DDP process starts.)
echo "=== Pre-downloading HuggingFace models ==="
python -c "
from transformers import Wav2Vec2Model, VideoMAEModel
Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base')
VideoMAEModel.from_pretrained('MCG-NJU/videomae-base')
print('Model download complete.')
"
echo ""

# Multi-node torchrun with rendezvous
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "=== Launching DDP training ==="
echo "MASTER_ADDR:  $MASTER_ADDR"
echo "MASTER_PORT:  $MASTER_PORT"
echo "Config:       multinode"
echo ""

torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc-per-node="$SLURM_GPUS_ON_NODE" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m scripts.train_hydra --config-name=multinode

echo ""
echo "=== Training complete ==="
echo "End time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
