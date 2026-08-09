#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --partition=gpu_a100
#SBATCH --time=24:00:00
#SBATCH --job-name=flik-train
#SBATCH --output=logdir/train-%j.out
#SBATCH --error=logdir/train-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=g.shen@tilburguniversity.edu

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
# on 4 GPUs; roughly half that on 8 GPUs.
#
# PITFALL: a SLURM batch script executes ONLY on the first allocated
# node. Multi-node launches MUST go through srun (see bottom of script)
# or the extra nodes idle and the rendezvous times out.
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

# NCCL network config (required for multi-node on Snellius).
# NCCL's bootstrap TCP connections must go over the high-speed node
# interconnect, not the management interface. "eno" prefix-matches the
# Snellius compute-node interface (eno1). Without this, NCCL can hang at
# init with no useful error. NCCL_DEBUG=INFO logs the chosen interface
# and transport (IB vs TCP) at startup - check it on the first run.
export NCCL_SOCKET_IFNAME="eno"
export NCCL_DEBUG=INFO

# Reduce PyTorch CUDA memory fragmentation over long runs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# SLURM_GPUS_PER_NODE reflects the --gpus-per-node request (used by the
# proven working job script); SLURM_GPUS_ON_NODE is the per-node variant.
GPUS_PER_NODE="${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"

echo "=== Job Info ==="
echo "Job ID:       $SLURM_JOB_ID"
echo "Nodes:        $SLURM_JOB_NUM_NODES"
echo "GPUs/node:    $GPUS_PER_NODE"
echo "Total GPUs:   $((SLURM_JOB_NUM_NODES * GPUS_PER_NODE))"
echo "Working dir:  $SLURM_SUBMIT_DIR"
echo "Start time:   $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# HF cache: must match what scripts/train_hydra.py uses (<repo>/cache).
# Without this export, the pre-download below populates ~/.cache/huggingface
# while the training script reads <repo>/cache - i.e. the pre-download would
# be useless and every rank would re-download simultaneously (GPFS race).
export HF_HOME="${SLURM_SUBMIT_DIR}/cache"

# Pre-download HuggingFace models BEFORE launching torchrun.
# Runs on the batch host only, but the cache is shared GPFS so all nodes
# see it. This avoids the GPFS cache race condition where multiple ranks
# call from_pretrained() simultaneously on the same cache directory.
# (The training script also has a rank-0-first barrier as a fallback.)
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

# srun is REQUIRED here: a SLURM batch script only executes on the first
# allocated node, so a bare `torchrun` would start a single agent on the
# head node, which then waits forever at the rendezvous for agents on the
# other nodes that never launch (RendezvousTimeoutError after 600s).
# srun launches one torchrun per node (--ntasks-per-node=1); each agent
# then spawns --nproc-per-node local workers.
# --rdzv-id ties the rendezvous to this job so a port collision with
# another job on the same node can't cross-wire the stores.
srun torchrun \
    --nnodes="$SLURM_JOB_NUM_NODES" \
    --nproc-per-node="$GPUS_PER_NODE" \
    --rdzv-backend=c10d \
    --rdzv-id="$SLURM_JOB_ID" \
    --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
    -m scripts.train_hydra --config-name=multinode

echo ""
echo "=== Training complete ==="
echo "End time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
