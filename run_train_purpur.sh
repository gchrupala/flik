#!/bin/bash

#SBATCH --nodes=1
#SBATCH --partition=GPU
#SBATCH --gres=gpu:4
#SBATCH --job-name="train"
#SBATCH --output="logdir/log_train_%A.out"
#SBATCH --error="logdir/log_train_%A.err"

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

echo "Setting up environment variables for CUDA and PyTorch"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export LD_LIBRARY_PATH=$(uv run --extra cu128 python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cudnn/lib')"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(uv run --extra cu128 python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cublas/lib')"):$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1  # GPU-bound training, prevents BLAS oversubscription
export OPENCV_FFMPEG_THREADS=1  # 1 FFmpeg decode thread per VideoCapture (not auto)
ulimit -u 65536 2>/dev/null || true  # raise thread limit (SLURM cgroups default low)

echo "Environment variables set:"
echo "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=$TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"

echo ""
echo "=== Launching 4-GPU DDP training (multinode config) ==="
NPROC="${SLURM_GPUS_ON_NODE:-4}"
echo "Using nproc-per-node=$NPROC"

torchrun --standalone --nnodes=1 --nproc-per-node="$NPROC" \
    -m scripts.train_hydra --config-name=multinode