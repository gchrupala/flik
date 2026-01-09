#!/bin/bash

#SBATCH --error=logdir/log.transcribe.%A.err
#SBATCH --output=logdir/log.transcribe.%A.out
#SBATCH --job-name="transcribe"
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1

cd /home/gshen/work_dir/flik
source .venv/bin/activate

echo "Setting up environment variables for CUDA and PyTorch"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export LD_LIBRARY_PATH=$(uv run python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cudnn/lib')"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(uv run python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cublas/lib')"):$LD_LIBRARY_PATH

echo "Environment variables set:"
echo "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=$TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" 

echo "Starting transcription process"
srun uv run src/transcribe.py