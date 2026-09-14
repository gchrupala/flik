#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --job-name=flik-zeroshot-full
#SBATCH --output=$HOME/work_dir/logs/zeroshot-full-%j.out
#SBATCH --error=$HOME/work_dir/logs/zeroshot-full-%j.err

# Full VGGSound test-split zero-shot eval (12,011 clips / 249 classes).
# GPU path: torch 2.14.0+cu126 in .venv-cu126 (Pascal sm_60, the box's
# known-good stack). The repo .venv (torch 2.8+cu128) has no sm_61 kernels.
# Decode on CPU with --workers, encode on the GTX 1080 Ti.
#
# Core budget: 12 CPUs = 10 decode workers + 2 BLAS threads for the encode
# process. This configuration is GPU-bound, so extra workers are headroom
# against decode stalls, not a speedup lever. Raise --workers further only if
# the encode moves to CPU or many more checkpoints are scored per clip.

set -euo pipefail
cd $HOME/work_dir/flik

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export OMP_NUM_THREADS=2

echo "=== Full zero-shot VGGSound eval (GPU) ==="
echo "Job ID:     $SLURM_JOB_ID"
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

.venv-cu126/bin/python -m scripts.eval_zeroshot_vggsound \
    --clips-manifest data/vggsound_full/clips_manifest.json \
    --checkpoint aug09_best:eval_assets/aug09_best.pth \
    --checkpoint aug11_epoch50:eval_assets/aug11_epoch50.pth \
    --checkpoint aug13_best:eval_assets/aug13_best.pth \
    --checkpoint aug13_latest:eval_assets/aug13_latest.pth \
    --random-init \
    --device cuda \
    --batch-size 16 \
    --workers 10 \
    --out zeroshot_results_full.json

echo "=== Eval complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
