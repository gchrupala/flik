#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_a100
#SBATCH --time=06:00:00
#SBATCH --mem=8G
#SBATCH --job-name=flik-vgg-full-dl
#SBATCH --output=$HOME/work_dir/logs/vgg-full-dl-%j.out
#SBATCH --error=$HOME/work_dir/logs/vgg-full-dl-%j.err

# Download the full VGGSound test split (15,446 clips) from the HF mirror.
# Network-bound: few CPUs, many workers.

set -euo pipefail
cd $HOME/work_dir/flik

uv run --extra cpu python -m scripts.download_vggsound_subset \
    --classes 310 \
    --per-class 50 \
    --out-dir data/vggsound_full \
    --workers 16

echo "=== Full download complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
