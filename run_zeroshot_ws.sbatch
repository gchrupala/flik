#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu_a100
#SBATCH --time=04:00:00
#SBATCH --mem=24G
#SBATCH --job-name=flik-zeroshot-ws
#SBATCH --output=logdir/zeroshot-ws-%j.out
#SBATCH --error=logdir/zeroshot-ws-%j.err

# ====================================================================
# Zero-shot VGGSound retrieval eval on the local cuda-workstation.
# CPU-only: torch 2.8+cu128 ships no sm_61 kernels, so the GTX 1080 Ti
# cannot run this build. CPU on 12 cores is fast enough for 600 clips.
#
# PREREQUISITES:
#   - eval_assets/ holds the checkpoints with DISTINCT names:
#       aug09_best.pth, aug11_epoch50.pth, aug13_best.pth, aug13_latest.pth
#   - eval_assets/vggsound_subset/clips_manifest.json exists
#       (downloaded subset, 600 clips / 40 classes).
#   - First run downloads wav2vec2/videomae weights from HF
#     (workstation has internet; no HF_HUB_OFFLINE here).
# ====================================================================

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logdir

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export OMP_NUM_THREADS=12
export OPENCV_FFMPEG_THREADS=1

echo "=== Zero-shot VGGSound eval (workstation, CPU) ==="
echo "Job ID:     $SLURM_JOB_ID"
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

uv run --extra cpu python -m scripts.eval_zeroshot_vggsound \
    --checkpoint aug09_best:eval_assets/aug09_best.pth \
    --checkpoint aug11_epoch50:eval_assets/aug11_epoch50.pth \
    --checkpoint aug13_best:eval_assets/aug13_best.pth \
    --checkpoint aug13_latest:eval_assets/aug13_latest.pth \
    --random-init \
    --clips-manifest eval_assets/clips_manifest.json \
    --device cpu \
    --batch-size 8 \
    --out zeroshot_results_ws.json

echo "=== Eval complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
