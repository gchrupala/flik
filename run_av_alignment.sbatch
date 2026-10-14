#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu_a100
#SBATCH --time=06:00:00
#SBATCH --mem=24G
#SBATCH --job-name=flik-av-align
#SBATCH --chdir=$HOME/work_dir/flik
#SBATCH --output=$HOME/work_dir/logs/av-alignment-%j.out
#SBATCH --error=$HOME/work_dir/logs/av-alignment-%j.err

# ====================================================================
# Audio-video alignment QC over the full corpus (CPU-only).
#
# PREREQUISITE: the machine must hold the full video corpus at the paths
# stored in the manifest (data/out/... for the cinedantan films).
#
# Reads data/expanded_manifest_segments.json, samples --samples-per-film
# windows per film, and writes data/av_alignment_scores.json.
# ====================================================================

set -euo pipefail

mkdir -p $HOME/work_dir/logs

# One thread per process for BLAS/OpenCV; parallelism comes from --workers.
export OMP_NUM_THREADS=1
export OPENCV_FFMPEG_THREADS=1

echo "=== audio-video alignment QC (full corpus) ==="
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

# OpenCV ships no YuNet weights. Fetch them once; the Haar cascade is the
# automatic fallback when the file is absent and the download fails.
if [ ! -f data/models/face_detection_yunet_2023mar.onnx ]; then
    mkdir -p data/models
    curl -sSL -o data/models/face_detection_yunet_2023mar.onnx \
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" \
        || rm -f data/models/face_detection_yunet_2023mar.onnx
fi

# --max-lag 10 is the default; stated here for clarity. A window searches at
# most window_seconds/2, so the ~7 s corpus windows top out near +/-3.4 s.
uv run --extra cpu python -m scripts.qc_av_alignment \
    --manifest data/expanded_manifest_segments.json \
    --samples-per-film 8 \
    --workers 12 \
    --seed 42 \
    --max-lag 10 \
    --out-dir data/

echo "=== audio-video alignment complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
