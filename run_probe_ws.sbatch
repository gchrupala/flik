#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpu_a100
#SBATCH --time=08:00:00
#SBATCH --mem=24G
#SBATCH --job-name=flik-probe
#SBATCH --output=$HOME/work_dir/logs/probe-%j.out
#SBATCH --error=$HOME/work_dir/logs/probe-%j.err

# ====================================================================
# Frozen-feature linear probe on the local cuda-workstation.
#
# Runs five backbones (pretrained HF baseline + four flik checkpoints):
#   pretrained:PRETRAINED, aug09_best, aug11_epoch50, aug13_best, aug13_latest
# on three (modality, dataset) combinations:
#   audio / esc50, audio / speech_commands, video / vggsound
#
# IMPORTANT: use .venv-cu126 (torch 2.14.0+cu126). The repo .venv has
# torch 2.8+cu128, which ships no sm_61 kernels and CANNOT run on the
# GTX 1080 Ti. Feature extraction runs on the GPU; the LogisticRegression
# probe runs on the CPU.
#
# PREREQUISITES:
#   - data/probe/ESC-50-master/{audio,meta/esc50.csv}
#   - data/probe/speech_commands_subset/<class>/*.wav (10 classes)
#   - data/vggsound_full/clips_manifest.json + clips/
#   - eval_assets/{aug09_best,aug11_epoch50,aug13_best,aug13_latest}.pth
#   - .venv-cu126 has scikit-learn installed
# ====================================================================

set -euo pipefail
cd $HOME/work_dir/flik
mkdir -p $HOME/work_dir/logs

export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export HF_HOME=$HOME/work_dir/flik/cache
export OMP_NUM_THREADS=12
export OPENCV_FFMPEG_THREADS=1

PY=.venv-cu126/bin/python

BACKBONES=(
    --backbone pretrained:PRETRAINED
    --backbone aug09_best:eval_assets/aug09_best.pth
    --backbone aug11_epoch50:eval_assets/aug11_epoch50.pth
    --backbone aug13_best:eval_assets/aug13_best.pth
    --backbone aug13_latest:eval_assets/aug13_latest.pth
)

echo "=== Frozen-feature probe (workstation, GPU) ==="
echo "Job ID:     ${SLURM_JOB_ID}"
echo "Start time: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Python:     $(${PY} --version 2>&1)"

echo "--- audio / esc50 ---"
"${PY}" -m scripts.probe_encoder \
    --modality audio --dataset esc50 \
    "${BACKBONES[@]}" \
    --dataset-root data/probe \
    --batch-size 8 --workers 4 --device cuda \
    --out probe_results_esc50.json

echo "--- audio / speech_commands ---"
"${PY}" -m scripts.probe_encoder \
    --modality audio --dataset speech_commands \
    "${BACKBONES[@]}" \
    --dataset-root data/probe \
    --batch-size 8 --workers 4 --device cuda \
    --out probe_results_speech_commands.json

echo "--- video / vggsound ---"
"${PY}" -m scripts.probe_encoder \
    --modality video --dataset vggsound --video-feature tower_mean \
    "${BACKBONES[@]}" \
    --dataset-root data/probe \
    --vggsound-manifest data/vggsound_full/clips_manifest.json \
    --batch-size 8 --workers 4 --device cuda \
    --out probe_results_vggsound.json

echo "=== Probe complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
