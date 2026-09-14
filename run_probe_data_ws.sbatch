#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu_a100
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --job-name=flik-probe-data
#SBATCH --output=$HOME/work_dir/logs/probe-data-%j.out
#SBATCH --error=$HOME/work_dir/logs/probe-data-%j.err

# Download the unimodal probe datasets:
#   - ESC-50 (2,000 clips, 50 classes, environmental sound)
#   - Speech Commands v2 (extract 10 keyword classes)
# Idempotent: existing archives and directories are kept.

set -euo pipefail
cd $HOME/work_dir/flik
mkdir -p data/probe

cd data/probe
if [ ! -d ESC-50-master ]; then
  [ -f master.zip ] || curl -sSL -o master.zip https://github.com/karoldvl/ESC-50/archive/master.zip
  unzip -q -o master.zip
fi
echo "ESC-50 clips: $(ls ESC-50-master/audio/*.wav 2>/dev/null | wc -l)"

if [ ! -d speech_commands_subset ]; then
  [ -f speech_commands_v0.02.tar.gz ] || curl -sSL -o speech_commands_v0.02.tar.gz http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
  mkdir -p speech_commands_subset
  # --no-anchored: the archive stores members as ./yes/..., so plain name
  # patterns do not match. GNU tar needs this flag for the subset extraction.
  tar -xzf speech_commands_v0.02.tar.gz --wildcards --no-anchored -C speech_commands_subset \
      yes no up down left right on off stop go 2>/dev/null || true
fi
echo "Speech Commands clips: $(find speech_commands_subset -name '*.wav' | wc -l)"
du -sh ESC-50-master speech_commands_subset 2>/dev/null

echo "=== Probe data download complete: $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="
