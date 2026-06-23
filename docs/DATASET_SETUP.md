# Dataset Setup

This guide covers the full data pipeline: downloading videos, transcribing audio, building manifests, scoring video-text correspondence with CLIP, and filtering to produce the final training manifest.

## Pipeline Overview

The pipeline has 6 stages. Stages 1-2 (transcription, manifest building) run individually. Stages 3-6 are orchestrated by `scripts/preprocess.py`.

| Stage | Script | Output | GPU? |
|-------|--------|--------|------|
| 1 | `src.preprocess.transcribe` | `data/transcripts/*.json` + `.srt` + `_language.txt` | Yes |
| 2 | `src.preprocess.filter_transcripts` | `data/batch_manifest.json` | No |
| 3 | `src.preprocess.check_correspondance --mode randomized` | `data/alignment_scores_randomized.json` | Yes |
| 4 | `src.preprocess.check_correspondance --mode main` | `data/alignment_scores.json` + `_segments.json` | Yes |
| 5 | `src.preprocess.filter_by_correspondence` | `data/filtered_manifest.json` + `_segments.json` | No |
| 6 | `scripts.validate_manifest` | `data/filtered_manifest_segments_validated.json` | No |

**Final output for training**: `data/filtered_manifest_segments_validated.json` (segment-level manifest with video-audio pairs that pass CLIP correspondence thresholds AND load-test validation).

## Prerequisites

### Installation

```bash
uv sync --extra cu128   # GPU (CUDA 12.8; cluster)
uv sync --extra cpu     # CPU-only (local debugging)
```

> Only `cu128` and `cpu` are available: `whisperx` pins `torch~=2.8.0`, and torch 2.8.x is only published for cu128 (+cpu).

### CUDA environment variables (GPU)

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export LD_LIBRARY_PATH=$(uv run --extra cu128 python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cudnn/lib')"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(uv run --extra cu128 python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cublas/lib')"):$LD_LIBRARY_PATH
```

### Data layout

- Raw videos: `data/out/cinedantan/` (each film in subdirectory by identifier)
- Transcripts: `data/transcripts/`
- Manifest: `data/batch_manifest.json` (list of video-transcript pairs)
- Input manifest: `data/in/cinedantan_movies.json`
- Logs: `logdir/` (git-ignored)

## Stage 1: Transcription

WhisperX transcription pipeline. Produces per-video transcript JSON, SRT, and language detection files.

```bash
uv run --extra cu128 python -m src.preprocess.transcribe
```

**Device detection**: Automatically uses CUDA if available; override with `--device cpu`.

**Outputs** (per video):
- `data/transcripts/<video_id>.json` — segment-level transcript with timestamps
- `data/transcripts/<video_id>.srt` — SRT subtitle file
- `data/transcripts/<video_id>_language.txt` — detected language

## Stage 2: Manifest building

Filters transcripts by language and duration, builds the batch manifest.

```bash
uv run --extra cu128 python -m src.preprocess.filter_transcripts
```

**Output**: `data/batch_manifest.json` — list of `{id, video_path, json_path}` entries.

## Stages 3-6: Correspondence scoring, filtering, and validation

Orchestrated by `scripts/preprocess.py`. This runs the CLIP randomized baseline, alignment scoring, filtering, and segment validation in one go.

### Full run (all 4 stages)

```bash
uv run --extra cu128 python -m scripts.preprocess
```

### Resume from a specific stage

```bash
uv run --extra cu128 python -m scripts.preprocess --from-stage alignment
```

### Stop after a specific stage

```bash
uv run --extra cu128 python -m scripts.preprocess --to-stage alignment
```

Stage names: `randomized` (3), `alignment` (4), `filter` (5), `validate` (6).

### CPU override for GPU stages

```bash
uv run --extra cu128 python -m scripts.preprocess --device cpu
```

### Dry-run (show plan without executing)

```bash
uv run --extra cu128 python -m scripts.preprocess --dry-run
```

### Custom thresholds

```bash
uv run --extra cu128 python -m scripts.preprocess --percentile-threshold 50 --segment-percentile-threshold 30
```

### Performance flags

Stages 3 and 4 support parallel video decoding and batched CLIP scoring:

| Flag | Default | Purpose |
|------|---------|---------|
| `--batch-size` | 256 | (text, frame) pairs per CLIP forward |
| `--num-workers` | 16 | CPU threads for parallel video decode |
| `--decoder` | cpu | Video decoder backend (`cpu` or `nvdec`) |
| `--no-fp16` | false | Disable FP16 autocast on CUDA |

```bash
uv run --extra cu128 python -m scripts.preprocess --batch-size 128 --num-workers 8
```

### What each stage does

**Stage 3 — Randomized baseline**: Computes CLIP scores on random video-text pairs to establish a null distribution. Output: `data/alignment_scores_randomized.json`.

**Stage 4 — Main alignment scoring**: Computes CLIP scores for matched video-transcript pairs. Produces z-scores (relative to the randomized baseline), percentiles, and segment-level pass/fail flags. Output: `data/alignment_scores.json` + `data/alignment_scores_segments.json`.

**Stage 5 — Filtering**: Keeps only videos/segments that meet percentile and segment-level thresholds. Output: `data/filtered_manifest.json` (video-level) + `data/filtered_manifest_segments.json` (segment-level).

**Stage 6 — Validation**: Load-tests every segment (video + audio decode) and removes segments that fail. This catches corrupt video files, missing audio streams, and bad AAC data that would cause runtime errors and retries during training. Output: `data/filtered_manifest_segments_validated.json` (the manifest used for training).

Can also be run standalone:

```bash
uv run --extra cu128 python -m scripts.validate_manifest
uv run --extra cu128 python -m scripts.validate_manifest --input data/filtered_manifest_segments.json --num-workers 16
```

A failure log is written to `data/filtered_manifest_segments_validated_failures.txt`.

### Key constants (in `src/CONSTANTS.py`)

- `SEGMENT_PERCENTILE_THRESHOLD` — Minimum percentile for a segment to "pass" (default 25)
- `VIDEO_PERCENTILE_THRESHOLD` — Minimum percentile for the whole video (default 25)
- `USE_SEGMENT_FILTER` — Whether to apply segment-level filtering (default `False`)
- `CLIP_BATCH_SIZE` — CLIP scoring batch size (default 256)
- `CLIP_NUM_WORKERS` — Parallel decode threads (default 16)

Thresholds are lenient by default; adjust for desired precision/recall trade-off.

## Cluster execution (SLURM)

SLURM job scripts: `run_transcription.sh`, `run_correpond.sh`.

They assume `.venv/bin/activate` exists and set the CUDA environment variables. Scripts use `$SLURM_SUBMIT_DIR` to auto-detect the working directory.

```bash
sbatch run_transcription.sh
```

> Any `uv`/`uv sync`/`uv run` calls inside SLURM scripts must include `--extra cu128`.

## Inspecting results

View statistics of computed alignment scores:

```bash
uv run --extra cu128 python -m src.preprocess.check_correspondance --mode output
```

## Dataset survey

See `data/README.md` for a survey of viable speech-video datasets (SayCam, AudioSet, Spoken Moments, Cinedantan, LSMDC, MSRVTT, Movienet, Condensed Movies).
