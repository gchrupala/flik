# Dataset Setup

This guide covers the full data pipeline: downloading videos, transcribing audio, building manifests, scoring video-text correspondence with CLIP, and filtering to produce the final training manifest.

## Pipeline Overview

The pipeline has 6 stages. Stages 1-2 (transcription, manifest building) run individually. Stages 3-6 are orchestrated by `scripts/preprocess.py`. **For training, the recommended fast path is stages 1-2 → `scripts.expand_and_validate_manifest`** (skips the CLIP filter, see [below](#expand--validate-fast-path--recommended)).

| Stage | Script | Output | GPU? |
|-------|--------|--------|------|
| 1 | `src.preprocess.transcribe` | `data/transcripts/*.json` + `.srt` + `_language.txt` | Yes |
| 2 | `src.preprocess.filter_transcripts` | `data/batch_manifest.json` | No |
| — | `scripts.expand_and_validate_manifest` | `data/expanded_manifest_segments.json` | No |
| 3 | `src.preprocess.check_correspondance --mode randomized` | `data/alignment_scores_randomized.json` | Yes |
| 4 | `src.preprocess.check_correspondance --mode main` | `data/alignment_scores.json` + `_segments.json` | Yes |
| 5 | `src.preprocess.filter_by_correspondence` | `data/filtered_manifest.json` + `_segments.json` | No |
| 6 | `scripts.validate_manifest` | `data/filtered_manifest_segments_validated.json` | No |

**Final output for training**: `data/expanded_manifest_segments.json` (segment-level manifest with ALL transcript segments that pass the duration filter). This is the **recommended** manifest — built via the [expand manifest](#expand-manifest-fast-path--recommended) script (optionally with [video-level CLIP QC](#hybrid-workflow-video-level-clip-qc--full-segment-expansion)). Pre-validation (decode-testing every segment) is off by default.

The legacy CLIP-filtered manifest `data/filtered_manifest_segments_validated.json` (stages 3-6) is still available but only keeps ~5 CLIP-scored segments per video. For audio↔video contrastive learning, the CLIP text↔video filter is unnecessary: audio and video come from the same file at the same timestamp, so they are inherently aligned by construction.

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

## Expand manifest (fast path) — RECOMMENDED

After stages 1-2, instead of running the CLIP correspondence pipeline (stages 3-6), use the expand script. This expands ALL transcript segments (with the 3-10s duration filter) into a segment-level manifest, producing a much larger training set. Pre-validation (decode-testing every segment) is **off by default** — use `--validate` to enable it.

```bash
uv run --extra cu128 python -m scripts.expand_and_validate_manifest
```

**Output**: `data/expanded_manifest_segments.json` — the default training manifest.

Options:
```bash
uv run --extra cu128 python -m scripts.expand_and_validate_manifest \
  --input data/batch_manifest.json \
  --output data/expanded_manifest_segments.json \
  --min-duration 3.0 \
  --max-duration 10.0 \
  --num-workers 16 \
  --validate              # run decode validation (slow, off by default)
```

**Why this exists**: The CLIP correspondence filter (stages 3-5) scores only ~5 segments per video against text, then filters by percentile. This is designed for text↔video alignment. But the contrastive task is audio↔video alignment — audio and video are sampled from the same file at the same timestamp, so they are inherently aligned and the text filter is irrelevant. The expand path keeps every segment that loads cleanly, yielding a far larger training set.

**Validation is off by default**: pre-validating every segment by decoding video+audio is slow (I/O-bound). The dataset retries corrupt segments at runtime (3 attempts with random fallback), and the CLIP correspondence stages (3-4) already decode video frames — so fully-corrupt videos are caught there. Use `--validate` if you want upfront corrupt-segment removal.

| Step | Script | Output | GPU? |
|------|--------|--------|------|
| Expand manifest | `scripts.expand_and_validate_manifest` | `data/expanded_manifest_segments.json` | No |

A failure log is written to `data/expanded_manifest_segments_failures.txt` (only when `--validate` is used).

### Hybrid workflow (video-level CLIP QC + full segment expansion)

The pure fast path above skips CLIP entirely. If you want to **filter out whole videos** where the transcript doesn't match the visual content (wrong language, hallucinated speech over silent films, mismatched audio tracks), run the CLIP video-level filter first, then expand all segments from the passing videos only:

```bash
# 1. Video-level CLIP QC (stages 3-5, GPU required)
#    Samples ~5 segments/video, scores text↔frame correspondence,
#    removes whole videos below the percentile threshold.
uv run --extra cu128 python -m scripts.preprocess
# → data/filtered_manifest.json (video-level allowlist of passing videos)

# 2. Expand ALL transcript segments from passing videos + validate
#    (NOT just the ~5 CLIP-scored segments — all of them)
uv run --extra cu128 python -m scripts.expand_and_validate_manifest \
  --input data/filtered_manifest.json
```

**What this preserves vs. what it drops:**
- ✅ **Keeps**: video-level QC — bad videos (desynced audio, wrong-language transcripts, unrelated voiceovers) are removed.
- ✅ **Keeps**: all transcript segments (3-10s) from passing videos — not just the ~5 that CLIP scored.
- ❌ **Drops**: segment-level CLIP filtering. This is intentional — the CLIP filter measures text↔video frame alignment, but the contrastive task is audio↔video alignment (same file, same timestamp, inherently aligned). Text isn't used in the loss.

**Tradeoff**: requires running stages 3-4 (GPU). The pure fast path (input = `batch_manifest.json`) skips the GPU cost entirely but has no video-level QC.

| Path | Input | Video QC? | Segment count | GPU needed? |
|------|-------|-----------|---------------|-------------|
| Pure fast path | `batch_manifest.json` | None | All segments (duration+load-test filtered) | No |
| Hybrid | `filtered_manifest.json` (stages 3-5) | CLIP video-level | All segments from passing videos | Yes (stages 3-4) |
| Legacy (stages 3-6) | `filtered_manifest_segments.json` | CLIP video+segment | ~5 CLIP-scored segments/video | Yes (stages 3-4) |

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
