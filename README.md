# flik

Video-grounded speech representation learning. Extends the [FaST-VGS](https://arxiv.org/abs/2109.08186) lineage to video-audio grounding using movie segments.

## Documentation

| Document | Description |
|----------|-------------|
| [docs/DATASET_SETUP.md](docs/DATASET_SETUP.md) | Data pipeline: download, transcribe, CLIP scoring, filtering, validation |
| [docs/TRAINING.md](docs/TRAINING.md) | Training: configuration, commands, monitoring, model architecture |
| [data/README.md](data/README.md) | Dataset survey (SayCam, AudioSet, Cinedantan, etc.) |

## Quick start

### Install

```bash
uv sync --extra cu128   # GPU (CUDA 12.8; cluster)
uv sync --extra cpu     # CPU-only (local debugging)
```

> Only `cu128` and `cpu` are available: `whisperx` pins `torch~=2.8.0`, and torch 2.8.x is only published for cu128 (+cpu).

Set CUDA environment variables (GPU only):

```bash
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export LD_LIBRARY_PATH=$(uv run --extra cu128 python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cudnn/lib')"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(uv run --extra cu128 python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cublas/lib')"):$LD_LIBRARY_PATH
```

> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set automatically by the training script to reduce CUDA fragmentation over long runs. Export it yourself to override. See [docs/TRAINING.md](docs/TRAINING.md#anti-collapse-measures) for the full set of anti-collapse and hyperparameter-tuning rationale.

### Prepare data

See [docs/DATASET_SETUP.md](docs/DATASET_SETUP.md) for the full pipeline. Short version:

```bash
# Stage 1-2: Transcribe + build manifest
uv run --extra cu128 python -m src.preprocess.transcribe
uv run --extra cu128 python -m src.preprocess.filter_transcripts

# Stage 3-6: CLIP correspondence scoring + filtering + validation
uv run --extra cu128 python -m scripts.preprocess
```

Output: `data/filtered_manifest_segments_validated.json` (validated segment-level manifest for training).

### Train

```bash
uv run --extra cu128 python -m scripts.train_hydra
```

Configuration: `src/configs/default.yaml`. See [docs/TRAINING.md](docs/TRAINING.md) for details.

CPU smoke test:

```bash
uv run --extra cu128 python -m scripts.train_hydra \
  dataset.dummy=true \
  dataloader.batch_size=4 \
  dataloader.num_workers=0 \
  training.num_epochs=2 \
  training.freeze_encoder_epochs=1 \
  training.gradient_checkpointing=false \
  hardware.device=cpu \
  logging.use_tensorboard=false \
  logging.use_wandb=false
```

## Datasets

### Cinedantan

Public domain movies from the Internet Archive.

- Number of movies: +2100
- Total duration: 2437 hours
- Size on disk: 705G

Data: https://github.com/casbah-ma/cinedantan/blob/master/public/database/movies.json

### Other datasets evaluated

See [data/README.md](data/README.md) for a survey of viable speech-video datasets (SayCam, AudioSet, Spoken Moments, LSMDC, MSRVTT, Movienet, Condensed Movies).

### Reference implementations

`scripts/eval_triplets.py` reimplements the two-alternative forced-choice
protocol of Nikolaus, Alishahi & Chrupała (2022), *Learning English with Peppa
Pig* (TACL 10:293-307, [doi:10.1162/tacl_a_00498](https://doi.org/10.1162/tacl_a_00498)).
It does **not** import that code — the logic is reimplemented and attributed in
the module docstring, so no extra checkout is needed to run it. To diff against
the reference locally:

```bash
git clone https://github.com/gchrupala/peppa vendor/peppa
git -C vendor/peppa checkout c92719d5e5abf5a29e776a09c42a2a2f6c74a10c
```

`vendor/` is gitignored, so that checkout is optional and never committed.

## Project structure

```
src/
  models/          Core model components
    audio_encoder.py    Wav2Vec2 wrapper
    video_encoder.py    VideoMAE wrapper + temporal transformer
    dual_encoder.py     Contrastive dual encoder + projection heads
    cross_modal.py      Cross-attention fusion (for MLM)
    flik_model.py       Top-level model
    losses.py           DCL/InfoNCE + VICReg variance reg + MLM loss
  datasets/        VideoAudioDataset
  eval/            Retrieval metrics (Recall@K)
  utils/           Video/audio preprocessing, logging
  configs/         Hydra configuration (default.yaml)
  preprocess/      Transcription, CLIP scoring, filtering
scripts/
  preprocess.py    Orchestrator for stages 3-6
  validate_manifest.py  Segment validation (Stage 6, also standalone)
  train_hydra.py   Main training script
  train.py         Legacy single-GPU dummy training
  test_pipeline.py Pipeline verification
  verify_contracts.py Contract verification
  debug_data.py    Data loading diagnostics
docs/
  DATASET_SETUP.md Data pipeline documentation
  TRAINING.md      Training documentation
```
