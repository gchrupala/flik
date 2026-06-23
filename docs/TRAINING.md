# Training

This guide covers training configuration, commands, monitoring, and the model architecture.

## Quick start

### Single-GPU training (Snellius A100 slice)

Snellius A100 slice: 1× A100 40GB GPU, 18 CPU cores, 120GB RAM.

**Prerequisite**: Run the data pipeline first (see [DATASET_SETUP.md](DATASET_SETUP.md)). The default config expects `data/filtered_manifest_segments_validated.json` (produced by Stage 6 validation). If you only have `data/filtered_manifest_segments.json` (unvalidated), either run validation first or override the path: `dataset.manifest_path=data/filtered_manifest_segments.json`.

```bash
uv run --extra cu128 python -m scripts.train_hydra
```

The default config (`batch_size=64`, `num_workers=12`, AMP, gradient checkpointing) is tuned for this setup.

### CPU smoke test (dummy data)

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

### Override config via CLI

Hydra allows overriding any config value from the command line:

```bash
uv run --extra cu128 python -m scripts.train_hydra \
  dataloader.batch_size=32 \
  optimizer.lr=5e-4 \
  training.num_epochs=100 \
  loss.temperature=0.1
```

## Configuration

All configuration is in `src/configs/default.yaml`. Key sections:

### Model (`model.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `audio_model_name` | `facebook/wav2vec2-base` | HuggingFace audio encoder |
| `video_model_name` | `MCG-NJU/videomae-base` | HuggingFace video encoder |
| `hidden_dim` | 768 | Shared hidden dimension |
| `audio_feature_layer` | 7 | Wav2Vec2 layer to extract features from (0-indexed) |
| `temporal_layers` | 2 | Transformer layers for video temporal aggregation |
| `cross_attention_layers` | 2 | Cross-attention layers (only used if MLM enabled) |
| `freeze_audio` | false | Freeze Wav2Vec2 weights |
| `freeze_video` | false | Freeze VideoMAE weights |
| `use_grounded_masked_prediction` | false | Enable MLM auxiliary loss (see warning below) |

> **Warning**: `use_grounded_masked_prediction` is disabled by default. The MLM teacher (`target_projection`) is a frozen random linear layer — it produces meaningless targets. Do not enable MLM until a real teacher is implemented (wav2vec2 quantizer or k-means clusters). The `mlm_weight` in the loss config is set to `0.0` to ensure MLM never activates.

### Dataloader (`dataloader.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 64 | Effective batch size for contrastive learning |
| `num_workers` | 12 | DataLoader workers (Snellius A100 slice has 18 CPU cores) |
| `pin_memory` | true | Pin memory for GPU transfer |
| `drop_last` | true | Drop incomplete last batch |
| `persistent_workers` | true | Keep workers alive across epochs |
| `prefetch_factor` | 4 | Prefetch batches to keep GPU fed |

### Training (`training.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_epochs` | 50 | Total training epochs |
| `gradient_accumulation_steps` | 1 | Gradient accumulation (note: does NOT increase InfoNCE negatives) |
| `clip_grad_norm` | 1.0 | Max gradient norm |
| `mixed_precision` | true | Enable AMP (Automatic Mixed Precision) |
| `gradient_checkpointing` | true | Trade compute for memory on wav2vec2 + videomae |
| `freeze_encoder_epochs` | 5 | Freeze pretrained encoders for first N epochs (projection head warmup) |
| `log_frequency` | 10 | Log every N steps |
| `run_retrieval_eval` | true | Run retrieval evaluation after each epoch |

### Optimizer (`optimizer.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-3 | Learning rate for new modules (projection heads, temporal transformer, cross-modal) |
| `encoder_lr` | 1e-5 | Learning rate for pretrained encoders (after warmup unfreeze) |
| `weight_decay` | 0.01 | AdamW weight decay |
| `betas` | [0.9, 0.999] | AdamW betas |

### Scheduler (`scheduler.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `type` | `cosine_with_warmup` | Linear warmup then cosine decay |
| `warmup_steps` | 500 | Linear warmup steps |
| `eta_min` | 1e-7 | Minimum LR for cosine decay |

### Loss (`loss.*`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `contrastive_weight` | 1.0 | Weight for contrastive loss |
| `mlm_weight` | 0.0 | Weight for MLM loss (0 = disabled) |
| `temperature` | 0.07 | InfoNCE/DCL temperature |
| `use_dcl` | true | Use DCL (Decoupled Contrastive Learning) instead of InfoNCE |
| `mlm_label_smoothing` | 0.0 | Label smoothing for MLM (unused if mlm_weight=0) |

## Key design decisions

### DCL instead of InfoNCE

The default loss is **DCL (Decoupled Contrastive Learning)**, not standard InfoNCE. DCL removes the positive pair from the denominator:

```
L_InfoNCE = -log( exp(s_ii/τ) / Σ_j exp(s_ij/τ) )
L_DCL     = -s_ii/τ + log( Σ_{j≠i} exp(s_ij/τ) )
```

**Why**: InfoNCE has a batch-size-dependent floor of `ln(batch_size)`. At batch size 8, the loss is stuck at `ln(8) ≈ 2.08` with no gradient signal. DCL removes this coupling, making smaller batch sizes viable. At batch size 64, DCL provides strong gradients from 63 negatives without the positive-negative coupling effect.

To switch back to InfoNCE: `loss.use_dcl=false`.

### Encoder warmup

Pretrained encoders (Wav2Vec2, VideoMAE) are **frozen for the first 5 epochs** by default (`freeze_encoder_epochs=5`). During this phase:

- Only the projection heads, temporal transformer, and cross-modal layers train
- The projection heads learn to map frozen encoder features to the contrastive space
- LR is 1e-3 (high) since only new modules are trainable

After epoch 5, encoders are unfrozen with a lower LR (`encoder_lr=1e-5`) to fine-tune without destroying pretrained features.

**Why**: Without warmup, random projection heads scramble pretrained features. The contrastive signal is too weak (especially at small batch sizes) for the model to bootstrap. Freezing encoders lets the projection heads learn a useful mapping first.

### Projection head expansion

Projection heads are 2-layer MLPs with hidden expansion: `768 → 1536 → 768` (LayerNorm → Linear → GELU → Linear). This follows the FaST-VGS recipe.

**Why**: SimCLR showed non-linear projection heads are >10% better than no projection, and hidden expansion (2× width) provides more capacity for the mapping between encoder and contrastive space.

### Mixed precision (AMP)

AMP is enabled by default on CUDA (`mixed_precision: true`). The training loop uses `torch.amp.autocast` and `GradScaler` for FP16 training.

**Why**: ~2× speedup and ~50% memory reduction on A100/H100. FP16 is safe for contrastive learning (cosine similarities are bounded in [-1, 1]).

### Gradient checkpointing

Gradient checkpointing is enabled by default on CUDA (`gradient_checkpointing: true`) for Wav2Vec2 and VideoMAE. This trades ~30% more compute for ~40% memory savings, allowing larger batch sizes.

**Why**: At batch size 64 with 16-frame video clips, memory is the bottleneck. Gradient checkpointing lets you fit larger batches on a single A100 40GB.

### Data loading performance

Several optimizations keep the GPU fed:

- **Video decoding** (`src/utils/video.py`): Sequential frame read in a single pass (no per-frame seeking), plus vectorized numpy resize/crop/normalize instead of per-frame PIL transforms. ~2.4× faster than the previous approach.
- **Audio decoding** (`src/utils/audio.py`): Tolerant of corrupt AAC frames — skips bad frames and returns partial audio instead of failing. libav warning logs suppressed.
- **DataLoader**: 12 workers with `persistent_workers=true` (workers stay alive across epochs) and `prefetch_factor=4` (prefetch batches to keep GPU fed).
- **Dataset retry**: On load failure, retries with a different random segment (up to 3 attempts) before raising. Prevents single corrupt files from crashing training.
- **cudnn benchmark**: Enabled on CUDA for consistent input sizes (video frames are always 224×224).

Monitor `train/data_time_avg` — if it exceeds `train/forward_time_avg`, the GPU is data-starved and you should increase `num_workers` or `prefetch_factor`.

## Monitoring

### Logged metrics

Every `log_frequency` steps, the following metrics are logged to TensorBoard/WandB:

| Metric | Description |
|--------|-------------|
| `train/loss` | Total loss |
| `train/contrastive_loss` | Contrastive loss (DCL or InfoNCE) |
| `train/contrastive_acc` | Batch retrieval accuracy (argmax of similarity matrix) |
| `train/lr` | Current learning rate |
| `train/audio_emb_std` | Std of audio embeddings (collapse detection) |
| `train/video_emb_std` | Std of video embeddings (collapse detection) |
| `train/sim_diag_mean` | Mean of diagonal similarities (positive pairs) |
| `train/sim_offdiag_mean` | Mean of off-diagonal similarities (negative pairs) |
| `train/data_time_avg` | Avg data loading time per step (last 10 steps) — if high, GPU is starved |
| `train/forward_time_avg` | Avg forward+backward time per step (last 10 steps) |

### Collapse detection

Watch these signals for embedding collapse:

- **`train/audio_emb_std` and `train/video_emb_std`**: Should be > 0.01. If approaching 0, embeddings are collapsing.
- **`train/sim_diag_mean` vs `train/sim_offdiag_mean`**: The gap should grow over training. If both converge to the same value, the model isn't learning to discriminate.
- **`train/contrastive_acc`**: Should increase above `1/batch_size` (1.56% for B=64). If stuck at chance, the model isn't learning.

### Retrieval evaluation

After each epoch, Recall@1/5/10 is computed on the training set:

```bash
# View in TensorBoard
tensorboard --logdir logdir

# Or in WandB (set logging.use_wandb=true)
```

> **Note**: Retrieval eval currently runs on the training dataloader, not a held-out validation set. This is a known limitation — add a real validation split for proper evaluation.

## Model architecture

### Components

```
[audio] → Wav2Vec2-base (layer 7) → mean-pool → audio_proj (768→1536→768) → L2-norm → audio_embedding
[video] → VideoMAE-base (16 frames) → mean-pool spatial → temporal transformer (2 layers) → CLS → video_proj (768→1536→768) → L2-norm → video_embedding

Loss: DCL(audio_embedding, video_embedding, τ=0.07)
```

### Audio encoder (`src/models/audio_encoder.py`)

- **Backbone**: `facebook/wav2vec2-base` (pretrained)
- **Feature extraction**: Layer 7 (8th encoder layer, 0-indexed)
- **Feature rate**: 50 Hz at 16 kHz input (stride product = 320)
- **Pooling**: Mean-pool over valid time steps (masked by padding mask)
- **Output**: `(batch, 768)` L2-normalized embedding

### Video encoder (`src/models/video_encoder.py`)

- **Backbone**: `MCG-NJU/videomae-base` (pretrained, tubelet_size=2)
- **Input**: 16 frames at 224×224
- **Spatial pooling**: Mean-pool over 196 spatial patches → 8 temporal tokens
- **Temporal aggregation**: Learnable CLS token + 2-layer transformer
- **Output**: `(batch, 768)` L2-normalized embedding (from CLS token)

### Projection heads (`src/models/dual_encoder.py`)

- **Architecture**: LayerNorm → Linear(768→1536) → GELU → Linear(1536→768)
- **Separate heads**: `audio_proj` and `video_proj` (each modality specializes)
- **Output**: L2-normalized before contrastive loss

### Cross-modal fusion (`src/models/cross_modal.py`)

- **Architecture**: 2-layer cross-attention (audio queries, video key/value)
- **Only active when MLM is enabled** (`use_grounded_masked_prediction=true`)
- **Current limitations**: Unidirectional (A→V only), no self-attention in cross-modal blocks, `audio_padding_mask` not applied

### Loss (`src/models/losses.py`)

- **Contrastive**: DCL (default) or InfoNCE, symmetric, temperature 0.07
- **MLM**: Disabled by default (`mlm_weight=0.0`). When enabled, uses a frozen random `target_projection` as teacher — **this is broken** and should not be enabled until a real teacher is implemented.

## Checkpointing

Checkpoints are saved at the end of training to `checkpoints/checkpoint_epoch{N}.pth`:

```python
{
    "epoch": int,
    "global_step": int,
    "model_state_dict": dict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "config": dict,
}
```

## Cluster execution

### SLURM

SLURM job scripts: `run_transcription.sh`, `run_correpond.sh`. Submit with:

```bash
sbatch run_transcription.sh
```

### Multi-GPU (DDP)

Distributed Data Parallel training is **not yet implemented**. For multi-GPU, the recommended path is:

1. Use `torchrun` with `DistributedDataParallel`
2. All-gather embeddings across GPUs before computing the contrastive loss (so each GPU sees all negatives)
3. Or use GradCache for memory-efficient large-batch training

## Known limitations

- **No validation split**: Retrieval eval runs on the training set
- **MLM teacher is broken**: `target_projection` is a frozen random linear — do not enable MLM
- **No hard negative mining**: FaST-VGS uses top-K=100 hard negatives; not yet implemented
- **Cross-modal is unidirectional**: Only A→V attention, no V→A or self-attention in cross-modal blocks
- **VideoMAE vs CLIP**: VideoMAE features are optimized for pixel reconstruction, not semantic alignment. CLIP ViT per-frame may be a stronger starting point for audio-video alignment.
- **No DDP**: Single-GPU only for now
