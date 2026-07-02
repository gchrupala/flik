# Training

This guide covers training configuration, commands, monitoring, and the model architecture.

## Quick start

### Single-GPU training (Snellius A100 slice)

Snellius A100 slice: 1× A100 40GB GPU, 18 CPU cores, 120GB RAM.

**Prerequisite**: Build the training manifest first (see [DATASET_SETUP.md](DATASET_SETUP.md)). The default config expects `data/expanded_manifest_segments.json`, produced by the expand script:

```bash
# After stages 1-2 (transcription + batch_manifest.json):
uv run --extra cu128 python -m scripts.expand_and_validate_manifest
```

This expands ALL transcript segments (3-10s duration filter) into a segment-level manifest, bypassing the CLIP correspondence segment filter (stages 3-5) which is unnecessary for audio↔video contrastive learning — audio and video are inherently aligned (same file, same timestamp). Pre-validation (decode-testing every segment) is **off by default** — the dataset retries corrupt segments at runtime (3 attempts with random fallback), and the CLIP stages already catch fully-corrupt videos. Use `--validate` to pre-validate if you want upfront corrupt-segment removal.

The legacy CLIP-filtered manifest `data/filtered_manifest_segments_validated.json` is still available if you prefer it: `dataset.manifest_path=data/filtered_manifest_segments_validated.json`.

**With video-level CLIP QC (hybrid)**: to filter out whole videos where the transcript doesn't match the visual content (wrong language, hallucinated speech), run CLIP stages 3-5 first, then expand from the filtered video list:

```bash
# 1. CLIP video-level filter (GPU) → data/filtered_manifest.json
uv run --extra cu128 python -m scripts.preprocess

# 2. Expand all segments from passing videos + validate
uv run --extra cu128 python -m scripts.expand_and_validate_manifest \
  --input data/filtered_manifest.json
```

See [DATASET_SETUP.md](DATASET_SETUP.md#hybrid-workflow-video-level-clip-qc--full-segment-expansion) for the full comparison of paths.

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
| `encoder_lr` | 5e-6 | Learning rate for pretrained encoders (after warmup unfreeze). Halved from 1e-5 to reduce the post-unfreeze shock that triggered embedding collapse (see [Anti-collapse measures](#anti-collapse-measures)). |
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
| `temperature` | 0.1 | InfoNCE/DCL temperature. Raised from 0.07 — a softer softmax reduces collapse risk under DCL with few negatives (see [Anti-collapse measures](#anti-collapse-measures)). |
| `use_dcl` | true | Use DCL (Decoupled Contrastive Learning) instead of InfoNCE |
| `mlm_label_smoothing` | 0.0 | Label smoothing for MLM (unused if mlm_weight=0) |
| `variance_weight` | 0.5 | VICReg variance regularization weight. Penalizes embedding collapse (per-dim std → 0). 0 disables. |
| `variance_gamma` | null | Target per-dim std for variance reg. `null` = auto `1/sqrt(hidden_dim)` ≈ 0.036 for L2-normalized 768-d embeddings. |

## Key design decisions

### DCL instead of InfoNCE

The default loss is **DCL (Decoupled Contrastive Learning)**, not standard InfoNCE. DCL removes the positive pair from the denominator:

```
L_InfoNCE = -log( exp(s_ii/τ) / Σ_j exp(s_ij/τ) )
L_DCL     = -s_ii/τ + log( Σ_{j≠i} exp(s_ij/τ) )
```

**Why**: InfoNCE has a batch-size-dependent floor of `ln(batch_size)`. At batch size 8, the loss is stuck at `ln(8) ≈ 2.08` with no gradient signal. DCL removes this coupling, making smaller batch sizes viable. At batch size 64, DCL provides strong gradients from 63 negatives without the positive-negative coupling effect.

**Collapse caveat**: DCL has a stable collapsed equilibrium at `ln(N-1)` (≈4.14 for batch 64). When all embeddings become identical, the positive term `-s_ii/τ` and the negative `logsumexp` term nearly cancel, leaving a near-zero gradient — the model can sit there indefinitely. This is exactly what was observed after encoder unfreezing (loss frozen at 4.143, `sim_diag ≈ sim_offdiag ≈ 0.9997`, accuracy = 1/64). The [anti-collapse measures](#anti-collapse-measures) below address this.

To switch back to InfoNCE: `loss.use_dcl=false`.

### Encoder warmup

Pretrained encoders (Wav2Vec2, VideoMAE) are **frozen for the first 5 epochs** by default (`freeze_encoder_epochs=5`). During this phase:

- Only the projection heads, temporal transformer, and cross-modal layers train
- The projection heads learn to map frozen encoder features to the contrastive space
- LR is 1e-3 (high) since only new modules are trainable

After epoch 5, encoders are unfrozen with a lower LR (`encoder_lr=5e-6`) to fine-tune without destroying pretrained features.

**Why**: Without warmup, random projection heads scramble pretrained features. The contrastive signal is too weak (especially at small batch sizes) for the model to bootstrap. Freezing encoders lets the projection heads learn a useful mapping first.

**How the unfreeze is done matters**: the encoder parameters are added to the *existing* optimizer via `optimizer.add_param_group()` rather than rebuilding a fresh optimizer. Rebuilding destroys the Adam momentum (`exp_avg` / `exp_avg_sq`) the projection heads accumulated during the freeze phase, so the first post-unfreeze step applies a raw `lr·grad` update with no momentum smoothing — a large shock that, combined with the low-temperature DCL loss, reliably triggers embedding collapse within ~2 epochs. `add_param_group` preserves the optimizer state, and the cosine scheduler's current lambda multiplier applies to the encoder's lower base LR, giving the encoder a natural warmup ramp too.

### Anti-collapse measures

A prior training run collapsed immediately after the encoder unfreeze at epoch 6: loss froze at `ln(63) ≈ 4.143`, `sim_diag_mean ≈ sim_offdiag_mean ≈ 0.9997` (all embeddings identical), and `contrastive_acc` stuck at `1/64` (random chance). The run then hit a CUDA OOM from memory fragmentation ~20 epochs later. Five changes address both issues:

1. **`add_param_group` instead of optimizer rebuild** (see [Encoder warmup](#encoder-warmup)). Preserves Adam momentum for the projection heads at unfreeze, eliminating the raw-`lr·grad` shock that kicked off collapse. This is the root-cause fix.

2. **VICReg variance regularization** (`loss.variance_weight=0.5`). Adds a term that penalizes low per-dimension std across the batch: `mean(relu(γ − std(z, dim=0)))`. This makes the collapsed solution a *high*-loss state instead of the near-zero-gradient equilibrium DCL settles into. `γ` auto-computes to `1/sqrt(hidden_dim) ≈ 0.036`, matching the healthy std of L2-normalized 768-d embeddings. Logged as `train/variance_loss`. Set `variance_weight=0` to disable.

3. **Lower `encoder_lr` (1e-5 → 5e-6)**. Halves the magnitude of the post-unfreeze perturbation to the pretrained weights, reducing the chance of a feature-distribution shift large enough to trigger collapse.

4. **Higher `temperature` (0.07 → 0.1)**. A softer softmax gives smoother, less peaky gradients, making it harder for the model to overshoot into the collapsed basin. 0.07 is aggressive for a setup with only 63 negatives and a tiny (2814-segment) dataset.

5. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**. Set automatically at the top of `train_hydra.py` (overridable via the environment). Reduces CUDA fragmentation over long runs, preventing the "reserved but unallocated" OOM that appeared after ~26 epochs. The earlier LR-scheduler cycling bug (cosine `progress > 1.0` wrapping back up, and a closure capturing `global_step` by reference) was also fixed in the same pass — `progress` is now clamped to 1.0.

These are belt-and-suspenders: fix 1 removes the trigger, fix 2 makes collapse non-viable regardless of trigger, and fixes 3–4 reduce general collapse susceptibility. Fix 5 is independent.

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
| `train/variance_loss` | VICReg variance regularization loss (0 when embeddings are well-spread; rises as std drops toward collapse) |
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
- **`train/sim_diag_mean` vs `train/sim_offdiag_mean`**: The gap should grow over training. If both converge to the same value (e.g. both ≈ 0.9997), the model has collapsed — all embeddings are nearly identical.
- **`train/contrastive_acc`**: Should increase above `1/batch_size` (1.56% for B=64). If stuck at chance, the model isn't learning.
- **`train/variance_loss`**: Should stay near 0 when embeddings are well-spread. If it rises, the variance regularizer is actively fighting collapse; if it stays high while `contrastive_loss` is frozen, the model is stuck in the collapsed basin.

### Retrieval evaluation

After each epoch, Recall@1/5/10 is computed on the training set:

```bash
# View in TensorBoard
tensorboard --logdir logdir

# Or in WandB (set logging.use_wandb=true)
```

> **Validation split**: A fraction of videos (`validation.split_ratio=0.1`, default) is held out from training and used for retrieval evaluation after each epoch. The split is grouped by `video_id` (deterministic, seeded by `validation.split_seed=42`) so no film appears in both train and validation — preventing leakage. Metrics are logged as `val/a2v_r@1`, `val/v2a_r@1`, `val/mean_r1`, etc. Set `validation.split_ratio=0` to disable and eval on the training set.

## Model architecture

### Components

```
[audio] → Wav2Vec2-base (layer 7) → mean-pool → audio_proj (768→1536→768) → L2-norm → audio_embedding
[video] → VideoMAE-base (16 frames) → mean-pool spatial → temporal transformer (2 layers) → CLS → video_proj (768→1536→768) → L2-norm → video_embedding

Loss: DCL(audio_embedding, video_embedding, τ=0.1) + 0.5 · VICReg-variance(γ=1/√768)
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

- **Contrastive**: DCL (default) or InfoNCE, symmetric, temperature 0.1
- **Variance (VICReg)**: Anti-collapse regularizer (`variance_weight=0.5`), target std `γ=1/sqrt(768)`. Penalizes the collapsed equilibrium where all embeddings become identical.
- **MLM**: Disabled by default (`mlm_weight=0.0`). When enabled, uses a frozen random `target_projection` as teacher — **this is broken** and should not be enabled until a real teacher is implemented.

## Checkpointing

Checkpoints are saved at the end of each epoch to `checkpoints/checkpoint_epoch{N}.pth`:

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

A **best checkpoint** `checkpoints/checkpoint_best.pth` tracks the model with the highest `validation.select_metric` (default `mean_r1` = mean of audio→video and video→audio Recall@1 on the held-out split). This is the checkpoint to use for downstream evaluation — DCL contrastive loss has no lower bound (it goes negative when positive similarity exceeds the negative logsumexp), so it is not a reliable model-selection signal.

> When `validation.split_ratio=0` (no held-out split), best-checkpoint tracking falls back to training-set `mean_r1`.

## Cluster execution

### SLURM

SLURM job scripts: `run_transcription.sh`, `run_correpond.sh`. Submit with:

```bash
sbatch run_transcription.sh
```

### SLURM Job Script

SLURM job scripts are not tracked in the repo (they live on the cluster).
Paste this block into your own `#SBATCH` script, after the directives and
`module load` lines:

```bash
# CUDA env vars (required for cudnn/cublas)
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
export LD_LIBRARY_PATH=$(uv run --extra cu128 python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cudnn/lib')"):$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$(uv run --extra cu128 python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cublas/lib')"):$LD_LIBRARY_PATH

cd "$SLURM_SUBMIT_DIR"
uv sync --extra cu128

# Build manifest if not present
[[ -f data/expanded_manifest_segments.json ]] || \
  uv run --extra cu128 python -m scripts.expand_and_validate_manifest

# --- Single-node (default) ---
torchrun --standalone --nnodes=1 --nproc-per-node="$SLURM_GPUS_ON_NODE" \
  -m scripts.train_hydra "$@"

# --- Multi-node (2+ nodes): comment out single-node above, uncomment below ---
# SLURM does NOT set MASTER_ADDR/MASTER_PORT — you must set them manually.
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_PORT=29500
# srun torchrun \
#   --nnodes=$SLURM_JOB_NUM_NODES \
#   --nproc-per-node=$SLURM_GPUS_PER_NODE \
#   --rdzv-backend=c10d \
#   --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
#   -m scripts.train_hydra "$@"
```

Pass config overrides as args: `sbatch myjob.sh training.resume_from=logdir/run1/checkpoint_latest.pth`

### Multi-GPU (DDP)

Distributed Data Parallel training is **implemented** and enabled automatically when launched via `torchrun` (detected via `RANK`/`WORLD_SIZE` env vars). Single-GPU runs (no `torchrun`) are fully backward-compatible — the DDP code paths are gated behind an `is_ddp` flag.

```bash
# Single-node, 4 GPUs (Snellius A100 ×4)
torchrun --standalone --nnodes=1 --nproc-per-node=4 -m scripts.train_hydra

# Single-node, 2 GPUs (smoke test)
torchrun --standalone --nnodes=1 --nproc-per-node=2 -m scripts.train_hydra
```

**How it works:**
- Per-GPU batch size stays at `dataloader.batch_size=64`; the **global** effective batch = `64 × world_size` (256 on 4 GPUs). This gives DCL far more negatives per step.
- Embeddings are all-gathered across GPUs (with gradients, via `torch.distributed.nn.all_gather`) before the contrastive loss, so every GPU sees all `world_size × 64` negatives. Loss is computed locally per-rank to avoid the O(N²) logits memory and gradient-scaling pitfalls of the naive all-gather approach (see OpenCLIP #1144).
- A `DistributedSampler` shards the dataset across GPUs; `train_sampler.set_epoch(epoch)` is called each epoch.
- Only rank 0 logs to TensorBoard/WandB/files and saves checkpoints; other ranks log at `WARNING` level to the console.
- `find_unused_parameters=True` is set on the DDP wrapper (needed because the MLM head is skipped when `mlm_weight=0` and encoders are frozen during warmup).
- Gradient checkpointing uses `use_reentrant=False` (required when `find_unused_parameters=True`).

**Distributed eval:** All ranks process their val shard via `DistributedSampler`, then embeddings are all-gathered so rank 0 computes R@K on the full validation set.

**Encoder warmup:** Encoder params start with `lr=0` (not `requires_grad=False`) so DDP's reducer tracks them from epoch 0. At the unfreeze epoch, the group LR is set to `encoder_lr` — no `add_param_group` mid-training.

**Multi-node (2+ nodes):**
```bash
# SLURM does NOT set MASTER_ADDR/MASTER_PORT — set them manually:
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

srun torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc-per-node=$SLURM_GPUS_PER_NODE \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m scripts.train_hydra
```
The NCCL init timeout defaults to 30 minutes (`hardware.ddp_timeout_min`); increase it for slow interconnects.

### Checkpoint Resume

Training saves three types of checkpoint:

| File | When | Purpose |
|------|------|---------|
| `checkpoint_latest.pth` | Every epoch + every `checkpoint_frequency` steps | Resume after crash |
| `checkpoint_best.pth` | New best `mean_r1` | Resume + inference |
| `checkpoint_epoch{N}.pth` | End of training | Final artifact |

All checkpoints include full state: model, optimizer, scheduler, AMP scaler,
RNG states, best metric, and config.

**Resume from a checkpoint:**
```bash
# Single-GPU
uv run --extra cu128 python -m scripts.train_hydra \
  training.resume_from=logdir/run1/checkpoint_latest.pth

# Multi-GPU
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  -m scripts.train_hydra \
  training.resume_from=logdir/run1/checkpoint_latest.pth

# Via SLURM (pass as args to your sbatch script)
sbatch myjob.sh training.resume_from=logdir/run1/checkpoint_latest.pth
```

On resume:
- All ranks load the checkpoint from the shared filesystem (GPFS).
- Optimizer, scheduler, and AMP scaler states are restored.
- RNG states (Python, PyTorch, NumPy, CUDA) are restored for reproducibility.
- If resuming past the encoder warmup phase, the encoder LR is automatically
  set to `encoder_lr` (no manual intervention needed).
- Training continues from `checkpoint_epoch + 1` to `num_epochs`.

## Known limitations

- **MLM teacher is broken**: `target_projection` is a frozen random linear — do not enable MLM
- **No hard negative mining**: FaST-VGS uses top-K=100 hard negatives; not yet implemented
- **Cross-modal is unidirectional**: Only A→V attention, no V→A or self-attention in cross-modal blocks
- **VideoMAE vs CLIP**: VideoMAE features are optimized for pixel reconstruction, not semantic alignment. CLIP ViT per-frame may be a stronger starting point for audio-video alignment.
