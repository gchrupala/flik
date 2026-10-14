#!/usr/bin/env python
"""
Training script for FlikModel with Hydra configuration management.

Key features:
- DCL (Decoupled Contrastive Learning) loss for small-batch viability
- Mixed precision (AMP) with GradScaler
- Gradient checkpointing for memory efficiency
- Encoder warmup (freeze pretrained encoders for first N epochs)
- Separate LRs for pretrained encoders vs new modules
- Cosine schedule with linear warmup
- Embedding statistics logging (collapse detection)
"""

import sys
import os
import math
import json
import random
import hashlib
from datetime import timedelta

import numpy as np

# Set HuggingFace cache to a local directory to avoid permission issues
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "..", "cache")
# Reduce CUDA memory fragmentation over long training runs (prevents the
# "reserved but unallocated" OOM that appears after many epochs). setdefault
# lets an externally-exported value win.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Restrict each DDP process to its assigned GPU. Without this, all processes
# on a node see ALL GPUs, and NCCL 2.27+ flags same-bus-ID GPUs on different
# nodes as duplicates (e.g. GPU 0 on node 0 and GPU 0 on node 1 both have bus
# ID 31000). Must be set before any CUDA call. With this set, each process
# sees one GPU remapped to index 0.
if "LOCAL_RANK" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from contextlib import nullcontext
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.flik_model import FlikModel
from src.models.losses import CombinedLoss
from src.datasets.video_audio_dataset import VideoAudioDataset
from src.eval.retrieval import retrieval_metrics
from src.utils.logging import setup_logging as flik_setup_logging, log_metrics, close_loggers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_encoder_param(name: str) -> bool:
    """Check if a parameter belongs to a pretrained encoder (wav2vec2 or videomae)."""
    return "wav2vec2" in name or "videomae" in name


def _sha256_file(path: str) -> str:
    """Return the hex sha256 of a file's bytes (content addressing)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_path(path: str, base: str) -> str:
    """Resolve a possibly-relative config path against the original cwd."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base, path))


def _load_movie_key_map(path: str) -> dict:
    """Load an ``id -> movie_key`` mapping from a dedup report.

    Accepts either the full report (uses its ``id_to_movie_key`` field) or a
    plain JSON object that is already the id -> movie_key mapping. Grouping by
    movie_key collapses duplicate copies / multi-part items (e.g. ``VTS_01_1``,
    ``Martha_Ivers``) into one hold-out unit so they cannot leak across
    train/val.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "id_to_movie_key" in data:
        mapping = data["id_to_movie_key"]
    elif isinstance(data, dict):
        mapping = data
    else:
        raise ValueError(f"{path}: expected a dedup report dict or an id->key dict")
    if not isinstance(mapping, dict):
        raise ValueError(f"{path}: 'id_to_movie_key' must be a dict")
    return {str(k): str(v) for k, v in mapping.items()}


def _group_key(item: dict, movie_key_map: dict = None) -> str:
    """Grouping key for a manifest item (movie_key when a dedup map is given)."""
    base = item.get("video_id") or item.get("video_path") or item.get("id") or ""
    if movie_key_map:
        return movie_key_map.get(base, base)
    return base


class StridedSampler(Sampler):
    """Partition dataset indices across ranks exactly (no padding/duplication).

    Rank ``r`` of ``world_size`` W yields indices ``r, r+W, r+2W, ...``. The
    union of all shards equals ``range(len(dataset))`` exactly once. Shards
    differ in size by at most one, so the eval all-gather must pad each rank's
    local tensor to the maximum shard size and trim by per-rank counts.
    """

    def __init__(self, dataset, rank: int, world_size: int):
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        if not (0 <= rank < world_size):
            raise ValueError(f"rank {rank} out of range for world_size {world_size}")
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        n = len(self.dataset)
        if n <= self.rank:
            return 0
        return (n - self.rank + self.world_size - 1) // self.world_size

    def set_epoch(self, epoch: int) -> None:
        """No-op (kept for API parity with DistributedSampler)."""
        return None


def _strided_gather_order(counts, world_size):
    """Source positions that restore dataset order after a strided gather.

    ``counts[r]`` is the number of local embeddings rank ``r`` contributed.
    StridedSampler gives rank ``r`` local position ``j`` the global index
    ``r + j*W``; to rebuild global order ``0..N-1`` we map global index ``i``
    to rank ``i % W`` at local position ``i // W``. Returns the positional
    index list into the rank-ordered concatenation.
    """
    n_pool = sum(counts)
    prefix = [0]
    for c in counts:
        prefix.append(prefix[-1] + c)
    return [prefix[i % world_size] + (i // world_size) for i in range(n_pool)]


def _setup_distributed(timeout_min: int = 30):
    """Initialize the DDP process group from torchrun env vars.

    Returns (rank, world_size, local_rank, is_ddp). No-op (returns 0,1,0,False)
    when not launched under torchrun / when CUDA is unavailable.

    Multi-node: torchrun --nnodes=N --rdzv-backend=c10d sets all required env
    vars (MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK). No code
    changes needed — just launch with the right torchrun flags.
    """
    if not (dist.is_available() and torch.cuda.is_available()):
        return 0, 1, 0, False
    # torchrun sets LOCAL_RANK / RANK / WORLD_SIZE; honor them.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1 and not os.environ.get("RANK"):
        return 0, 1, 0, False
    # Set device BEFORE init_process_group so NCCL binds to the correct GPU.
    # (Not using device_id= arg: NCCL's duplicate-GPU check flags same bus ID
    # across nodes in multi-node — see NCCLUtils.cpp "Duplicate GPU detected".)
    # With CUDA_VISIBLE_DEVICES set, each process sees one GPU at index 0.
    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=timeout_min),
    )
    return rank, world_size, local_rank, True


def build_param_groups(model: nn.Module, lr: float, encoder_lr: float, weight_decay: float):
    """Create two parameter groups: new modules (lr) and pretrained encoders (encoder_lr)."""
    encoder_params = []
    encoder_param_names = []
    new_params = []
    new_param_names = []

    for name, param in model.named_parameters():
        if _is_encoder_param(name):
            encoder_params.append(param)
            encoder_param_names.append(name)
        else:
            new_params.append(param)
            new_param_names.append(name)

    param_groups = [
        {"params": new_params, "lr": lr, "weight_decay": weight_decay, "name": "new"},
    ]
    if encoder_params:
        param_groups.append(
            {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay, "name": "encoder"}
        )

    return param_groups, len(new_params), len(encoder_params)


def _save_checkpoint(
    path, model, optimizer, scheduler, scaler,
    epoch, global_step, best_metric, best_epoch, select_metric,
    cfg, is_ddp,
):
    """Save full training state for resume.

    Includes model, optimizer, scheduler, AMP scaler, RNG states, best metric,
    and config. All three checkpoint types (latest, best, final) use this format.
    """
    model_to_save = model.module if is_ddp else model
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "select_metric": select_metric,
        "config": OmegaConf.to_container(cfg, resolve=True),
        "rng_states": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(),
        },
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(checkpoint, path)


def _load_checkpoint(
    path, model, optimizer, scheduler, scaler,
    is_ddp, logger,
):
    """Load checkpoint and restore all state.

    Returns (start_epoch, global_step, best_metric, best_epoch).
    All ranks load from shared filesystem (standard for HPC with GPFS).
    """
    logger.info(f"Resuming from {path}")
    # weights_only=False: checkpoint contains config dict, RNG states, optimizer
    # state — not just tensors.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    # Load model state (strip module. prefix if checkpoint was saved from DDP)
    model_to_load = model.module if is_ddp else model
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[7:] if k.startswith("module.") else k
        new_state_dict[new_key] = v
    model_to_load.load_state_dict(new_state_dict)
    logger.info("  Restored model state")

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("  Restored optimizer state")
    else:
        logger.warning("  No optimizer state in checkpoint — starting fresh")

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("  Restored scheduler state")

    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        logger.info("  Restored AMP scaler state")

    if "rng_states" in checkpoint:
        rng = checkpoint["rng_states"]
        if rng.get("python") is not None:
            random.setstate(rng["python"])
        if rng.get("torch") is not None:
            torch.set_rng_state(rng["torch"])
        if rng.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])
        if rng.get("numpy") is not None:
            np.random.set_state(rng["numpy"])
        logger.info("  Restored RNG states")

    start_epoch = checkpoint["epoch"] + 1
    global_step = checkpoint["global_step"]
    best_metric = checkpoint.get("best_metric", float("-inf"))
    best_epoch = checkpoint.get("best_epoch", -1)

    logger.info(
        f"  Resumed at epoch {start_epoch}, global_step {global_step}, "
        f"best_metric={best_metric:.4f} @ epoch {best_epoch}"
    )
    return start_epoch, global_step, best_metric, best_epoch


def cosine_with_warmup_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """LR multiplier: linear warmup → cosine decay → clamp at eta_min."""
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / max(1.0, float(warmup_steps))
    progress = float(step - warmup_steps) / max(1.0, float(total_steps - warmup_steps))
    # Clamp to prevent cosine from cycling back up after total_steps
    progress = min(1.0, progress)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _split_manifest_by_video(manifest, val_ratio: float, seed: int, movie_key_map: dict = None):
    """Split a segment-level manifest into (train_items, val_items) grouped by
    movie so no film appears in both splits (prevents cross-film leakage).

    Groups segments by ``_group_key``: the dedup report's movie_key when
    ``movie_key_map`` is given (so duplicate copies / multi-part ids of one film
    stay together), else ``video_id`` (falling back to ``video_path``). Sorts
    the keys for determinism, then shuffles with ``seed`` and takes the first
    ``ceil(n_keys * val_ratio)`` keys as the val set.
    Returns ([...], [...]). If val_ratio <= 0, returns (manifest, []).
    Guarantees at least 1 key in each side when val_ratio > 0 and there are
    >= 2 keys.
    """
    if val_ratio <= 0 or len(manifest) == 0:
        return list(manifest), []

    def _key(item):
        return _group_key(item, movie_key_map)

    groups: dict = {}
    for item in manifest:
        groups.setdefault(_key(item), []).append(item)

    video_keys = sorted(groups.keys())
    n_val = max(1, math.ceil(len(video_keys) * val_ratio))
    n_val = min(n_val, len(video_keys) - 1) if len(video_keys) >= 2 else n_val
    rng = random.Random(seed)
    rng.shuffle(video_keys)
    val_keys = set(video_keys[:n_val])

    train_items, val_items = [], []
    for k in video_keys:
        (val_items if k in val_keys else train_items).extend(groups[k])
    return train_items, val_items


def _selection_metric_value(metrics, sel: str) -> float:
    """Map a config ``select_metric`` name to a scalar from retrieval metrics.

    ``mean_r1``   -- (a2v_r@1 + v2a_r@1) / 2 (legacy default)
    ``hits5_sum`` -- a2v_hits@5 + v2a_hits@5 (frozen-pool protocol)
    Otherwise the metric is looked up by key (0.0 when missing).
    """
    if sel == "mean_r1":
        return (metrics.get("a2v_r@1", 0.0) + metrics.get("v2a_r@1", 0.0)) / 2.0
    if sel == "hits5_sum":
        return float(metrics.get("a2v_hits@5", 0) + metrics.get("v2a_hits@5", 0))
    return float(metrics.get(sel, 0.0))


def _trailing_mean(trailing, epoch: int, window: int = 3) -> float:
    """Mean over the trailing ``window`` epochs ending at ``epoch`` (inclusive)."""
    vals = [v for (e, v) in trailing if epoch - window < e <= epoch]
    return sum(vals) / len(vals) if vals else float("-inf")


def _frozen_live_accept(m, best_metric, best_epoch, n_pool, min_improve_sigma) -> bool:
    """Frozen live-update gate: accept iff ``m > best + sigma_mult * sigma``.

    ``sigma = sqrt(n*p*(1-p))`` uses the incumbent's own hits5 rate
    ``p = best_metric / (2*n_pool)`` (hits5_sum counts both directions over
    n_pool queries). The first accepted epoch (no incumbent) always passes.
    """
    if best_epoch < 0 or not math.isfinite(best_metric):
        return True
    p_inc = best_metric / (2.0 * n_pool) if n_pool else 0.0
    sigma = math.sqrt(max(0.0, n_pool * p_inc * (1.0 - p_inc)))
    return bool(m > best_metric + min_improve_sigma * sigma)


def _append_val_history(history_path: str, epoch: int, metrics: dict) -> None:
    """Append/replace one epoch's record in the history JSON.

    Record: {"epoch", "n_pool", "metrics" (scalars), "hit_vectors" (lists)}.
    Replacing any existing record for the same epoch keeps resume idempotent.
    """
    record = {
        "epoch": int(epoch),
        "n_pool": int(metrics.get("n_pool", 0)),
        "metrics": {
            k: v
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        },
        "hit_vectors": {k: v for k, v in metrics.items() if isinstance(v, list)},
    }
    records = []
    if os.path.exists(history_path):
        try:
            with open(history_path, "r", encoding="utf-8") as f:
                records = json.load(f)
            if not isinstance(records, list):
                records = []
        except (json.JSONDecodeError, OSError):
            records = []
    records = [r for r in records if r.get("epoch") != epoch]
    records.append(record)
    records.sort(key=lambda r: r.get("epoch", 0))
    os.makedirs(os.path.dirname(os.path.abspath(history_path)) or ".", exist_ok=True)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(cfg, log_dir, rank=0):
    loggers = flik_setup_logging(
        log_dir=log_dir,
        name="flik",
        tensorboard=cfg.logging.use_tensorboard,
        wandb=cfg.logging.use_wandb,
        wandb_project=cfg.logging.wandb.project,
        wandb_entity=cfg.logging.wandb.entity,
        wandb_offline=cfg.logging.wandb.offline,
        config=OmegaConf.to_container(cfg, resolve=True),
        rank=rank,
    )
    loggers["logger"].info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    return loggers


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    cfg,
    model,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    logger,
    epoch,
    global_step,
    scaler=None,
    tb_writer=None,
    wandb_run=None,
    is_ddp=False,
    rank=0,
    best_metric=float("-inf"),
    best_epoch=-1,
    select_metric="mean_r1",
):
    model.train()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_contrastive_acc = 0.0
    total_mlm_loss = 0.0
    total_mlm_acc = 0.0
    total_variance_loss = 0.0
    num_steps = 0

    use_amp = scaler is not None
    accum_steps = cfg.training.gradient_accumulation_steps

    import time
    data_times = []
    forward_times = []

    for step, batch in enumerate(dataloader):
        data_t = time.time()
        audio = batch["audio"].to(device)  # (B, 1, T)
        video = batch["video"].to(device)  # (B, F, C, H, W)
        audio_padding_mask = batch["audio_padding_mask"].to(device)  # (B, T_max)
        data_times.append(time.time() - data_t)

        # Forward pass (with optional AMP)
        fwd_t = time.time()
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(
                audio,
                video,
                audio_padding_mask=audio_padding_mask,
                mlm_mask=None,
                return_features=False,
            )
            loss, metrics = loss_fn(
                audio_embeddings=out["audio_embedding"],
                video_embeddings=out["video_embedding"],
                mlm_logits=out.get("mlm_logits"),
                mlm_targets=out.get("mlm_targets"),
                mlm_mask=out.get("mlm_mask"),
            )

        # Scale loss for gradient accumulation
        loss_scaled = loss / accum_steps

        # Under DDP with gradient accumulation (accum_steps > 1), skip the
        # all-reduce gradient sync on intermediate accumulation steps. Only
        # the final step triggers a normal backward() which syncs gradients.
        # With accum_steps=1 (default), every step syncs — no change.
        is_sync_step = (step + 1) % accum_steps == 0
        grad_sync_ctx = (
            model.no_sync()
            if (is_ddp and accum_steps > 1 and not is_sync_step)
            else nullcontext()
        )

        # Backward pass (with optional GradScaler)
        with grad_sync_ctx:
            if use_amp:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

        forward_times.append(time.time() - fwd_t)

        # Update weights
        if (step + 1) % accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)

            if cfg.training.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.clip_grad_norm
                )

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        # Accumulate metrics
        total_loss += loss.item()
        total_contrastive_loss += metrics.get("contrastive_loss", 0.0)
        total_contrastive_acc += metrics.get("contrastive_acc", 0.0)
        total_mlm_loss += metrics.get("mlm_loss", 0.0)
        total_mlm_acc += metrics.get("mlm_acc", 0.0)
        total_variance_loss += metrics.get("variance_loss", 0.0)
        num_steps += 1

        # Log according to frequency
        if global_step % cfg.training.log_frequency == 0:
            # Embedding statistics for collapse detection
            with torch.no_grad():
                audio_emb = out["audio_embedding"]
                video_emb = out["video_embedding"]
                sim_matrix = audio_emb @ video_emb.t()
                batch_size = audio_emb.shape[0]
                diag_mean = sim_matrix.diag().mean().item()
                if batch_size > 1:
                    off_diag_sum = sim_matrix.sum() - sim_matrix.diag().sum()
                    off_diag_mean = (off_diag_sum / (batch_size * (batch_size - 1))).item()
                else:
                    off_diag_mean = 0.0

            log_metrics(
                global_step,
                {
                    "train/loss": loss.item(),
                    "train/contrastive_loss": metrics.get("contrastive_loss", 0.0),
                    "train/contrastive_acc": metrics.get("contrastive_acc", 0.0),
                    "train/mlm_loss": metrics.get("mlm_loss", 0.0),
                    "train/mlm_acc": metrics.get("mlm_acc", 0.0),
                    "train/variance_loss": metrics.get("variance_loss", 0.0),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/lr_encoder": (
                        scheduler.get_last_lr()[1]
                        if len(scheduler.get_last_lr()) > 1
                        else 0.0
                    ),
                    "train/audio_emb_std": audio_emb.std().item(),
                    "train/video_emb_std": video_emb.std().item(),
                    "train/sim_diag_mean": diag_mean,
                    "train/sim_offdiag_mean": off_diag_mean,
                    "train/data_time_avg": sum(data_times[-10:]) / max(1, len(data_times[-10:])),
                    "train/forward_time_avg": sum(forward_times[-10:]) / max(1, len(forward_times[-10:])),
                    "epoch": epoch,
                },
                tb_writer=tb_writer,
                wandb_run=wandb_run,
                logger=logger,
            )

        global_step += 1

        # Periodic checkpoint (every checkpoint_frequency steps).
        # Saves full state for crash recovery. Rank 0 writes, then barrier
        # ensures all ranks wait before proceeding.
        checkpoint_frequency = cfg.training.get("checkpoint_frequency", 0)
        if checkpoint_frequency > 0 and global_step % checkpoint_frequency == 0:
            if not is_ddp or rank == 0:
                latest_path = os.path.join(cfg.training.checkpoint_dir, "checkpoint_latest.pth")
                _save_checkpoint(
                    latest_path, model, optimizer, scheduler, scaler,
                    epoch, global_step, best_metric, best_epoch, select_metric,
                    cfg, is_ddp,
                )
                logger.info(f"Periodic checkpoint saved at step {global_step}")
            if is_ddp:
                dist.barrier()

    # Compute epoch averages
    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
    avg_contrastive_loss = total_contrastive_loss / num_steps if num_steps > 0 else 0.0
    avg_contrastive_acc = total_contrastive_acc / num_steps if num_steps > 0 else 0.0
    avg_mlm_loss = total_mlm_loss / num_steps if num_steps > 0 else 0.0
    avg_mlm_acc = total_mlm_acc / num_steps if num_steps > 0 else 0.0
    avg_variance_loss = total_variance_loss / num_steps if num_steps > 0 else 0.0

    log_metrics(
        global_step,
        {
            "epoch/avg_loss": avg_loss,
            "epoch/avg_contrastive_loss": avg_contrastive_loss,
            "epoch/avg_contrastive_acc": avg_contrastive_acc,
            "epoch/avg_mlm_loss": avg_mlm_loss,
            "epoch/avg_mlm_acc": avg_mlm_acc,
            "epoch/avg_variance_loss": avg_variance_loss,
            "epoch": epoch,
        },
        tb_writer=tb_writer,
        wandb_run=wandb_run,
        logger=logger,
    )
    return global_step


@torch.no_grad()
def evaluate_retrieval(cfg, model, dataloader, device, logger, tb_writer=None, wandb_run=None, step=0, is_ddp=False, val_sampler=None, epoch=None):
    """Compute bidirectional retrieval metrics on the validation pool.

    Under DDP each rank processes its val shard (a StridedSampler partitions
    indices exactly), then embeddings are all-gathered. Shards differ in size
    by at most one, so each rank pads its local tensor to the maximum shard
    size with zeros, gathers, and trims by per-rank counts. Rank 0 computes
    and logs the final metrics; non-zero ranks return {}.

    Returns the full metric dict (scalars + per-query hit vectors) on rank 0.
    """
    model.eval()
    # Eval forward uses AMP when training does — ~2x faster on A100.
    # Convert embeddings to FP32 for ranking precision (cosine sims over
    # ~19k candidates can accumulate FP16 rounding in the sort).
    use_amp = cfg.training.get("mixed_precision", False) and device.type == "cuda"
    audio_embs = []
    video_embs = []

    with torch.no_grad():
        for batch in dataloader:
            audio = batch["audio"].to(device)
            video = batch["video"].to(device)
            audio_padding_mask = batch["audio_padding_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                audio_embs.append(model.encode_audio(audio, audio_padding_mask).detach().float())
                video_embs.append(model.encode_video(video).detach().float())

    if not audio_embs:
        return {}

    audio_embeddings = torch.cat(audio_embs, dim=0)  # (local_N, D)
    video_embeddings = torch.cat(video_embs, dim=0)  # (local_N, D)
    # The pool is the full validation dataset; each rank sees the same dataset.
    expected_n_pool = len(dataloader.dataset)
    is_dist = is_ddp and dist.is_available() and dist.is_initialized()

    if is_dist:
        world_size = dist.get_world_size()
        device_ = audio_embeddings.device
        # Gather per-rank local counts so each rank can trim the concatenation.
        local_count = torch.tensor([audio_embeddings.shape[0]], dtype=torch.long, device=device_)
        gathered_counts = [torch.zeros(1, dtype=torch.long, device=device_) for _ in range(world_size)]
        dist.all_gather(gathered_counts, local_count)
        counts = [int(t.item()) for t in gathered_counts]
        max_n = max(counts)

        # Pad every rank's local tensor to max_n so all_gather shapes match.
        pad = max_n - audio_embeddings.shape[0]
        if pad > 0:
            audio_send = torch.cat(
                [audio_embeddings,
                 torch.zeros(pad, audio_embeddings.shape[1], dtype=audio_embeddings.dtype, device=device_)],
                dim=0,
            )
            video_send = torch.cat(
                [video_embeddings,
                 torch.zeros(pad, video_embeddings.shape[1], dtype=video_embeddings.dtype, device=device_)],
                dim=0,
            )
        else:
            audio_send, video_send = audio_embeddings, video_embeddings

        gathered_audio = [torch.zeros_like(audio_send) for _ in range(world_size)]
        gathered_video = [torch.zeros_like(video_send) for _ in range(world_size)]
        dist.all_gather(gathered_audio, audio_send)
        dist.all_gather(gathered_video, video_send)

        # Trim each rank's shard to its true count, then concatenate.
        audio_embeddings = torch.cat([gathered_audio[r][:counts[r]] for r in range(world_size)], dim=0)
        video_embeddings = torch.cat([gathered_video[r][:counts[r]] for r in range(world_size)], dim=0)

        # Gathered shards are in rank-strided order (r, r+W, ...). Reorder to
        # dataset order so hit vectors are comparable across runs/checkpoints.
        src_order = _strided_gather_order(counts, world_size)
        order_idx = torch.tensor(src_order, dtype=torch.long, device=device_)
        audio_embeddings = audio_embeddings[order_idx]
        video_embeddings = video_embeddings[order_idx]

        assert audio_embeddings.shape[0] == expected_n_pool, (
            f"all-gather n_pool={audio_embeddings.shape[0]} != len(val_dataset)={expected_n_pool}"
        )
        if dist.get_rank() != 0:
            return {}
    else:
        assert audio_embeddings.shape[0] == expected_n_pool, (
            f"n_pool={audio_embeddings.shape[0]} != len(val_dataset)={expected_n_pool}"
        )

    metrics = retrieval_metrics(
        audio_embeddings,
        video_embeddings,
        ks=cfg.validation.retrieval_top_k,
    )
    # Log scalars only (hit vectors are lists, not loggable scalars).
    scalar_metrics = {
        k: v
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    log_metrics(
        step,
        scalar_metrics,
        tb_writer=tb_writer,
        wandb_run=wandb_run,
        logger=logger,
        prefix="retrieval/",
    )

    hits5_sum = metrics.get("a2v_hits@5", 0) + metrics.get("v2a_hits@5", 0)
    logger.info(
        f"Retrieval n_pool={metrics['n_pool']} "
        f"a2v_hits@1/5/10={metrics.get('a2v_hits@1')}/{metrics.get('a2v_hits@5')}/{metrics.get('a2v_hits@10')} "
        f"v2a_hits@1/5/10={metrics.get('v2a_hits@1')}/{metrics.get('v2a_hits@5')}/{metrics.get('v2a_hits@10')} "
        f"hits5_sum={hits5_sum}"
    )

    # Optional embedding dump (off by default) for offline diagnostics.
    if cfg.validation.get("dump_embeddings", False):
        dump_path = os.path.join(
            cfg.training.checkpoint_dir,
            f"val_embeddings_epoch{epoch}.pt" if epoch is not None else "val_embeddings.pt",
        )
        os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)
        torch.save(
            {
                "audio": audio_embeddings.cpu(),
                "video": video_embeddings.cpu(),
                "n_pool": int(metrics["n_pool"]),
                "epoch": epoch,
            },
            dump_path,
        )
        logger.info(f"Dumped validation embeddings to {dump_path}")

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../src/configs", config_name="default")
def main(cfg: DictConfig):
    from hydra.utils import get_original_cwd
    log_dir = os.path.abspath(os.path.join(get_original_cwd(), cfg.logging.log_dir))

    # DDP init (no-op on single GPU / CPU). Must happen before any CUDA context.
    ddp_timeout_min = cfg.hardware.get("ddp_timeout_min", 30)
    rank, world_size, local_rank, is_ddp = _setup_distributed(timeout_min=ddp_timeout_min)

    loggers = setup_logging(cfg, log_dir, rank=rank)
    logger = loggers["logger"]
    tb_writer = loggers["tb_writer"]
    wandb_run = loggers["wandb_run"]
    logger.info("Starting training with Hydra configuration")
    if is_ddp:
        logger.info(f"DDP initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    # Device
    if is_ddp:
        device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES remaps to index 0
    elif cfg.hardware.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.hardware.device)
    logger.info(f"Using device: {device}")

    # Enable cudnn benchmark for consistent input sizes (video frames are always 224x224)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("cudnn benchmark enabled")

    # Dataset & DataLoader
    # In dummy mode, skip manifest I/O entirely and let the dataset generate
    # random tensors (fast smoke test of the training loop without disk I/O).
    # Otherwise, load manifest once and split into train/val at the video
    # level so no film appears in both (prevents leakage). split_ratio=0
    # disables the split and evaluates on the training set (legacy behavior).
    original_cwd = get_original_cwd()
    val_manifest_path = cfg.dataset.get("val_manifest_path", None)
    # Audio window alignment (defect fix). Historical default was the buggy
    # keyframe pre-roll (false); runs logged before 2026-09-16 used false.
    audio_align_to_start = bool(cfg.dataset.get("audio_align_to_start", True))
    logger.info(
        "Audio window alignment: "
        + (
            "timestamp-accurate, starts at start_sec (audio_align_to_start=true)"
            if audio_align_to_start
            else "legacy keyframe pre-roll retained (audio_align_to_start=false, bug-compatible)"
        )
    )
    train_items, val_items = None, []
    if cfg.dataset.dummy:
        logger.info("Dummy mode: skipping manifest load (random tensors)")
    else:
        manifest_sha = _sha256_file(cfg.dataset.manifest_path)
        with open(cfg.dataset.manifest_path, "r", encoding="utf-8") as _f:
            full_manifest = json.load(_f)
        logger.info(
            f"Training manifest sha256: {manifest_sha} ({cfg.dataset.manifest_path})"
        )
        # Optional movie-level dedup mapping: group ids that are copies/parts of
        # one film so they cannot straddle the train/val boundary.
        dedup_report_path = cfg.dataset.get("dedup_report_path", None)
        movie_key_map = None
        if dedup_report_path:
            dedup_path = _resolve_path(dedup_report_path, original_cwd)
            movie_key_map = _load_movie_key_map(dedup_path)
            logger.info(
                f"Dedup/movie-key map sha256: {_sha256_file(dedup_path)} "
                f"({dedup_path}, {len(movie_key_map)} ids)"
            )
        logger.info(
            "Hold-out grouping: "
            + ("movie_key (dedup report)" if movie_key_map else "video_id")
        )
        if val_manifest_path:
            # Frozen validation pool: no runtime split. Hold out the pool's
            # movies from training so no film leaks into both sides.
            pool_path = _resolve_path(val_manifest_path, original_cwd)
            pool_sha = _sha256_file(pool_path)
            with open(pool_path, "r", encoding="utf-8") as _f:
                val_items = json.load(_f)
            pool_keys = {_group_key(i, movie_key_map) for i in val_items}
            train_items = [
                i
                for i in full_manifest
                if _group_key(i, movie_key_map) not in pool_keys
            ]
            logger.info(
                f"Validation pool sha256: {pool_sha} ({pool_path}, "
                f"{len(val_items)} segments, {len(pool_keys)} movies)"
            )
            dropped = len(full_manifest) - len(train_items)
            logger.info(
                f"Frozen pool holds out {len(pool_keys)} movies; "
                f"removed {dropped} segments from training"
            )
            logger.info(
                f"Frozen pool split: {len(train_items)} train / {len(val_items)} val "
                f"segments (runtime split skipped)"
            )
        else:
            val_ratio = cfg.validation.get("split_ratio", 0.0)
            split_seed = cfg.validation.get("split_seed", 42)
            train_items, val_items = _split_manifest_by_video(
                full_manifest, val_ratio, split_seed, movie_key_map
            )
            n_units = len({_group_key(i, movie_key_map) for i in full_manifest})
            logger.info(
                f"Manifest split: {len(train_items)} train / {len(val_items)} val segments "
                f"(val_ratio={val_ratio}, seed={split_seed}, "
                f"{'movies' if movie_key_map else 'videos'}={n_units})"
            )
            logger.info(
                "Validation pool: runtime split (dataset.val_manifest_path is null); "
                "no frozen sha256"
            )
        if not val_items:
            logger.warning("Validation split is empty — retrieval eval will run on the TRAIN set")

    dataset = VideoAudioDataset(
        manifest_path=cfg.dataset.manifest_path,
        sample_rate=cfg.dataset.sample_rate,
        num_frames=cfg.dataset.num_frames,
        min_duration=cfg.dataset.min_duration,
        max_duration=cfg.dataset.max_duration,
        dummy=cfg.dataset.dummy,
        manifest_items=train_items,
        audio_align_to_start=audio_align_to_start,
    )
    # Under DDP, use a DistributedSampler (disjoint shards per rank). The
    # sampler handles shuffling; DataLoader.shuffle must be False when a
    # sampler is provided. set_epoch() is called in the training loop.
    if is_ddp:
        train_sampler = DistributedSampler(
            dataset, shuffle=True, drop_last=cfg.dataloader.drop_last
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = cfg.dataloader.shuffle
    # DataLoader: prefetch_factor only valid when num_workers > 0
    dl_kwargs = dict(
        batch_size=cfg.dataloader.batch_size,
        shuffle=shuffle,
        collate_fn=VideoAudioDataset.collate_fn,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        drop_last=cfg.dataloader.drop_last,
    )
    if train_sampler is not None:
        dl_kwargs["sampler"] = train_sampler
    if cfg.dataloader.num_workers > 0:
        dl_kwargs["persistent_workers"] = cfg.dataloader.get("persistent_workers", False)
        dl_kwargs["prefetch_factor"] = cfg.dataloader.get("prefetch_factor", None)
    dataloader = DataLoader(dataset, **dl_kwargs)
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Batch size: {cfg.dataloader.batch_size}" + (f" (per-rank; global={cfg.dataloader.batch_size * world_size})" if is_ddp else ""))

    # Validation dataset/dataloader. Under DDP, ALL ranks get a val_dataloader
    # with a DistributedSampler so eval is parallelized across GPUs. Each rank
    # computes embeddings for its val shard, then embeddings are all-gathered
    # and rank 0 computes R@K on the full set.
    val_dataloader = None
    val_sampler = None
    if cfg.training.run_retrieval_eval and val_items:
        val_dataset = VideoAudioDataset(
            manifest_path=cfg.dataset.manifest_path,
            sample_rate=cfg.dataset.sample_rate,
            num_frames=cfg.dataset.num_frames,
            min_duration=cfg.dataset.min_duration,
            max_duration=cfg.dataset.max_duration,
            dummy=cfg.dataset.dummy,
            manifest_items=val_items,
            audio_align_to_start=audio_align_to_start,
        )
        if is_ddp:
            if len(val_dataset) < world_size:
                raise ValueError(
                    f"validation pool ({len(val_dataset)}) smaller than world_size "
                    f"({world_size}); strided eval needs at least one sample per rank"
                )
            # Exact strided partition: rank r gets r, r+W, r+2W, ... No padding
            # and no duplicated samples (DistributedSampler would pad shards).
            val_sampler = StridedSampler(val_dataset, rank=rank, world_size=world_size)
            val_shuffle = False
        else:
            val_sampler = None
            val_shuffle = False
        val_dl_kwargs = dict(
            batch_size=cfg.validation.eval_batch_size,
            shuffle=val_shuffle,
            collate_fn=VideoAudioDataset.collate_fn,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
            drop_last=False,
        )
        if val_sampler is not None:
            val_dl_kwargs["sampler"] = val_sampler
        if cfg.dataloader.num_workers > 0:
            val_dl_kwargs["persistent_workers"] = cfg.dataloader.get("persistent_workers", False)
            val_dl_kwargs["prefetch_factor"] = cfg.dataloader.get("prefetch_factor", None)
        val_dataloader = DataLoader(val_dataset, **val_dl_kwargs)
        logger.info(f"Validation dataset size: {len(val_dataset)} (batch_size={cfg.validation.eval_batch_size})")

    # Pre-download HuggingFace models on rank 0 before all ranks construct
    # FlikModel. Without this, all ranks call from_pretrained() simultaneously,
    # racing on the same HF cache directory (os.environ["HF_HOME"]). On GPFS
    # and other shared filesystems this causes partial writes, corrupt cache
    # entries, and DDP parameter count mismatches (e.g. rank 0 has 467 params
    # while rank 1 has 0). Fix: rank 0 downloads first, barrier, then all ranks
    # construct the model from the now-complete cache.
    if is_ddp:
        if rank == 0:
            logger.info("Rank 0: pre-downloading HuggingFace models to cache...")
            from transformers import Wav2Vec2Model, VideoMAEModel
            Wav2Vec2Model.from_pretrained(cfg.model.audio_model_name)
            VideoMAEModel.from_pretrained(cfg.model.video_model_name)
            logger.info("Rank 0: model download complete.")
        dist.barrier()
        logger.info(f"Rank {rank}: constructing FlikModel (cache populated)")

    # Model
    model = FlikModel(
        audio_model_name=cfg.model.audio_model_name,
        video_model_name=cfg.model.video_model_name,
        hidden_dim=cfg.model.hidden_dim,
        audio_feature_layer=cfg.model.audio_feature_layer,
        temporal_layers=cfg.model.temporal_layers,
        cross_attention_layers=cfg.model.cross_attention_layers,
        num_heads=cfg.model.num_heads,
        ff_dim=cfg.model.ff_dim,
        dropout=cfg.model.dropout,
        freeze_audio=cfg.model.freeze_audio,
        freeze_video=cfg.model.freeze_video,
        use_grounded_masked_prediction=cfg.model.use_grounded_masked_prediction,
        mlm_mask_prob=cfg.model.mlm_mask_prob,
        mlm_mask_length=cfg.model.mlm_mask_length,
        num_codebook_entries=cfg.model.num_codebook_entries,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model initialized: {total_params:,} total params, {trainable_params:,} trainable")

    # Gradient checkpointing (saves memory at the cost of recomputation).
    # CRITICAL under DDP: use_reentrant=False is required for DDP's autograd
    # graph analysis to work with find_unused_parameters=True.
    if cfg.training.get("gradient_checkpointing", False) and device.type == "cuda":
        try:
            model.dual_encoder.audio_encoder.wav2vec2.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.dual_encoder.video_encoder.videomae.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            logger.info("Gradient checkpointing enabled on wav2vec2 + videomae (use_reentrant=False)")
        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing on encoders: {e}")

        # enable_input_require_grads() registers a hook so checkpointed
        # submodules' inputs require grad. Only needed for NLP models where
        # input is token IDs → embedding lookup (no grad through the lookup).
        # For audio (float waveform) and video (float frames), inputs already
        # require grad. Wav2Vec2Model doesn't implement get_input_embeddings(),
        # so this call raises — wrap in try/except.
        try:
            model.dual_encoder.audio_encoder.wav2vec2.enable_input_require_grads()
            model.dual_encoder.video_encoder.videomae.enable_input_require_grads()
        except Exception:
            pass  # Not needed for audio/video — inputs are float tensors

    # Encoder warmup: encoder params use lr=0 for first N epochs,
    # then their param group LR is set to encoder_lr. This keeps all params
    # in the DDP autograd graph from epoch 0 (no requires_grad toggling,
    # no add_param_group mid-training — both are DDP-fragile).
    freeze_encoder_epochs = cfg.training.get("freeze_encoder_epochs", 0)

    # Wrap with DDP after moving to device and enabling gradient checkpointing.
    # find_unused_parameters=True is needed because the MLM head is skipped
    # when mlm_weight=0. With the lr=0 warmup approach, all params are always
    # in the autograd graph (no requires_grad toggling) — DDP-robust.
    # gradient_as_bucket_view reuses gradient bucket memory (small perf win).
    if is_ddp:
        model = DDP(
            model,
            device_ids=[0],  # CUDA_VISIBLE_DEVICES remaps to index 0
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
        logger.info(f"Model wrapped with DDP (find_unused_parameters=True)")

    # Loss function
    loss_fn = CombinedLoss(
        contrastive_weight=cfg.loss.contrastive_weight,
        mlm_weight=cfg.loss.mlm_weight,
        temperature=cfg.loss.temperature,
        use_dcl=cfg.loss.get("use_dcl", True),
        mlm_label_smoothing=cfg.loss.mlm_label_smoothing,
        variance_weight=cfg.loss.get("variance_weight", 0.0),
        variance_gamma=cfg.loss.get("variance_gamma", None),
        hidden_dim=cfg.model.hidden_dim,
    ).to(device)
    logger.info(
        f"Loss: contrastive_weight={cfg.loss.contrastive_weight}, "
        f"mlm_weight={cfg.loss.mlm_weight}, "
        f"use_dcl={cfg.loss.get('use_dcl', True)}, "
        f"temperature={cfg.loss.temperature}, "
        f"variance_weight={cfg.loss.get('variance_weight', 0.0)}"
    )

    # Optimizer with separate param groups.
    # During encoder warmup (freeze_encoder_epochs > 0), the encoder param
    # group has base_lr=encoder_lr but a per-group LR lambda returns 0.0
    # during freeze steps, then follows the cosine schedule after. This
    # keeps requires_grad=True (DDP reducer tracks from start) while applying
    # no gradients during warmup. The per-group lambda replaces the old
    # manual unfreeze blocks that were defeated by LambdaLR's base_lr snapshot.
    encoder_lr = cfg.optimizer.get("encoder_lr", cfg.optimizer.lr)
    encoder_init_lr = encoder_lr
    param_groups, n_new, n_enc = build_param_groups(
        model, cfg.optimizer.lr, encoder_init_lr, cfg.optimizer.weight_decay
    )
    logger.info(
        f"Param groups: {n_new} new-module params (lr={cfg.optimizer.lr}), "
        f"{n_enc} encoder params (lr={encoder_init_lr})"
        + (f" [warmup: lr=0 for {freeze_encoder_epochs} epochs]" if freeze_encoder_epochs > 0 else "")
    )

    if cfg.optimizer.type == "adamw":
        optimizer = AdamW(
            param_groups,
            weight_decay=cfg.optimizer.weight_decay,
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {cfg.optimizer.type}")

    # Scheduler: cosine with linear warmup
    if cfg.scheduler.total_steps is None:
        total_steps = (
            len(dataloader) * cfg.training.num_epochs
            // cfg.training.gradient_accumulation_steps
        )
    else:
        total_steps = cfg.scheduler.total_steps

    warmup_steps = cfg.scheduler.get("warmup_steps", 0)

    if cfg.scheduler.type in ("cosine_with_warmup", "cosine"):
        sched_warmup = warmup_steps if cfg.scheduler.type == "cosine_with_warmup" else 0
        new_lambda = lambda step: cosine_with_warmup_lambda(
            step, sched_warmup, total_steps
        )
        lr_lambdas = [new_lambda]
        if n_enc > 0:
            if freeze_encoder_epochs > 0:
                freeze_steps = (
                    len(dataloader) * freeze_encoder_epochs
                    // cfg.training.gradient_accumulation_steps
                )
                def enc_lambda(step, fs=freeze_steps, ws=sched_warmup, ts=total_steps):
                    if step < fs:
                        return 0.0
                    return cosine_with_warmup_lambda(step, ws, ts)
                lr_lambdas.append(enc_lambda)
            else:
                lr_lambdas.append(new_lambda)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambdas)
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.scheduler.type}")

    logger.info(
        f"Scheduler: {cfg.scheduler.type}, total_steps={total_steps}, "
        f"warmup_steps={warmup_steps}"
    )

    # Mixed precision (AMP)
    use_amp = (
        cfg.training.get("mixed_precision", False)
        and device.type == "cuda"
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    if use_amp:
        logger.info("Mixed precision (AMP) enabled")
    else:
        logger.info("Mixed precision disabled (CPU or config flag off)")

    # Resume from checkpoint (all ranks load from shared filesystem).
    # Restores model, optimizer, scheduler, scaler, RNG states, best metric.
    # If resuming past encoder warmup, encoder LR is set to encoder_lr automatically.
    start_epoch = 1
    global_step = 0
    best_metric = float("-inf")
    best_epoch = -1
    select_metric = cfg.validation.get("select_metric", "mean_r1")
    resume_from = cfg.training.get("resume_from", None)
    if resume_from:
        start_epoch, global_step, best_metric, best_epoch = _load_checkpoint(
            resume_from, model, optimizer, scheduler, scaler, is_ddp, logger,
        )
        # Fix base_lrs for encoder group (migration from old scheduler bug
        # where encoder base_lr was pinned to 0.0 by LambdaLR). The per-group
        # lambda now handles the freeze→unfreeze transition automatically.
        for i, group in enumerate(optimizer.param_groups):
            if group.get("name") == "encoder":
                scheduler.base_lrs[i] = encoder_lr
                break

    # Frozen-pool checkpoint selection (opt-in). When disabled, the legacy
    # "max metric" rule is used and behaviour is unchanged.
    frozen_selection = bool(cfg.validation.get("enable_frozen_selection", False))
    min_improve_sigma = float(cfg.validation.get("min_improve_sigma", 2.0))
    history_path_cfg = cfg.validation.get("history_path", None)
    history_path = (
        _resolve_path(history_path_cfg, original_cwd)
        if history_path_cfg
        else os.path.join(cfg.training.checkpoint_dir, "val_history.json")
    )
    candidate_path = os.path.join(cfg.training.checkpoint_dir, "checkpoint_best_candidate.pth")
    trailing = []  # [(epoch, hits5_sum)] for the trailing-3-epoch rule
    best_trailing_mean = float("-inf")
    winner_epoch = -1
    if frozen_selection:
        select_metric = "hits5_sum"
        logger.info(
            f"Frozen-pool selection ENABLED: select_metric=hits5_sum, "
            f"min_improve_sigma={min_improve_sigma}, history={history_path}"
        )

    # Training loop
    for epoch in range(start_epoch, cfg.training.num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Encoder unfreeze: the per-group LR lambda automatically transitions
        # from 0 (during freeze steps) to the cosine schedule. No manual LR
        # override needed — the lambda handles it.
        if freeze_encoder_epochs > 0 and epoch == freeze_encoder_epochs + 1:
            logger.info(
                f"Encoders UNFROZEN at epoch {epoch} (per-group LR lambda: "
                f"encoder lr now {encoder_lr:.2e} * cosine multiplier)."
            )

        logger.info(f"--- Epoch {epoch}/{cfg.training.num_epochs} ---")
        global_step = train_one_epoch(
            cfg,
            model,
            dataloader,
            optimizer,
            scheduler,
            loss_fn,
            device,
            logger,
            epoch,
            global_step,
            scaler=scaler,
            tb_writer=tb_writer,
            wandb_run=wandb_run,
            is_ddp=is_ddp,
            rank=rank,
            best_metric=best_metric,
            best_epoch=best_epoch,
            select_metric=select_metric,
        )

        eval_every_n = cfg.training.get("eval_every_n_epochs", 1)
        is_last_epoch = epoch >= cfg.training.num_epochs
        if (cfg.training.run_retrieval_eval and val_dataloader is not None
                and (is_last_epoch or epoch % max(1, eval_every_n) == 0)):
            eval_model = model.module if is_ddp else model
            # Under DDP, set the val sampler epoch so sharding is deterministic
            # (though shuffle=False, set_epoch is good practice).
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            val_metrics = evaluate_retrieval(
                cfg, eval_model, val_dataloader, device, logger, tb_writer, wandb_run, global_step,
                is_ddp=is_ddp, val_sampler=val_sampler, epoch=epoch,
            )
            # Track best checkpoint by the configured selection metric.
            # Default select_metric="mean_r1" = (a2v_r@1 + v2a_r@1)/2.
            if val_metrics:
                sel = "hits5_sum" if frozen_selection else cfg.validation.get("select_metric", "mean_r1")
                m = _selection_metric_value(val_metrics, sel)
                is_writer = not is_ddp or rank == 0
                n_pool = int(val_metrics.get("n_pool", 0))

                # Record every eval (metric scalars + per-query hit vectors).
                if is_writer:
                    _append_val_history(history_path, epoch, val_metrics)

                new_trailing_leader = False
                if frozen_selection:
                    h5 = _selection_metric_value(val_metrics, "hits5_sum")
                    trailing.append((epoch, h5))
                    tmean = _trailing_mean(trailing, epoch, window=3)
                    if tmean > best_trailing_mean:
                        best_trailing_mean = tmean
                        winner_epoch = epoch
                        new_trailing_leader = True

                # Live update rule. Legacy: any improvement. Frozen: require the
                # new hits5_sum to beat the incumbent by min_improve_sigma
                # binomial sigma (sigma = sqrt(n*p*(1-p)) from the incumbent's
                # own rate p), so a fluke does not become "best".
                if frozen_selection:
                    accept = _frozen_live_accept(
                        m, best_metric, best_epoch, n_pool, min_improve_sigma
                    )
                else:
                    accept = m > best_metric
                if accept:
                    best_metric = m
                    best_epoch = epoch
                    if is_writer:
                        best_path = os.path.join(cfg.training.checkpoint_dir, "checkpoint_best.pth")
                        _save_checkpoint(
                            best_path, model, optimizer, scheduler, scaler,
                            epoch, global_step, best_metric, best_epoch, sel,
                            cfg, is_ddp,
                        )
                        logger.info(f"New best {sel}={best_metric:.4f} @ epoch {epoch} — saved {best_path}")

                # Save the trailing-3 leader's state AFTER the live update so the
                # candidate checkpoint carries current best_metric/best_epoch.
                if frozen_selection and new_trailing_leader and is_writer:
                    _save_checkpoint(
                        candidate_path, model, optimizer, scheduler, scaler,
                        epoch, global_step, best_metric, best_epoch, sel,
                        cfg, is_ddp,
                    )
                    logger.info(
                        f"Post-hoc leader epoch {epoch}: trailing-3 mean "
                        f"hits5_sum={best_trailing_mean:.1f}"
                    )

        # Save latest checkpoint (every epoch, rank 0 only under DDP).
        # Full state for crash recovery. Barrier ensures all ranks wait for save.
        if not is_ddp or rank == 0:
            latest_path = os.path.join(cfg.training.checkpoint_dir, "checkpoint_latest.pth")
            _save_checkpoint(
                latest_path, model, optimizer, scheduler, scaler,
                epoch, global_step, best_metric, best_epoch, select_metric,
                cfg, is_ddp,
            )
            logger.info(f"Latest checkpoint saved to {latest_path}")
        if is_ddp:
            dist.barrier()

    best_sel_name = "hits5_sum" if frozen_selection else cfg.validation.get("select_metric", "mean_r1")
    logger.info(f"Best val {best_sel_name}={best_metric:.4f} at epoch {best_epoch}")
    logger.info("Training finished")

    # Post-hoc selection: pick the epoch with the maximal trailing-3-epoch mean
    # of hits5_sum (ties go to the earlier epoch) and rewrite checkpoint_best.pth
    # if it differs from the live best. Runs on rank 0 only.
    if frozen_selection:
        best_path = os.path.join(cfg.training.checkpoint_dir, "checkpoint_best.pth")
        is_writer = not is_ddp or rank == 0
        if is_writer:
            if winner_epoch > 0 and winner_epoch != best_epoch:
                if os.path.exists(candidate_path):
                    os.replace(candidate_path, best_path)
                    logger.info(
                        f"Post-hoc selection: epoch {winner_epoch} (trailing-3 mean "
                        f"hits5_sum={best_trailing_mean:.1f}) differs from live best "
                        f"epoch {best_epoch}; rewrote {best_path}"
                    )
                else:
                    logger.warning(
                        f"Post-hoc winner epoch {winner_epoch} has no candidate "
                        f"checkpoint at {candidate_path}; keeping live best"
                    )
            elif os.path.exists(candidate_path):
                os.remove(candidate_path)

    # Save final checkpoint (rank 0 only under DDP). Full state for resume/inference.
    if not is_ddp or rank == 0:
        final_path = os.path.join(
            cfg.training.checkpoint_dir, f"checkpoint_epoch{cfg.training.num_epochs}.pth"
        )
        _save_checkpoint(
            final_path, model, optimizer, scheduler, scaler,
            cfg.training.num_epochs, global_step, best_metric, best_epoch, select_metric,
            cfg, is_ddp,
        )
        logger.info(f"Final checkpoint saved to {final_path}")

    # Sync all ranks before tearing down the process group so the save
    # completes on rank 0 before any rank exits.
    if is_ddp:
        dist.barrier()

    close_loggers(tb_writer=tb_writer, wandb_run=wandb_run)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
