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

# Set HuggingFace cache to a local directory to avoid permission issues
os.environ["TRANSFORMERS_CACHE"] = os.path.join(
    os.path.dirname(__file__), "..", "cache"
)
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "..", "cache")
# Reduce CUDA memory fragmentation over long training runs (prevents the
# "reserved but unallocated" OOM that appears after many epochs). setdefault
# lets an externally-exported value win.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.flik_model import FlikModel
from src.models.losses import CombinedLoss
from src.datasets.video_audio_dataset import VideoAudioDataset
from src.eval.retrieval import retrieval_recall_at_k
from src.utils.logging import setup_logging as flik_setup_logging, log_metrics, close_loggers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_encoder_param(name: str) -> bool:
    """Check if a parameter belongs to a pretrained encoder (wav2vec2 or videomae)."""
    return "wav2vec2" in name or "videomae" in name


def _setup_distributed():
    """Initialize the DDP process group from torchrun env vars.

    Returns (rank, world_size, local_rank, is_ddp). No-op (returns 0,1,0,False)
    when not launched under torchrun / when CUDA is unavailable.
    """
    if not (dist.is_available() and torch.cuda.is_available()):
        return 0, 1, 0, False
    # torchrun sets LOCAL_RANK / RANK / WORLD_SIZE; honor them.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size <= 1 and not os.environ.get("RANK"):
        return 0, 1, 0, False
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, True


def freeze_encoders(model: nn.Module):
    """Freeze pretrained encoder parameters."""
    for name, param in model.named_parameters():
        if _is_encoder_param(name):
            param.requires_grad = False


def unfreeze_encoders(model: nn.Module):
    """Unfreeze pretrained encoder parameters."""
    for name, param in model.named_parameters():
        if _is_encoder_param(name):
            param.requires_grad = True


def build_param_groups(model: nn.Module, lr: float, encoder_lr: float, weight_decay: float):
    """Create two parameter groups: new modules (lr) and pretrained encoders (encoder_lr)."""
    encoder_params = []
    encoder_param_names = []
    new_params = []
    new_param_names = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
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


def cosine_with_warmup_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """LR multiplier: linear warmup → cosine decay → clamp at eta_min."""
    if warmup_steps > 0 and step < warmup_steps:
        return float(step) / max(1.0, float(warmup_steps))
    progress = float(step - warmup_steps) / max(1.0, float(total_steps - warmup_steps))
    # Clamp to prevent cosine from cycling back up after total_steps
    progress = min(1.0, progress)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(cfg, log_dir):
    loggers = flik_setup_logging(
        log_dir=log_dir,
        name="flik",
        tensorboard=cfg.logging.use_tensorboard,
        wandb=cfg.logging.use_wandb,
        wandb_project=cfg.logging.wandb.project,
        wandb_entity=cfg.logging.wandb.entity,
        wandb_offline=cfg.logging.wandb.offline,
        config=OmegaConf.to_container(cfg, resolve=True),
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

        # Backward pass (with optional GradScaler)
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
def evaluate_retrieval(cfg, model, dataloader, device, logger, tb_writer=None, wandb_run=None, step=0):
    model.eval()
    audio_embs = []
    video_embs = []

    for batch in dataloader:
        audio = batch["audio"].to(device)
        video = batch["video"].to(device)
        audio_padding_mask = batch["audio_padding_mask"].to(device)

        audio_embs.append(model.encode_audio(audio, audio_padding_mask).detach())
        video_embs.append(model.encode_video(video).detach())

    if not audio_embs:
        return {}

    audio_embeddings = torch.cat(audio_embs, dim=0)
    video_embeddings = torch.cat(video_embs, dim=0)
    metrics = retrieval_recall_at_k(
        audio_embeddings,
        video_embeddings,
        ks=cfg.validation.retrieval_top_k,
    )
    log_metrics(
        step,
        metrics,
        tb_writer=tb_writer,
        wandb_run=wandb_run,
        logger=logger,
        prefix="retrieval/",
    )
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../src/configs", config_name="default")
def main(cfg: DictConfig):
    from hydra.utils import get_original_cwd
    log_dir = os.path.abspath(os.path.join(get_original_cwd(), cfg.logging.log_dir))

    # DDP init (no-op on single GPU / CPU). Must happen before any CUDA context.
    rank, world_size, local_rank, is_ddp = _setup_distributed()

    loggers = setup_logging(cfg, log_dir, rank=rank)
    logger = loggers["logger"]
    tb_writer = loggers["tb_writer"]
    wandb_run = loggers["wandb_run"]
    logger.info("Starting training with Hydra configuration")
    if is_ddp:
        logger.info(f"DDP initialized: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    # Device
    if is_ddp:
        device = torch.device(f"cuda:{local_rank}")
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
    dataset = VideoAudioDataset(
        manifest_path=cfg.dataset.manifest_path,
        sample_rate=cfg.dataset.sample_rate,
        num_frames=cfg.dataset.num_frames,
        min_duration=cfg.dataset.min_duration,
        max_duration=cfg.dataset.max_duration,
        dummy=cfg.dataset.dummy,
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
    # graph analysis to work with find_unused_parameters=True. Also call
    # enable_input_require_grads() so the checkpointed submodules' inputs
    # require grad (otherwise DDP silently skips their gradients).
    if cfg.training.get("gradient_checkpointing", False) and device.type == "cuda":
        try:
            model.dual_encoder.audio_encoder.wav2vec2.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.dual_encoder.video_encoder.videomae.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
            logger.info("Gradient checkpointing enabled on wav2vec2 + videomae (use_reentrant=False)")
        except Exception as e:
            logger.warning(f"Failed to enable gradient checkpointing: {e}")

    # Encoder warmup: freeze pretrained encoders for first N epochs
    freeze_encoder_epochs = cfg.training.get("freeze_encoder_epochs", 0)
    if freeze_encoder_epochs > 0:
        freeze_encoders(model)
        logger.info(
            f"Pretrained encoders FROZEN for first {freeze_encoder_epochs} epochs "
            f"(projection heads + temporal transformer will train)"
        )

    # Wrap with DDP after moving to device, enabling gradient checkpointing,
    # and freezing encoders. find_unused_parameters=True is required because
    # the MLM head is skipped when mlm_weight=0 and the encoders are frozen
    # pre-unfreeze — those params receive no gradient. gradient_as_bucket_view
    # reuses gradient bucket memory (small perf win). Do NOT use static_graph
    # (incompatible with mid-training encoder unfreeze).
    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
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

    # Optimizer with separate param groups
    encoder_lr = cfg.optimizer.get("encoder_lr", cfg.optimizer.lr)
    param_groups, n_new, n_enc = build_param_groups(
        model, cfg.optimizer.lr, encoder_lr, cfg.optimizer.weight_decay
    )
    logger.info(
        f"Param groups: {n_new} new-module params (lr={cfg.optimizer.lr}), "
        f"{n_enc} encoder params (lr={encoder_lr})"
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

    if cfg.scheduler.type == "cosine_with_warmup":
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_with_warmup_lambda(step, warmup_steps, total_steps),
        )
    elif cfg.scheduler.type == "cosine":
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_with_warmup_lambda(step, 0, total_steps),
        )
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

    # Training loop
    global_step = 0
    for epoch in range(1, cfg.training.num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Unfreeze encoders after warmup
        if freeze_encoder_epochs > 0 and epoch == freeze_encoder_epochs + 1:
            unfreeze_encoders(model)
            # Add encoder params as a NEW param group instead of rebuilding the
            # optimizer. Rebuilding destroys the Adam momentum (exp_avg /
            # exp_avg_sq) that the projection heads built up over the freeze
            # phase, so the first post-unfreeze step applies a raw lr*grad
            # update with no momentum smoothing — a large shock that, combined
            # with the low-temperature DCL loss, reliably triggers embedding
            # collapse within ~2 epochs. add_param_group preserves the
            # existing optimizer state for the new-module params and lets the
            # scheduler's current lambda multiplier apply to the encoder's
            # lower base_lr, giving the encoder a natural warmup ramp too.
            encoder_params = [
                p for n, p in model.named_parameters()
                if _is_encoder_param(n) and p.requires_grad
            ]
            optimizer.add_param_group(
                {
                    "params": encoder_params,
                    "lr": encoder_lr,
                    "weight_decay": cfg.optimizer.weight_decay,
                    "name": "encoder",
                }
            )
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(
                f"Encoders UNFROZEN at epoch {epoch}. "
                f"Trainable params: {trainable_params:,}. "
                f"Added encoder param group (lr={encoder_lr}) to existing optimizer "
                f"(Adam momentum preserved)."
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
        )

        if cfg.training.run_retrieval_eval and (not is_ddp or rank == 0):
            evaluate_retrieval(
                cfg, model, dataloader, device, logger, tb_writer, wandb_run, global_step
            )

    logger.info("Training finished")

    # Save checkpoint (rank 0 only under DDP). Use model.module.state_dict()
    # when DDP-wrapped so checkpoint keys match a plain (non-DDP) model.
    if not is_ddp or rank == 0:
        checkpoint_path = os.path.join(
            cfg.training.checkpoint_dir, f"checkpoint_epoch{cfg.training.num_epochs}.pth"
        )
        os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
        model_to_save = model.module if is_ddp else model
        torch.save(
            {
                "epoch": cfg.training.num_epochs,
                "global_step": global_step,
                "model_state_dict": model_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": OmegaConf.to_container(cfg, resolve=True),
            },
            checkpoint_path,
        )
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    # Sync all ranks before tearing down the process group so the save
    # completes on rank 0 before any rank exits.
    if is_ddp:
        dist.barrier()

    close_loggers(tb_writer=tb_writer, wandb_run=wandb_run)

    if is_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
