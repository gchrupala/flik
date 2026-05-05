#!/usr/bin/env python
"""
Training script for FlikModel with Hydra configuration management.
"""

import sys
import os

# Set HuggingFace cache to a local directory to avoid permission issues
os.environ["TRANSFORMERS_CACHE"] = os.path.join(
    os.path.dirname(__file__), "..", "cache"
)
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "..", "cache")

import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.flik_model import FlikModel
from src.models.losses import CombinedLoss
from src.datasets.video_audio_dataset import VideoAudioDataset
from src.eval.retrieval import retrieval_recall_at_k


def setup_logging(cfg):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    return logger


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
):
    model.train()
    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_mlm_loss = 0.0
    total_contrastive_acc = 0.0
    total_mlm_acc = 0.0
    num_steps = 0

    for step, batch in enumerate(dataloader):
        # Move batch to device
        audio = batch["audio"].to(device)  # (B, 1, T)
        video = batch["video"].to(device)  # (B, F, C, H, W)
        audio_padding_mask = batch["audio_padding_mask"].to(device)  # (B, T_max)
        audio_lengths = batch["audio_lengths"].to(device)  # (B,)

        # Forward pass
        out = model(
            audio,
            video,
            audio_padding_mask=audio_padding_mask,
            mlm_mask=None,  # let model create random mask
            return_features=False,
        )

        # Compute loss
        loss, metrics = loss_fn(
            audio_embeddings=out["audio_embedding"],
            video_embeddings=out["video_embedding"],
            mlm_logits=out.get("mlm_logits"),
            mlm_targets=out.get("mlm_targets"),
            mlm_mask=out.get("mlm_mask"),
        )

        # Scale loss for gradient accumulation
        loss = loss / cfg.training.gradient_accumulation_steps
        loss.backward()

        # Update weights
        if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
            if cfg.training.clip_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.clip_grad_norm
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Accumulate metrics
        total_loss += loss.item() * cfg.training.gradient_accumulation_steps
        total_contrastive_loss += metrics.get("contrastive_loss", 0.0)
        total_contrastive_acc += metrics.get("contrastive_acc", 0.0)
        total_mlm_loss += metrics.get("mlm_loss", 0.0)
        total_mlm_acc += metrics.get("mlm_acc", 0.0)
        num_steps += 1

        # Log according to frequency
        if step % cfg.training.log_frequency == 0:
            logger.info(
                f"Epoch {epoch} | Step {step} | Loss: {loss.item() * cfg.training.gradient_accumulation_steps:.4f} | "
                f"Contrastive: {metrics.get('contrastive_loss', 0.0):.4f} | "
                f"MLM: {metrics.get('mlm_loss', 0.0):.4f}"
            )

    # Compute epoch averages
    avg_loss = total_loss / num_steps if num_steps > 0 else 0.0
    avg_contrastive_loss = total_contrastive_loss / num_steps if num_steps > 0 else 0.0
    avg_contrastive_acc = total_contrastive_acc / num_steps if num_steps > 0 else 0.0
    avg_mlm_loss = total_mlm_loss / num_steps if num_steps > 0 else 0.0
    avg_mlm_acc = total_mlm_acc / num_steps if num_steps > 0 else 0.0

    logger.info(
        f"Epoch {epoch} summary: "
        f"Loss = {avg_loss:.4f}, "
        f"Contrastive = {avg_contrastive_loss:.4f} (acc {avg_contrastive_acc:.3f}), "
        f"MLM = {avg_mlm_loss:.4f} (acc {avg_mlm_acc:.3f})"
    )
    return avg_loss


@torch.no_grad()
def evaluate_retrieval(cfg, model, dataloader, device, logger):
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
    logger.info(f"Retrieval metrics: {metrics}")
    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    logger = setup_logging(cfg)
    logger.info("Starting training with Hydra configuration")

    # Device
    if cfg.hardware.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.hardware.device)
    logger.info(f"Using device: {device}")

    # Dataset & DataLoader
    dataset = VideoAudioDataset(
        manifest_path=cfg.dataset.manifest_path,
        sample_rate=cfg.dataset.sample_rate,
        num_frames=cfg.dataset.num_frames,
        min_duration=cfg.dataset.min_duration,
        max_duration=cfg.dataset.max_duration,
        dummy=cfg.dataset.dummy,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=cfg.dataloader.shuffle,
        collate_fn=VideoAudioDataset.collate_fn,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        drop_last=cfg.dataloader.drop_last,
    )
    logger.info(f"Dataset size: {len(dataset)}")

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
    logger.info(
        f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Loss function
    loss_fn = CombinedLoss(
        contrastive_weight=cfg.loss.contrastive_weight,
        mlm_weight=cfg.loss.mlm_weight,
        temperature=cfg.loss.temperature,
        mlm_label_smoothing=cfg.loss.mlm_label_smoothing,
    ).to(device)

    # Optimizer
    if cfg.optimizer.type == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            betas=tuple(cfg.optimizer.betas),
            eps=cfg.optimizer.eps,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {cfg.optimizer.type}")

    # Scheduler
    if cfg.scheduler.total_steps is None:
        total_steps = (
            len(dataloader)
            * cfg.training.num_epochs
            // cfg.training.gradient_accumulation_steps
        )
    else:
        total_steps = cfg.scheduler.total_steps

    if cfg.scheduler.type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=cfg.scheduler.eta_min,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.scheduler.type}")

    # Training loop
    for epoch in range(1, cfg.training.num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{cfg.training.num_epochs} ---")
        avg_loss = train_one_epoch(
            cfg,
            model,
            dataloader,
            optimizer,
            scheduler,
            loss_fn,
            device,
            logger,
            epoch,
        )

        if cfg.training.run_retrieval_eval:
            evaluate_retrieval(cfg, model, dataloader, device, logger)

    logger.info("Training finished")
    # Save checkpoint
    checkpoint_path = os.path.join(
        cfg.training.checkpoint_dir, f"checkpoint_epoch{cfg.training.num_epochs}.pth"
    )
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    torch.save(
        {
            "epoch": cfg.training.num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        checkpoint_path,
    )
    logger.info(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
