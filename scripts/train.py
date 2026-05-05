#!/usr/bin/env python
"""
Training script for FlikModel (single‑GPU, dummy data).
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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.flik_model import FlikModel
from src.models.losses import CombinedLoss
from src.datasets.video_audio_dataset import VideoAudioDataset


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    logger,
    epoch,
    gradient_accumulation_steps=1,
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
        loss = loss / gradient_accumulation_steps
        loss.backward()

        # Update weights
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # Accumulate metrics
        total_loss += loss.item() * gradient_accumulation_steps
        total_contrastive_loss += metrics.get("contrastive_loss", 0.0)
        total_contrastive_acc += metrics.get("contrastive_acc", 0.0)
        total_mlm_loss += metrics.get("mlm_loss", 0.0)
        total_mlm_acc += metrics.get("mlm_acc", 0.0)
        num_steps += 1

        # Log every 10 steps
        if step % 10 == 0:
            logger.info(
                f"Epoch {epoch} | Step {step} | Loss: {loss.item() * gradient_accumulation_steps:.4f} | "
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


def main():
    logger = setup_logging()
    logger.info("Starting training with dummy data")

    # Hyperparameters (can be moved to config later)
    batch_size = 8
    num_epochs = 2
    learning_rate = 5e-5
    weight_decay = 0.01
    gradient_accumulation_steps = 1
    num_workers = 0  # for dummy data

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Dataset & DataLoader (dummy mode)
    dataset = VideoAudioDataset("", dummy=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=VideoAudioDataset.collate_fn,
        num_workers=num_workers,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    # Model
    model = FlikModel(use_grounded_masked_prediction=False).to(device)
    logger.info(
        f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # Loss function
    loss_fn = CombinedLoss(contrastive_weight=1.0, mlm_weight=1.0).to(device)

    # Optimizer & scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    total_steps = len(dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    # Training loop
    for epoch in range(1, num_epochs + 1):
        logger.info(f"--- Epoch {epoch}/{num_epochs} ---")
        avg_loss = train_one_epoch(
            model,
            dataloader,
            optimizer,
            scheduler,
            loss_fn,
            device,
            logger,
            epoch,
            gradient_accumulation_steps,
        )

    logger.info("Training finished")
    # Save checkpoint
    checkpoint_path = "checkpoint_dummy.pth"
    torch.save(
        {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
        },
        checkpoint_path,
    )
    logger.info(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
