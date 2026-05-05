#!/usr/bin/env python
"""
Test the full pipeline with random weights (no pretrained downloads).
"""

import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.flik_model import FlikModel
from src.models.losses import CombinedLoss
from src.datasets.video_audio_dataset import VideoAudioDataset


def test_dataset():
    print("Testing Dataset...")
    dataset = VideoAudioDataset("", dummy=True)
    print(f"  Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"  Audio shape: {sample['audio'].shape}")
    print(f"  Video shape: {sample['video'].shape}")
    # Test collate
    batch = [dataset[i] for i in range(4)]
    collated = VideoAudioDataset.collate_fn(batch)
    print(f"  Collated audio shape: {collated['audio'].shape}")
    print(f"  Collated video shape: {collated['video'].shape}")
    print(f"  Collated padding mask shape: {collated['audio_padding_mask'].shape}")
    print("  Dataset test passed.")
    return dataset


def test_model():
    print("Testing Model (pretrained=False)...")
    model = FlikModel(
        pretrained_audio=False,
        pretrained_video=False,
        mlm_mask_prob=0.15,
    )
    batch_size = 2
    audio = torch.randn(batch_size, 1, 16000)  # 1 second at 16kHz
    video = torch.randn(batch_size, 16, 3, 224, 224)
    audio_padding_mask = torch.zeros(batch_size, 16000, dtype=torch.bool)

    out = model(audio, video, audio_padding_mask, return_features=False)
    print(f"  Audio embedding shape: {out['audio_embedding'].shape}")
    print(f"  Video embedding shape: {out['video_embedding'].shape}")
    print(f"  Grounded MLM enabled: {'mlm_logits' in out}")
    # Test encode methods
    audio_emb = model.encode_audio(audio)
    video_emb = model.encode_video(video)
    print(f"  Encode audio shape: {audio_emb.shape}")
    print(f"  Encode video shape: {video_emb.shape}")
    print("  Model test passed.")
    return model


def test_loss():
    print("Testing Loss functions...")
    batch, seq_len, hidden = 4, 50, 768
    audio_emb = torch.randn(batch, hidden)
    video_emb = torch.randn(batch, hidden)
    logits = torch.randn(batch, seq_len, 320)
    targets = torch.randint(0, 320, (batch, seq_len))
    mask = torch.rand(batch, seq_len) > 0.85

    # Contrastive loss
    from src.models.losses import ContrastiveLoss

    cont_loss, metrics = ContrastiveLoss()(audio_emb, video_emb)
    print(
        f"  Contrastive loss: {cont_loss.item():.4f}, acc: {metrics['contrastive_acc']:.3f}"
    )

    # MLM loss
    from src.models.losses import MLMLoss

    mlm_loss = MLMLoss()
    loss_mlm, metrics_mlm = mlm_loss(logits, targets, mask)
    print(f"  MLM loss: {loss_mlm.item():.4f}, acc: {metrics_mlm['mlm_acc']:.3f}")

    # Combined loss
    combined = CombinedLoss()
    total, metrics = combined(audio_emb, video_emb, logits, targets, mask)
    print(f"  Combined total loss: {total.item():.4f}")
    for k, v in metrics.items():
        print(f"    {k}: {v}")
    print("  Loss test passed.")
    return combined


def test_training_step():
    print("Simulating training step...")
    model = FlikModel(
        pretrained_audio=False,
        pretrained_video=False,
        mlm_mask_prob=0.15,
    )
    loss_fn = CombinedLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    dataset = VideoAudioDataset("", dummy=True)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=VideoAudioDataset.collate_fn,
        num_workers=0,
    )
    batch = next(iter(dataloader))
    audio = batch["audio"]
    video = batch["video"]
    audio_padding_mask = batch["audio_padding_mask"]

    # Forward
    out = model(audio, video, audio_padding_mask=audio_padding_mask)
    loss, metrics = loss_fn(
        audio_embeddings=out["audio_embedding"],
        video_embeddings=out["video_embedding"],
        mlm_logits=out.get("mlm_logits"),
        mlm_targets=out.get("mlm_targets"),
        mlm_mask=out.get("mlm_mask"),
    )
    print(f"  Loss: {loss.item():.4f}")
    # Backward
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("  Training step passed.")


def main():
    print("=== Pipeline Test (no pretrained downloads) ===")
    torch.manual_seed(42)
    test_dataset()
    print()
    test_model()
    print()
    test_loss()
    print()
    test_training_step()
    print("\nAll tests passed successfully!")


if __name__ == "__main__":
    main()
