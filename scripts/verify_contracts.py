#!/usr/bin/env python

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.datasets.video_audio_dataset import VideoAudioDataset
from src.models.audio_encoder import AudioEncoder
from src.models.video_encoder import VideoEncoder
from src.models.flik_model import FlikModel
from src.models.losses import CombinedLoss


def check_dataset_masks():
    ds = VideoAudioDataset("", dummy=True)
    batch = [ds[i] for i in range(3)]
    collated = VideoAudioDataset.collate_fn(batch)
    audio = collated["audio"]
    padding_mask = collated["audio_padding_mask"]
    assert audio.ndim == 3
    assert padding_mask.ndim == 2
    assert padding_mask.dtype == torch.bool

    for i in range(audio.shape[0]):
        true_len = batch[i]["audio"].shape[1]
        assert torch.all(padding_mask[i, :true_len] == 0)
        assert torch.all(padding_mask[i, true_len:] == 1)


def check_audio_encoder():
    enc = AudioEncoder(pretrained=False)
    audio = torch.randn(2, 1, 16000)
    padding_mask = torch.zeros(2, 16000, dtype=torch.bool)
    padding_mask[:, 12000:] = True
    out = enc(audio, padding_mask)
    assert out["features"].ndim == 3
    if out["padding_mask"] is not None:
        assert out["padding_mask"].shape[:2] == out["features"].shape[:2]


def check_video_encoder():
    enc = VideoEncoder(pretrained=False)
    video = torch.randn(2, 16, 3, 224, 224)
    out = enc(video)
    assert out["cls_embedding"].shape == (2, 768)
    assert out["frame_embeddings"].ndim == 3


def check_training_step():
    ds = VideoAudioDataset("", dummy=True)
    batch = VideoAudioDataset.collate_fn([ds[0], ds[1]])
    model = FlikModel(pretrained_audio=False, pretrained_video=False)
    loss_fn = CombinedLoss(contrastive_weight=1.0, mlm_weight=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    out = model(
        batch["audio"],
        batch["video"],
        audio_padding_mask=batch["audio_padding_mask"],
    )
    loss, _ = loss_fn(out["audio_embedding"], out["video_embedding"])
    loss.backward()
    opt.step()
    opt.zero_grad()


def main():
    torch.manual_seed(0)
    check_dataset_masks()
    check_audio_encoder()
    check_video_encoder()
    check_training_step()
    print("verify_contracts: OK")


if __name__ == "__main__":
    main()
