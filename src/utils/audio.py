import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
from typing import Tuple, Optional


def load_audio_from_video(
    video_path: str,
    start_sec: float,
    end_sec: float,
    target_sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Load audio segment from video file.
    Returns waveform of shape (1, samples) at target_sample_rate.
    """
    # Use torchaudio's avbackend
    try:
        # torchaudio.load can load video files and extract audio
        waveform, sr = torchaudio.load(video_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio from {video_path}: {e}")

    # Ensure mono: (channels, samples) -> average if multiple channels
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sample_rate:
        waveform = F.resample(waveform, sr, target_sample_rate)

    # Extract segment
    start_sample = int(start_sec * target_sample_rate)
    end_sample = int(end_sec * target_sample_rate)
    if end_sample > waveform.shape[1]:
        # Pad with zeros if segment extends beyond file
        pad_right = end_sample - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_right))

    segment = waveform[:, start_sample:end_sample]

    # Normalize to zero mean, unit variance (per segment)
    if segment.numel() > 0:
        segment = (segment - segment.mean()) / (segment.std() + 1e-8)

    return segment


def load_audio_from_wav(
    wav_path: str,
    start_sec: float,
    end_sec: float,
    target_sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Load audio segment from .wav file.
    """
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sample_rate:
        waveform = F.resample(waveform, sr, target_sample_rate)

    start_sample = int(start_sec * target_sample_rate)
    end_sample = int(end_sec * target_sample_rate)
    if end_sample > waveform.shape[1]:
        pad_right = end_sample - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_right))

    segment = waveform[:, start_sample:end_sample]
    if segment.numel() > 0:
        segment = (segment - segment.mean()) / (segment.std() + 1e-8)
    return segment


def audio_to_tensor(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Generic audio loading; detects video vs wav by extension.
    """
    if audio_path.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
        return load_audio_from_wav(audio_path, start_sec, end_sec, sample_rate)
    else:
        # Assume video file
        return load_audio_from_video(audio_path, start_sec, end_sec, sample_rate)


if __name__ == "__main__":
    # Quick test
    try:
        dummy_path = "/tmp/test.wav"
        # Create a dummy sine wave for testing if file exists
        import warnings

        warnings.warn("No real audio file, skipping test")
    except Exception as e:
        print(f"Test skipped: {e}")
