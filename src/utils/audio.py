import torch
import numpy as np
from typing import Optional


def _load_audio_segment(
    path: str,
    start_sec: float,
    end_sec: float,
    target_sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Load audio segment using PyAV (ffmpeg bindings).
    Works for both video and audio files (.mp4, .mkv, .avi, .mov, .webm,
    .wav, .flac, .mp3, .ogg).
    Returns waveform of shape (1, samples) at target_sample_rate.
    """
    import av

    duration = end_sec - start_sec
    num_samples = max(1, int(duration * target_sample_rate))

    container = av.open(path)
    try:
        audio_stream = next(s for s in container.streams if s.type == "audio")
    except StopIteration:
        container.close()
        raise RuntimeError(f"No audio stream found in {path}")

    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=target_sample_rate
    )

    container.seek(int(start_sec * av.time_base))

    samples_list = []
    for frame in container.decode(audio_stream):
        frame_pts_sec = float(frame.pts * audio_stream.time_base)
        if frame_pts_sec > end_sec:
            break
        for resampled in resampler.resample(frame):
            arr = resampled.to_ndarray().ravel()
            samples_list.append(arr)

    # Flush remaining samples from resampler
    for resampled in resampler.resample(None):
        arr = resampled.to_ndarray().ravel()
        samples_list.append(arr)

    container.close()

    if samples_list:
        all_samples = np.concatenate(samples_list)
    else:
        all_samples = np.zeros(0, dtype=np.int16)

    # Trim/pad to exact length
    if len(all_samples) > num_samples:
        all_samples = all_samples[:num_samples]
    elif len(all_samples) < num_samples:
        all_samples = np.pad(all_samples, (0, num_samples - len(all_samples)))

    waveform = torch.from_numpy(all_samples).float().unsqueeze(0)

    # Normalize to zero mean, unit variance
    if waveform.numel() > 0:
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)

    return waveform


def load_audio_from_video(
    video_path: str,
    start_sec: float,
    end_sec: float,
    target_sample_rate: int = 16000,
) -> torch.Tensor:
    return _load_audio_segment(video_path, start_sec, end_sec, target_sample_rate)


def load_audio_from_wav(
    wav_path: str,
    start_sec: float,
    end_sec: float,
    target_sample_rate: int = 16000,
) -> torch.Tensor:
    return _load_audio_segment(wav_path, start_sec, end_sec, target_sample_rate)


def audio_to_tensor(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """
    Generic audio loading from video or audio files via PyAV.
    """
    return _load_audio_segment(audio_path, start_sec, end_sec, sample_rate)


if __name__ == "__main__":
    # Quick test
    try:
        dummy_path = "/tmp/test.wav"
        # Create a dummy sine wave for testing if file exists
        import warnings

        warnings.warn("No real audio file, skipping test")
    except Exception as e:
        print(f"Test skipped: {e}")
