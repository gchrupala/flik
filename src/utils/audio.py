import torch
import numpy as np
from typing import Optional


def _load_audio_segment(
    path: str,
    start_sec: float,
    end_sec: float,
    target_sample_rate: int = 16000,
    align_to_start: bool = False,
) -> torch.Tensor:
    """
    Load audio segment using PyAV (ffmpeg bindings).
    Works for both video and audio files (.mp4, .mkv, .avi, .mov, .webm,
    .wav, .flac, .mp3, .ogg).
    Returns waveform of shape (1, samples) at target_sample_rate.

    Tolerant of corrupt frames: if decoding fails partway through, returns
    whatever samples were collected so far (padded to target length).

    Note:
        ``align_to_start=False`` (default) keeps the historical behaviour:
        the container is seeked to the previous keyframe, every decoded
        sample is concatenated, and the buffer is trimmed from the FRONT to
        ``num_samples``. The pre-roll between the keyframe and ``start_sec``
        is therefore retained and the returned window starts early by the
        keyframe distance (measured 0.07-8.0 s on the local films; the gap is
        the video keyframe spacing).

        ``align_to_start=True`` opts in to timestamp-accurate trimming: the
        first decoded frame's PTS defines sample 0 of the buffer, and the
        pre-roll is dropped so the window starts at ``start_sec``. Callers
        must opt in explicitly so default training behaviour is unchanged.

        Run mapping: runs logged before 2026-09-16 used
        ``align_to_start=False`` (the buggy behaviour). Training and
        validation now pass ``dataset.audio_align_to_start`` (default true).
    """
    import av

    # Suppress libav log output (harmless warnings from old/corrupt AAC frames:
    # "decode_pce", "Referenced QT chapter track not found", "illegal icc").
    # set_level(None) replaces the FFmpeg callback with a no-op — total silence.
    # Errors are still handled per-frame via try/except + zero-pad fallback below.
    av.logging.set_level(None)

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
    base_pts_sec = None  # PTS of the first frame that contributed samples
    try:
        for frame in container.decode(audio_stream):
            frame_pts_sec = float(frame.pts * audio_stream.time_base)
            if frame_pts_sec > end_sec:
                break
            try:
                for resampled in resampler.resample(frame):
                    arr = resampled.to_ndarray().ravel()
                    if base_pts_sec is None:
                        base_pts_sec = frame_pts_sec
                    samples_list.append(arr)
            except Exception:
                # Skip corrupt frame, continue with next
                continue
    except Exception:
        # Decode generator raised (e.g. "Input buffer exhausted").
        # Use whatever samples we've collected so far.
        pass

    # Flush remaining samples from resampler
    try:
        for resampled in resampler.resample(None):
            arr = resampled.to_ndarray().ravel()
            samples_list.append(arr)
    except Exception:
        pass

    container.close()

    if samples_list:
        all_samples = np.concatenate(samples_list)
    else:
        all_samples = np.zeros(0, dtype=np.int16)

    # Opt-in: drop the keyframe pre-roll by decoded timestamps. Sample 0 of the
    # buffer is base_pts_sec, so start_sec is `skip` samples into the buffer.
    if align_to_start and base_pts_sec is not None:
        skip = int(round((start_sec - base_pts_sec) * target_sample_rate))
        if skip >= 0:
            all_samples = all_samples[skip:]
        else:
            all_samples = np.concatenate(
                [np.zeros(-skip, dtype=all_samples.dtype), all_samples]
            )

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
    align_to_start: bool = False,
) -> torch.Tensor:
    return _load_audio_segment(
        video_path, start_sec, end_sec, target_sample_rate,
        align_to_start=align_to_start,
    )


def load_audio_from_wav(
    wav_path: str,
    start_sec: float,
    end_sec: float,
    target_sample_rate: int = 16000,
    align_to_start: bool = False,
) -> torch.Tensor:
    return _load_audio_segment(
        wav_path, start_sec, end_sec, target_sample_rate,
        align_to_start=align_to_start,
    )


def audio_to_tensor(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    sample_rate: int = 16000,
    align_to_start: bool = False,
) -> torch.Tensor:
    """
    Generic audio loading from video or audio files via PyAV.

    Set ``align_to_start=True`` for timestamp-accurate windows (drops the
    keyframe pre-roll). Default ``False`` preserves historical behaviour.
    """
    return _load_audio_segment(
        audio_path, start_sec, end_sec, sample_rate,
        align_to_start=align_to_start,
    )


if __name__ == "__main__":
    # Quick test
    try:
        dummy_path = "/tmp/test.wav"
        # Create a dummy sine wave for testing if file exists
        import warnings

        warnings.warn("No real audio file, skipping test")
    except Exception as e:
        print(f"Test skipped: {e}")
