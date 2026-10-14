#!/usr/bin/env python
"""Regression tests for audio window alignment (keyframe pre-roll defect).

Background
----------
``src/utils/audio.py::_load_audio_segment`` seeks to the previous video
keyframe. The historical path then trims the decoded buffer from the FRONT to
the requested length, so the returned window starts early by the keyframe gap
(measured mean ~2.3-2.5 s, worst ~8 s on the two local films). The opt-in
``align_to_start=True`` path drops that pre-roll by decoded PTS.

These tests use an independent ground truth: an accurate ``ffmpeg -ss``
extraction of the same interval, cross-correlated against the loader output.

Notes
-----
* Tests skip cleanly when the two local films are absent.
* ``align_to_start=False`` must stay bit-identical to the historical
  behaviour; a frozen reference implementation is inlined below and compared
  with ``torch.equal``.

Runnable directly (``python tests/test_audio_alignment.py``) or via pytest.
"""

import os
import subprocess
import sys
import unittest

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from src.utils.audio import audio_to_tensor  # noqa: E402

try:  # allow both pytest and a plain-python run
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

VIDEO_DIR = os.environ.get(
    "FLIK_TEST_VIDEO_DIR", os.path.join(_REPO_ROOT, "data", "videos")
)
FILMS = {
    "Black_Brigade": os.path.join(VIDEO_DIR, "Black_Brigade.mp4"),
    "TheStilwellRoad": os.path.join(VIDEO_DIR, "TheStilwellRoad.mp4"),
}
SAMPLE_RATE = 16000
MISSING = [name for name, path in FILMS.items() if not os.path.exists(path)]

# (film, start_sec). TheStilwellRoad at 300 s has a ~8 s keyframe gap.
START_TIMES = [
    ("Black_Brigade", 30.0),
    ("Black_Brigade", 300.0),
    ("TheStilwellRoad", 100.0),
    ("TheStilwellRoad", 300.0),
]
LARGE_GAP = ("TheStilwellRoad", 300.0)


def _skip(reason):
    if pytest is not None:
        pytest.skip(reason)
    raise unittest.SkipTest(reason)


def _require_films():
    if MISSING:
        _skip(
            "local films not found (missing: "
            + ", ".join(MISSING)
            + f"); set FLIK_TEST_VIDEO_DIR or place them in {VIDEO_DIR}"
        )


# ---------------------------------------------------------------------------
# Independent ground truth + lag estimation
# ---------------------------------------------------------------------------

def ffmpeg_ground_truth(path, start_sec, end_sec, sample_rate=SAMPLE_RATE):
    """Sample-accurate mono PCM for [start_sec, end_sec) via ffmpeg.

    ``-ss`` before ``-i`` uses input seeking; ffmpeg's accurate_seek (default)
    discards every sample before the seek target, so output sample 0 is
    ``start_sec``.
    """
    duration = end_sec - start_sec
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-nostdin",
            "-ss", f"{start_sec:.6f}", "-i", path,
            "-t", f"{duration:.6f}",
            "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "-",
        ],
        capture_output=True,
        check=True,
    )
    x = np.frombuffer(proc.stdout, dtype="<i2").astype(np.float64)
    n = max(1, int(round(duration * sample_rate)))
    if len(x) > n:
        x = x[:n]
    elif len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    return (x - x.mean()) / (x.std() + 1e-8)


def lag_seconds(a, b, sample_rate=SAMPLE_RATE):
    """Cross-correlation lag of ``a`` relative to ``b`` in seconds.

    Positive => ``a`` is delayed relative to ``b`` (its content appears later).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    n = len(a) + len(b)
    cc = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    # Re-order so index 0 is lag -(len(b)-1): negative lags then positive.
    cc = np.concatenate([cc[-(len(b) - 1):], cc[: len(a)]])
    lags = np.arange(-(len(b) - 1), len(a))
    return int(lags[int(np.argmax(cc))]) / sample_rate


def video_keyframes(path):
    """Video keyframe PTS values (metadata-only, packet flags)."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v",
            "-show_packets", "-show_entries", "packet=pts_time,flags",
            "-of", "csv=p=0", path,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    ts = []
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) > 1 and "K" in parts[1]:
            ts.append(float(parts[0]))
    return np.array(sorted(ts))


# ---------------------------------------------------------------------------
# Frozen reference for the historical (align_to_start=False) path
# ---------------------------------------------------------------------------

def _legacy_load_audio_segment(path, start_sec, end_sec, target_sample_rate=16000):
    """Verbatim copy of the pre-fix front-trim behaviour (no PTS pre-roll drop).

    Kept in the test so a change to the production ``align_to_start=False``
    path is detected as a regression.
    """
    import av

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
    try:
        for frame in container.decode(audio_stream):
            frame_pts_sec = float(frame.pts * audio_stream.time_base)
            if frame_pts_sec > end_sec:
                break
            try:
                for resampled in resampler.resample(frame):
                    samples_list.append(resampled.to_ndarray().ravel())
            except Exception:
                continue
    except Exception:
        pass
    try:
        for resampled in resampler.resample(None):
            samples_list.append(resampled.to_ndarray().ravel())
    except Exception:
        pass
    container.close()

    if samples_list:
        all_samples = np.concatenate(samples_list)
    else:
        all_samples = np.zeros(0, dtype=np.int16)
    if len(all_samples) > num_samples:
        all_samples = all_samples[:num_samples]
    elif len(all_samples) < num_samples:
        all_samples = np.pad(all_samples, (0, num_samples - len(all_samples)))
    waveform = torch.from_numpy(all_samples).float().unsqueeze(0)
    if waveform.numel() > 0:
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)
    return waveform


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_aligned_window_starts_at_requested_timestamp():
    """align_to_start=True must start every window at the requested time."""
    _require_films()
    for film, start in START_TIMES:
        path = FILMS[film]
        end = start + 5.0
        truth = ffmpeg_ground_truth(path, start, end)
        window = audio_to_tensor(
            path, start, end, sample_rate=SAMPLE_RATE, align_to_start=True
        )[0].numpy()
        lag = lag_seconds(window, truth)
        assert abs(lag) <= 0.005, (
            f"{film}@{start}s: aligned window lag {lag*1000:.2f} ms "
            f"(tolerance 5 ms)"
        )


def test_unaligned_window_keeps_legacy_preroll():
    """align_to_start=False must stay bit-identical to historical behaviour.

    Also checks the observable consequence at the large-gap location: the
    window is late by roughly the full keyframe gap.
    """
    _require_films()

    # Bit-identical to the frozen reference for every covered start time.
    for film, start in START_TIMES:
        path = FILMS[film]
        end = start + 5.0
        prod = audio_to_tensor(
            path, start, end, sample_rate=SAMPLE_RATE, align_to_start=False
        )
        ref = _legacy_load_audio_segment(path, start, end, SAMPLE_RATE)
        assert torch.equal(prod, ref), (
            f"{film}@{start}s: align_to_start=False differs from the frozen "
            f"legacy reference (max abs diff "
            f"{float((prod - ref).abs().max()):.6f})"
        )

    # The bug the fix targets: a large keyframe gap produced a large offset.
    # A 5 s window shifted by ~8 s has no overlap with a ground truth that
    # starts at start_sec, so cross-correlation there is meaningless. Extract a
    # longer ground truth that spans the whole gap instead, and locate each
    # window in absolute file time.
    film, start = LARGE_GAP
    path = FILMS[film]
    keyframes = video_keyframes(path)
    gap = start - float(keyframes[np.searchsorted(keyframes, start) - 1])
    assert gap > 5.0, f"expected a large keyframe gap at {film}@{start}s, got {gap:.3f}s"

    gt_start = start - gap - 1.0
    gt_end = start + 5.0
    truth = ffmpeg_ground_truth(path, gt_start, gt_end)

    aligned = audio_to_tensor(
        path, start, start + 5.0, sample_rate=SAMPLE_RATE, align_to_start=True
    )[0].numpy()
    unaligned = audio_to_tensor(
        path, start, start + 5.0, sample_rate=SAMPLE_RATE, align_to_start=False
    )[0].numpy()

    # lag_seconds(a, b) is positive when ``a`` is delayed relative to ``b``, so
    # the window's absolute start is gt_start minus the lag.
    abs_aligned = gt_start - lag_seconds(aligned, truth)
    abs_unaligned = gt_start - lag_seconds(unaligned, truth)

    assert abs(abs_aligned - start) <= 0.02, (
        f"{film}@{start}s: aligned window starts at {abs_aligned:.3f}s, "
        f"expected {start:.3f}s"
    )
    # The decoded audio frame can sit up to ~0.1 s before the video keyframe,
    # so allow 0.25 s of slack against the metadata gap.
    assert abs(abs_unaligned - (start - gap)) <= 0.25, (
        f"{film}@{start}s: legacy window starts at {abs_unaligned:.3f}s, "
        f"expected about {start - gap:.3f}s (gap {gap:.3f}s)"
    )


if __name__ == "__main__":
    _failures = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"PASS {_name}")
            except unittest.SkipTest as _e:
                print(f"SKIP {_name}: {_e}")
            except Exception as _e:  # noqa: BLE001
                _failures += 1
                print(f"FAIL {_name}: {type(_e).__name__}: {_e}")
    raise SystemExit(1 if _failures else 0)
