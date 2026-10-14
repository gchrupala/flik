#!/usr/bin/env python
"""
Audio-video alignment QC (offset, drift, speech-vs-face).

Implements the extension proposed in docs/MEETING_2026-09-15.md, item 7
("Proposed extension - audio-video alignment").

Check 1 - synchrony (offset and drift)
    Cross-correlate the audio energy envelope against a visual activity
    envelope (mean absolute frame difference) over a configurable lag range
    (--max-lag, default +/-10 s), per sampled window. A dense signal is needed
    for lag estimation, so video decodes at native frame rate (~24-30 fps) and
    audio builds a 40 ms RMS envelope. Sub-sample parabolic interpolation
    resolves lags below one 40 ms step. Aggregation is per film, because
    synchrony is a film-level property: lag distribution, median lag, IQR,
    fraction of windows with a significant correlation peak, and an OLS fit of
    lag against absolute window time (frame-rate drift).

    Window length bounds the search: the overlap constraint allows a lag of at
    most (window_seconds / 2). A 7 s window therefore searches about +/-3.4 s
    even when --max-lag is larger. Use windows longer than twice --max-lag to
    search the full range.

Check 2 - semantic (speech against faces)
    Detect faces on sampled frames and cross-tabulate with speech presence
    (energy VAD). Speech with no visible face, or a face over music-only
    audio, points to a displaced track.

Targeted feature test
    Three audio envelopes (broadband RMS, 300-3400 Hz speech band, spectral
    flux) are crossed with three visual envelopes (whole-frame difference,
    face-region motion, mouth-region motion). Windows are stratified by
    whether the manifest text field carries speech. The decisive cell is
    speech-bearing window x speech-band audio x face/mouth visual: a peak at
    |lag| <= 0.5 s there means the method works; its absence means the audio
    does not contain the speech the transcript claims.

Unit of analysis: the film. The output JSON holds one record per film with the
per-window records nested under it, so downstream filtering can use window
resolution. Every window records the effective search range
(min(max_lag, window_seconds/2)). A 7 s window searches only about +/-3.5 s
even when --max-lag is 10.

Usage:
    uv run --extra cpu python -m scripts.qc_av_alignment \
        --manifest data/qc_pilot_manifest.json \
        --samples-per-film 8 --workers 10 --out-dir data/
"""

import argparse
import json
import logging
import os
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root so src.* / scripts.* imports work under `python -m`.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import cv2
import numpy as np

# OpenCV's internal thread pool exhausts the SLURM cgroup thread limit when
# many workers run. Parallelism comes from the thread pool below, not from
# OpenCV.
cv2.setNumThreads(0)

from src.CONSTANTS import PROJECT_ROOT
from src.utils.paths import resolve_path
from scripts.qc_segments import compute_vad, NO_SPEECH_VOICED_FRAC

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENV_FPS = 25.0                 # 40 ms envelope grid
HOP_SEC = 1.0 / ENV_FPS        # 40 ms
DEFAULT_MAX_LAG_SEC = 10.0     # default correlation search range (+/-10 s)
LAG_HIST_BINS = 20             # bins across the search range for the lag histogram
OFFSET_FLAG_SEC = 0.080        # |median lag| above this is a constant offset
DRIFT_FLAG_SLOPE = 0.01        # |s lag per s film| above this is drift (1%)
PEAK_MIN = 0.6                 # per-window peak strength for a real correlation
PEAK_SIG_Z = 4.0               # reported only: z of the peak over the lag axis
CONSISTENT_FRAC = 0.5          # fraction of significant windows for a real peak
MIN_SIG_FOR_DRIFT = 5          # minimum usable windows before trusting a slope
MIN_CELL = 5                   # minimum windows in a stratum before a rate is read
MIN_WINDOW_SEC = 4.5           # shorter windows cannot host the full lag range

AUDIO_ENV_FRAME_SEC = 0.04     # STFT / RMS frame length == hop (non-overlapping)
AUDIO_SMOOTH_SIGMA = 2.0       # envelope smoothing, in 40 ms samples (80 ms)
SPEECH_SMOOTH_SIGMA = 2.0
FLUX_SMOOTH_SIGMA = 1.0
VIDEO_SCALE = (160, 120)       # decode downscale before whole-frame differencing
VIDEO_BLUR = 3                 # Gaussian blur kernel (kills film grain)
VIDEO_SMOOTH_SIGMA = 1.0       # whole-frame activity smoothing, in 40 ms samples
FACE_SMOOTH_SIGMA = 1.0
MOUTH_SMOOTH_SIGMA = 1.0

SPEECH_BAND_HZ = (300.0, 3400.0)  # speech-band envelope
NEAR_ZERO_LAG_SEC = 0.5           # "at lag ~ 0" window
MIN_SPEECH_CHARS = 10             # a window is speech-bearing above this text length

AUDIO_FEATURES = ("rms", "speech", "flux")
VISUAL_FEATURES = ("whole", "face", "mouth")
AUDIO_SMOOTH = {
    "rms": AUDIO_SMOOTH_SIGMA,
    "speech": SPEECH_SMOOTH_SIGMA,
    "flux": FLUX_SMOOTH_SIGMA,
}
VISUAL_SMOOTH = {
    "whole": VIDEO_SMOOTH_SIGMA,
    "face": FACE_SMOOTH_SIGMA,
    "mouth": MOUTH_SMOOTH_SIGMA,
}

DEFAULT_FACE_MODEL = os.path.join(
    PROJECT_ROOT, "data/models/face_detection_yunet_2023mar.onnx"
)
HAAR_CASCADE = "haarcascade_frontalface_default.xml"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    """Configure file + stream logging (mirrors qc_segments.py)."""
    log_dir = os.path.join(PROJECT_ROOT, "logdir")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("qc_av_alignment")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(os.path.join(log_dir, "qc_av_alignment.log"))
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


logger = logging.getLogger("qc_av_alignment")


# ---------------------------------------------------------------------------
# Face detector selection
# ---------------------------------------------------------------------------

def build_face_detector(model_path):
    """Return (kind, detector). YuNet when weights exist, else Haar cascade.

    YuNet needs an external ONNX file that OpenCV does not ship. When the file
    is missing, fall back to the bundled Haar cascade.
    """
    if model_path and os.path.isfile(model_path):
        try:
            det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), 0.6, 0.3, 5000)
            return "yunet", det
        except Exception as exc:  # noqa: BLE001 - fall back, never crash QC
            logger.warning("YuNet load failed (%s); using Haar cascade", exc)
    path = os.path.join(cv2.data.haarcascades, HAAR_CASCADE)
    return "haar", cv2.CascadeClassifier(path)


def _mouth_box_from_landmarks(face_row):
    """Mouth region from YuNet's 5 landmarks (last two are mouth corners)."""
    x, y, w, h = (float(v) for v in face_row[:4])
    corners = np.asarray(face_row[10:14], dtype=np.float64).reshape(2, 2)
    cx, cy = float(corners[:, 0].mean()), float(corners[:, 1].mean())
    dx = abs(float(corners[0, 0] - corners[1, 0]))
    dy = abs(float(corners[0, 1] - corners[1, 1]))
    mw = max(1.5 * dx, 0.25 * w, 6.0)
    mh = max(2.0 * dy, 2.0, 0.15 * h)
    return (cx - mw / 2.0, cy - mh / 2.0, mw, mh)


def detect_largest_face(kind, detector, frame):
    """Return (face_box, mouth_box) in frame pixels, or (None, None)."""
    if kind == "yunet":
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)
        if faces is None or len(faces) == 0:
            return None, None
        i = int(np.argmax(faces[:, 2] * faces[:, 3]))
        return (
            tuple(float(v) for v in faces[i, :4]),
            _mouth_box_from_landmarks(faces[i]),
        )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = detector.detectMultiScale(gray, 1.1, 5, minSize=(24, 24))
    if len(boxes) == 0:
        return None, None
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    # Haar gives no landmarks; approximate the mouth as the lower face half.
    return (float(x), float(y), float(w), float(h)), (
        float(x + 0.2 * w), float(y + 0.6 * h), float(0.6 * w), float(0.4 * h)
    )


def _region_mean(diff, box, height, width):
    """Mean of a difference image inside a clipped box, or None if empty."""
    if box is None:
        return None
    x, y, bw, bh = box
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(width, int(round(x + bw))), min(height, int(round(y + bh)))
    if x1 <= x0 or y1 <= y0:
        return None
    return float(np.mean(diff[y0:y1, x0:x1]))


def _fill_nan(values):
    """Linearly interpolate NaNs; an all-NaN array becomes zeros."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    good = ~np.isnan(arr)
    if good.all():
        return arr
    if not good.any():
        return np.zeros_like(arr)
    idx = np.arange(arr.size)
    return np.interp(idx, idx[good], arr[good])


class FaceDetectorPool:
    """One face detector per worker thread.

    cv2.FaceDetectorYN is not thread-safe: concurrent setInputSize/detect calls
    segfault. Each thread builds and owns its detector via threading.local.
    """

    def __init__(self, model_path, kind):
        self.model_path = model_path
        self.kind = kind
        self._local = threading.local()

    def get(self):
        det = getattr(self._local, "detector", None)
        if det is None:
            kind, det = build_face_detector(self.model_path)
            self._local.kind = kind
            self._local.detector = det
        return self._local.kind, self._local.detector


# ---------------------------------------------------------------------------
# Audio: accurate decode + envelope
# ---------------------------------------------------------------------------

def decode_audio_window(path, start_sec, end_sec):
    """Decode mono audio for [start_sec, end_sec) at the native sample rate.

    Returns (waveform_float32, sample_rate). Never normalises and never
    resamples, so absolute level survives.

    NOTE: this does NOT reuse src.utils.audio.audio_to_tensor. That helper
    seeks with PyAV and decodes from the previous keyframe without skipping
    the leading samples, so its windows can start more than a second early.
    It also normalises every window to zero mean / unit variance, which
    destroys absolute level. Neither is acceptable for lag estimation.
    """
    import av

    av.logging.set_level(None)
    container = av.open(path)
    try:
        stream = next(s for s in container.streams if s.type == "audio")
    except StopIteration:
        container.close()
        raise RuntimeError(f"No audio stream in {path}")

    rate = float(stream.rate) if stream.rate else 48000.0
    # any_frame seeks to the closest packet before start; leading samples are
    # trimmed below by timestamp.
    container.seek(int(start_sec * av.time_base), any_frame=True)

    chunks = []
    base_pts = None
    try:
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            pts = float(frame.pts) * float(stream.time_base)
            if pts > end_sec:
                break
            arr = frame.to_ndarray()
            if arr.ndim == 2:
                mono = arr.mean(axis=0) if arr.shape[0] <= 8 else arr.mean(axis=1)
            else:
                mono = arr
            if base_pts is None:
                base_pts = pts
            chunks.append(np.asarray(mono, dtype=np.float32))
    except Exception:  # noqa: BLE001 - corrupt frames must not kill the run
        pass
    finally:
        container.close()

    if not chunks:
        return np.zeros(0, dtype=np.float32), rate

    wav = np.concatenate(chunks)
    if base_pts is not None:
        lead = int(round((start_sec - base_pts) * rate))
        if lead > 0:
            wav = wav[lead:]
        elif lead < 0:
            wav = np.pad(wav, (-lead, 0))

    n = int(round((end_sec - start_sec) * rate))
    if len(wav) > n:
        wav = wav[:n]
    elif len(wav) < n:
        wav = np.pad(wav, (0, n - len(wav)))
    return wav, rate


def audio_envelope(wav, rate, frame_sec=AUDIO_ENV_FRAME_SEC):
    """RMS envelope at 1/frame_sec. Bin i covers [i*hop, (i+1)*hop)."""
    frame_len = max(1, int(round(rate * frame_sec)))
    n = len(wav) // frame_len
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    frames = wav[: n * frame_len].reshape(n, frame_len).astype(np.float64)
    return np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)


def spectral_envelopes(wav, rate, frame_sec=AUDIO_ENV_FRAME_SEC):
    """Speech-band energy and spectral-flux envelopes at 1/frame_sec.

    One short-time Fourier pass over non-overlapping frames returns both:
      speech - RMS of the magnitude spectrum inside SPEECH_BAND_HZ;
      flux   - sum of positive magnitude changes from the previous frame
               (onset strength), which responds to speech and percussive
               onsets but not to a steady music bed.
    """
    frame_len = max(8, int(round(rate * frame_sec)))
    n = len(wav) // frame_len
    if n < 1:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    frames = wav[: n * frame_len].reshape(n, frame_len).astype(np.float64)
    window = np.hanning(frame_len)
    mag = np.abs(np.fft.rfft(frames * window, axis=1))
    freqs = np.fft.rfftfreq(frame_len, 1.0 / rate)
    band = (freqs >= SPEECH_BAND_HZ[0]) & (freqs <= SPEECH_BAND_HZ[1])
    if band.any():
        speech = np.sqrt(np.sum(mag[:, band] ** 2, axis=1) + 1e-12)
    else:
        speech = np.zeros(n, dtype=np.float64)
    delta = np.diff(mag, axis=0, prepend=mag[:1])
    flux = np.sum(np.clip(delta, 0.0, None), axis=1)
    return speech, flux


# ---------------------------------------------------------------------------
# Video: three visual envelopes in one decode pass
# ---------------------------------------------------------------------------

def decode_video_window(path, start_sec, end_sec, face_kind, face_det):
    """Decode a window once and build three visual activity envelopes.

    Returns (times, feats, valid):
      times - midpoint timestamp of each frame pair, seconds.
      feats - {"whole", "face", "mouth"}: mean absolute frame difference over
              the whole downscaled frame, inside the largest detected face
              box, and inside the mouth box. Face and mouth are NaN where no
              face was detected in the frame pair.
      valid - fraction of decoded frames with a face box / mouth box.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    times, whole, face, mouth = [], [], [], []
    n_frames = n_face = n_mouth = 0
    prev_small = prev_native = prev_t = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 24.0
        start_frame = max(0, int(start_sec * fps) - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = (start_frame + idx) / fps
            idx += 1
            if t < start_sec - 1.0:
                prev_small = prev_native = None
                continue
            if t > end_sec:
                break

            # Whole-frame envelope on a downscaled, blurred gray (grain-robust).
            small = cv2.cvtColor(cv2.resize(frame, VIDEO_SCALE), cv2.COLOR_BGR2GRAY)
            if VIDEO_BLUR:
                k = VIDEO_BLUR if VIDEO_BLUR % 2 == 1 else VIDEO_BLUR + 1
                small = cv2.GaussianBlur(small, (k, k), 0)
            small = small.astype(np.float32)
            native = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face and mouth boxes every frame (YuNet landmarks give the mouth).
            box, mouth_box = detect_largest_face(face_kind, face_det, frame)
            n_frames += 1
            n_face += int(box is not None)
            n_mouth += int(mouth_box is not None)

            if prev_small is not None:
                whole.append(float(np.mean(np.abs(small - prev_small))))
                diff = np.abs(
                    native.astype(np.float32) - prev_native.astype(np.float32)
                )
                h, w = native.shape
                fb = _region_mean(diff, box, h, w)
                mb = _region_mean(diff, mouth_box, h, w)
                face.append(np.nan if fb is None else fb)
                mouth.append(np.nan if mb is None else mb)
                times.append(0.5 * (t + prev_t))
            prev_small, prev_native, prev_t = small, native, t
    finally:
        cap.release()

    feats = {
        "whole": np.asarray(whole, dtype=np.float64),
        "face": np.asarray(face, dtype=np.float64),
        "mouth": np.asarray(mouth, dtype=np.float64),
    }
    valid = {
        "face": n_face / n_frames if n_frames else 0.0,
        "mouth": n_mouth / n_frames if n_frames else 0.0,
    }
    return np.asarray(times, dtype=np.float64), feats, valid


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------

def smooth(x, sigma):
    """Gaussian smoothing with edge padding, in samples."""
    if sigma <= 0 or x.size == 0:
        return x
    radius = int(3 * sigma)
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(x, radius, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _norm(x):
    """Zero mean, unit variance."""
    arr = np.asarray(x, dtype=np.float64)
    return (arr - arr.mean()) / (arr.std() + 1e-9)


def normalized_xcorr(a, b, max_lag_samples):
    """Normalized cross-correlation over +/-max_lag_samples.

    Returns (lags_sec, r). Positive lag means the audio feature matches a
    later video feature, i.e. the audio track is late by that much.
    """
    n = len(a)
    k_max = min(max_lag_samples, (n - 5) // 2)
    if k_max < 1:
        return np.zeros(0), np.zeros(0)
    ks = np.arange(-k_max, k_max + 1)
    r = np.zeros(len(ks), dtype=np.float64)
    for i, k in enumerate(ks):
        if k >= 0:
            x, y = a[k:], b[: n - k]
        else:
            x, y = a[: n + k], b[-k:]
        if x.size < 5:
            continue
        xs, ys = x.std(), y.std()
        if xs < 1e-9 or ys < 1e-9:
            continue
        r[i] = float(np.mean((x - x.mean()) * (y - y.mean())) / (xs * ys))
    return ks * HOP_SEC, r


def peak_of(lags, r):
    """Return (lag_sec, peak, z) with parabolic sub-sample refinement."""
    if r.size == 0:
        return None, None, None
    j = int(np.argmax(r))
    peak = float(r[j])
    z = float((peak - r.mean()) / (r.std() + 1e-12))
    refined = float(lags[j])
    if 0 < j < len(r) - 1:
        denom = r[j - 1] - 2.0 * r[j] + r[j + 1]
        if abs(denom) > 1e-12:
            delta = 0.5 * (r[j - 1] - r[j + 1]) / denom
            refined = float(lags[j]) + float(np.clip(delta, -1.0, 1.0)) * HOP_SEC
    return refined, peak, z


# ---------------------------------------------------------------------------
# Per-window analysis
# ---------------------------------------------------------------------------

PAIRING_NAMES = tuple(
    f"{a}__{v}" for a in AUDIO_FEATURES for v in VISUAL_FEATURES
)
HEADLINE_PAIRING = "rms__whole"


def _pairing(lag, peak, z):
    """Per-pairing result. `near_zero` is the dialogue cell (|lag| <= 0.5 s)."""
    significant = bool(peak is not None and peak >= PEAK_MIN)
    near_zero = bool(
        significant and lag is not None and abs(lag) <= NEAR_ZERO_LAG_SEC
    )
    return {
        "lag_sec": lag,
        "corr_peak": peak,
        "corr_peak_z": z,
        "significant": significant,
        "near_zero": near_zero,
    }


def analyze_window(seg, face_pool, max_lag_sec, min_speech_chars):
    """Decode one window and test every audio x visual feature pairing.

    Never raises: failures are recorded in the row. The top-level lag/peak/
    significant fields remain the broadband-RMS x whole-frame headline, for
    backward compatibility; all pairings are under row["pairings"].
    """
    face_kind, face_det = face_pool.get()
    start = float(seg.get("start_sec", 0.0))
    end = float(seg.get("end_sec", 0.0))
    text = (seg.get("text") or "").strip()
    row = {
        "id": seg.get("id"),
        "video_id": seg.get("video_id"),
        "start_sec": start,
        "end_sec": end,
        "duration": max(0.0, end - start),
        "text_len": len(text),
        "speech_bearing": bool(len(text) >= min_speech_chars),
        "speech_frac": seg.get("speech_frac"),
        "search_max_lag_sec": None,
        "lag_sec": None,
        "corr_peak": None,
        "corr_peak_z": None,
        "significant": False,
        "speech_present": None,
        "voiced_frac": None,
        "mean_dbfs": None,
        "face_valid_frac": None,
        "mouth_valid_frac": None,
        "pairings": {},
        "status": "ok",
    }
    _r = None

    try:
        if (end - start) < MIN_WINDOW_SEC:
            row["status"] = "skipped: window shorter than MIN_WINDOW_SEC"
            return row, _r

        video_path = resolve_path(seg.get("video_path", ""))
        audio_path = resolve_path(seg.get("audio_path") or seg.get("video_path", ""))
        if not video_path or not os.path.isfile(video_path):
            raise FileNotFoundError(f"video not found: {video_path}")
        if not audio_path or not os.path.isfile(audio_path):
            raise FileNotFoundError(f"audio not found: {audio_path}")

        # --- audio: broadband RMS, speech band, spectral flux ---
        wav, rate = decode_audio_window(audio_path, start, end)
        if wav.size == 0:
            raise RuntimeError("empty audio decode")
        voiced_frac, mean_dbfs = compute_vad(wav, rate)
        a_rms = audio_envelope(wav, rate)
        a_speech, a_flux = spectral_envelopes(wav, rate)

        # --- video: whole-frame, face-region, mouth-region motion ---
        act_times, feats, valid = decode_video_window(
            video_path, start, end, face_kind, face_det
        )
        if feats["whole"].size < 10:
            raise RuntimeError("too few decoded video frames")

        # --- common 40 ms grid ---
        grid_n = int(round((end - start) * ENV_FPS))
        grid_rel = (np.arange(grid_n) + 0.5) * HOP_SEC
        a_times_rel = (np.arange(a_rms.size) + 0.5) * HOP_SEC
        v_times_rel = act_times - start

        audio_envs = {
            "rms": np.interp(grid_rel, a_times_rel, a_rms),
            "speech": np.interp(grid_rel, a_times_rel, a_speech),
            "flux": np.interp(grid_rel, a_times_rel, a_flux),
        }
        visual_envs = {
            "whole": np.interp(grid_rel, v_times_rel, feats["whole"]),
            "face": np.interp(grid_rel, v_times_rel, _fill_nan(feats["face"])),
            "mouth": np.interp(grid_rel, v_times_rel, _fill_nan(feats["mouth"])),
        }
        audio_n = {k: _norm(smooth(v, AUDIO_SMOOTH[k])) for k, v in audio_envs.items()}
        visual_n = {k: _norm(smooth(v, VISUAL_SMOOTH[k])) for k, v in visual_envs.items()}

        k_lag = int(max_lag_sec * ENV_FPS)
        pairings = {}
        headline_r = None
        headline = (None, None, None, False)
        search_range = None
        for an in AUDIO_FEATURES:
            for vn in VISUAL_FEATURES:
                lags, r = normalized_xcorr(audio_n[an], visual_n[vn], k_lag)
                lag, peak, z = peak_of(lags, r)
                pairings[f"{an}__{vn}"] = _pairing(lag, peak, z)
                if search_range is None:
                    search_range = float(
                        min(max_lag_sec, lags[-1] if lags.size else 0.0)
                    )
                if f"{an}__{vn}" == HEADLINE_PAIRING:
                    headline_r = r
                    headline = (lag, peak, z, pairings[HEADLINE_PAIRING]["significant"])

        hlag, hpeak, hz, hsig = headline
        row.update(
            {
                "search_max_lag_sec": search_range,
                "lag_sec": hlag,
                "corr_peak": hpeak,
                "corr_peak_z": hz,
                "significant": hsig,
                "speech_present": bool(voiced_frac >= NO_SPEECH_VOICED_FRAC),
                "voiced_frac": float(voiced_frac),
                "mean_dbfs": float(mean_dbfs),
                "face_valid_frac": float(valid["face"]),
                "mouth_valid_frac": float(valid["mouth"]),
                "pairings": pairings,
            }
        )
        _r = headline_r
    except Exception as exc:  # noqa: BLE001 - bad media must not kill the run
        row["status"] = f"error: {exc}"

    return row, _r


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_windows_per_film(manifest, samples_per_film, seed):
    """Pick `samples_per_film` windows per film, spread across the film.

    Windows are sorted by start time, split into that many contiguous time
    bins, and one window is drawn at random from each bin. This covers the
    film's timeline, which the drift fit needs.
    """
    groups = defaultdict(list)
    for seg in manifest:
        groups[seg.get("video_id", "unknown")].append(seg)

    rng = random.Random(seed)
    sampled = []
    for vid in sorted(groups.keys(), key=str):
        segs = sorted(groups[vid], key=lambda s: float(s.get("start_sec", 0.0)))
        k = min(samples_per_film, len(segs))
        if k <= 0:
            continue
        film_seed = rng.randrange(2**63)
        film_rng = random.Random(film_seed)
        bins = [[] for _ in range(k)]
        for i, seg in enumerate(segs):
            bins[min(k - 1, i * k // len(segs))].append(seg)
        for b in bins:
            sampled.append(film_rng.choice(b))
    return sampled


# ---------------------------------------------------------------------------
# Per-film aggregation
# ---------------------------------------------------------------------------

def _lag_distribution(lags, sig, max_lag_sec):
    """Lag histogram, full lag list, and a plain co-occurrence statement.

    `co_occurs_within_search` answers one question only: does the audio
    correlate with the picture at ANY lag inside the search range. A real
    offset larger than max_lag_sec stays invisible; the residual_limitation
    field says so.
    """
    n_ok = int(len(lags))
    n_sig = int(sig.sum()) if n_ok else 0
    frac_sig = (n_sig / n_ok) if n_ok else 0.0

    edges = np.linspace(-max_lag_sec, max_lag_sec, LAG_HIST_BINS + 1)
    counts = np.histogram(lags, bins=edges)[0].tolist() if n_ok else []
    sig_lags = [float(v) for v, s in zip(lags, sig) if s]

    co_occurs = bool(n_ok > 0 and frac_sig >= CONSISTENT_FRAC)
    if n_ok == 0:
        statement = "no decodable windows; co-occurrence not assessed"
    elif co_occurs:
        statement = (
            f"audio co-occurs with picture at some lag within +/-{max_lag_sec:g} s: "
            f"{n_sig}/{n_ok} windows show a significant correlation peak"
        )
    else:
        statement = (
            f"audio does not co-occur with picture at any lag within "
            f"+/-{max_lag_sec:g} s: only {n_sig}/{n_ok} windows show a "
            "significant correlation peak"
        )

    return {
        "search_max_lag_sec": float(max_lag_sec),
        "lag_values_sec": [float(v) for v in lags],
        "significant_lag_values_sec": sig_lags,
        "n_significant_global": n_sig,
        "frac_significant_global": float(frac_sig),
        "lag_histogram": {
            "edges_sec": [float(e) for e in edges],
            "counts": counts,
        },
        "co_occurs_within_search": co_occurs,
        "co_occurs_statement": statement,
        "residual_limitation": (
            f"search bounded to +/-{max_lag_sec:g} s, and per window to "
            "window_seconds/2; a real offset beyond that range would not be seen"
        ),
    }


def _pairing_aggregate(ok, max_lag_sec):
    """Per feature pairing, split into speech-bearing and non-speech windows."""
    edges = np.linspace(-max_lag_sec, max_lag_sec, LAG_HIST_BINS + 1)
    out = {}
    for name in PAIRING_NAMES:
        rows = [r for r in ok if name in r.get("pairings", {})]
        entry = {}
        for stratum, subset in (
            ("all", rows),
            ("speech", [r for r in rows if r.get("speech_bearing")]),
            ("nonspeech", [r for r in rows if not r.get("speech_bearing")]),
        ):
            n = len(subset)
            lags = [
                r["pairings"][name]["lag_sec"]
                for r in subset
                if r["pairings"][name]["lag_sec"] is not None
            ]
            n_sig = sum(1 for r in subset if r["pairings"][name]["significant"])
            n_near = sum(1 for r in subset if r["pairings"][name]["near_zero"])
            entry[stratum] = {
                "n": n,
                "n_significant": n_sig,
                "frac_significant": (n_sig / n) if n else None,
                "n_near_zero": n_near,
                "frac_near_zero": (n_near / n) if n else None,
                "median_lag_sec": float(np.median(lags)) if lags else None,
                "lag_values_sec": [float(v) for v in lags],
                "lag_histogram": {
                    "edges_sec": [float(e) for e in edges],
                    "counts": (
                        np.histogram(lags, bins=edges)[0].tolist() if lags else []
                    ),
                },
            }
        out[name] = entry
    return out


def _cooccurrence_verdict(pairings):
    """Decide whether dialogue-bearing windows co-occur at lag ~ 0.

    The decisive cells are speech-band x face and speech-band x mouth on
    speech-bearing windows. A rate at or above CONSISTENT_FRAC means the
    method works and the corpus can be filtered on it; below means the audio
    does not contain the speech the transcript claims.
    """
    cells = {}
    for name in ("speech__face", "speech__mouth"):
        stats = pairings.get(name, {}).get("speech", {})
        cells[name] = {
            "n_speech_windows": stats.get("n", 0),
            "n_significant_near_zero": stats.get("n_near_zero", 0),
            "frac_significant_near_zero": stats.get("frac_near_zero"),
        }
    usable = [
        c for c in cells.values()
        if c["n_speech_windows"] >= MIN_CELL
        and c["frac_significant_near_zero"] is not None
    ]
    best = max((c["frac_significant_near_zero"] for c in usable), default=None)
    if best is not None and best >= CONSISTENT_FRAC:
        verdict = "dialogue_correlates_at_lag0"
        statement = (
            "speech-bearing windows show a significant speech-band x face/mouth "
            "peak at |lag| <= 0.5 s; the alignment method works"
        )
    elif best is None:
        verdict = "insufficient_dialogue_windows"
        statement = (
            "fewer than MIN_CELL speech-bearing windows with a usable face/mouth "
            "signal; the decisive cell cannot be read"
        )
    else:
        verdict = "no_dialogue_cooccurrence"
        statement = (
            "speech-bearing windows do NOT show a significant speech-band x "
            "face/mouth peak at |lag| <= 0.5 s; the audio does not contain the "
            "speech the transcript claims (added score or mismatched track)"
        )
    return {
        "decisive_pairings": ["speech__face", "speech__mouth"],
        "cells": cells,
        "verdict": verdict,
        "statement": statement,
    }


def aggregate_film(video_id, rows, corr_curves, max_lag_sec):
    """Aggregate window rows into one film record with a verdict."""
    ok = [r for r in rows if r["status"] == "ok"]
    film = {
        "video_id": video_id,
        "n_windows": len(rows),
        "n_ok": len(ok),
        "n_failed": len(rows) - len(ok),
        "windows": rows,
    }

    if not ok:
        film.update(
            {
                "median_lag_sec": None,
                "iqr_lag_sec": None,
                "mad_lag_sec": None,
                "frac_significant": 0.0,
                "n_significant": 0,
                "film_corr_peak": None,
                "film_corr_peak_z": None,
                "film_lag_sec": None,
                "drift_slope": None,
                "drift_pct": None,
                "drift_sec_per_10min": None,
                "drift_r": None,
                "drift_n": 0,
                "face_speech_crosstab": _crosstab(rows),
                "speech_no_face_frac": None,
                "face_no_speech_frac": None,
                "semantic_flags": [],
                "feature_pairings": {},
                "dialogue_cooccurrence": _cooccurrence_verdict({}),
                "flags": ["no_data"],
                "verdict": "error",
            }
        )
        film.update(_lag_distribution(np.array([]), np.array([], dtype=bool), max_lag_sec))
        return film

    lags = np.array([r["lag_sec"] for r in ok], dtype=np.float64)
    sig = np.array([r["significant"] for r in ok], dtype=bool)
    n_sig = int(sig.sum())

    film["median_lag_sec"] = float(np.median(lags))
    film["iqr_lag_sec"] = float(np.percentile(lags, 75) - np.percentile(lags, 25))
    film["mad_lag_sec"] = float(np.median(np.abs(lags - np.median(lags))))
    film["n_significant"] = n_sig
    film["frac_significant"] = float(n_sig / len(ok))

    # Lag distribution: full list, histogram, and a plain co-occurrence verdict.
    film.update(_lag_distribution(lags, sig, max_lag_sec))

    # Per feature pairing, stratified by whether the window carries speech text.
    pairings = _pairing_aggregate(ok, max_lag_sec)
    film["feature_pairings"] = pairings
    film["dialogue_cooccurrence"] = _cooccurrence_verdict(pairings)

    # Film-level correlation = mean of the per-window curves (shared lag axis).
    k_full = int(max_lag_sec * ENV_FPS)
    film_lag = film_peak = film_z = None
    same_len = [c for c in corr_curves if c is not None and c.size == 2 * k_full + 1]
    if same_len:
        mean_r = np.mean(np.stack(same_len, axis=0), axis=0)
        lags_axis = np.arange(-k_full, k_full + 1) * HOP_SEC
        film_lag, film_peak, film_z = peak_of(lags_axis, mean_r)
    film["film_lag_sec"] = film_lag
    film["film_corr_peak"] = film_peak
    film["film_corr_peak_z"] = film_z

    # Drift: lag against absolute window time.
    drift = _fit_drift(ok, sig)
    film.update(drift)

    film.update(_crosstab(rows))
    flags, verdict = _verdict(film)
    film["flags"] = flags
    film["verdict"] = verdict
    return film


def _fit_drift(ok, sig):
    """OLS of lag (s) against window time (s). Slope is s lag per s film.

    Fits only windows with a real correlation peak. A fit over noise peaks
    would report a slope with no meaning.
    """
    fit_rows = [r for r, s in zip(ok, sig) if s]
    if len(fit_rows) < MIN_SIG_FOR_DRIFT:
        return {
            "drift_slope": None,
            "drift_pct": None,
            "drift_sec_per_10min": None,
            "drift_r": None,
            "drift_n": len(fit_rows),
        }
    t = np.array([0.5 * (r["start_sec"] + r["end_sec"]) for r in fit_rows])
    y = np.array([r["lag_sec"] for r in fit_rows])
    if np.ptp(t) < 1e-6:
        return {
            "drift_slope": None,
            "drift_pct": None,
            "drift_sec_per_10min": None,
            "drift_r": None,
            "drift_n": len(fit_rows),
        }
    slope, intercept = np.polyfit(t, y, 1)
    pred = slope * t + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "drift_slope": float(slope),
        "drift_pct": float(slope * 100.0),
        "drift_sec_per_10min": float(slope * 600.0),
        "drift_r": float(np.sqrt(max(0.0, r2))),
        "drift_n": len(fit_rows),
    }


def _crosstab(rows):
    """Cross-tabulate speech presence against face presence."""
    speech_face = speech_no_face = no_speech_face = neither = 0
    n_speech = n_face = 0
    for r in rows:
        if r.get("speech_present") is None or r.get("face_valid_frac") is None:
            continue
        sp, fc = bool(r["speech_present"]), bool(r["face_valid_frac"] > 0)
        n_speech += int(sp)
        n_face += int(fc)
        if sp and fc:
            speech_face += 1
        elif sp and not fc:
            speech_no_face += 1
        elif not sp and fc:
            no_speech_face += 1
        else:
            neither += 1

    speech_no_face_frac = (speech_no_face / n_speech) if n_speech else None
    face_no_speech_frac = (no_speech_face / n_face) if n_face else None

    semantic_flags = []
    if n_speech >= 3 and speech_no_face_frac is not None and speech_no_face_frac > 0.9:
        semantic_flags.append("speech_without_face")
    if n_face >= 3 and face_no_speech_frac is not None and face_no_speech_frac > 0.5:
        semantic_flags.append("face_over_music")

    return {
        "face_speech_crosstab": {
            "speech_face": speech_face,
            "speech_no_face": speech_no_face,
            "no_speech_face": no_speech_face,
            "neither": neither,
            "n_speech": n_speech,
            "n_face": n_face,
        },
        "speech_no_face_frac": speech_no_face_frac,
        "face_no_speech_frac": face_no_speech_frac,
        "semantic_flags": semantic_flags,
    }


def _verdict(film):
    """Combine synchrony and semantic evidence into flags + a primary verdict."""
    flags, semantic_flags = [], film.get("semantic_flags", [])
    flags.extend(semantic_flags)

    if film["n_ok"] < 3:
        flags.append("insufficient_data")
        return flags, "insufficient_data"
    if film["frac_significant"] < CONSISTENT_FRAC:
        flags.append("no_consistent_peak")
        return flags, "no_consistent_peak"

    median_lag = film.get("median_lag_sec")
    slope = film.get("drift_slope")
    if median_lag is not None and abs(median_lag) > OFFSET_FLAG_SEC:
        flags.append("offset")
    if slope is not None and abs(slope) > DRIFT_FLAG_SLOPE:
        flags.append("drift")

    # A large slope accumulates into a large median lag. Drift is the root
    # cause in that case, so it takes priority over the constant-offset flag.
    if "drift" in flags:
        return flags, "drift"
    if "offset" in flags:
        return flags, "offset"
    return flags, "ok"


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(films):
    print("")
    print("=== audio-video alignment (per film) ===")
    print(
        f"{'film':<26}{'n':>4}{'ok':>4}{'sig':>5}{'med_lag':>10}"
        f"{'iqr':>9}{'slope%':>9}{'sp_no_face':>12}{'face_no_sp':>12}"
        f"{'search':>9}{'co_occ':>8}  dialogue-verdict / verdict"
    )
    for f in films:
        med = f"{f['median_lag_sec']*1000:.0f}ms" if f["median_lag_sec"] is not None else "-"
        iqr = f"{f['iqr_lag_sec']*1000:.0f}" if f["iqr_lag_sec"] is not None else "-"
        sl = f"{f['drift_pct']:.2f}" if f["drift_pct"] is not None else "-"
        snf = f"{f['speech_no_face_frac']:.2f}" if f["speech_no_face_frac"] is not None else "-"
        fns = f"{f['face_no_speech_frac']:.2f}" if f["face_no_speech_frac"] is not None else "-"
        search = f"+/-{f['search_max_lag_sec']:g}s"
        co = "yes" if f.get("co_occurs_within_search") else "no"
        dlg = f.get("dialogue_cooccurrence", {}).get("verdict", "-")
        print(
            f"{str(f['video_id'])[:25]:<26}{f['n_windows']:>4}{f['n_ok']:>4}"
            f"{f['n_significant']:>5}{med:>10}{iqr:>9}{sl:>9}{snf:>12}{fns:>12}"
            f"{search:>9}{co:>8}  {dlg} / {f['verdict']}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audio-video alignment QC: offset, drift, speech-vs-face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/expanded_manifest_segments.json",
        help="Input segment manifest (default: data/expanded_manifest_segments.json)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/",
        help="Output directory (default: data/)",
    )
    parser.add_argument(
        "--samples-per-film",
        type=int,
        default=8,
        help="Windows sampled per film (default: 8)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Thread pool size for window decode (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for per-film window sampling (default: 42)",
    )
    parser.add_argument(
        "--max-lag",
        type=float,
        default=DEFAULT_MAX_LAG_SEC,
        help="Correlation search range in seconds, +/- this value "
        "(default: 10.0). A window searches at most window_seconds/2.",
    )
    parser.add_argument(
        "--min-speech-chars",
        type=int,
        default=MIN_SPEECH_CHARS,
        help="A window is speech-bearing when its transcript text is at least "
        f"this many characters (default: {MIN_SPEECH_CHARS})",
    )
    parser.add_argument(
        "--face-model",
        type=str,
        default=os.environ.get("FLIK_YUNET_MODEL", DEFAULT_FACE_MODEL),
        help="YuNet ONNX path. Missing file falls back to the Haar cascade.",
    )
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    manifest_path = resolve_path(args.manifest)
    out_dir = resolve_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Manifest: %s", manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    logger.info("Loaded %d windows", len(manifest))

    face_kind, _ = build_face_detector(args.face_model)
    face_pool = FaceDetectorPool(args.face_model, face_kind)
    logger.info("Face detector: %s", face_kind)
    print(f"face detector: {face_kind}")

    sampled = sample_windows_per_film(manifest, args.samples_per_film, args.seed)
    logger.info("Sampled %d windows across %d films", len(sampled), len({s.get("video_id") for s in sampled}))
    print(f"sampled windows: {len(sampled)}")

    started = time.time()
    rows = [None] * len(sampled)
    curves = [None] * len(sampled)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                analyze_window, seg, face_pool, args.max_lag, args.min_speech_chars
            ): i
            for i, seg in enumerate(sampled)
        }
        done = 0
        for future in as_completed(futures):
            i = futures[future]
            rows[i], curves[i] = future.result()
            done += 1
            if done % 100 == 0 or done == len(sampled):
                print(f"  analyzed {done}/{len(sampled)}")
    elapsed = time.time() - started
    logger.info("Analysis finished in %.1f s", elapsed)

    by_film = defaultdict(list)
    curves_by_film = defaultdict(list)
    for row, curve in zip(rows, curves):
        vid = row["video_id"]
        by_film[vid].append(row)
        if curve is not None:
            curves_by_film[vid].append(curve)

    films = [
        aggregate_film(vid, by_film[vid], curves_by_film.get(vid, []), args.max_lag)
        for vid in sorted(by_film.keys(), key=str)
    ]

    out = {
        "meta": {
            "manifest": manifest_path,
            "out_dir": out_dir,
            "samples_per_film": args.samples_per_film,
            "workers": args.workers,
            "seed": args.seed,
            "elapsed_sec": elapsed,
            "n_films": len(films),
            "n_sampled": len(sampled),
            "env_fps": ENV_FPS,
            "max_lag_sec": args.max_lag,
            "offset_flag_sec": OFFSET_FLAG_SEC,
            "drift_flag_slope": DRIFT_FLAG_SLOPE,
            "peak_min": PEAK_MIN,
            "peak_sig_z": PEAK_SIG_Z,
            "consistent_frac": CONSISTENT_FRAC,
            "near_zero_lag_sec": NEAR_ZERO_LAG_SEC,
            "min_speech_chars": args.min_speech_chars,
            "audio_features": list(AUDIO_FEATURES),
            "visual_features": list(VISUAL_FEATURES),
            "speech_band_hz": list(SPEECH_BAND_HZ),
            "face_detector": face_kind,
            "face_model": args.face_model if face_kind == "yunet" else None,
        },
        "films": films,
    }

    out_path = os.path.join(out_dir, "av_alignment_scores.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print_summary(films)
    print("")
    print(f"wrote {out_path}")
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
