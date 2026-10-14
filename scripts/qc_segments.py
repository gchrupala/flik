#!/usr/bin/env python
"""
Screen training windows for data quality.

Phase 1 computes cheap manifest-wide statistics over every segment:
durations, text lengths, and windows-per-video. Phase 2 decodes a
stratified sample of windows and flags no-speech audio, static video,
and mostly-black video.

Usage:
    uv run --extra cu128 python -m scripts.qc_segments
    uv run --extra cu128 python -m scripts.qc_segments --total-samples 5000 --workers 32
    uv run --extra cpu python -m scripts.qc_segments --total-samples 20 --workers 4
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root so src.* imports work under `python -m`.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import torch

from src.CONSTANTS import PROJECT_ROOT
from src.utils.audio import audio_to_tensor
from src.utils.paths import resolve_path
from src.utils.video import VIDEOMAE_MEAN, VIDEOMAE_STD, video_to_tensor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_FRAMES = 16
SAMPLE_RATE = 16000

# Energy VAD: 25 ms frames, 10 ms hop.
VAD_FRAME_MS = 25
VAD_HOP_MS = 10
SILENCE_FLOOR_DBFS = -45.0
PEAK_OFFSET_DB = 30.0
NO_SPEECH_VOICED_FRAC = 0.1

# Static / black video thresholds on [0, 1] frames.
STATIC_DIFF_THRESHOLD = 0.005
BLACK_PIXEL_THRESHOLD = 0.02
MOSTLY_BLACK_FRAC = 0.8
MIN_DBFS = -100.0

# video_to_tensor returns ImageNet-normalized frames. Denormalize to [0, 1]
# so the static and black thresholds above keep their meaning.
_MEAN_TENSOR = torch.tensor(VIDEOMAE_MEAN).view(1, 3, 1, 1)
_STD_TENSOR = torch.tensor(VIDEOMAE_STD).view(1, 3, 1, 1)

DURATION_BUCKETS = [0, 2, 3, 4, 5, 6, 8, 10, 15, 30, float("inf")]
DBFS_HIST_EDGES = [-100, -60, -50, -40, -30, -20, -10, 0, 10]
FRAME_DIFF_HIST_EDGES = [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 1.0]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    """Configure file + stream logging (mirrors validate_manifest.py)."""
    log_dir = os.path.join(PROJECT_ROOT, "logdir")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("qc_segments")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh = logging.FileHandler(os.path.join(log_dir, "qc_segments.log"))
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)
    return logger


logger = logging.getLogger("qc_segments")


# ---------------------------------------------------------------------------
# Manifest loading and Phase 1
# ---------------------------------------------------------------------------

def load_manifest(path):
    """Load a segment manifest JSON list."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _percentiles(values):
    """Return p5/p25/p50/p75/p95 plus min/max/mean for a value list."""
    if not values:
        return {}
    arr = np.asarray(values, dtype=np.float64)
    p5, p25, p50, p75, p95 = np.percentile(arr, [5, 25, 50, 75, 95])
    return {
        "min": float(arr.min()),
        "p5": float(p5),
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
    }


def phase1_stats(manifest):
    """Compute cheap manifest-wide statistics. No decoding."""
    durations = []
    text_lengths = []
    empty_text = 0
    per_video = Counter()
    n_segments_counter = Counter()

    for seg in manifest:
        start = float(seg.get("start_sec", 0.0))
        end = float(seg.get("end_sec", 0.0))
        durations.append(max(0.0, end - start))

        text = seg.get("text", "") or ""
        text_lengths.append(len(text))
        if not text.strip():
            empty_text += 1

        per_video[seg.get("video_id", "unknown")] += 1

        n_seg = seg.get("n_segments")
        if n_seg is None:
            n_segments_counter["unknown"] += 1
        elif n_seg <= 1:
            n_segments_counter["1"] += 1
        elif n_seg == 2:
            n_segments_counter["2"] += 1
        elif n_seg <= 4:
            n_segments_counter["3-4"] += 1
        else:
            n_segments_counter["5+"] += 1

    total = len(manifest)
    dur_hist = np.histogram(durations, bins=DURATION_BUCKETS)[0].tolist()
    windows_per_video = list(per_video.values())

    return {
        "n_windows": total,
        "n_videos": len(per_video),
        "duration_sec": _percentiles(durations),
        "duration_buckets": {
            "edges": [str(e) for e in DURATION_BUCKETS],
            "counts": dur_hist,
        },
        "text_chars": _percentiles(text_lengths),
        "n_empty_text": empty_text,
        "frac_empty_text": (empty_text / total) if total else 0.0,
        "windows_per_video": _percentiles(windows_per_video),
        "n_segments_dist": dict(n_segments_counter),
        "top_videos": per_video.most_common(10),
    }


def print_phase1(stats):
    """Print Phase 1 statistics as a compact table."""
    print("")
    print("=== Phase 1: manifest-wide stats (no decoding) ===")
    print(f"windows: {stats['n_windows']}   videos: {stats['n_videos']}")
    d = stats["duration_sec"]
    if d:
        print(
            "duration (s): "
            f"p5={d['p5']:.2f} p25={d['p25']:.2f} p50={d['p50']:.2f} "
            f"p75={d['p75']:.2f} p95={d['p95']:.2f} "
            f"min={d['min']:.2f} max={d['max']:.2f} mean={d['mean']:.2f}"
        )
    t = stats["text_chars"]
    if t:
        print(
            "text chars:   "
            f"p5={t['p5']:.0f} p25={t['p25']:.0f} p50={t['p50']:.0f} "
            f"p75={t['p75']:.0f} p95={t['p95']:.0f} mean={t['mean']:.1f}"
        )
    print(
        f"empty text:   {stats['n_empty_text']} "
        f"({100.0 * stats['frac_empty_text']:.2f}%)"
    )
    w = stats["windows_per_video"]
    if w:
        print(
            "windows/video: "
            f"min={w['min']:.0f} p50={w['p50']:.0f} mean={w['mean']:.1f} "
            f"max={w['max']:.0f}"
        )
    print(f"n_segments:   {stats['n_segments_dist']}")
    print("duration buckets (s: count):")
    edges, counts = stats["duration_buckets"]["edges"], stats["duration_buckets"]["counts"]
    for i, c in enumerate(counts):
        print(f"  [{edges[i]}, {edges[i + 1]}): {c}")


# ---------------------------------------------------------------------------
# Phase 2: stratified sampling
# ---------------------------------------------------------------------------

def stratified_sample(manifest, total_samples, seed):
    """Sample windows proportionally by video. Deterministic for a seed.

    Every video gets at least one window when total_samples allows it.
    """
    rng = random.Random(seed)
    groups = defaultdict(list)
    for seg in manifest:
        groups[seg.get("video_id", "unknown")].append(seg)

    video_ids = sorted(groups.keys())
    n_videos = len(video_ids)
    n_total = len(manifest)
    if n_videos == 0 or n_total == 0 or total_samples <= 0:
        return []

    target = min(total_samples, n_total)

    counts = {v: 0 for v in video_ids}
    remaining = target
    if target >= n_videos:
        for v in video_ids:
            counts[v] = 1
        remaining = target - n_videos

    # Largest-remainder allocation of the remaining slots by group size.
    quota = {v: remaining * len(groups[v]) / n_total for v in video_ids}
    floors = {v: int(quota[v]) for v in video_ids}
    for v in video_ids:
        counts[v] += floors[v]

    leftover = target - sum(counts.values())
    order = sorted(video_ids, key=lambda v: quota[v] - floors[v], reverse=True)
    i = 0
    guard = 0
    while leftover > 0 and order and guard < target * len(order) + len(order):
        v = order[i % len(order)]
        if counts[v] < len(groups[v]):
            counts[v] += 1
            leftover -= 1
        i += 1
        guard += 1

    sampled = []
    for v in video_ids:
        k = min(counts[v], len(groups[v]))
        if k > 0:
            sampled.extend(rng.sample(groups[v], k))
    rng.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Phase 2: per-window metrics
# ---------------------------------------------------------------------------

def compute_vad(waveform, sample_rate):
    """Energy VAD on a mono waveform.

    Returns (voiced_fraction, mean_dbfs). The audio loader normalizes every
    window to unit variance, so dBFS is relative to the window, not absolute.
    """
    frame_len = int(sample_rate * VAD_FRAME_MS / 1000)
    hop = int(sample_rate * VAD_HOP_MS / 1000)
    wav = np.asarray(waveform, dtype=np.float64).reshape(-1)
    if wav.size == 0:
        return 0.0, MIN_DBFS
    if wav.size < frame_len:
        wav = np.pad(wav, (0, frame_len - wav.size))

    frames = np.lib.stride_tricks.sliding_window_view(wav, frame_len)[::hop]
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    dbfs = 20.0 * np.log10(rms)
    dbfs = np.maximum(dbfs, MIN_DBFS)

    peak_dbfs = float(dbfs.max())
    threshold = max(SILENCE_FLOOR_DBFS, peak_dbfs - PEAK_OFFSET_DB)
    voiced_frac = float(np.mean(dbfs > threshold))
    return voiced_frac, float(dbfs.mean())


def analyze_segment(seg, align_to_start: bool = True):
    """Decode one window and compute QC metrics. Never raises.

    ``align_to_start`` is forwarded to ``audio_to_tensor`` so the voiced /
    energy metrics describe the requested ``[start_sec, end_sec)`` window.
    ``False`` keeps the historical keyframe pre-roll (runs before 2026-09-16).
    """
    start = float(seg.get("start_sec", 0.0))
    end = float(seg.get("end_sec", 0.0))
    text = seg.get("text", "") or ""
    row = {
        "id": seg.get("id"),
        "video_id": seg.get("video_id"),
        "start_sec": start,
        "end_sec": end,
        "duration": max(0.0, end - start),
        "n_segments": seg.get("n_segments"),
        "text_len": len(text),
        "voiced_frac": None,
        "mean_dbfs": None,
        "mean_frame_diff": None,
        "black_frac": None,
        "flags": [],
        "status": "ok",
    }

    try:
        video_path = resolve_path(seg.get("video_path", ""))
        video = video_to_tensor(video_path, start, end, num_frames=NUM_FRAMES)
        audio = audio_to_tensor(video_path, start, end, sample_rate=SAMPLE_RATE,
                                align_to_start=align_to_start)
    except Exception as exc:  # noqa: BLE001 - bad media must not kill the run
        row["status"] = f"error: {exc}"
        row["flags"] = ["decode_fail"]
        return row

    # Video metrics on [0, 1] frames. Denormalize the loader output.
    frames01 = (video * _STD_TENSOR + _MEAN_TENSOR).clamp(0.0, 1.0)
    if frames01.shape[0] >= 2:
        diffs = (frames01[1:] - frames01[:-1]).abs().mean(dim=(1, 2, 3))
        mean_frame_diff = float(diffs.mean())
    else:
        mean_frame_diff = 0.0
    frame_means = frames01.mean(dim=(1, 2, 3))
    black_frac = float((frame_means < BLACK_PIXEL_THRESHOLD).float().mean())

    # Audio metrics. One channel at sample_rate.
    voiced_frac, mean_dbfs = compute_vad(audio[0].numpy(), SAMPLE_RATE)

    flags = []
    if mean_frame_diff < STATIC_DIFF_THRESHOLD:
        flags.append("static")
    if black_frac > MOSTLY_BLACK_FRAC:
        flags.append("mostly_black")
    if voiced_frac < NO_SPEECH_VOICED_FRAC:
        flags.append("no_speech")

    row.update(
        {
            "voiced_frac": voiced_frac,
            "mean_dbfs": mean_dbfs,
            "mean_frame_diff": mean_frame_diff,
            "black_frac": black_frac,
            "flags": flags,
        }
    )
    return row


# ---------------------------------------------------------------------------
# Phase 2: aggregation and output
# ---------------------------------------------------------------------------

def _pct(count, total):
    return (100.0 * count / total) if total else 0.0


def summarize(rows):
    """Aggregate sampled rows into overall rates, per-video table, histograms."""
    n = len(rows)
    n_decode_fail = sum(1 for r in rows if r["status"] != "ok")
    n_no_speech = sum(1 for r in rows if "no_speech" in r["flags"])
    n_static = sum(1 for r in rows if "static" in r["flags"])
    n_black = sum(1 for r in rows if "mostly_black" in r["flags"])
    n_empty_text = sum(1 for r in rows if r["text_len"] == 0)

    ok_rows = [r for r in rows if r["status"] == "ok"]

    overall = {
        "n_sampled": n,
        "n_decode_ok": len(ok_rows),
        "n_decode_fail": n_decode_fail,
        "pct_no_speech": _pct(n_no_speech, n),
        "pct_static": _pct(n_static, n),
        "pct_mostly_black": _pct(n_black, n),
        "pct_decode_fail": _pct(n_decode_fail, n),
        "pct_empty_text": _pct(n_empty_text, n),
    }

    per_video = {}
    by_video = defaultdict(list)
    for r in rows:
        by_video[r["video_id"]].append(r)
    for vid in sorted(by_video.keys(), key=lambda x: str(x)):
        vrows = by_video[vid]
        m = len(vrows)
        vok = [r for r in vrows if r["status"] == "ok"]
        per_video[str(vid)] = {
            "n_sampled": m,
            "pct_no_speech": _pct(sum("no_speech" in r["flags"] for r in vrows), m),
            "pct_static": _pct(sum("static" in r["flags"] for r in vrows), m),
            "pct_mostly_black": _pct(sum("mostly_black" in r["flags"] for r in vrows), m),
            "pct_decode_fail": _pct(sum(r["status"] != "ok" for r in vrows), m),
            "mean_voiced_frac": float(
                np.mean([r["voiced_frac"] for r in vok])
            ) if vok else None,
            "mean_dbfs": float(
                np.mean([r["mean_dbfs"] for r in vok])
            ) if vok else None,
            "mean_frame_diff": float(
                np.mean([r["mean_frame_diff"] for r in vok])
            ) if vok else None,
        }

    dbfs_vals = [r["mean_dbfs"] for r in ok_rows if r["mean_dbfs"] is not None]
    diff_vals = [
        r["mean_frame_diff"] for r in ok_rows if r["mean_frame_diff"] is not None
    ]
    dbfs_hist = np.histogram(dbfs_vals, bins=DBFS_HIST_EDGES)[0].tolist()
    diff_hist = np.histogram(diff_vals, bins=FRAME_DIFF_HIST_EDGES)[0].tolist()

    return {
        "overall": overall,
        "per_video": per_video,
        "histograms": {
            "mean_dbfs": {
                "edges": DBFS_HIST_EDGES,
                "counts": dbfs_hist,
            },
            "mean_frame_diff": {
                "edges": FRAME_DIFF_HIST_EDGES,
                "counts": diff_hist,
            },
        },
    }


def print_summary(summary):
    """Print the Phase 2 summary as a compact table."""
    o = summary["overall"]
    print("")
    print("=== Phase 2: decode-based QC summary ===")
    print(f"sampled windows: {o['n_sampled']}   decode ok: {o['n_decode_ok']}")
    print(
        f"no_speech: {o['pct_no_speech']:.2f}%   "
        f"static: {o['pct_static']:.2f}%   "
        f"mostly_black: {o['pct_mostly_black']:.2f}%   "
        f"decode_fail: {o['pct_decode_fail']:.2f}%   "
        f"empty_text: {o['pct_empty_text']:.2f}%"
    )
    print("")
    print(
        f"{'video_id':<28}{'n':>6}{'no_sp%':>9}{'static%':>9}"
        f"{'black%':>9}{'fail%':>8}{'vox_frac':>10}{'frame_diff':>12}"
    )
    for vid, v in summary["per_video"].items():
        vf = f"{v['mean_voiced_frac']:.3f}" if v["mean_voiced_frac"] is not None else "-"
        fd = f"{v['mean_frame_diff']:.4f}" if v["mean_frame_diff"] is not None else "-"
        print(
            f"{vid[:27]:<28}{v['n_sampled']:>6}"
            f"{v['pct_no_speech']:>9.2f}{v['pct_static']:>9.2f}"
            f"{v['pct_mostly_black']:>9.2f}{v['pct_decode_fail']:>8.2f}"
            f"{vf:>10}{fd:>12}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Screen training windows for data quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="data/expanded_manifest_segments.json",
        help="Input segment manifest (default: data/expanded_manifest_segments.json)",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=5000,
        help="Number of windows to decode in Phase 2 (default: 5000)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Thread pool size for Phase 2 decode (default: 16)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for stratified sampling (default: 42)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="qc_results/",
        help="Output directory (default: qc_results/)",
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip Phase 1 manifest-wide statistics",
    )
    parser.add_argument(
        "--audio-align-to-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim the audio windows to start_sec by decoded PTS (default: "
        "true; matches dataset.audio_align_to_start). "
        "--no-audio-align-to-start keeps the historical keyframe pre-roll "
        "(runs logged before 2026-09-16).",
    )
    return parser.parse_args()


def main():
    setup_logging()
    args = parse_args()

    manifest_path = resolve_path(args.manifest)
    out_dir = resolve_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    logger.info("Manifest: %s", manifest_path)
    logger.info(
        "Audio window alignment: "
        + (
            "timestamp-accurate, starts at start_sec (audio_align_to_start=true)"
            if args.audio_align_to_start
            else "legacy keyframe pre-roll retained (audio_align_to_start=false, bug-compatible)"
        )
    )
    manifest = load_manifest(manifest_path)
    logger.info("Loaded %d windows", len(manifest))

    phase1 = None
    if not args.skip_phase1:
        phase1 = phase1_stats(manifest)
        print_phase1(phase1)

    print("")
    print("=== Phase 2: stratified sampling ===")
    sampled = stratified_sample(manifest, args.total_samples, args.seed)
    logger.info("Sampled %d windows with seed %d", len(sampled), args.seed)
    print(f"sampled windows: {len(sampled)} (seed={args.seed})")

    started = time.time()
    rows = [None] * len(sampled)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(analyze_segment, seg, args.audio_align_to_start): idx
            for idx, seg in enumerate(sampled)
        }
        done = 0
        for future in as_completed(futures):
            idx = futures[future]
            rows[idx] = future.result()
            done += 1
            if done % 250 == 0 or done == len(sampled):
                print(f"  decoded {done}/{len(sampled)}")
    elapsed = time.time() - started
    logger.info("Phase 2 decode finished in %.1f s", elapsed)

    # Write one JSON object per sampled window.
    jsonl_path = os.path.join(out_dir, "qc_windows.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows)
    summary["meta"] = {
        "manifest": manifest_path,
        "seed": args.seed,
        "total_samples": args.total_samples,
        "n_sampled": len(sampled),
        "workers": args.workers,
        "elapsed_sec": elapsed,
    }
    if phase1 is not None:
        summary["phase1"] = phase1

    summary_path = os.path.join(out_dir, "qc_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print_summary(summary)
    print("")
    print(f"wrote {jsonl_path}")
    print(f"wrote {summary_path}")
    logger.info("Done in %.1f s", elapsed)


if __name__ == "__main__":
    main()
