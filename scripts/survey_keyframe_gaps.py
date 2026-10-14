#!/usr/bin/env python
"""Survey predicted audio offsets from video keyframe gaps (metadata only).

Background
----------
``_load_audio_segment`` (historical path) seeks to the previous video keyframe
and trims the decoded buffer from the FRONT. The returned window therefore
starts early by the keyframe distance:

    offset ~= -(start_sec - previous_video_keyframe_pts)

This script predicts that offset for every window in a manifest without
decoding a single audio or video frame. It reads only container metadata
(``ffprobe`` packet/frame tables), so it is cheap enough to run over a whole
corpus.

The relation is approximate. The demuxer's first audio frame after the seek
can start before the video keyframe (by tens of milliseconds, and up to about
one second on some files), so the prediction tracks the *distribution* of
measured offsets well (means within ~0.1 s and the exposure fractions within a
couple of points) but is not a per-window exact offset. Measured on the local
films: TheStilwellRoad had a median error of ~13 ms (92% of windows within one
audio frame); Black_Brigade had a systematic ~78 ms bias (about 3.6 AAC
frames).

Usage
-----
    uv run --extra cpu python -m scripts.survey_keyframe_gaps \
        --manifest data/expanded_manifest_segments.json \
        --output data/keyframe_gap_survey.json

    # Local films: remap manifest paths to an existing directory.
    uv run --extra cpu python -m scripts.survey_keyframe_gaps \
        --manifest /tmp/local_fixture.json --video-dir data/videos
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.paths import resolve_path  # noqa: E402

logger = logging.getLogger("survey_keyframe_gaps")


# ---------------------------------------------------------------------------
# Metadata probes (no frame decoding)
# ---------------------------------------------------------------------------

def ffprobe_keyframe_times(path: str) -> np.ndarray:
    """Video keyframe presentation timestamps in seconds.

    Primary probe: packet table with the K flag (metadata only, fast).
    Fallback: keyframe-only frame table via ``-skip_frame nokey``.
    """
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v",
            "-show_packets", "-show_entries", "packet=pts_time,flags",
            "-of", "csv=p=0", path,
        ],
        capture_output=True, text=True, check=True,
    ).stdout
    ts = []
    for line in out.splitlines():
        parts = line.split(",")
        if len(parts) > 1 and "K" in parts[1]:
            try:
                ts.append(float(parts[0]))
            except ValueError:
                continue
    if not ts:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v",
                "-skip_frame", "nokey", "-show_frames",
                "-show_entries", "frame=pts_time", "-of", "csv=p=0", path,
            ],
            capture_output=True, text=True, check=True,
        ).stdout
        for line in out.splitlines():
            try:
                ts.append(float(line.split(",")[0]))
            except ValueError:
                continue
    return np.array(sorted(ts), dtype=np.float64)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_offsets(keyframe_times: np.ndarray, windows) -> np.ndarray:
    """Predicted offset per window, signed (negative = window starts early).

    ``windows`` is a sequence of (start_sec, end_sec).
    """
    if len(keyframe_times) == 0:
        return np.zeros(len(windows), dtype=np.float64)
    offsets = []
    for start_sec, _end_sec in windows:
        idx = int(np.searchsorted(keyframe_times, start_sec, side="right")) - 1
        prev = float(keyframe_times[idx]) if idx >= 0 else 0.0
        offsets.append(-(start_sec - prev))
    return np.array(offsets, dtype=np.float64)


def summarize(offsets: np.ndarray, durations: np.ndarray) -> dict:
    """Aggregate offset and duration-normalised exposure statistics."""
    offsets = np.asarray(offsets, dtype=np.float64)
    durations = np.asarray(durations, dtype=np.float64)
    if len(offsets) == 0:
        return {"n_windows": 0}
    abs_off = np.abs(offsets)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac_dur = np.where(durations > 0, abs_off / durations, np.inf)
    worst = int(np.argmax(abs_off))
    return {
        "n_windows": int(len(offsets)),
        "mean_offset_sec": float(offsets.mean()),
        "median_offset_sec": float(np.median(offsets)),
        "worst_offset_sec": float(offsets[worst]),
        "mean_abs_offset_sec": float(abs_off.mean()),
        "max_abs_offset_sec": float(abs_off.max()),
        "frac_offset_gt_50pct_window": float(np.mean(frac_dur > 0.5)),
        "frac_offset_gt_100pct_window": float(np.mean(frac_dur > 1.0)),
        "mean_window_sec": float(durations.mean()),
        "median_window_sec": float(np.median(durations)),
    }


def load_manifest(path: str) -> dict:
    """Group manifest items by video_path -> list of (start_sec, end_sec)."""
    with open(resolve_path(path), "r", encoding="utf-8") as f:
        items = json.load(f)
    films: dict = {}
    for item in items:
        if "start_sec" not in item or "end_sec" not in item:
            continue
        raw = item.get("video_path") or item.get("video_id") or "?"
        films.setdefault(raw, []).append((float(item["start_sec"]), float(item["end_sec"])))
    return films


def local_video_path(manifest_path: str, video_dir: str) -> str:
    """Use <video_dir>/<basename> when the manifest path is unavailable."""
    resolved = resolve_path(manifest_path)
    if os.path.exists(resolved):
        return resolved
    candidate = os.path.join(video_dir, os.path.basename(manifest_path))
    return candidate if os.path.exists(candidate) else resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="data/expanded_manifest_segments.json",
        help="Segment manifest whose start_sec values define the windows.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional JSON output path. Default: print a summary only.",
    )
    parser.add_argument(
        "--video-dir", default=None,
        help="Fallback directory for manifest paths that do not exist "
        "(matched by basename); useful for local validation.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=8,
        help="Parallel ffprobe workers (default: 8).",
    )
    parser.add_argument(
        "--limit-films", type=int, default=None,
        help="Only survey the first N films (debug/validation).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    films = load_manifest(args.manifest)
    film_items = list(films.items())
    if args.limit_films is not None:
        film_items = film_items[: args.limit_films]
    logger.info(
        f"Surveying {len(film_items)} films "
        f"({sum(len(w) for _, w in film_items)} windows) from {args.manifest}"
    )

    results: dict = {}
    corpus_offsets = []
    corpus_durations = []
    t0 = time.time()
    errors: dict = {}

    def _work(item):
        raw_path, windows = item
        path = raw_path
        if args.video_dir:
            path = local_video_path(raw_path, args.video_dir)
        else:
            path = resolve_path(raw_path)
        keyframes = ffprobe_keyframe_times(path)
        offsets = predict_offsets(keyframes, windows)
        durations = np.array([e - s for s, e in windows], dtype=np.float64)
        res = summarize(offsets, durations)
        if len(keyframes) > 1:
            gaps = np.diff(keyframes)
            res["keyframe_gaps"] = {
                "n_keyframes": int(len(keyframes)),
                "mean_gap_sec": float(gaps.mean()),
                "median_gap_sec": float(np.median(gaps)),
                "max_gap_sec": float(gaps.max()),
            }
        else:
            res["keyframe_gaps"] = {"n_keyframes": int(len(keyframes))}
        return res, offsets, durations

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as ex:
        futures = {ex.submit(_work, it): it[0] for it in film_items}
        for future in as_completed(futures):
            raw_path = futures[future]
            try:
                res, offsets, durations = future.result()
                results[raw_path] = res
                corpus_offsets.append(offsets)
                corpus_durations.append(durations)
            except Exception as e:  # noqa: BLE001 - one bad film must not kill the run
                errors[raw_path] = f"{type(e).__name__}: {e}"
                logger.warning(f"Failed to probe {raw_path}: {e}")

    if corpus_offsets:
        corpus = summarize(
            np.concatenate(corpus_offsets), np.concatenate(corpus_durations)
        )
    else:
        corpus = {"n_windows": 0}

    logger.info("=" * 68)
    for raw_path, res in sorted(results.items()):
        logger.info(
            f"{os.path.basename(raw_path)}: n={res['n_windows']} "
            f"mean={res['mean_offset_sec']:+.3f}s "
            f"median={res['median_offset_sec']:+.3f}s "
            f"worst={res['worst_offset_sec']:+.3f}s "
            f">50%={res['frac_offset_gt_50pct_window']:.1%} "
            f">100%={res['frac_offset_gt_100pct_window']:.1%}"
        )
    logger.info(
        f"CORPUS: n={corpus['n_windows']} "
        f"mean={corpus['mean_offset_sec']:+.3f}s "
        f"median={corpus['median_offset_sec']:+.3f}s "
        f"worst={corpus['worst_offset_sec']:+.3f}s "
        f">50%={corpus['frac_offset_gt_50pct_window']:.1%} "
        f">100%={corpus['frac_offset_gt_100pct_window']:.1%}"
    )
    if errors:
        logger.warning(f"{len(errors)} films failed to probe")
    logger.info(f"Done in {time.time() - t0:.1f}s (metadata only, no decodes)")

    if args.output:
        payload = {
            "manifest": args.manifest,
            "relation": "offset = -(start_sec - previous_video_keyframe_pts)",
            "note": "Predicted only; accurate to about one audio frame.",
            "films": results,
            "film_errors": errors,
            "corpus": corpus,
        }
        with open(resolve_path(args.output), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Wrote {resolve_path(args.output)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
