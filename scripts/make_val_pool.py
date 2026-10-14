#!/usr/bin/env python
"""Build a frozen, content-addressed validation retrieval pool.

Why: the internal held-out retrieval metric is used for checkpoint selection
but is not comparable across runs because the runtime video-level split picks
a different pool (in size and composition) each run. This script freezes one
pool once, then addresses it by sha256 so every run measures the same metric.

The pool holds out WHOLE movies (reusing ``_split_manifest_by_video`` from
``scripts/train_hydra.py``), so no film appears in both the training manifest
and the pool. It then caps segments per movie and in total, deduplicates by
``(video_id, start_sec, end_sec)``, and sorts deterministically.

Outputs:
    <output>        JSON list of segment dicts (usable as a manifest)
    <sha-output>    hex sha256 of the exact bytes written to <output>

The same input always produces the same sha256 (deterministic ordering and
serialization).

Usage:
    python -m scripts.make_val_pool \
        --input data/expanded_manifest_segments.json \
        --output data/val_pool.json \
        --sha-output data/val_pool.sha256 \
        --dedup-report data/dedup_report.json \
        --per-movie-cap 50 --total-cap 20000 --val-ratio 0.1 --seed 42

With ``--dedup-report`` the hold-out unit is the movie key from the dedup
report (1,336 groups for 1,428 ids), so multiple video_ids of one film (e.g.
``Martha_Ivers`` + ``VTS_01_1..4``) stay together and cannot leak across the
train/val boundary.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse the exact grouping/hold-out logic used by training so the frozen pool
# matches the runtime split semantics (whole movies held out, deterministic seed).
from scripts.train_hydra import (  # noqa: E402
    _group_key,
    _load_movie_key_map,
    _split_manifest_by_video,
)


def build_val_pool(manifest, val_ratio=0.1, seed=42, per_movie_cap=50, total_cap=20000,
                   movie_key_map=None):
    """Return the frozen pool as a deterministic list of segment dicts.

    Steps:
      1. Hold out whole movies with ``_split_manifest_by_video`` (same seed).
         With ``movie_key_map`` (dedup report) the hold-out unit is the movie
         key, so duplicate copies / multi-part ids of one film stay together.
      2. Deduplicate by ``(video_id, start_sec, end_sec)`` (original contract;
         keeps distinct multi-part parts that happen to share timestamps).
      3. Sort by ``(group_key, start_sec, end_sec, video_id)``.
      4. Keep at most ``per_movie_cap`` segments per movie (first N in sort order).
      5. Truncate the sorted list to ``total_cap`` segments.

    ``group_key`` is the movie key when ``movie_key_map`` is given, else the
    item's ``video_id`` (so both modes are backwards compatible).
    """
    _, val_items = _split_manifest_by_video(manifest, val_ratio, seed, movie_key_map)

    def _gkey(item):
        return _group_key(item, movie_key_map)

    def _vid(item):
        return item.get("video_id") or item.get("video_path") or item.get("id") or ""

    seen = set()
    deduped = []
    for item in val_items:
        key = (_vid(item), item.get("start_sec"), item.get("end_sec"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    deduped.sort(
        key=lambda it: (
            _gkey(it),
            it.get("start_sec", 0.0),
            it.get("end_sec", 0.0),
            _vid(it),
        )
    )

    per_movie_counts = defaultdict(int)
    capped = []
    for item in deduped:
        group = _gkey(item)
        if per_movie_counts[group] >= per_movie_cap:
            continue
        per_movie_counts[group] += 1
        capped.append(item)

    return capped[:total_cap]


def _serialize(pool):
    """Canonical, deterministic JSON bytes for the pool."""
    return (json.dumps(pool, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _duration_stats(pool):
    durations = [
        float(it["end_sec"]) - float(it["start_sec"])
        for it in pool
        if "start_sec" in it and "end_sec" in it
    ]
    if not durations:
        return {}
    durations.sort()
    q1, _, q3 = statistics.quantiles(durations, n=4, method="inclusive")
    return {
        "n": len(durations),
        "min": durations[0],
        "p25": q1,
        "median": statistics.median(durations),
        "p75": q3,
        "max": durations[-1],
        "mean": statistics.fmean(durations),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/expanded_manifest_segments.json",
        help="Source segment manifest (default: data/expanded_manifest_segments.json)",
    )
    parser.add_argument("--output", default="data/val_pool.json")
    parser.add_argument("--sha-output", default="data/val_pool.sha256")
    parser.add_argument("--per-movie-cap", type=int, default=50)
    parser.add_argument("--total-cap", type=int, default=20000)
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of movies held out (default: 0.1, matches split_ratio)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dedup-report",
        default=None,
        help="Movie-level dedup report (e.g. data/dedup_report.json). When set, "
             "hold out by movie_key instead of video_id, so duplicate copies / "
             "multi-part ids of one film cannot leak across train/val.",
    )
    args = parser.parse_args(argv)

    with open(args.input, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    movie_key_map = None
    if args.dedup_report:
        movie_key_map = _load_movie_key_map(args.dedup_report)
        with open(args.dedup_report, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        print(f"dedup_report={args.dedup_report} sha256={digest} ids={len(movie_key_map)}")
    else:
        print("dedup_report=None; grouping by video_id (can leak across train/val)")

    pool = build_val_pool(
        manifest,
        val_ratio=args.val_ratio,
        seed=args.seed,
        per_movie_cap=args.per_movie_cap,
        total_cap=args.total_cap,
        movie_key_map=movie_key_map,
    )

    payload = _serialize(pool)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "wb") as f:
        f.write(payload)
    digest = hashlib.sha256(payload).hexdigest()
    with open(args.sha_output, "w", encoding="utf-8") as f:
        f.write(digest + "\n")

    n_movies = len({_group_key(it, movie_key_map) for it in pool})
    stats = _duration_stats(pool)
    print(f"Wrote {args.output}: {len(pool)} segments, {n_movies} movies")
    print(f"sha256={digest}")
    if stats:
        print(
            "duration(s): "
            f"min={stats['min']:.2f} p25={stats['p25']:.2f} "
            f"median={stats['median']:.2f} p75={stats['p75']:.2f} "
            f"max={stats['max']:.2f} mean={stats['mean']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
