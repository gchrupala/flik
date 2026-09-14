#!/usr/bin/env python3
"""Add confidence intervals to the QC screening table.

Reads the outputs of scripts/qc_segments.py and reports, for each quality
metric, the observed rate with two 95% intervals:

  1. Wilson score interval (naive: treats every window as independent).
  2. Film-cluster bootstrap: resamples the sampled FILMS with replacement and
     recomputes the pooled rate. Windows inside one film are correlated, so
     this interval is the honest one to quote.

Usage:
  python -m scripts.qc_significance \
      --summary qc_results_ws/qc_summary.json \
      --windows qc_results_ws/qc_windows.jsonl
"""

import argparse
import json
import logging
import math
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("qc_significance")

Z = 1.959963984540054  # 95%
METRICS = [
    ("decode_fail", "decode failure"),
    ("no_speech", "no speech"),
    ("static", "static video"),
    ("mostly_black", "mostly black"),
    ("empty_text", "empty text"),
]


def wilson_ci(k: int, n: int, z: float = Z) -> tuple:
    """Wilson score interval for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def load_windows(path: str) -> dict:
    """Group per-window flag dicts by film (video_id)."""
    by_film = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            w = json.loads(line)
            by_film.setdefault(w["video_id"], []).append(w)
    return by_film


def flag_of(window: dict, metric: str) -> bool:
    """True if this window fails the given metric."""
    if metric == "decode_fail":
        return window.get("status", "ok") != "ok"
    flags = window.get("flags") or []
    return metric in flags


def cluster_bootstrap(by_film: dict, metric: str, n_iter: int = 2000,
                      seed: int = 42) -> tuple:
    """Film-level bootstrap 95% interval for the pooled rate."""
    rng = random.Random(seed)
    films = list(by_film.keys())
    if not films:
        return (0.0, 0.0)
    rates = []
    for _ in range(n_iter):
        k = 0
        n = 0
        for _ in range(len(films)):
            film = films[rng.randrange(len(films))]
            for w in by_film[film]:
                n += 1
                k += int(flag_of(w, metric))
        rates.append(k / n if n else 0.0)
    rates.sort()
    lo = rates[int(0.025 * n_iter)]
    hi = rates[min(n_iter - 1, int(0.975 * n_iter))]
    return (lo, hi)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", default="qc_results_ws/qc_summary.json")
    p.add_argument("--windows", default="qc_results_ws/qc_windows.jsonl")
    p.add_argument("--iterations", type=int, default=2000)
    args = p.parse_args()

    with open(args.summary, "r", encoding="utf-8") as f:
        summary = json.load(f)
    overall = summary["overall"]
    n = overall["n_sampled"]
    by_film = load_windows(args.windows)

    logger.info("n windows: %d   n films: %d", n, len(by_film))
    print()
    print("metric            k/n            rate       Wilson 95% CI        "
          "film-bootstrap 95% CI")
    print("-" * 88)
    for key, label in METRICS:
        # summary stores pct_* as percentage numbers (0.30 = 0.30%)
        k = int(round(overall.get("pct_" + key, 0.0) * n / 100.0))
        lo_w, hi_w = wilson_ci(k, n)
        lo_b, hi_b = cluster_bootstrap(by_film, key, args.iterations)
        print("%-16s %5d/%-6d %8.3f%%   [%6.3f%%, %6.3f%%]   [%6.3f%%, %6.3f%%]"
              % (label, k, n, 100 * k / n,
                 100 * lo_w, 100 * hi_w, 100 * lo_b, 100 * hi_b))
    print()
    print("Wilson: every window independent (naive).")
    print("Film bootstrap: resamples films, %d iterations - quote this one."
          % args.iterations)


if __name__ == "__main__":
    main()
