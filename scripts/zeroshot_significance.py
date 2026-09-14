#!/usr/bin/env python3
"""Confidence intervals and paired tests for the zero-shot retrieval table.

Reads the JSON written by scripts/eval_zeroshot_vggsound.py and reports, per
model and metric:

  1. The pooled value, with a 95% class-cluster bootstrap interval. Classes
     are the resampling unit because clips in one class share content; treating
     clips as independent would understate the uncertainty.
  2. Paired comparisons between models: same class resample for both models,
     difference of pooled values, 95% interval. A difference whose interval
     excludes 0 is reported as a real effect at this sample size.

Usage:
  python -m scripts.zeroshot_significance --results zeroshot_results_full.json
"""

import argparse
import json
import logging
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("zeroshot_significance")

METRICS = [
    "a2v_r@1", "v2a_r@1", "a2v_r@5", "v2a_r@5", "a2v_r@10", "v2a_r@10",
]
N_ITER = 2000
SEED = 42


def class_stats(checkpoint: dict) -> dict:
    """Per-class (n_clips, per-metric rate) for one checkpoint."""
    out = {}
    for label, c in (checkpoint.get("per_class") or {}).items():
        n = c.get("n_clips")
        if not n:
            continue
        out[label] = {m: (n, c[m]) for m in METRICS if m in c}
    return out


def has_metric(stats: dict, metric: str) -> bool:
    """True if any class carries this metric (else report level value only)."""
    return any(metric in label_metrics
               for model_stats in stats.values()
               for label_metrics in model_stats.values())


def pooled(stats: dict, classes, metric: str) -> float:
    """Clips-weighted pooled rate over the given classes."""
    hits = 0.0
    total = 0.0
    for label in classes:
        entry = stats.get(label, {}).get(metric)
        if entry is None:
            continue
        n, rate = entry
        hits += rate * n
        total += n
    return hits / total if total else 0.0


def pooled_counts(stats: dict, metric: str) -> tuple:
    """Total hits and clips for a metric across all classes."""
    hits = 0.0
    total = 0.0
    for classes in stats.values():
        entry = classes.get(metric)
        if entry is None:
            continue
        n, rate = entry
        hits += rate * n
        total += n
    return hits, total


def wilson_ci(k: float, n: float, z: float = 1.959963984540054) -> tuple:
    """Wilson score interval; treats clips as independent (naive)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci(stats: dict, metric: str, labels, rng, n_iter=N_ITER) -> tuple:
    """95% class-cluster bootstrap interval for the pooled rate."""
    vals = sorted(
        pooled(stats, [labels[rng.randrange(len(labels))] for _ in labels], metric)
        for _ in range(n_iter)
    )
    return vals[int(0.025 * n_iter)], vals[int(0.975 * n_iter)]


def paired_diff(stats_a: dict, stats_b: dict, metric: str, labels, rng,
                n_iter=N_ITER) -> tuple:
    """95% interval for pooled(A) - pooled(B), resampling classes once per iter."""
    diffs = []
    for _ in range(n_iter):
        sample = [labels[rng.randrange(len(labels))] for _ in labels]
        diffs.append(pooled(stats_a, sample, metric) - pooled(stats_b, sample, metric))
    diffs.sort()
    return diffs[int(0.025 * n_iter)], diffs[int(0.975 * n_iter)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="zeroshot_results_full.json")
    p.add_argument("--iterations", type=int, default=N_ITER)
    args = p.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        data = json.load(f)
    checkpoints = data["checkpoints"]
    n_clips = data.get("n_clips_used", data.get("n_clips_manifest", "?"))
    logger.info("models: %d   clips: %s", len(checkpoints), n_clips)

    stats = {name: class_stats(c) for name, c in checkpoints.items()}
    labels = sorted(set().union(*(set(s.keys()) for s in stats.values())))
    logger.info("classes: %d", len(labels))
    rng = random.Random(SEED)

    print()
    print("Pooled retrieval with 95% class-cluster bootstrap CI")
    print("-" * 92)
    header = "model".ljust(16) + "".join(m.rjust(12) for m in METRICS)
    print(header)
    for name, s in stats.items():
        cells = []
        for m in METRICS:
            if has_metric(stats, m):
                lo, hi = bootstrap_ci(s, m, labels, rng, args.iterations)
                cells.append("%.4f+/-%.4f" % (pooled(s, labels, m), (hi - lo) / 2))
            else:
                # No per-class data: fall back to pooled value with a naive
                # Wilson interval computed from the checkpoint-level counts.
                k = checkpoints[name].get(m, float("nan")) * (n_clips if isinstance(n_clips, int) else 0)
                lo, hi = wilson_ci(k, n_clips) if isinstance(n_clips, int) else (0.0, 0.0)
                cells.append(("%.4f+/-%.4f~" % (checkpoints[name].get(m, float("nan")),
                                                (hi - lo) / 2)).rjust(12))
        print(name.ljust(16) + "".join(c.rjust(12) for c in cells))

    print()
    print("Paired comparisons (95% CI of difference; * = excludes 0)")
    print("-" * 92)
    names = list(stats.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            cells = []
            for m in METRICS:
                if not has_metric(stats, m):
                    cells.append("     n/a".rjust(12))
                    continue
                lo, hi = paired_diff(stats[a], stats[b], m, labels, rng, args.iterations)
                star = "*" if (lo > 0 or hi < 0) else " "
                cells.append(("%+.4f[%s]" % (pooled(stats[a], labels, m)
                                             - pooled(stats[b], labels, m), star)).rjust(12))
            print(("%s - %s" % (a, b)).ljust(16) + "".join(cells))
    print()
    print("Interval: 2.5-97.5 percentiles of %d class resamples." % args.iterations)
    print("Classes are the resampling unit; clips within a class are correlated.")


if __name__ == "__main__":
    main()
