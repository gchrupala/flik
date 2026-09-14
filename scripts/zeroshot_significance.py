#!/usr/bin/env python3
"""Significance for the zero-shot VGGSound retrieval table.

Retrieval over N clips: chance R@K = K/N. Observed hits are single-digit, so
the right test is exact (binomial tail), not a bootstrap: resampling cannot
resolve differences of 1-3 clips, and bootstrap intervals look spuriously
narrow when the hits sit in a few classes.

The eval JSON also carries a per_class block, but those rates are CLASS-MATCH
accuracy (fraction of a class's clips whose top-1 match comes from the same
class; chance = 1/n_classes). They are reported as a separate line and are NOT
pooled with the global R@K metrics.

Usage:
  python -m scripts.zeroshot_significance --results zeroshot_results_full.json
"""

import argparse
import json
import logging
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("zeroshot_significance")

RETRIEVAL = ["a2v_r@1", "v2a_r@1", "a2v_r@5", "v2a_r@5", "a2v_r@10", "v2a_r@10"]
CLASS_MATCH = ["a2v_cls_r@1", "v2a_cls_r@1"]


def wilson_ci(k: float, n: float, z: float = 1.959963984540054) -> tuple:
    """Wilson score interval, 95%. Treats clips as independent."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / denom
    return (max(0.0, center - half), min(1.0, center + half))


def binom_tail_ge(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), computed in log space."""
    k, n = int(k), int(n)
    if k <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0
    log_p = math.log(p)
    log_q = math.log1p(-p)

    def log_term(i: int) -> float:
        return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                + i * log_p + (n - i) * log_q)

    terms = [log_term(i) for i in range(k, n + 1)]
    m = max(terms)
    return min(1.0, math.exp(m + math.log(sum(math.exp(t - m) for t in terms))))


def chance_for(metric: str, n_clips: int, n_classes: int) -> float:
    """Chance rate: K/N for R@K, 1/n_classes for class-match accuracy."""
    if metric in CLASS_MATCH:
        return 1.0 / n_classes if n_classes else float("nan")
    k = 1 if metric.endswith("r@1") else (5 if metric.endswith("r@5") else 10)
    return k / n_clips if n_clips else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="zeroshot_results_full.json")
    args = p.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        data = json.load(f)
    checkpoints = data["checkpoints"]
    n_clips = data.get("n_clips_used") or data.get("n_clips_manifest")
    n_classes = len(next(iter(checkpoints.values())).get("per_class") or {}) or data.get("n_classes")
    logger.info("models: %d   clips: %s   classes: %s",
                len(checkpoints), n_clips, n_classes)

    metrics = [m for m in RETRIEVAL if any(m in c for c in checkpoints.values())]
    metrics += [m for m in CLASS_MATCH if any(m in c for c in checkpoints.values())]

    print()
    print("Hits vs chance (one-sided exact binomial p; * = p<0.05)")
    print("-" * 104)
    print("model".ljust(16) + "".join(m.rjust(15) for m in metrics))
    for name, c in checkpoints.items():
        cells = []
        for m in metrics:
            rate = c.get(m)
            if rate is None:
                cells.append("n/a".rjust(15))
                continue
            k = int(round(rate * n_clips)) if m in RETRIEVAL else None
            n = n_clips
            if m in CLASS_MATCH:
                # class-match hits are per clip as well: k = rate * n_clips
                k = int(round(rate * n_clips))
            chance = chance_for(m, n_clips, n_classes)
            pval = binom_tail_ge(k, n, chance)
            lo, hi = wilson_ci(k, n)
            star = "*" if pval < 0.05 else " "
            cells.append(("%.4f %d/%.1f%s" % (rate, k, chance * n, star)).rjust(15))
        print(name.ljust(16) + "".join(c.rjust(15) for c in cells))

    print()
    print("95% Wilson intervals on the same rates")
    print("-" * 104)
    print("model".ljust(16) + "".join(m.rjust(15) for m in metrics))
    for name, c in checkpoints.items():
        cells = []
        for m in metrics:
            rate = c.get(m)
            if rate is None:
                cells.append("n/a".rjust(15))
                continue
            k = int(round(rate * n_clips))
            lo, hi = wilson_ci(k, n_clips)
            cells.append(("[%.4f, %.4f]" % (lo, hi)).rjust(15))
        print(name.ljust(16) + "".join(c.rjust(15) for c in cells))

    print()
    print("Exact p-values for the same cells")
    print("-" * 104)
    print("model".ljust(16) + "".join(m.rjust(15) for m in metrics))
    for name, c in checkpoints.items():
        cells = []
        for m in metrics:
            rate = c.get(m)
            if rate is None:
                cells.append("n/a".rjust(15))
                continue
            k = int(round(rate * n_clips))
            cells.append(("p=%.1e" % binom_tail_ge(k, n_clips, chance_for(m, n_clips, n_classes))).rjust(15))
        print(name.ljust(16) + "".join(c.rjust(15) for c in cells))
    print()
    print("Chance: K/N for R@K, 1/%s for class-match. N clips = %s." % (n_classes, n_clips))
    print("With hits this sparse, differences of 1-3 clips between models are noise.")


if __name__ == "__main__":
    main()
