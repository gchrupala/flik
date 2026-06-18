#!/usr/bin/env python
"""
Preprocessing orchestrator — runs the CLIP correspondence + filtering pipeline
(stages 3-5) that produces data/filtered_manifest_segments.json for training.

Prerequisites (already run on the cluster):
    Stage 1: uv run --extra cu128 python -m src.preprocess.transcribe
    Stage 2: uv run --extra cu128 python -m src.preprocess.filter_transcripts

This script orchestrates:
    Stage 3 (randomized): random CLIP baseline → null distribution for normalization
    Stage 4 (alignment):  main CLIP scoring on real video-text pairs → z-scores, percentiles
    Stage 5 (filter):     threshold filtering → filtered_manifest_segments.json

Usage:
    uv run --extra cu128 python -m scripts.preprocess                     # run all 3 stages
    uv run --extra cu128 python -m scripts.preprocess --from-stage alignment   # resume at stage 4
    uv run --extra cu128 python -m scripts.preprocess --to-stage alignment     # stop after stage 4
    uv run --extra cu128 python -m scripts.preprocess --device cpu             # CPU override
    uv run --extra cu128 python -m scripts.preprocess --dry-run                # show plan, don't execute
"""

import argparse
import os
import subprocess
import sys
import time

# Resolve project root (works whether launched as module or script)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.CONSTANTS import (
    OUTPUT_MANIFEST,
    ALIGNMENT_SCORES_FILE,
    PROJECT_ROOT as CONSTANTS_ROOT,
)


# ---------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(CONSTANTS_ROOT, "data")

STAGES = [
    {
        "name": "randomized",
        "title": "Stage 3: Randomized CLIP baseline (null distribution)",
        "module": "src.preprocess.check_correspondance",
        "extra_args": lambda cfg: (
            ["--mode", "randomized"]
            + (["--device", cfg.device] if cfg.device else [])
        ),
        "inputs": [OUTPUT_MANIFEST],
        "outputs": [ALIGNMENT_SCORES_FILE.replace(".json", "_randomized.json")],
        "gpu": True,
    },
    {
        "name": "alignment",
        "title": "Stage 4: Main alignment scoring (z-scores, percentiles)",
        "module": "src.preprocess.check_correspondance",
        "extra_args": lambda cfg: (
            ["--mode", "main"]
            + (["--device", cfg.device] if cfg.device else [])
        ),
        "inputs": [
            OUTPUT_MANIFEST,
            ALIGNMENT_SCORES_FILE.replace(".json", "_randomized.json"),
        ],
        "outputs": [
            ALIGNMENT_SCORES_FILE,
            ALIGNMENT_SCORES_FILE.replace(".json", "_segments.json"),
        ],
        "gpu": True,
    },
    {
        "name": "filter",
        "title": "Stage 5: Filter by correspondence + build segment manifest",
        "module": "src.preprocess.filter_by_correspondence",
        "extra_args": lambda cfg: [
            "--build-segment-manifest",
            "--percentile-threshold", str(cfg.percentile_threshold),
            "--segment-percentile-threshold", str(cfg.segment_percentile_threshold),
        ] + (["--use-segment-filter"] if cfg.use_segment_filter else []),
        "inputs": [
            ALIGNMENT_SCORES_FILE,
            ALIGNMENT_SCORES_FILE.replace(".json", "_segments.json"),
            OUTPUT_MANIFEST,
        ],
        "outputs": [
            os.path.join(DATA_DIR, "filtered_manifest.json"),
            os.path.join(DATA_DIR, "filtered_manifest_segments.json"),
        ],
        "gpu": False,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stage_index(name: str) -> int:
    for i, s in enumerate(STAGES):
        if s["name"] == name:
            return i
    raise ValueError(f"Unknown stage '{name}'. Valid: {[s['name'] for s in STAGES]}")


def check_inputs(stage: dict) -> list:
    """Return list of missing input files (empty if all present)."""
    return [f for f in stage["inputs"] if not os.path.exists(f)]


def check_outputs(stage: dict) -> list:
    """Return list of expected output files that were NOT produced."""
    return [f for f in stage["outputs"] if not os.path.exists(f)]


def run_stage(stage: dict, cfg) -> bool:
    """Execute a single stage via subprocess. Returns True on success."""
    print()
    print("=" * 72)
    print(f"  {stage['title']}")
    print("=" * 72)

    # --- Pre-flight: input checks ---
    missing = check_inputs(stage)
    if missing:
        print(f"  ERROR: Missing required inputs for '{stage['name']}':")
        for f in missing:
            print(f"    - {f}")
        if stage["name"] == "alignment" and any("randomized" in f for f in missing):
            print("\n  Hint: Run stage 'randomized' first to generate the baseline.")
        return False

    print(f"  Module: {stage['module']}")
    print(f"  Inputs:")
    for f in stage["inputs"]:
        print(f"    - {f}  ({'OK' if os.path.exists(f) else 'MISSING'})")
    print(f"  Expected outputs:")
    for f in stage["outputs"]:
        print(f"    - {f}")

    # --- Build command ---
    cmd = [sys.executable, "-m", stage["module"]] + stage["extra_args"](cfg)

    if cfg.dry_run:
        print(f"\n  [DRY-RUN] Would execute:")
        print(f"    {' '.join(cmd)}")
        return True

    print(f"\n  Running: {' '.join(cmd)}")
    print("-" * 72)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=CONSTANTS_ROOT)
    elapsed = time.time() - t0

    print("-" * 72)
    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        return False

    # --- Post-flight: output verification ---
    missing_out = check_outputs(stage)
    if missing_out:
        print(f"  WARNING: Stage exited 0 but expected outputs missing:")
        for f in missing_out:
            print(f"    - {f}")
        return False

    print(f"  OK ({elapsed:.1f}s) — produced {len(stage['outputs'])} file(s)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate CLIP correspondence scoring + filtering (stages 3-5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Prerequisites: stages 1 (transcribe) and 2 (filter_transcripts) must already be run.",
    )
    parser.add_argument(
        "--from-stage",
        type=str,
        default=STAGES[0]["name"],
        choices=[s["name"] for s in STAGES],
        help=f"Stage to start from (default: {STAGES[0]['name']})",
    )
    parser.add_argument(
        "--to-stage",
        type=str,
        default=STAGES[-1]["name"],
        choices=[s["name"] for s in STAGES],
        help=f"Stage to end at, inclusive (default: {STAGES[-1]['name']})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device override for GPU stages (default: auto-detect via torch.accelerator)",
    )
    parser.add_argument(
        "--percentile-threshold",
        type=float,
        default=25.0,
        help="Video-level minimum percentile for filter stage (default: 25)",
    )
    parser.add_argument(
        "--segment-percentile-threshold",
        type=float,
        default=25.0,
        help="Segment-level minimum percentile for filter stage (default: 25)",
    )
    parser.add_argument(
        "--use-segment-filter",
        action="store_true",
        help="Enable segment-fraction filtering in addition to segment-manifest building",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the execution plan without running anything",
    )
    args = parser.parse_args()

    start_i = stage_index(args.from_stage)
    end_i = stage_index(args.to_stage)
    if start_i > end_i:
        parser.error(f"--from-stage ({args.from_stage}) must come before --to-stage ({args.to_stage})")

    selected = STAGES[start_i : end_i + 1]

    # --- Header ---
    print()
    print("#" * 72)
    print("#  FLIK PREPROCESSING ORCHESTRATOR (stages 3-5)")
    print("#" * 72)
    print(f"  Project root: {CONSTANTS_ROOT}")
    print(f"  Stages to run: {[s['name'] for s in selected]}")
    if args.device:
        print(f"  Device override: {args.device}")
    if args.dry_run:
        print(f"  Mode: DRY-RUN (no execution)")
    print(f"  Filter thresholds: percentile={args.percentile_threshold}, "
          f"segment={args.segment_percentile_threshold}, "
          f"use_segment_filter={args.use_segment_filter}")

    # --- Sanity: prerequisite (batch_manifest.json from stage 2) ---
    if start_i == 0 and not os.path.exists(OUTPUT_MANIFEST):
        print(f"\nERROR: {OUTPUT_MANIFEST} not found.")
        print("  Stage 2 (filter_transcripts) must be run first to generate it.")
        print("  Run: uv run --extra cu128 python -m src.preprocess.filter_transcripts")
        sys.exit(1)

    # --- Execute stages ---
    all_ok = True
    for stage in selected:
        ok = run_stage(stage, args)
        if not ok and not args.dry_run:
            all_ok = False
            print(f"\nAborting: stage '{stage['name']}' failed.")
            break

    # --- Summary ---
    print()
    print("#" * 72)
    if args.dry_run:
        print("#  DRY-RUN COMPLETE — no files were modified")
    elif all_ok:
        print("#  ALL STAGES COMPLETE")
        if end_i == len(STAGES) - 1:
            target = os.path.join(DATA_DIR, "filtered_manifest_segments.json")
            n = 0
            try:
                import json
                with open(target) as f:
                    n = len(json.load(f))
            except Exception:
                pass
            print(f"#  Target file: {target}")
            print(f"#  Segments in manifest: {n}")
            print(f"#  Ready for training: uv run --extra cu128 python -m scripts.train_hydra")
    else:
        print("#  PIPELINE FAILED — see errors above")
    print("#" * 72)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
