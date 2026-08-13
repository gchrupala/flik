#!/usr/bin/env python
"""
Validate a segment manifest by attempting to load every segment's video and audio.

Removes segments that fail to load (corrupt video, missing audio stream, bad AAC
data, etc.) and produces a clean manifest for training.

Usage:
    uv run --extra cu128 python -m scripts.validate_manifest
    uv run --extra cu128 python -m scripts.validate_manifest --input data/filtered_manifest_segments.json
    uv run --extra cu128 python -m scripts.validate_manifest --num-workers 16
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.CONSTANTS import PROJECT_ROOT
from src.utils.validation import validate_segment
from src.utils.paths import resolve_path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(PROJECT_ROOT, "logdir/validate_manifest.log")
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Validate segment manifest by loading every segment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/filtered_manifest_segments.json",
        help="Input manifest path (default: data/filtered_manifest_segments.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output manifest path (default: <input>_validated.json)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of video frames to load per segment (default: 16)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate (default: 16000)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16)",
    )
    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_validated{ext}"

    # Load manifest
    if not os.path.exists(args.input):
        logger.error(f"Input manifest not found: {args.input}")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        segments = json.load(f)

    logger.info(f"Validating {len(segments)} segments from {args.input}")
    logger.info(f"  Workers: {args.num_workers}")
    logger.info(f"  Output: {args.output}")

    # Validate in parallel
    valid_segments = []
    failed = []
    t0 = time.time()

    from tqdm.auto import tqdm

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                validate_segment, seg, args.num_frames, args.sample_rate
            ): seg
            for seg in segments
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Validating segments"
        ):
            seg = futures[future]
            seg_id, success, error = future.result()

            if success:
                valid_segments.append(seg)
            else:
                failed.append((seg_id, error))

    elapsed = time.time() - t0

    # Save validated manifest
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(valid_segments, f, indent=2, ensure_ascii=False)

    # Summary
    logger.info("=" * 60)
    logger.info(f"VALIDATION COMPLETE ({elapsed:.1f}s)")
    logger.info(f"  Total segments:   {len(segments)}")
    logger.info(f"  Valid:            {len(valid_segments)} ({100*len(valid_segments)/len(segments):.1f}%)")
    logger.info(f"  Failed:           {len(failed)} ({100*len(failed)/len(segments):.1f}%)")
    logger.info(f"  Throughput:       {len(segments)/elapsed:.1f} seg/s")
    logger.info(f"  Output:           {args.output}")

    if failed:
        logger.info(f"Failed segments (first 20):")
        for seg_id, error in failed[:20]:
            logger.info(f"  {seg_id}: {error[:80]}")
        if len(failed) > 20:
            logger.info(f"  ... and {len(failed) - 20} more")

        # Save failure log
        fail_log = args.output.replace(".json", "_failures.txt")
        with open(fail_log, "w") as f:
            for seg_id, error in failed:
                f.write(f"{seg_id}\t{error}\n")
        logger.info(f"  Failure log: {fail_log}")


if __name__ == "__main__":
    main()
