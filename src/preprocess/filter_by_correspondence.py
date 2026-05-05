#!/usr/bin/env python3
"""
Filter video‑transcript pairs based on CLIP correspondence scores.
Uses normalized scores (percentile relative to random baseline) and optional segment‑level filtering.
"""

import argparse
import json
import logging
import os
import sys

from src.CONSTANTS import (
    PROJECT_ROOT,
    OUTPUT_MANIFEST,
    ALIGNMENT_SCORES_FILE,
    VIDEO_PERCENTILE_THRESHOLD,
    SEGMENT_PERCENTILE_THRESHOLD,
    USE_SEGMENT_FILTER,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                os.path.join(PROJECT_ROOT, "logdir/filter_correspondence.log")
            ),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_segment_manifest(
    scores_file, output_path, segment_percentile_threshold, logger
):
    """Build a segment-level manifest from alignment score output."""
    segment_scores_path = scores_file.replace(".json", "_segments.json")
    if not os.path.exists(segment_scores_path):
        logger.warning(
            f"Segment score file not found at {segment_scores_path}; skipping segment manifest"
        )
        return

    segment_scores = load_json(segment_scores_path)
    filtered_segments = []
    for seg in segment_scores:
        percentile = seg.get("clip_score_percentile")
        if percentile is None:
            continue
        if percentile < segment_percentile_threshold:
            continue
        filtered_segments.append(
            {
                "id": seg["id"],
                "video_id": seg.get("video_id"),
                "video_path": seg["video_path"],
                "json_path": seg.get("json_path", ""),
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "text": seg.get("text", ""),
                "clip_score_percentile": percentile,
            }
        )

    segment_output = output_path.replace(".json", "_segments.json")
    save_json(filtered_segments, segment_output)
    logger.info(f"Segment manifest saved to {segment_output}")
    logger.info(f"Segment manifest size: {len(filtered_segments)}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter videos by CLIP correspondence scores"
    )
    parser.add_argument(
        "--scores",
        type=str,
        default=ALIGNMENT_SCORES_FILE,
        help="Path to alignment scores JSON (default: data/alignment_scores.json)",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=OUTPUT_MANIFEST,
        help="Path to original manifest JSON (default: data/batch_manifest.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data/filtered_manifest.json"),
        help="Output filtered manifest path (default: data/filtered_manifest.json)",
    )
    parser.add_argument(
        "--percentile-threshold",
        type=float,
        default=VIDEO_PERCENTILE_THRESHOLD,
        help=f"Minimum percentile for video‑level score (default: {VIDEO_PERCENTILE_THRESHOLD})",
    )
    parser.add_argument(
        "--segment-fraction-threshold",
        type=float,
        default=0.5,
        help="Minimum fraction of segments that must pass segment‑level threshold (default: 0.5)",
    )
    parser.add_argument(
        "--use-segment-filter",
        action="store_true",
        default=USE_SEGMENT_FILTER,
        help="Enable segment‑level filtering (default: from CONSTANTS)",
    )
    parser.add_argument(
        "--segment-percentile-threshold",
        type=float,
        default=SEGMENT_PERCENTILE_THRESHOLD,
        help=f"Minimum percentile for segment‑level pass (default: {SEGMENT_PERCENTILE_THRESHOLD})",
    )
    parser.add_argument(
        "--build-segment-manifest",
        action="store_true",
        help="Also write a segment-level manifest based on segment percentile threshold",
    )
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("Starting correspondence filtering")
    logger.info(f"Score file: {args.scores}")
    logger.info(f"Manifest file: {args.manifest}")
    logger.info(f"Percentile threshold: {args.percentile_threshold}")
    logger.info(f"Use segment filter: {args.use_segment_filter}")
    if args.use_segment_filter:
        logger.info(
            f"Segment percentile threshold: {args.segment_percentile_threshold}"
        )
        logger.info(f"Segment fraction threshold: {args.segment_fraction_threshold}")

    # Load data
    try:
        scores = load_json(args.scores)
    except Exception as e:
        logger.error(f"Failed to load scores file: {e}")
        sys.exit(1)

    try:
        manifest = load_json(args.manifest)
    except Exception as e:
        logger.error(f"Failed to load manifest file: {e}")
        sys.exit(1)

    # Index manifest by video ID (assuming 'id' field matches scores['id'])
    manifest_by_id = {item["id"]: item for item in manifest}

    # Filtering
    passed_ids = []
    failed_ids = []
    failed_reasons = {"low_percentile": 0, "low_segment_fraction": 0, "missing_data": 0}

    for item in scores:
        vid = item["id"]
        if vid not in manifest_by_id:
            logger.warning(f"Video ID {vid} in scores not found in manifest, skipping")
            failed_reasons["missing_data"] += 1
            continue

        # Check percentile threshold
        percentile = item.get("avg_clip_score_percentile")
        if percentile is None:
            logger.warning(f"Video {vid} has no percentile score, skipping")
            failed_reasons["missing_data"] += 1
            continue
        if percentile < args.percentile_threshold:
            failed_ids.append((vid, "low_percentile"))
            failed_reasons["low_percentile"] += 1
            continue

        # Check segment‑level filter if enabled
        if args.use_segment_filter:
            segment_passed = item.get("segment_passed")
            if segment_passed is None:
                logger.warning(f"Video {vid} missing segment_passed field, skipping")
                failed_reasons["missing_data"] += 1
                continue
            fraction_passed = sum(segment_passed) / len(segment_passed)
            if fraction_passed < args.segment_fraction_threshold:
                failed_ids.append((vid, "low_segment_fraction"))
                failed_reasons["low_segment_fraction"] += 1
                continue

        # All checks passed
        passed_ids.append(vid)

    # Build filtered manifest
    filtered = [manifest_by_id[vid] for vid in passed_ids]

    # Save results
    save_json(filtered, args.output)
    logger.info(f"Filtered manifest saved to {args.output}")
    logger.info(f"Total videos processed: {len(scores)}")
    logger.info(f"Passed: {len(passed_ids)}")
    logger.info(f"Failed: {len(failed_ids)}")
    for reason, count in failed_reasons.items():
        if count > 0:
            logger.info(f"  {reason}: {count}")

    # Optionally write failed list
    failed_output = args.output.replace(".json", "_failed.json")
    save_json(
        [{"id": vid, "reason": reason} for vid, reason in failed_ids], failed_output
    )
    logger.info(f"Failed list saved to {failed_output}")

    # Summary statistics
    if passed_ids:
        avg_percentile = sum(
            item["avg_clip_score_percentile"]
            for item in scores
            if item["id"] in passed_ids
            and item.get("avg_clip_score_percentile") is not None
        ) / len(passed_ids)
        logger.info(f"Average percentile of kept videos: {avg_percentile:.1f}")

    if args.build_segment_manifest:
        build_segment_manifest(
            scores_file=args.scores,
            output_path=args.output,
            segment_percentile_threshold=args.segment_percentile_threshold,
            logger=logger,
        )


if __name__ == "__main__":
    main()
