#!/usr/bin/env python
"""
Expand a video-level batch_manifest.json into a segment-level manifest
by expanding ALL transcript segments (with duration filter). Optionally
validates segments with actual video+audio decode (use --validate).

Validation is OFF by default — the training dataset retries corrupt segments
at runtime (3 attempts with random fallback). The CLIP correspondence stages
(3-4) already decode video frames, so fully-corrupt videos are caught there.

This bypasses the CLIP correspondence filter (stages 3-5) which kneecaps
training data by only scoring ~5 segments/video. For contrastive audio↔video
learning the audio and video come from the same file at the same timestamp
— inherently aligned — so the CLIP text↔video filter is unnecessary.

Usage:
    uv run --extra cu128 python -m scripts.expand_and_validate_manifest
    uv run --extra cu128 python -m scripts.expand_and_validate_manifest --input data/filtered_manifest.json
    uv run --extra cu128 python -m scripts.expand_and_validate_manifest --validate
    uv run --extra cu128 python -m scripts.expand_and_validate_manifest --min-duration 2.0 --max-duration 15.0
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root and scripts parent to path so src.* and scripts.* are importable
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from src.CONSTANTS import PROJECT_ROOT
from src.utils.paths import resolve_path
from src.utils.segments import merge_segments
from src.utils.validation import validate_segment


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    """Configure logging with FileHandler + StreamHandler (mirrors validate_manifest.py)."""
    log_dir = os.path.join(PROJECT_ROOT, "logdir")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("expand_and_validate_manifest")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers on repeated calls
    if not logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh = logging.FileHandler(
            os.path.join(log_dir, "expand_and_validate_manifest.log")
        )
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------

def expand_segments(
    manifest_path: str,
    min_duration: float,
    max_duration: float,
    max_gap: float,
    logger: logging.Logger,
) -> list:
    """
    Expand a video-level manifest into a segment-level manifest.

    For each video in the input manifest:
      - Load the transcript JSON from item["json_path"]
      - Merge consecutive transcript segments into [min, max] windows
        (breaks on gaps > max_gap and on max-length overflow), which rescues
        the sub-minimum utterance fragments Whisper produces in dialogue
        - Build segment dicts matching the train dataset format, tagging each
        window with its constituent segment count (n_segments)

    Returns (segments, n_videos, n_transcript_segments, n_kept).
    """
    # Load video-level manifest
    if not os.path.exists(manifest_path):
        logger.error(f"Input manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        videos = json.load(f)

    logger.info(f"Loaded {len(videos)} videos from {manifest_path}")

    all_segments = []
    total_transcript_segments = 0
    total_kept = 0

    for item in videos:
        video_id = item.get("id", "unknown")
        json_path_stored = item.get("json_path", "")
        json_path = resolve_path(json_path_stored)

        if not json_path or not os.path.exists(json_path):
            logger.warning(f"Missing transcript JSON for {video_id}: {json_path_stored}")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                transcript = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load transcript for {video_id} ({json_path}): {e}")
            continue

        n_total = len(transcript)
        total_transcript_segments += n_total
        n_kept = 0

        windows = merge_segments(
            transcript,
            min_duration=min_duration,
            max_duration=max_duration,
            max_gap=max_gap,
        )

        for w in windows:
            # Include end time + finer precision to avoid 0.1s-rounding collisions
            # between windows that start within ~0.05s of each other.
            segment_id = f"{video_id}_{w['start_sec']:.2f}_{w['end_sec']:.2f}"
            all_segments.append({
                "id": segment_id,
                "video_id": video_id,
                "video_path": item["video_path"],  # kept as stored (relative) for portability
                "json_path": json_path_stored,
                "start_sec": w["start_sec"],
                "end_sec": w["end_sec"],
                "text": w["text"],
                "n_segments": w["n_segments"],
            })
            n_kept += 1

        total_kept += n_kept
        logger.info(
            f"Expanded {video_id}: {n_kept} windows from {n_total} transcript "
            f"segments (merged with gap threshold {max_gap}s)"
        )

    logger.info(
        f"Expansion complete: {total_kept} training windows from "
        f"{total_transcript_segments} transcript segments "
        f"({100*total_kept/total_transcript_segments:.1f}%)"
    )

    return all_segments, len(videos), total_transcript_segments, total_kept


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_segments(
    segments: list,
    num_frames: int,
    sample_rate: int,
    num_workers: int,
    logger: logging.Logger,
) -> tuple:
    """
    Validate every segment by actually decoding video+audio.

    Returns (valid_segments, failed_list).
    """
    valid_segments = []
    failed = []

    from tqdm.auto import tqdm

    logger.info(
        f"Validating {len(segments)} segments "
        f"(workers={num_workers}, num_frames={num_frames}, sample_rate={sample_rate})"
    )

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(validate_segment, seg, num_frames, sample_rate): seg
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
    logger.info(f"Validation complete ({elapsed:.1f}s)")

    return valid_segments, failed, elapsed, len(segments)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(
    logger: logging.Logger,
    n_videos: int,
    n_transcript: int,
    n_dur: int,
    n_valid: int,
    n_failed: int,
    output_path: str,
    elapsed: float = 0.0,
    validated: bool = False,
):
    """Log a funnel summary table."""
    logger.info("=" * 60)
    logger.info("EXPANSION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Input videos:              {n_videos:>8d}")
    logger.info(f"  Total transcript segments: {n_transcript:>8d}")
    if n_transcript > 0:
        logger.info(f"  After duration filter:     {n_dur:>8d}  ({100*n_dur/n_transcript:.1f}%)")
    else:
        logger.info(f"  After duration filter:     {n_dur:>8d}  (N/A%)")
    if validated:
        if n_dur > 0:
            logger.info(f"  After validation:          {n_valid:>8d}  ({100*n_valid/n_dur:.1f}%)")
            if n_failed:
                logger.info(f"  Failed:                    {n_failed:>8d}  ({100*n_failed/n_dur:.1f}%)")
        else:
            logger.info(f"  After validation:          {n_valid:>8d}  (N/A%)")
    else:
        logger.info(f"  Validation:                SKIPPED (use --validate to enable)")
    logger.info(f"  Output:                    {output_path}")
    if elapsed > 0:
        logger.info(f"  Total time:                {elapsed:.1f}s")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Expand video-level manifest to segment-level manifest (validation opt-in via --validate)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run --extra cu128 python -m scripts.expand_and_validate_manifest
  uv run --extra cu128 python -m scripts.expand_and_validate_manifest --input data/filtered_manifest.json
  uv run --extra cu128 python -m scripts.expand_and_validate_manifest --validate
  uv run --extra cu128 python -m scripts.expand_and_validate_manifest --min-duration 2.0 --max-duration 15.0
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/batch_manifest.json",
        help="Video-level input manifest (default: data/batch_manifest.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/expanded_manifest_segments.json",
        help="Output segment-level manifest (default: data/expanded_manifest_segments.json)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=3.0,
        help="Minimum segment duration in seconds (default: 3.0)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=10.0,
        help="Maximum segment duration in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=1.0,
        help="Max allowed gap (s) between merged consecutive segments; larger "
             "gaps break the window (scene change / music / silence). "
             "(default: 1.0)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=16,
        help="Number of video frames per segment (default: 16)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate in Hz (default: 16000)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="Number of parallel validation workers (default: 16)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run decode validation on each segment (slow). Off by default — "
        "the dataset's runtime retry logic (3 attempts with random fallback) "
        "handles corrupt segments during training.",
    )
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("EXPAND AND VALIDATE MANIFEST")
    logger.info("=" * 60)
    logger.info(f"Input manifest:      {args.input}")
    logger.info(f"Output manifest:     {args.output}")
    logger.info(f"Duration filter:     {args.min_duration}s – {args.max_duration}s")
    logger.info(f"Validate:            {'YES' if args.validate else 'NO'}")
    if args.validate:
        logger.info(f"  num_frames:        {args.num_frames}")
        logger.info(f"  sample_rate:       {args.sample_rate}")
        logger.info(f"  num_workers:       {args.num_workers}")

    t_start = time.time()

    # -- Step 1: Expand --
    input_path = os.path.join(PROJECT_ROOT, args.input)
    output_path = os.path.join(PROJECT_ROOT, args.output)

    segments, n_videos, n_transcript, n_dur = expand_segments(
        input_path, args.min_duration, args.max_duration, args.max_gap, logger
    )

    if not segments:
        logger.warning("No segments passed the duration filter — nothing to output.")
        print_summary(logger, n_videos, n_transcript, 0, 0, 0, output_path, validated=False)
        sys.exit(0)

    # -- Step 2: Validate (optional) --
    validated = args.validate
    if validated:
        valid_segments, failed, val_elapsed, n_val_total = validate_segments(
            segments, args.num_frames, args.sample_rate, args.num_workers, logger
        )
        n_failed = len(failed)

        # Write failures
        if failed:
            fail_log = output_path.replace(".json", "_failures.txt")
            with open(fail_log, "w", encoding="utf-8") as f:
                for seg_id, error in failed:
                    f.write(f"{seg_id}\t{error}\n")
            logger.info(f"Failures written to {fail_log} ({n_failed} segments)")
    else:
        valid_segments = segments
        n_failed = 0
        val_elapsed = 0.0
        logger.info(f"Validation skipped — writing {len(segments)} segments directly")

    # -- Step 3: Write output --
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(valid_segments, f, indent=2, ensure_ascii=False)

    total_elapsed = time.time() - t_start

    # -- Step 4: Summary --
    print_summary(
        logger,
        n_videos,
        n_transcript,
        n_dur,
        len(valid_segments),
        n_failed,
        output_path,
        total_elapsed,
        validated=validated,
    )


if __name__ == "__main__":
    main()
