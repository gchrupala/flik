#!/usr/bin/env python
"""
Expand a video-level batch_manifest.json into a curated segment-level manifest
by expanding ALL transcript segments (with duration filter) and validating them
with actual video+audio decode.

This bypasses the CLIP correspondence filter (stages 3-5) which kneecaps
training data by only scoring ~5 segments/video. For contrastive audio↔video
learning the audio and video come from the same file at the same timestamp
— inherently aligned — so the CLIP text↔video filter is unnecessary.

Usage:
    uv run --extra cu128 python -m scripts.expand_and_validate_manifest
    uv run --extra cu128 python -m scripts.expand_and_validate_manifest --input data/batch_manifest.json
    uv run --extra cu128 python -m scripts.expand_and_validate_manifest --skip-validate
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
# Fallback validate_segment (imported by preference)
# ---------------------------------------------------------------------------

def _fallback_validate_segment(seg: dict, num_frames: int = 16, sample_rate: int = 16000) -> tuple:
    """
    Try to load a single segment's video and audio.
    Copied verbatim from scripts/validate_manifest.py (lines 43-69).

    Returns (segment_id, success, error_message).
    """
    from src.utils.video import video_to_tensor
    from src.utils.audio import audio_to_tensor

    seg_id = seg.get("id", f"{seg.get('video_path', '?')}_{seg['start_sec']:.1f}")
    try:
        video = video_to_tensor(
            seg["video_path"],
            seg["start_sec"],
            seg["end_sec"],
            num_frames=num_frames,
        )
        audio = audio_to_tensor(
            seg["video_path"],
            seg["start_sec"],
            seg["end_sec"],
            sample_rate=sample_rate,
        )
        # Basic sanity checks
        if video.shape[0] != num_frames:
            return (seg_id, False, f"Wrong frame count: {video.shape[0]}")
        if audio.shape[1] < sample_rate:  # less than 1 second
            return (seg_id, False, f"Audio too short: {audio.shape[1]} samples")
        return (seg_id, True, None)
    except Exception as e:
        return (seg_id, False, str(e))


def _get_validate_segment():
    """Return the validate_segment function (imported or fallback).

    Tries to import from the sibling ``scripts/validate_manifest.py`` at
    call time (lazy import) so that ``--skip-validate`` runs without
    triggering the torchvision import chain. Falls back to a local copy
    if the import fails for ANY reason (not just ImportError).
    """
    try:
        from scripts.validate_manifest import validate_segment
        return validate_segment
    except Exception:
        return _fallback_validate_segment


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------

def expand_segments(
    manifest_path: str,
    min_duration: float,
    max_duration: float,
    logger: logging.Logger,
) -> list:
    """
    Expand a video-level manifest into a segment-level manifest.

    For each video in the input manifest:
      - Load the transcript JSON from item["json_path"]
      - For each transcript segment, apply the duration filter
      - Build segment dicts matching the train dataset format

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
        json_path = item.get("json_path", "")

        if not json_path or not os.path.exists(json_path):
            logger.warning(f"Missing transcript JSON for {video_id}: {json_path}")
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

        for seg in transcript:
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            dur = end - start

            # Duration filter — mirrors VideoAudioDataset._build_segment_list()
            if dur < min_duration or dur > max_duration:
                continue

            segment_id = f"{video_id}_{start:.1f}"
            all_segments.append({
                "id": segment_id,
                "video_id": video_id,
                "video_path": item["video_path"],
                "json_path": json_path,
                "start_sec": start,
                "end_sec": end,
                "text": seg.get("text", ""),
            })
            n_kept += 1

        total_kept += n_kept
        logger.info(
            f"Expanded {video_id}: {n_kept}/{n_total} segments pass duration filter"
        )

    logger.info(
        f"Expansion complete: {total_kept}/{total_transcript_segments} "
        f"segments kept ({100*total_kept/total_transcript_segments:.1f}%)"
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
    validate_fn = _get_validate_segment()
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
            executor.submit(validate_fn, seg, num_frames, sample_rate): seg
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
):
    """Log a funnel summary table."""
    logger.info("=" * 60)
    logger.info("EXPANSION + VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Input videos:              {n_videos:>8d}")
    logger.info(f"  Total transcript segments: {n_transcript:>8d}")
    if n_transcript > 0:
        logger.info(f"  After duration filter:     {n_dur:>8d}  ({100*n_dur/n_transcript:.1f}%)")
    else:
        logger.info(f"  After duration filter:     {n_dur:>8d}  (N/A%)")
    if n_dur > 0:
        logger.info(f"  After validation:          {n_valid:>8d}  ({100*n_valid/n_dur:.1f}%)")
        if n_failed:
            logger.info(f"  Failed:                    {n_failed:>8d}  ({100*n_failed/n_dur:.1f}%)")
    else:
        logger.info(f"  After validation:          {n_valid:>8d}  (N/A%)")
    logger.info(f"  Output:                    {output_path}")
    if elapsed > 0:
        logger.info(f"  Total time:                {elapsed:.1f}s")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Expand video-level manifest to validated segment-level manifest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run --extra cu128 python -m scripts.expand_and_validate_manifest
  uv run --extra cu128 python -m scripts.expand_and_validate_manifest --input data/batch_manifest.json
  uv run --extra cu128 python -m scripts.expand_and_validate_manifest --skip-validate
  uv run --extra cu128 python -m scripts.expand_and_validate_manifest --min-duration 2.0 --max-duration 15.0 --num-workers 32
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
        default="data/expanded_manifest_segments_validated.json",
        help="Output segment-level manifest (default: data/expanded_manifest_segments_validated.json)",
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
        "--skip-validate",
        action="store_true",
        help="Skip the decode validation step (just expand + write)",
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
    logger.info(f"Validate:            {'SKIP' if args.skip_validate else 'YES'}")
    if not args.skip_validate:
        logger.info(f"  num_frames:        {args.num_frames}")
        logger.info(f"  sample_rate:       {args.sample_rate}")
        logger.info(f"  num_workers:       {args.num_workers}")

    t_start = time.time()

    # -- Step 1: Expand --
    input_path = os.path.join(PROJECT_ROOT, args.input)
    output_path = os.path.join(PROJECT_ROOT, args.output)

    segments, n_videos, n_transcript, n_dur = expand_segments(
        input_path, args.min_duration, args.max_duration, logger
    )

    if not segments:
        logger.warning("No segments passed the duration filter — nothing to output.")
        print_summary(logger, n_videos, n_transcript, 0, 0, 0, output_path)
        sys.exit(0)

    # -- Step 2: Validate (optional) --
    if args.skip_validate:
        valid_segments = segments
        n_failed = 0
        n_val_total = 0
        val_elapsed = 0.0
        logger.info(f"Skipping validation — writing {len(segments)} segments directly")
    else:
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
    )


if __name__ == "__main__":
    main()
