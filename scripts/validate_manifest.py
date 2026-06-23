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
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.video import video_to_tensor
from src.utils.audio import audio_to_tensor


def validate_segment(seg: dict, num_frames: int = 16, sample_rate: int = 16000) -> tuple:
    """
    Try to load a single segment's video and audio.
    Returns (segment_id, success, error_message).
    """
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
        print(f"Error: Input manifest not found: {args.input}")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        segments = json.load(f)

    print(f"Validating {len(segments)} segments from {args.input}")
    print(f"  Workers: {args.num_workers}")
    print(f"  Output: {args.output}")
    print()

    # Validate in parallel
    valid_segments = []
    failed = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                validate_segment, seg, args.num_frames, args.sample_rate
            ): seg
            for seg in segments
        }

        for i, future in enumerate(as_completed(futures)):
            seg = futures[future]
            seg_id, success, error = future.result()

            if success:
                valid_segments.append(seg)
            else:
                failed.append((seg_id, error))

            # Progress every 100 segments
            if (i + 1) % 100 == 0 or (i + 1) == len(segments):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(
                    f"  [{i+1}/{len(segments)}] "
                    f"valid={len(valid_segments)} failed={len(failed)} "
                    f"({rate:.1f} seg/s, {elapsed:.0f}s elapsed)"
                )

    elapsed = time.time() - t0

    # Save validated manifest
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(valid_segments, f, indent=2, ensure_ascii=False)

    # Summary
    print()
    print("=" * 60)
    print(f"VALIDATION COMPLETE ({elapsed:.0f}s)")
    print(f"  Total segments:   {len(segments)}")
    print(f"  Valid:            {len(valid_segments)} ({100*len(valid_segments)/len(segments):.1f}%)")
    print(f"  Failed:           {len(failed)} ({100*len(failed)/len(segments):.1f}%)")
    print(f"  Output:           {args.output}")
    print()

    if failed:
        print("Failed segments (first 20):")
        for seg_id, error in failed[:20]:
            print(f"  {seg_id}: {error[:80]}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")

        # Save failure log
        fail_log = args.output.replace(".json", "_failures.txt")
        with open(fail_log, "w") as f:
            for seg_id, error in failed:
                f.write(f"{seg_id}\t{error}\n")
        print(f"\n  Failure log: {fail_log}")


if __name__ == "__main__":
    main()
