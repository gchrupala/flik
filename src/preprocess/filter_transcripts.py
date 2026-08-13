import glob
import json
import logging
import os

from tqdm.auto import tqdm

from src.CONSTANTS import (
    ENGLISH_LANGUAGE_CODE,
    ENGLISH_LANGUAGE_PREFIX,
    OUTPUT_MANIFEST,
    PROJECT_ROOT,
    TRANSCRIPT_ROOT,
    VIDEO_ROOT,
    VIDEO_EXTENSIONS,
)
from src.utils.paths import to_relative

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logdir/transcript_filter.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def _build_video_index() -> dict:
    """One-time index of every video file under VIDEO_ROOT, keyed by basename.

    Walk VIDEO_ROOT once (instead of re-walking a directory for every SRT),
    turning the per-SRT O(N*M) lookup into an O(N+M) dict lookup.
    """
    index: dict = {}
    for root, _, files in os.walk(VIDEO_ROOT):
        for fname in files:
            if fname.endswith(VIDEO_EXTENSIONS):
                base = os.path.splitext(fname)[0]
                index.setdefault(base, []).append(os.path.join(root, fname))
    return index


def _find_video(base_name: str, json_path: str, video_index: dict):
    """Return the video path for a transcript, or None if not found.

    Prefers the video in the subdir mapped from the transcript's location
    (mirrors the old per-SRT tree-walk behavior); falls back to any match when
    the mapped dir has no hit.
    """
    candidates = video_index.get(base_name, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    video_dir = os.path.dirname(json_path.replace(TRANSCRIPT_ROOT, VIDEO_ROOT))
    for vp in candidates:
        if os.path.dirname(vp) == video_dir:
            return vp
    return candidates[0]


def check_transcript_language(srt_path: str) -> bool:
    """Checks the language of the transcript using the _language.txt file.
    The file is expected to be in the same directory as the .srt transcript
    with the same base name.
    Args:
        srt_path (str): Path to the transcript srt file.
    Returns:
        bool: True if the language is English, False otherwise.
    """

    language_txt = srt_path.replace(".srt", "_language.txt")
    if not os.path.isfile(language_txt):
        return False
    try:
        with open(language_txt, "r", encoding="utf-8") as f:
            lang = f.read().strip().lower()
            return lang == ENGLISH_LANGUAGE_CODE or lang.startswith(
                ENGLISH_LANGUAGE_PREFIX
            )
    except Exception as e:
        logger.warning(f"Could not read language file {language_txt}: {e}")
        return False


def main():
    valid_pairs = []
    skipped_reason = {
        "non_english": 0,
        "missing_json": 0,
        "video_not_found": 0,
    }

    # Get all SRT files with glob
    logger.info("Scanning for SRT files...")
    all_srt_files = glob.glob(
        os.path.join(TRANSCRIPT_ROOT, "**", "*.srt"), recursive=True
    )

    logger.info(f"Found {len(all_srt_files)} SRT files. Processing...")

    # Pre-index all videos once (avoids re-walking the video tree per SRT).
    video_index = _build_video_index()
    logger.info(f"Indexed {sum(len(v) for v in video_index.values())} video files.")

    for srt_file in tqdm(all_srt_files, desc="Processing SRTs"):
        # First check the Whisperx language ID output to see if the transcript is English
        if not check_transcript_language(srt_file):
            logger.warning(
                f"Skipping {srt_file}: Non-English content detected via language file."
            )
            skipped_reason["non_english"] += 1
            continue

        base_name = os.path.splitext(os.path.basename(srt_file))[0]
        json_path = srt_file.replace(".srt", ".json")

        if os.path.isfile(json_path):
            video_path = _find_video(base_name, json_path, video_index)

            if video_path is None:
                logger.warning(f"Skipping {base_name}: Video not found.")
                skipped_reason["video_not_found"] += 1
                continue

            # Store paths relative to PROJECT_ROOT so the manifest stays valid
            # across machines.
            valid_pairs.append(
                {
                    "id": base_name,
                    "video_path": to_relative(video_path),
                    "json_path": to_relative(json_path),  # We pass the JSON to the next step
                    "srt_path": to_relative(srt_file),  # Kept for reference
                }
            )
        else:
            logger.warning(f"Skipping {base_name}: Found SRT but missing JSON.")
            skipped_reason["missing_json"] += 1

    logger.info("\nScan complete.")
    logger.info(f"Valid English pairs: {len(valid_pairs)}")
    logger.info(f"Skipped/Non-English: {skipped_reason['non_english']}")
    logger.info(f"Skipped/Missing JSON: {skipped_reason['missing_json']}")
    logger.info(f"Skipped/Video Not Found: {skipped_reason['video_not_found']}")

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(valid_pairs, f, indent=2)

    logger.info(f"Manifest saved to {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
