import glob
import json
import logging
import os
import re

from langdetect import LangDetectException, detect, detect_langs
from tqdm.auto import tqdm

from CONSTANTS import (
    CHAR_SAMPLE_SIZE,
    ENGLISH_LANGUAGE_CODE,
    ENGLISH_LANGUAGE_PREFIX,
    MIN_TEXT_LENGTH_FOR_DETECTION,
    OUTPUT_MANIFEST,
    PROJECT_ROOT,
    TRANSCRIPT_ROOT,
    VIDEO_ROOT,
    VIDEO_EXTENSIONS,
)

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


def parse_srt_to_text(srt_path):
    """
    Parses an SRT file, strips indices and timestamps,
    and returns a clean block of text.
    """
    clean_lines = []

    try:
        with open(srt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip numeric indices (e.g., "1", "145")
            if line.isdigit():
                continue

            # Skip timestamp lines (e.g., "00:00:12,000 --> 00:00:15,000")
            if "-->" in line:
                continue

            # Remove HTML tags if present (e.g., <i>text</i>)
            line = re.sub(r"<[^>]+>", "", line)

            clean_lines.append(line)

    except Exception as e:
        print(f"Error parsing SRT {srt_path}: {e}")
        return ""

    # Join them into a single string
    return " ".join(clean_lines)


def is_english_content(text):
    """
    Uses langdetect to verify text is English.
    """
    if len(text) < MIN_TEXT_LENGTH_FOR_DETECTION:
        return False  # Too short to decide

    try:
        # We take a slice of the text to speed up processing
        # Using the middle of the text is often safer than the start (intros/music)
        mid_point = len(text) // 2
        start = max(0, mid_point - (CHAR_SAMPLE_SIZE // 2))
        end = min(len(text), mid_point + (CHAR_SAMPLE_SIZE // 2))
        sample = text[start:end]

        # Detect
        lang = detect(sample)
        return lang == ENGLISH_LANGUAGE_CODE

    except LangDetectException:
        return False


def find_video_file(base_name, search_root):
    extensions = list(VIDEO_EXTENSIONS)
    for ext in extensions:
        # Direct check
        path = os.path.join(search_root, base_name + ext)
        if os.path.exists(path) and os.path.isfile(path):
            return path

    # Recursive check
    for root, _, files in os.walk(search_root):
        for file in files:
            if file.rsplit(".", 1)[0] == base_name and file.endswith(tuple(extensions)):
                return os.path.join(root, file)
    return None


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
        "misc_error": 0,
    }

    # Get all SRT files with glob
    logger.info("Scanning for SRT files...")
    all_srt_files = glob.glob(
        os.path.join(TRANSCRIPT_ROOT, "**", "*.srt"), recursive=True
    )

    logger.info(f"Found {len(all_srt_files)} SRT files. Processing...")

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
            video_dir = os.path.dirname(json_path.replace(TRANSCRIPT_ROOT, VIDEO_ROOT))
            if os.path.isdir(video_dir):
                for root, _, files in os.walk(video_dir):
                    for file in files:
                        if file.rsplit(".", 1)[0] == base_name and file.endswith(
                            VIDEO_EXTENSIONS
                        ):
                            video_path = os.path.join(root, file)

                            valid_pairs.append(
                                {
                                    "id": base_name,
                                    "video_path": video_path,
                                    "json_path": json_path,  # We pass the JSON to the next step
                                    "srt_path": srt_file,  # Kept for reference
                                }
                            )
                            break

            else:
                logger.warning(f"Skipping {base_name}: Video not found.")
                skipped_reason["video_not_found"] += 1
        else:
            logger.warning(f"Skipping {base_name}: Found SRT but missing JSON.")
            skipped_reason["missing_json"] += 1

    logger.info("\nScan complete.")
    logger.info(f"Valid English pairs: {len(valid_pairs)}")
    logger.info(f"Skipped/Non-English: {skipped_reason['non_english']}")
    logger.info(f"Skipped/Missing JSON: {skipped_reason['missing_json']}")
    logger.info(f"Skipped/Video Not Found: {skipped_reason['video_not_found']}")
    logger.info(f"Skipped/Misc Errors: {skipped_reason['misc_error']}")

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(valid_pairs, f, indent=2)

    logger.info(f"Manifest saved to {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
