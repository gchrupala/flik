import argparse
import gc
import glob
import json
import logging
import os

import torch
import whisperx
from tqdm import tqdm

from src.CONSTANTS import (
    PROJECT_ROOT,
    VIDEO_ROOT as SOURCE_FOLDER,
    TRANSCRIPT_ROOT as OUTPUT_FOLDER,
    WHISPER_BATCH_SIZE as BATCH_SIZE,
    WHISPER_COMPUTE_TYPE as COMPUTE_TYPE,
    WHISPER_MODEL,
    ALIGNMENT_LANGUAGE,
    VIDEO_EXTENSIONS,
)

# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logdir/transcribe.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def process_video(video_path, output_path, model, align_model, metadata, device):
    base_name = os.path.basename(video_path)
    file_name = os.path.splitext(base_name)[0]
    alignment_output_path = os.path.join(output_path, f"{file_name}.json")

    if os.path.isfile(alignment_output_path):
        logger.info(f"Skipping (already processed): {video_path}")
        return

    logger.info(f"Processing: {video_path}")

    # 1. Transcribe with original whisper (batched)
    audio = whisperx.load_audio(video_path)
    result = model.transcribe(audio, batch_size=BATCH_SIZE)

    language_code = result["language"]

    # 2. Align whisper output
    # This aligns the text segments to the audio waveforms for word-level precision
    result_aligned = whisperx.align(
        result["segments"],
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # Cleanup audio memory
    del audio
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save transcription as JSON (best for data analysis/search)
    with open(alignment_output_path, "w", encoding="utf-8") as f:
        json.dump(result_aligned["segments"], f, indent=2, ensure_ascii=False)

    # Helper: Save as SRT (if you want to watch the alignment)
    save_as_srt(
        result_aligned["segments"], os.path.join(output_path, f"{file_name}.srt")
    )

    # Save language info to text file
    with open(
        os.path.join(output_path, f"{file_name}_language.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(language_code)

    logger.info(f"Done: {file_name}")


def save_as_srt(segments, output_file):
    def format_timestamp(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    with open(output_file, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments):
            start = format_timestamp(segment["start"])
            end = format_timestamp(segment["end"])
            text = segment["text"].strip()
            f.write(f"{i + 1}\n{start} --> {end}\n{text}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Transcribe videos using WhisperX")
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Device to use for inference (default: {DEVICE})",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="This shard's index (0-based). Use with --num-shards for SLURM "
        "array jobs. Default 0 = no sharding.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards (SLURM array size). Default 1 = no "
        "sharding (process all videos).",
    )
    args = parser.parse_args()

    device = args.device
    if device not in ["cuda", "cpu"]:
        logger.warning(f"Device '{device}' not recognized, defaulting to 'cpu'")
        device = "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    logger.info(f"Using device: {device}")

    # 1. Load Whisper Model
    # Options: "tiny", "base", "small", "medium", "large-v2", "large-v3"
    logger.info("Loading Whisper model...")
    model = whisperx.load_model(WHISPER_MODEL, device, compute_type=COMPUTE_TYPE)

    # 2. Process Files
    logger.info("Searching for video files...")
    # We walk through the directory to find video files
    video_extensions = VIDEO_EXTENSIONS

    files_to_process = []
    for ext in video_extensions:
        pattern = os.path.join(SOURCE_FOLDER, "**", f"*{ext}")
        files_to_process.extend(glob.glob(pattern, recursive=True))
        pattern = os.path.join(SOURCE_FOLDER, "**", f"*{ext.upper()}")
        files_to_process.extend(glob.glob(pattern, recursive=True))

    # Deterministic order is REQUIRED for SLURM array sharding: every array
    # task must see the same list so that (index % num_tasks) assigns each
    # video to exactly one task.
    files_to_process = sorted(set(files_to_process))

    logger.info(f"Found {len(files_to_process)} video files to process.")

    if not files_to_process:
        logger.info("No video files found.")
        return

    # --- SLURM array sharding (optional) ---
    # --shard-index i --num-shards N -> process videos[i], videos[i+N], ...
    # Defaults (0 / 1) reproduce the original single-process behavior.
    shard_index = args.shard_index
    num_shards = args.num_shards
    if num_shards > 1:
        if not (0 <= shard_index < num_shards):
            raise ValueError(
                f"--shard-index must be in [0, {num_shards}), got {shard_index}"
            )
        files_to_process = files_to_process[shard_index::num_shards]
        logger.info(
            f"Shard {shard_index}/{num_shards}: processing {len(files_to_process)} videos"
        )

    # 3. Load Alignment Model (English usually, or detect automatically)
    # Note: If your movies are mixed languages, you might need to handle language code dynamically per file.
    logger.info("Loading Alignment model...")
    # This loads a localized Wav2Vec2 model for timestamp alignment
    align_model, metadata = whisperx.load_align_model(
        language_code=ALIGNMENT_LANGUAGE, device=device
    )

    for video_file in tqdm(files_to_process, desc="Processing videos"):
        try:
            output_path = os.path.join(
                OUTPUT_FOLDER,
                os.path.relpath(os.path.dirname(video_file), SOURCE_FOLDER),
            )
            os.makedirs(output_path, exist_ok=True)

            process_video(video_file, output_path, model, align_model, metadata, device)

        except Exception as e:
            logger.error(f"Error processing {video_file}: {e}")


if __name__ == "__main__":
    main()
