import gc
import glob
import json
import logging
import os

import torch
import whisperx
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration
SOURCE_FOLDER = os.path.join(
    PROJECT_ROOT, "data/out/cinedantan/"
)  # Where your video files are
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data/transcripts")  # Where to save results
DEVICE = "cuda"
BATCH_SIZE = 16  # Reduce if you run out of VRAM
COMPUTE_TYPE = "float16"  # Change to "int8" if you have older GPU/low VRAM

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


def process_video(video_path, output_path, model, align_model, metadata):
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
        DEVICE,
        return_char_alignments=False,
    )

    # Cleanup audio memory
    del audio
    gc.collect()
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
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 1. Load Whisper Model
    # Options: "tiny", "base", "small", "medium", "large-v2", "large-v3"
    logger.info("Loading Whisper model...")
    model = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE)

    # 2. Process Files
    logger.info("Searching for video files...")
    # We walk through the directory to find video files
    video_extensions = (".mp4", ".mkv", ".avi", ".mov")

    files_to_process = []
    for ext in video_extensions:
        pattern = os.path.join(SOURCE_FOLDER, "**", f"*{ext}")
        files_to_process.extend(glob.glob(pattern, recursive=True))
        pattern = os.path.join(SOURCE_FOLDER, "**", f"*{ext.upper()}")
        files_to_process.extend(glob.glob(pattern, recursive=True))

    logger.info(f"Found {len(files_to_process)} video files to process.")

    if not files_to_process:
        logger.info("No video files found.")
        return

    # 3. Load Alignment Model (English usually, or detect automatically)
    # Note: If your movies are mixed languages, you might need to handle language code dynamically per file.
    logger.info("Loading Alignment model...")
    # This loads a localized Wav2Vec2 model for timestamp alignment
    align_model, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)

    for video_file in tqdm(files_to_process, desc="Processing videos"):
        try:
            output_path = os.path.join(
                OUTPUT_FOLDER,
                os.path.relpath(os.path.dirname(video_file), SOURCE_FOLDER),
            )
            os.makedirs(output_path, exist_ok=True)

            process_video(video_file, output_path, model, align_model, metadata)

        except Exception as e:
            logger.error(f"Error processing {video_file}: {e}")


if __name__ == "__main__":
    main()
