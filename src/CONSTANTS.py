import json
import os

# Config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSCRIPT_ROOT = os.path.join(PROJECT_ROOT, "data/transcripts")
VIDEO_ROOT = os.path.join(
    PROJECT_ROOT, "data/out/cinedantan"
)  # Where the actual video files is stored
OUTPUT_MANIFEST = os.path.join(PROJECT_ROOT, "data/batch_manifest.json")

# How many characters to sample to speed up detection?
# 2000 chars is usually enough to be certain of the language without reading the whole file.
CHAR_SAMPLE_SIZE = 2000


def calculate_total_duration(json_path: str = OUTPUT_MANIFEST) -> None:
    """Calculate total duration of the json that

    Args:
        json_path (str): Path to the JSON file containing transcript data.
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_duration = 0.0
    for item in data:
        json_file = item.get("json_path")
        if json_file and os.path.isfile(json_file):
            with open(json_file, "r", encoding="utf-8") as jf:
                segments = json.load(jf)
                for segment in segments:
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    total_duration += max(0, end - start)

    total_hours = total_duration / 3600
    print(f"Total duration of valid transcripts: {total_hours:.2f} hours")


def calculate_total_size(json_path: str = OUTPUT_MANIFEST) -> None:
    """Calculate total size of the video files in the JSON manifest.

    Args:
        json_path (str): Path to the JSON file containing transcript data.
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_size_bytes = 0
    for item in data:
        video_file = item.get("video_path")
        if video_file and os.path.isfile(video_file):
            total_size_bytes += os.path.getsize(video_file)

    total_size_gb = total_size_bytes / (1024**3)
    print(f"Total size of valid video files: {total_size_gb:.2f} GB")
