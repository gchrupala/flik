import json
import os

# Config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data root: explicit override wins, hostname detection is the fallback.
# FLIK_DATA_ROOT should point at the folder that holds the cinedantan data.
#   - Split layout: <root>/transcripts + <root>/videos  (detected when
#     <root>/transcripts exists; videos are then assumed under <root>/videos)
#   - Flat layout:  transcripts and videos both directly under <root>
#
# Snellius (project prjs1586): videos persist in project space and are
# symlinked into the repo; transcripts/manifests are small and live in-repo.
# The cluster uses the split layout via a symlink pair:
#     data/transcripts/                  (in-repo, small)
#     data/videos  -> /projects/prjs1586/cinedantan_videos   (project space)
#     data/out     -> /projects/prjs1586/cinedantan_videos   (legacy alias)
# and FLIK_DATA_ROOT is set to the repo's data/ dir:
#     ln -sfn /projects/prjs1586/cinedantan_videos data/videos
#     export FLIK_DATA_ROOT="$PWD/data"
# Split detection then finds data/transcripts and resolves videos to
# data/videos (project space). Set FLIK_DATA_ROOT in every job script:
# compute/login node hostnames (gcn*/int*) do NOT contain "snellius", so the
# hostname heuristic below never fires there.
_DATA_ROOT = os.environ.get("FLIK_DATA_ROOT")
if _DATA_ROOT:
    if os.path.isdir(os.path.join(_DATA_ROOT, "transcripts")):
        TRANSCRIPT_ROOT = os.path.join(_DATA_ROOT, "transcripts")
        VIDEO_ROOT = os.path.join(_DATA_ROOT, "videos")
    else:
        TRANSCRIPT_ROOT = _DATA_ROOT
        VIDEO_ROOT = _DATA_ROOT
elif "snellius" in os.uname().nodename:
    TRANSCRIPT_ROOT = os.path.join(PROJECT_ROOT, "data/out")
    VIDEO_ROOT = os.path.join(PROJECT_ROOT, "data/out")
else:
    TRANSCRIPT_ROOT = os.path.join(PROJECT_ROOT, "data/transcripts")
    VIDEO_ROOT = os.path.join(PROJECT_ROOT, "data/videos")
OUTPUT_MANIFEST = os.path.join(PROJECT_ROOT, "data/batch_manifest.json")
ALIGNMENT_SCORES_FILE = os.path.join(PROJECT_ROOT, "data/alignment_scores.json")
RANDOM_BASELINE_STATS_FILE = os.path.join(
    PROJECT_ROOT, "data/clip_random_baseline_stats.json"
)

# How many characters to sample to speed up detection?
# 2000 chars is usually enough to be certain of the language without reading the whole file.
CHAR_SAMPLE_SIZE = 2000

# Transcription (WhisperX) constants
WHISPER_DEVICE = "cuda"  # Default device; scripts dynamically detect CUDA availability
WHISPER_BATCH_SIZE = 48  # raised from 16: large-v3 uses ~6GB VRAM, 40GB headroom
WHISPER_COMPUTE_TYPE = "float16"
WHISPER_MODEL = "large-v3"
ALIGNMENT_LANGUAGE = "en"

# CLIP alignment constants
CLIP_MODEL = "openai/clip-vit-base-patch32"
SAMPLES_PER_VIDEO = 5
TARGET_FPS = 3
SEGMENT_PERCENTILE_THRESHOLD = 25  # Lenient threshold for segment-level filtering
VIDEO_PERCENTILE_THRESHOLD = 25  # Lenient threshold for video-level filtering
USE_SEGMENT_FILTER = False  # Whether to apply segment-level filtering

# Video file extensions
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".avi", ".mov", ".webm")

# Language detection constants
MIN_TEXT_LENGTH_FOR_DETECTION = 50
ENGLISH_LANGUAGE_CODE = "en"
ENGLISH_LANGUAGE_PREFIX = "en-"

# CLIP model constants
CLIP_TOKEN_LIMIT = 77

# CLIP scoring performance (parallel decode + batched GPU inference)
CLIP_BATCH_SIZE = 256  # (text, frame) pairs per CLIP forward pass
CLIP_NUM_WORKERS = 16  # CPU threads for parallel video decoding

# Video processing constants
DEFAULT_FPS = 24
MIN_STRIDE = 1


def calculate_total_duration(json_path: str = OUTPUT_MANIFEST) -> None:
    """Calculate total duration of the json that

    Args:
        json_path (str): Path to the JSON file containing transcript data.
    """

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    from src.utils.paths import resolve_path

    total_duration = 0.0
    for item in data:
        json_file = resolve_path(item.get("json_path", ""))
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

    from src.utils.paths import resolve_path

    total_size_bytes = 0
    for item in data:
        video_file = resolve_path(item.get("video_path", ""))
        if video_file and os.path.isfile(video_file):
            total_size_bytes += os.path.getsize(video_file)

    total_size_gb = total_size_bytes / (1024**3)
    print(f"Total size of valid video files: {total_size_gb:.2f} GB")
