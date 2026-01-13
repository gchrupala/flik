# Take snippet from video at random intervals according to the srt file and check if the content of the video frames correspond to the transcription
import json
import logging
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

# Config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_FILE = os.path.join(PROJECT_ROOT, "data/batch_manifest.json")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data/alignment_scores.json")
SAMPLES_PER_VIDEO = 5
TARGET_FPS = 3  # Downsample
DEVICE = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)
CLIP_MODEL = "openai/clip-vit-base-patch32"

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(PROJECT_ROOT, "logdir/clip_correspondance.log")
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def load_video_frames(video_path, start_sec, end_sec, target_fps=3):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    stride = max(1, int(round(fps / target_fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_pos = start_frame

    while current_pos < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # Capture frame if it matches stride
        if (current_pos - start_frame) % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

        current_pos += 1

    cap.release()
    return frames


def main():
    # 1. Load Data
    with open(MANIFEST_FILE, "r") as f:
        tasks = json.load(f)

    logger.info(f"Loading CLIP ({DEVICE})...")
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    results = []

    # 2. Process Loop
    logger.info(f"Processing {len(tasks)} videos...")

    for task in tqdm(tasks):
        try:
            # Re-load transcript just to get segments
            with open(task["json_path"], "r") as f:
                data = json.load(f)
            segments = data.get("segments", []) if isinstance(data, dict) else data

            # Filter for segments > 2 seconds (better visual context)
            valid_segs = [s for s in segments if (s["end"] - s["start"]) > 2.0]
            if not valid_segs:
                valid_segs = segments

            # Random sample
            selected = random.sample(
                valid_segs, min(SAMPLES_PER_VIDEO, len(valid_segs))
            )

            segment_scores = []

            for seg in selected:
                frames = load_video_frames(
                    task["video_path"], seg["start"], seg["end"], TARGET_FPS
                )

                if not frames:
                    continue

                text_input = seg["text"][:77]  # CLIP limit

                # # For debug only we show the frames in a 6x6 grid
                # grid_size = (6, 6)
                # if len(frames) > grid_size[0] * grid_size[1]:
                #     frames = frames[: grid_size[0] * grid_size[1]]
                # grid_img = Image.new(
                #     "RGB",
                #     (grid_size[1] * frames[0].width, grid_size[0] * frames[0].height),
                # )
                # for idx, frame in enumerate(frames):
                #     row = idx // grid_size[1]
                #     col = idx % grid_size[1]
                #     grid_img.paste(
                #         frame, (col * frame.width, row * frame.height)
                #     )
                # print(f"Segment Text: {text_input}")
                # grid_img.show()

                inputs = processor(
                    text=[text_input], images=frames, return_tensors="pt", padding=True
                ).to(DEVICE)

                with torch.no_grad():
                    out = model(**inputs)

                # logits_per_image: [1, num_frames]
                # We want the max alignment per segment (did the frame match the text at any point?)
                score = out.logits_per_image.max().item()
                segment_scores.append(score)

            avg_score = np.mean(segment_scores) if segment_scores else 0.0

            results.append(
                {
                    "id": task["id"],
                    "video_path": task["video_path"],
                    "avg_clip_score": round(avg_score, 4),
                    "detail_scores": [round(s, 2) for s in segment_scores],
                }
            )

        except Exception as e:
            logger.warning(f"\nFailed on {task['id']}: {e}")

    # 3. Save Results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nDone. Results saved to {OUTPUT_FILE}")


def check_output():
    with open(OUTPUT_FILE, "r") as f:
        alignment_score = json.load(f)

    import pandas as pd

    alignment_score_df = pd.DataFrame(alignment_score)

    # Order the df with avg_clip_score
    alignment_score_df = alignment_score_df.sort_values(
        by=["avg_clip_score"], ascending=False
    ).reset_index(drop=True)

    # Check the statistics of the alignments scores:
    print(alignment_score_df["avg_clip_score"].describe())


if __name__ == "__main__":
    main()
