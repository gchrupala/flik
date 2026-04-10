# Take snippet from video at random intervals according to the srt file and check if the content of the video frames correspond to the transcription
import argparse
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
from CONSTANTS import (
    PROJECT_ROOT,
    OUTPUT_MANIFEST as MANIFEST_FILE,
    ALIGNMENT_SCORES_FILE as OUTPUT_FILE,
    SAMPLES_PER_VIDEO,
    TARGET_FPS,
    CLIP_MODEL,
    CLIP_TOKEN_LIMIT,
    DEFAULT_FPS,
    MIN_STRIDE,
)

DEVICE = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)

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


def load_video_frames(video_path, start_sec, end_sec, target_fps=TARGET_FPS):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = DEFAULT_FPS

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    stride = max(MIN_STRIDE, int(round(fps / target_fps)))

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


def establish_clip_score_baseline():
    """
    Establish a baseline CLIP score by using the MSCOCO dataset captions and images.
    Use matched pairs for the topline score, and random pairs for the baseline score.
    This helps to understand what a "low" or "random" alignment score looks like.
    """

    # Load MSCOCO 2017 validation set from json and into HF dataset
    dataset_json = os.path.expanduser("~/corpora/MSCOCO-2017/captions_val2017.json")
    images_root = os.path.expanduser("~/corpora/MSCOCO-2017/val2017/")
    with open(dataset_json, "r") as f:
        data = json.load(f)
    annotations = data["annotations"]
    images_info = {img["id"]: img for img in data["images"]}
    from datasets import Dataset

    records = []
    for ann in annotations:
        img_info = images_info[ann["image_id"]]
        records.append(
            {
                "image_path": os.path.join(images_root, img_info["file_name"]),
                "caption": ann["caption"],
            }
        )
    dataset = Dataset.from_list(records)

    # Load CLIP model
    logger.info(f"Loading CLIP ({DEVICE}) for baseline establishment...")
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    # Compute matched scores
    matched_scores = {"text-to-image": [], "image-to-text": []}
    for item in tqdm(dataset, desc="Computing matched scores"):
        image = Image.open(item["image_path"]).convert("RGB")
        caption = item["caption"][:CLIP_TOKEN_LIMIT]  # CLIP limit

        inputs = processor(
            text=[caption], images=[image], return_tensors="pt", padding=True
        ).to(DEVICE)

        with torch.no_grad():
            out = model(**inputs)

        itt_score = out.logits_per_image.item()
        tti_score = out.logits_per_text.item()
        matched_scores["image-to-text"].append(itt_score)
        matched_scores["text-to-image"].append(tti_score)
    avg_matched_itt_score = np.mean(matched_scores["image-to-text"])
    avg_matched_tti_score = np.mean(matched_scores["text-to-image"])
    logger.info(
        f"Average matched image-to-text CLIP score: {avg_matched_itt_score:.4f}"
    )
    logger.info(
        f"Average matched text-to-image CLIP score: {avg_matched_tti_score:.4f}"
    )

    # Compute random scores
    random_scores = {
        "text-to-image": [],
        "image-to-text": [],
    }
    captions = [item["caption"][:CLIP_TOKEN_LIMIT] for item in dataset]
    for item in tqdm(dataset, desc="Computing random scores"):
        image = Image.open(item["image_path"]).convert("RGB")
        random_caption = random.choice(captions)

        inputs = processor(
            text=[random_caption], images=[image], return_tensors="pt", padding=True
        ).to(DEVICE)

        with torch.no_grad():
            out = model(**inputs)

        itt_score = out.logits_per_image.item()
        tti_score = out.logits_per_text.item()
        random_scores["image-to-text"].append(itt_score)
        random_scores["text-to-image"].append(tti_score)
    avg_random_itt_score = np.mean(random_scores["image-to-text"])
    avg_random_tti_score = np.mean(random_scores["text-to-image"])
    logger.info(f"Average random image-to-text CLIP score: {avg_random_itt_score:.4f}")
    logger.info(f"Average random text-to-image CLIP score: {avg_random_tti_score:.4f}")

    # Save both random and normal results for future reference
    combined_results = {
        "matched": matched_scores,
        "random": random_scores,
    }
    with open(
        os.path.join(PROJECT_ROOT, "data/clip_baseline_scores_coco.json"), "w"
    ) as f:
        json.dump(combined_results, f, indent=2)


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

                text_input = seg["text"][:CLIP_TOKEN_LIMIT]  # CLIP limit

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


def check_randomized():
    # 1. Load Data
    with open(MANIFEST_FILE, "r") as f:
        tasks = json.load(f)

    logger.info(f"Loading CLIP ({DEVICE})...")
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    results = []

    # 2. Process Loop
    logger.info(f"Processing {len(tasks)} videos...")

    # Choose a random number of videos to process:

    random_tasks = random.sample(tasks, k=min(1000, len(tasks)))
    all_text_input, all_frames = [], []

    # First we load all the data
    for task in tqdm(random_tasks):
        # Re-load transcript just to get segments
        with open(task["json_path"], "r") as f:
            data = json.load(f)
        segments = data.get("segments", []) if isinstance(data, dict) else data

        # Filter for segments > 2 seconds (better visual context)
        valid_segs = [s for s in segments if (s["end"] - s["start"]) > 2.0]
        if not valid_segs:
            valid_segs = segments

        # Random sample
        selected = random.sample(valid_segs, min(SAMPLES_PER_VIDEO, len(valid_segs)))

        for seg in selected:
            frames = load_video_frames(
                task["video_path"], seg["start"], seg["end"], TARGET_FPS
            )

            if not frames:
                continue

            text_input = seg["text"][:CLIP_TOKEN_LIMIT]  # CLIP limit

            # Save the input to lists
            all_text_input.append(text_input)
            all_frames.append(frames)

    # Then we scramble the text inputs
    random.shuffle(all_text_input)
    idx = 0
    for text_input, frames in tqdm(
        zip(all_text_input, all_frames),
        total=len(all_text_input),
        desc="Processing randomized pairs",
    ):
        try:
            inputs = processor(
                text=[text_input], images=frames, return_tensors="pt", padding=True
            ).to(DEVICE)

            with torch.no_grad():
                out = model(**inputs)

            # logits_per_image: [1, num_frames]
            # We want the max alignment per segment (did the frame match the text at any point?)
            score = out.logits_per_image.max().item()

            results.append(score)
        except Exception as e:
            logger.warning(f"\nFailed on randomized pair {idx}: {e}")
        idx += 1

    # 3. Save Results
    output_file = OUTPUT_FILE.replace(".json", "_randomized.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nDone. Results saved to {output_file}")


def check_randomized_output():
    randomized_output_file = OUTPUT_FILE.replace(".json", "_randomized.json")
    with open(randomized_output_file, "r") as f:
        randomized_scores = json.load(f)

    import pandas as pd

    randomized_scores_df = pd.DataFrame(randomized_scores, columns=["clip_score"])

    # Check the statistics of the alignments scores:
    print(randomized_scores_df["clip_score"].describe())


def check_output():
    with open(OUTPUT_FILE, "r") as f:
        alignment_score = json.load(f)

    import pandas as pd

    alignment_score_df = pd.DataFrame(alignment_score)
    # Ignore the 0 scores
    alignment_score_df = alignment_score_df[alignment_score_df["avg_clip_score"] > 0.0]

    # Order the df with avg_clip_score
    alignment_score_df = alignment_score_df.sort_values(
        by=["avg_clip_score"], ascending=False
    ).reset_index(drop=True)

    # Check the statistics of the alignments scores:
    print(alignment_score_df["avg_clip_score"].describe())


def check_coco_clip_scores():
    baseline_file = os.path.join(PROJECT_ROOT, "data/clip_baseline_scores_coco.json")
    with open(baseline_file, "r") as f:
        baseline_scores = json.load(f)

    import pandas as pd

    matched_itt_df = pd.DataFrame(
        baseline_scores["matched"]["image-to-text"], columns=["clip_score"]
    )
    random_itt_df = pd.DataFrame(
        baseline_scores["random"]["image-to-text"], columns=["clip_score"]
    )

    print("Matched Image-to-Text CLIP Scores:")
    print(matched_itt_df["clip_score"].describe())

    print("\nRandom Image-to-Text CLIP Scores:")
    print(random_itt_df["clip_score"].describe())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check video-text correspondence using CLIP"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (cuda/cpu). Default: auto-detect using torch.accelerator",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["main", "randomized", "output", "coco"],
        default="randomized",
        help="Which function to run: main (alignment), randomized (random pairs), output (stats), coco (baseline)",
    )
    args = parser.parse_args()

    # Override DEVICE if specified
    if args.device is not None:
        if args.device == "cuda":
            if torch.cuda.is_available():
                DEVICE = torch.device("cuda")
            else:
                logger.warning("CUDA requested but not available, using CPU")
                DEVICE = torch.device("cpu")
        else:
            DEVICE = torch.device("cpu")
        logger.info(f"Using device: {DEVICE}")

    if args.mode == "main":
        main()
    elif args.mode == "randomized":
        check_randomized()
    elif args.mode == "output":
        check_output()
    elif args.mode == "coco":
        check_coco_clip_scores()
