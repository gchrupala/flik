# Take snippet from video at random intervals according to the srt file and check if the content of the video frames correspond to the transcription
import argparse
import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

# Config
from src.CONSTANTS import (
    PROJECT_ROOT,
    OUTPUT_MANIFEST as MANIFEST_FILE,
    ALIGNMENT_SCORES_FILE as OUTPUT_FILE,
    RANDOM_BASELINE_STATS_FILE,
    SAMPLES_PER_VIDEO,
    TARGET_FPS,
    CLIP_MODEL,
    CLIP_TOKEN_LIMIT,
    CLIP_BATCH_SIZE,
    CLIP_NUM_WORKERS,
    DEFAULT_FPS,
    MIN_STRIDE,
    SEGMENT_PERCENTILE_THRESHOLD,
    VIDEO_PERCENTILE_THRESHOLD,
    USE_SEGMENT_FILTER,
)

DEVICE = (
    torch.accelerator.current_accelerator()
    if torch.accelerator.is_available()
    else torch.device("cpu")
)

# Performance settings (overridable via CLI)
_BATCH_SIZE = CLIP_BATCH_SIZE
_NUM_WORKERS = CLIP_NUM_WORKERS
_USE_FP16 = True
_DECODER = "cpu"  # "cpu" or "nvdec"

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


def load_video_frames_nvdec(video_path, start_sec, end_sec, target_fps=TARGET_FPS):
    """NVDEC hardware video decode — stub for future implementation.

    Future implementation should use one of:
      - cv2.cudacodec (OpenCV CUDA module)
      - PyNvVideoCodec (NVIDIA's Python bindings)
      - NVIDIA DALI (Data Loading Library)
      - ffmpeg with h264_cuvid via PyAV

    The interface must match load_video_frames: return list[PIL.Image].
    """
    raise NotImplementedError(
        "NVDEC hardware decode not yet implemented. Use --decoder cpu (default)."
    )


def _get_decoder_fn():
    """Return the active decoder function based on _DECODER setting."""
    if _DECODER == "nvdec":
        return load_video_frames_nvdec
    return load_video_frames


def _score_pairs_batched(model, processor, texts, images, device, use_fp16):
    """Score a batch of (text, image) pairs using CLIP.

    Computes CLIP similarity for each pair via the diagonal of the similarity
    matrix. This is mathematically equivalent to scoring each pair individually,
    since CLIP encodes texts and images independently (no cross-attention).

    Args:
        texts: list of B caption strings.
        images: list of B PIL Images.
        device: torch device.
        use_fp16: if True and on CUDA, use autocast for FP16 inference.

    Returns:
        list of B float scores (one per pair).
    """
    inputs = processor(
        text=texts, images=images, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        if use_fp16 and device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(**inputs)
        else:
            out = model(**inputs)
    # Diagonal = per-pair similarity
    return out.logits_per_image.diag().cpu().tolist()


def _run_decode_score_pipeline(model, processor, decode_units, device, desc):
    """Parallel decode + batched CLIP scoring pipeline.

    Args:
        decode_units: list of (global_idx, video_path, start, end, text).
        device: torch device for CLIP inference.
        desc: tqdm description string.

    Returns:
        dict: global_idx -> max frame score for that segment.
    """
    decoder_fn = _get_decoder_fn()
    seg_frame_scores = {}  # global_idx -> list of per-frame scores

    with ThreadPoolExecutor(max_workers=_NUM_WORKERS) as pool:
        # Submit all decode tasks
        futures = {}
        for unit in decode_units:
            gidx, vpath, start, end, text = unit
            fut = pool.submit(decoder_fn, vpath, start, end, TARGET_FPS)
            futures[fut] = unit

        # Consume completed decodes, batch for CLIP
        batch_texts, batch_images, batch_seg_ids = [], [], []

        for fut in tqdm(
            as_completed(futures), total=len(futures), desc=desc
        ):
            unit = futures[fut]
            gidx, _, _, _, text = unit
            try:
                frames = fut.result()
            except Exception as e:
                logger.warning(f"Decode failed for seg {gidx}: {e}")
                frames = []

            if not frames:
                continue

            for frame in frames:
                batch_texts.append(text)
                batch_images.append(frame)
                batch_seg_ids.append(gidx)

                if len(batch_texts) >= _BATCH_SIZE:
                    scores = _score_pairs_batched(
                        model, processor, batch_texts, batch_images, device, _USE_FP16
                    )
                    for sid, sc in zip(batch_seg_ids, scores):
                        seg_frame_scores.setdefault(sid, []).append(sc)
                    batch_texts, batch_images, batch_seg_ids = [], [], []

        # Score remaining
        if batch_texts:
            scores = _score_pairs_batched(
                model, processor, batch_texts, batch_images, device, _USE_FP16
            )
            for sid, sc in zip(batch_seg_ids, scores):
                seg_frame_scores.setdefault(sid, []).append(sc)

    # Max per segment (best frame-text alignment)
    return {sid: max(scores) for sid, scores in seg_frame_scores.items()}


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

    # Load random baseline scores for normalization
    random_scores = load_random_scores()
    if random_scores is None:
        logger.warning(
            "Random baseline scores not found. Normalized scores will not be computed."
        )
        random_scores = []

    # 2. First pass: read transcripts, sample segments (no video decode)
    logger.info(f"Sampling segments from {len(tasks)} videos...")
    decode_units = []  # (global_idx, video_path, start, end, text)
    task_segments = {}  # task_idx -> list of (global_idx, seg_dict)

    for task_idx, task in enumerate(tqdm(tasks, desc="Reading transcripts")):
        try:
            with open(task["json_path"], "r") as f:
                data = json.load(f)
            segments = data.get("segments", []) if isinstance(data, dict) else data

            valid_segs = [s for s in segments if (s["end"] - s["start"]) > 2.0]
            if not valid_segs:
                valid_segs = segments

            selected = random.sample(
                valid_segs, min(SAMPLES_PER_VIDEO, len(valid_segs))
            )

            task_segs = []
            for seg in selected:
                gidx = len(decode_units)
                text = seg["text"][:CLIP_TOKEN_LIMIT]
                decode_units.append(
                    (gidx, task["video_path"], seg["start"], seg["end"], text)
                )
                task_segs.append((gidx, seg))
            task_segments[task_idx] = task_segs
        except Exception as e:
            logger.warning(f"Failed to load transcript for {task['id']}: {e}")
            task_segments[task_idx] = []

    logger.info(f"Total segments to score: {len(decode_units)}")
    logger.info(
        f"Pipeline config: batch_size={_BATCH_SIZE}, num_workers={_NUM_WORKERS}, "
        f"fp16={_USE_FP16}, decoder={_DECODER}"
    )

    # 3. Parallel decode + batched CLIP scoring
    seg_max_scores = _run_decode_score_pipeline(
        model, processor, decode_units, DEVICE, "Decoding + scoring"
    )
    logger.info(f"Scored {len(seg_max_scores)} segments")

    # 4. Build results (second pass)
    results = []
    segment_results = []

    for task_idx, task in enumerate(tasks):
        task_segs = task_segments.get(task_idx, [])
        if not task_segs:
            continue

        segment_scores_raw = []
        segment_scores_z = []
        segment_scores_percentile = []
        segment_passed = []

        for gidx, seg in task_segs:
            score_raw = seg_max_scores.get(gidx, 0.0)
            segment_scores_raw.append(score_raw)

            if random_scores:
                z, perc = compute_normalized_scores(score_raw, random_scores)
                segment_scores_z.append(z)
                segment_scores_percentile.append(perc)
                if USE_SEGMENT_FILTER:
                    segment_passed.append(perc >= SEGMENT_PERCENTILE_THRESHOLD)
                else:
                    segment_passed.append(True)
            else:
                segment_scores_z.append(None)
                segment_scores_percentile.append(None)
                segment_passed.append(True)

            seg_z = segment_scores_z[-1]
            seg_percentile = segment_scores_percentile[-1]
            seg_ok = segment_passed[-1]
            segment_results.append(
                {
                    "id": f"{task['id']}_{seg['start']:.2f}_{seg['end']:.2f}",
                    "video_id": task["id"],
                    "video_path": task["video_path"],
                    "json_path": task["json_path"],
                    "start_sec": seg["start"],
                    "end_sec": seg["end"],
                    "text": seg.get("text", ""),
                    "clip_score_raw": round(score_raw, 4),
                    "clip_score_z": round(seg_z, 4) if seg_z is not None else None,
                    "clip_score_percentile": round(seg_percentile, 2)
                    if seg_percentile is not None
                    else None,
                    "segment_passed": bool(seg_ok),
                }
            )

        avg_score_raw = np.mean(segment_scores_raw) if segment_scores_raw else 0.0
        avg_score_z = None
        avg_score_percentile = None
        if random_scores:
            avg_score_z, avg_score_percentile = compute_normalized_scores(
                avg_score_raw, random_scores
            )

        video_passed_by_percentile = False
        video_passed_by_segment_fraction = False
        if random_scores and avg_score_percentile is not None:
            video_passed_by_percentile = (
                avg_score_percentile >= VIDEO_PERCENTILE_THRESHOLD
            )
        if segment_passed:
            fraction_passed = sum(segment_passed) / len(segment_passed)
            video_passed_by_segment_fraction = (
                fraction_passed >= 0.5
            )

        results.append(
            {
                "id": task["id"],
                "video_path": task["video_path"],
                "avg_clip_score": round(avg_score_raw, 4),
                "avg_clip_score_raw": round(avg_score_raw, 4),
                "avg_clip_score_z": round(avg_score_z, 4)
                if avg_score_z is not None
                else None,
                "avg_clip_score_percentile": round(avg_score_percentile, 2)
                if avg_score_percentile is not None
                else None,
                "segment_scores_raw": [round(s, 4) for s in segment_scores_raw],
                "segment_scores_z": [
                    round(z, 4) if z is not None else None for z in segment_scores_z
                ],
                "segment_scores_percentile": [
                    round(p, 2) if p is not None else None
                    for p in segment_scores_percentile
                ],
                "segment_passed": segment_passed,
                "video_passed_by_percentile": video_passed_by_percentile,
                "video_passed_by_segment_fraction": video_passed_by_segment_fraction,
            }
        )

    # 5. Save Results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    segment_output_file = OUTPUT_FILE.replace(".json", "_segments.json")
    with open(segment_output_file, "w") as f:
        json.dump(segment_results, f, indent=2)

    logger.info(f"\nDone. Results saved to {OUTPUT_FILE}")
    logger.info(f"Segment-level results saved to {segment_output_file}")


def check_randomized():
    # 1. Load Data
    with open(MANIFEST_FILE, "r") as f:
        tasks = json.load(f)

    logger.info(f"Loading CLIP ({DEVICE})...")
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    logger.info(f"Processing {len(tasks)} videos...")

    # 2. Sample segments and collect texts (no decode yet)
    random_tasks = random.sample(tasks, k=min(1000, len(tasks)))
    decode_units = []  # (idx, video_path, start, end, text)

    for task in tqdm(random_tasks, desc="Sampling segments"):
        with open(task["json_path"], "r") as f:
            data = json.load(f)
        segments = data.get("segments", []) if isinstance(data, dict) else data

        valid_segs = [s for s in segments if (s["end"] - s["start"]) > 2.0]
        if not valid_segs:
            valid_segs = segments

        selected = random.sample(valid_segs, min(SAMPLES_PER_VIDEO, len(valid_segs)))

        for seg in selected:
            text = seg["text"][:CLIP_TOKEN_LIMIT]
            decode_units.append(
                (len(decode_units), task["video_path"], seg["start"], seg["end"], text)
            )

    # 3. Shuffle texts to create random text-frame pairings
    texts = [u[4] for u in decode_units]
    random.shuffle(texts)
    decode_units = [
        (u[0], u[1], u[2], u[3], texts[i]) for i, u in enumerate(decode_units)
    ]

    logger.info(f"Total randomized pairs: {len(decode_units)}")
    logger.info(
        f"Pipeline config: batch_size={_BATCH_SIZE}, num_workers={_NUM_WORKERS}, "
        f"fp16={_USE_FP16}, decoder={_DECODER}"
    )

    # 4. Parallel decode + batched CLIP scoring (streaming, no RAM accumulation)
    seg_max_scores = _run_decode_score_pipeline(
        model, processor, decode_units, DEVICE, "Decoding + scoring (randomized)"
    )

    # 5. Save results (one score per decode unit, in original order)
    results = [
        seg_max_scores.get(i, 0.0) for i in range(len(decode_units))
    ]

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
    import numpy as np

    df = pd.DataFrame(alignment_score)
    # Ignore the 0 scores
    df = df[df["avg_clip_score_raw"] > 0.0].copy()

    # Compute additional statistics
    print("=== Raw CLIP scores (max segment alignment) ===")
    print(df["avg_clip_score_raw"].describe())

    if "avg_clip_score_z" in df.columns:
        print("\n=== Z-scores (relative to random baseline) ===")
        print(df["avg_clip_score_z"].describe())

        # Percentage of videos with z > 0 (better than random mean)
        z_positive = (df["avg_clip_score_z"] > 0).sum()
        print(
            f"Videos better than random mean (z > 0): {z_positive}/{len(df)} ({z_positive / len(df) * 100:.1f}%)"
        )

    if "avg_clip_score_percentile" in df.columns:
        print("\n=== Percentiles (relative to random baseline) ===")
        print(df["avg_clip_score_percentile"].describe())

        # Count by percentile bins
        bins = [0, 25, 50, 75, 95, 100]
        labels = ["0-25", "25-50", "50-75", "75-95", "95-100"]
        if df["avg_clip_score_percentile"].notna().any():
            df["percentile_bin"] = pd.cut(
                df["avg_clip_score_percentile"],
                bins=bins,
                labels=labels,
                include_lowest=True,
            )
            print("\nPercentile distribution:")
            print(df["percentile_bin"].value_counts().sort_index())

    # Top 10 videos by raw score
    print("\n=== Top 10 videos by raw CLIP score ===")
    top_raw = df.sort_values("avg_clip_score_raw", ascending=False).head(10)
    for _, row in top_raw.iterrows():
        print(
            f"{row['id']}: raw={row['avg_clip_score_raw']:.3f}, z={row.get('avg_clip_score_z', 'NA')}, percentile={row.get('avg_clip_score_percentile', 'NA')}"
        )


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


def compute_random_baseline_stats():
    """
    Compute statistics (mean, std, percentiles) from the randomized alignment scores
    and save to RANDOM_BASELINE_STATS_FILE.
    """
    randomized_file = OUTPUT_FILE.replace(".json", "_randomized.json")
    if not os.path.exists(randomized_file):
        logger.warning(
            f"Randomized scores file {randomized_file} not found. Run with --mode randomized first."
        )
        return None

    with open(randomized_file, "r") as f:
        random_scores = json.load(f)

    if not random_scores:
        logger.warning("Randomized scores list is empty.")
        return None

    stats = {
        "mean": float(np.mean(random_scores)),
        "std": float(np.std(random_scores)),
        "min": float(np.min(random_scores)),
        "max": float(np.max(random_scores)),
        "percentiles": {
            "p5": float(np.percentile(random_scores, 5)),
            "p25": float(np.percentile(random_scores, 25)),
            "p50": float(np.percentile(random_scores, 50)),
            "p75": float(np.percentile(random_scores, 75)),
            "p95": float(np.percentile(random_scores, 95)),
        },
        "count": len(random_scores),
    }

    with open(RANDOM_BASELINE_STATS_FILE, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Random baseline stats saved to {RANDOM_BASELINE_STATS_FILE}")
    return stats


def load_baseline_stats(compute_if_missing=True):
    """
    Load random baseline stats from file. If missing, optionally compute them.
    Returns dict or None.
    """
    if os.path.exists(RANDOM_BASELINE_STATS_FILE):
        with open(RANDOM_BASELINE_STATS_FILE, "r") as f:
            return json.load(f)

    if compute_if_missing:
        logger.info("Random baseline stats not found, computing...")
        return compute_random_baseline_stats()

    return None


def load_random_scores():
    """
    Load the raw randomized scores from the randomized JSON file.
    Returns list of floats or None if file missing.
    """
    randomized_file = OUTPUT_FILE.replace(".json", "_randomized.json")
    if not os.path.exists(randomized_file):
        logger.warning(f"Randomized scores file {randomized_file} not found.")
        return None
    with open(randomized_file, "r") as f:
        return json.load(f)


def compute_normalized_scores(raw_score, random_scores):
    """
    Compute z-score and percentile rank of raw_score relative to random_scores.
    Returns (z_score, percentile) where percentile is percentage of random scores <= raw_score.
    """
    if not random_scores:
        return None, None
    mean = np.mean(random_scores)
    std = np.std(random_scores)
    z_score = (raw_score - mean) / std if std != 0 else 0.0

    # Compute percentile: proportion of random scores <= raw_score
    sorted_random = np.sort(random_scores)
    percentile = (
        np.searchsorted(sorted_random, raw_score) / len(random_scores)
    ) * 100.0
    return float(z_score), float(percentile)


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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CLIP_BATCH_SIZE,
        help=f"Number of (text, frame) pairs per CLIP forward pass (default: {CLIP_BATCH_SIZE})",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=CLIP_NUM_WORKERS,
        help=f"CPU threads for parallel video decoding (default: {CLIP_NUM_WORKERS})",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        choices=["cpu", "nvdec"],
        default="cpu",
        help="Video decoder backend: cpu (cv2) or nvdec (stub, not yet implemented)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Use FP16 autocast on CUDA for faster inference (default: True)",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Disable FP16, use FP32 (slower but bit-exact)",
    )
    args = parser.parse_args()

    # Apply performance settings
    _BATCH_SIZE = args.batch_size
    _NUM_WORKERS = args.num_workers
    _USE_FP16 = args.fp16
    _DECODER = args.decoder

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
