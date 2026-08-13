# Take snippet from video at random intervals according to the srt file and check if the content of the video frames correspond to the transcription
import argparse
import json
import logging
import os
import random
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

# Config
from src.utils.paths import resolve_path

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

# COCO baseline config (overridable via CLI, used by --mode coco)
COCO_ANNOTATIONS = "~/corpora/MSCOCO-2017/captions_val2017.json"
COCO_IMAGES = "~/corpora/MSCOCO-2017/val2017/"
COCO_LIMIT = 0  # 0 = score all images; >0 caps the number of images (smoke tests)

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

    try:
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
    finally:
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
    # Full text is passed here; the tokenizer truncates to CLIP_TOKEN_LIMIT
    # tokens (77) via truncation=True. (Earlier code sliced text to 77
    # CHARACTERS, discarding most of each segment's content.)
    inputs = processor(
        text=texts, images=images, return_tensors="pt", padding=True,
        truncation=True, max_length=CLIP_TOKEN_LIMIT,
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

    Uses a bounded sliding window of outstanding decode futures to prevent
    CPU RAM exhaustion. Without bounding, all segments are submitted at once
    and their decoded PIL frames accumulate in completed-but-unconsumed
    futures, which can exceed system RAM on large manifests (7309 segments
    × ~15 frames × ~1-3 MB/frame = 100+ GB).

    Args:
        decode_units: list of (global_idx, video_path, start, end, text).
        device: torch device for CLIP inference.
        desc: tqdm description string.

    Returns:
        dict: global_idx -> max frame score for that segment.
    """
    decoder_fn = _get_decoder_fn()
    seg_frame_scores = {}  # global_idx -> list of per-frame scores

    # Bounded outstanding futures to limit RAM (decoded frames in-flight).
    # At most MAX_PENDING segments' frames exist in memory simultaneously.
    MAX_PENDING = _NUM_WORKERS * 4

    batch_texts, batch_images, batch_seg_ids = [], [], []

    with ThreadPoolExecutor(max_workers=_NUM_WORKERS) as pool:
        futures = {}  # fut -> unit (only pending futures, popped on completion)
        unit_iter = iter(decode_units)
        exhausted = False

        # Submit initial batch
        for _ in range(MAX_PENDING):
            try:
                unit = next(unit_iter)
            except StopIteration:
                exhausted = True
                break
            gidx, vpath, start, end, text = unit
            fut = pool.submit(decoder_fn, vpath, start, end, TARGET_FPS)
            futures[fut] = unit

        pbar = tqdm(total=len(decode_units), desc=desc)

        while futures:
            # Wait for at least one future to complete
            done, _ = wait(set(futures), return_when=FIRST_COMPLETED)

            for fut in done:
                # Pop to allow GC of decoded frames (future caches its result)
                unit = futures.pop(fut)
                gidx, _, _, _, text = unit
                pbar.update(1)

                try:
                    frames = fut.result()
                except Exception as e:
                    logger.warning(f"Decode failed for seg {gidx}: {e}")
                    frames = []

                if frames:
                    for frame in frames:
                        batch_texts.append(text)
                        batch_images.append(frame)
                        batch_seg_ids.append(gidx)

                        if len(batch_texts) >= _BATCH_SIZE:
                            scores = _score_pairs_batched(
                                model, processor, batch_texts,
                                batch_images, device, _USE_FP16,
                            )
                            for sid, sc in zip(batch_seg_ids, scores):
                                seg_frame_scores.setdefault(sid, []).append(sc)
                            batch_texts, batch_images, batch_seg_ids = [], [], []

                # Submit replacement unit to keep the pipeline full
                if not exhausted:
                    try:
                        unit = next(unit_iter)
                    except StopIteration:
                        exhausted = True
                    else:
                        gidx, vpath, start, end, text = unit
                        fut = pool.submit(
                            decoder_fn, vpath, start, end, TARGET_FPS
                        )
                        futures[fut] = unit

        pbar.close()

    # Score remaining batch
    if batch_texts:
        scores = _score_pairs_batched(
            model, processor, batch_texts, batch_images, device, _USE_FP16
        )
        for sid, sc in zip(batch_seg_ids, scores):
            seg_frame_scores.setdefault(sid, []).append(sc)

    # Max per segment (best frame-text alignment)
    return {sid: max(scores) for sid, scores in seg_frame_scores.items()}

def check_coco_clip_scores():
    """
    Establish a CLIP calibration baseline on MSCOCO 2017 val captions+images.

    Computes (a) *matched* scores on true image<->caption pairs (the topline)
    and (b) *random* scores on shuffled pairings that EXCLUDE the true caption
    (the null). Reports summary stats and saves both lists + stats to
    ``data/clip_baseline_scores_coco.json``.

    This is a calibration reference: it tells us what an "aligned" vs a
    "random" text<->image CLIP score looks like on a known-aligned dataset, so
    we can interpret the flik video<->transcript alignment scores (and sanity
    check percentile/z thresholds).

    Run:
        uv run --extra cu128 python -m src.preprocess.check_correspondance \
            --mode coco [--coco-annotations ...] [--coco-images ...] [--coco-limit N]
    """
    dataset_json = os.path.expanduser(COCO_ANNOTATIONS)
    images_root = os.path.expanduser(COCO_IMAGES)
    if not os.path.isfile(dataset_json):
        logger.error(f"COCO annotations not found: {dataset_json}")
        return
    if not os.path.isdir(images_root):
        logger.error(f"COCO images root not found: {images_root}")
        return

    with open(dataset_json, "r") as f:
        data = json.load(f)
    annotations = data["annotations"]
    images_info = {img["id"]: img for img in data["images"]}

    records = [
        {
            "image_id": ann["image_id"],
            "image_path": os.path.join(images_root, images_info[ann["image_id"]]["file_name"]),
            "caption": ann["caption"],
        }
        for ann in annotations
    ]

    # Optional cap on the number of *images* (keeps an image's 5 captions together).
    if COCO_LIMIT > 0:
        seen = set()
        capped = []
        for r in records:
            if len(seen) >= COCO_LIMIT:
                break
            seen.add(r["image_id"])
            capped.append(r)
        records = capped

    n_images = len({r["image_id"] for r in records})
    logger.info(f"Loaded {len(records)} COCO caption<->image pairs ({n_images} images)")

    # Load CLIP once.
    logger.info(f"Loading CLIP ({DEVICE}) for COCO baseline...")
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL)

    # Preload all images once (avoids re-opening per pair).
    logger.info("Loading images...")
    images = [Image.open(r["image_path"]).convert("RGB") for r in records]
    captions = [r["caption"] for r in records]
    image_ids = [r["image_id"] for r in records]

    def _batched_scores(texts, imgs):
        """Score all (text, image) pairs in batches via the diagonal."""
        scores = []
        for s in range(0, len(imgs), _BATCH_SIZE):
            scores.extend(
                _score_pairs_batched(
                    model, processor, texts[s:s + _BATCH_SIZE], imgs[s:s + _BATCH_SIZE],
                    DEVICE, _USE_FP16,
                )
            )
        return scores

    def _stats(vals):
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "percentiles": {
                "p5": float(np.percentile(vals, 5)),
                "p25": float(np.percentile(vals, 25)),
                "p50": float(np.percentile(vals, 50)),
                "p75": float(np.percentile(vals, 75)),
                "p95": float(np.percentile(vals, 95)),
            },
            "count": len(vals),
        }

    # --- Matched scores (true image<->caption) ---
    logger.info("Computing matched text-image scores...")
    matched_itt = _batched_scores(captions, images)
    matched_stats = _stats(matched_itt)
    logger.info(
        f"Matched image-to-text CLIP: mean={matched_stats['mean']:.4f} "
        f"std={matched_stats['std']:.4f} (n={matched_stats['count']})"
    )

    # --- Random scores (shuffled pairing, EXCLUDING the true caption) ---
    logger.info("Computing random (shuffled, true-caption-excluded) scores...")
    pool = list(zip(captions, image_ids))
    n_unique_ids = n_images
    random_captions = []
    for img_id in tqdm(image_ids, desc="Random pairing"):
        while True:
            cap, cid = random.choice(pool)
            if cid != img_id:
                break
            if n_unique_ids == 1:
                # Degenerate: only one image's captions exist, so no "other".
                cap, cid = pool[0]
                break
        random_captions.append(cap)
    random_itt = _batched_scores(random_captions, images)
    random_stats = _stats(random_itt)
    logger.info(
        f"Random image-to-text CLIP: mean={random_stats['mean']:.4f} "
        f"std={random_stats['std']:.4f} (n={random_stats['count']})"
    )

    # --- Save + report ---
    combined = {
        "matched": matched_itt,
        "random": random_itt,
        "matched_stats": matched_stats,
        "random_stats": random_stats,
    }
    out_path = os.path.join(PROJECT_ROOT, "data/clip_baseline_scores_coco.json")
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=float)
    logger.info(f"Saved COCO baseline to {out_path}")
    logger.info(
        f"Calibration: random {random_stats['mean']:.3f} vs matched "
        f"{matched_stats['mean']:.3f} (delta {matched_stats['mean'] - random_stats['mean']:+.3f})"
    )



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
            with open(resolve_path(task["json_path"]), "r") as f:
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
                text = seg["text"]  # full text; tokenizer truncates to 77 tokens
                decode_units.append(
                    (gidx, resolve_path(task["video_path"]), seg["start"], seg["end"], text)
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
        try:
            with open(resolve_path(task["json_path"]), "r") as f:
                data = json.load(f)
            segments = data.get("segments", []) if isinstance(data, dict) else data
        except Exception as e:
            logger.warning(f"Failed to load transcript for {task.get('id', '?')}: {e}")
            continue

        valid_segs = [s for s in segments if (s["end"] - s["start"]) > 2.0]
        if not valid_segs:
            valid_segs = segments

        selected = random.sample(valid_segs, min(SAMPLES_PER_VIDEO, len(valid_segs)))

        for seg in selected:
            text = seg["text"]  # full text; tokenizer truncates to 77 tokens
            decode_units.append(
                (len(decode_units), resolve_path(task["video_path"]), seg["start"], seg["end"], text)
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible segment sampling and null baseline "
             "(default: 42). Use the same seed for --mode randomized and "
             "--mode main so the null distribution is stable across runs.",
    )
    parser.add_argument(
        "--coco-annotations",
        type=str,
        default="~/corpora/MSCOCO-2017/captions_val2017.json",
        help="Path to MSCOCO 2017 captions JSON (used by --mode coco)",
    )
    parser.add_argument(
        "--coco-images",
        type=str,
        default="~/corpora/MSCOCO-2017/val2017/",
        help="Root directory of MSCOCO 2017 val images (used by --mode coco)",
    )
    parser.add_argument(
        "--coco-limit",
        type=int,
        default=0,
        help="Cap on number of COCO images to score in --mode coco (0 = all)",
    )
    args = parser.parse_args()

    # Seed RNG for reproducible segment sampling / null baseline.
    # Each mode runs in its own process, so a fixed seed here makes the
    # sampled segments and the randomized null deterministic run-to-run.
    random.seed(args.seed)
    logger.info(f"RNG seed: {args.seed}")

    # Apply performance settings
    _BATCH_SIZE = args.batch_size
    _NUM_WORKERS = args.num_workers
    _USE_FP16 = args.fp16
    _DECODER = args.decoder

    # COCO baseline overrides (module-level assignment rebinds the globals)
    COCO_ANNOTATIONS = args.coco_annotations
    COCO_IMAGES = args.coco_images
    COCO_LIMIT = args.coco_limit

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
