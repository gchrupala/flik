#!/usr/bin/env python
"""
Download a class-stratified subset of the VGGSound TEST split.

Source: HuggingFace mirror ``11hu83/vggsound`` (public, CC-BY-4.0). The mirror
only contains the VGGSound test split and lays files out as::

    video/{video_id}/video.mp4
    audio/{video_id}/audio.wav

where ``video_id = f"{youtube_id}_{start_seconds:06d}"`` (verified: the mirror
``metadata.csv`` column ``file_name`` is ``video/<video_id>/video.mp4`` and the
official VGGSound CSV maps ``(youtube_id, start_seconds) -> label/split``).

The mirror ``metadata.csv`` has columns ``file_name,text`` and already covers
the test split only. The official CSV is fetched as well to confirm split
membership and labels; if it is unreachable the mirror list is used as-is.

This script uses only the standard library + ``requests`` (no torch, no GPU),
so it runs on a Snellius login node.

Output layout (default ``data/vggsound_subset/``)::

    clips/{video_id}/video.mp4
    clips/{video_id}/audio.wav
    clips_manifest.json   # list of {video_id, label, video_path, audio_path}
    download_failures.json

``video_path`` / ``audio_path`` in the manifest are RELATIVE to the manifest
directory (portable), so the eval script resolves them against the manifest
location.

Usage:
    python scripts/download_vggsound_subset.py --classes 40 --per-class 15
    python scripts/download_vggsound_subset.py --classes 2 --per-class 2 --out-dir /tmp/vgg_smoke
"""

import argparse
import csv
import io
import json
import logging
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

HF_BASE = "https://huggingface.co/datasets/11hu83/vggsound/resolve/main"
MIRROR_METADATA_URL = f"{HF_BASE}/metadata.csv"
OFFICIAL_CSV_URL = (
    "https://raw.githubusercontent.com/hche11/VGGSound/master/data/vggsound.csv"
)

_USER_AGENT = "flik-vggsound-downloader/1.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging():
    logger = logging.getLogger("download_vggsound_subset")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


_thread_local = threading.local()


def _thread_session() -> requests.Session:
    """One HTTP session per worker thread (connection reuse)."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        session.headers.update({"User-Agent": _USER_AGENT})
        _thread_local.session = session
    return session


def _get(session: requests.Session, url: str, retries: int = 3,
         timeout: int = 120, stream: bool = False):
    """GET with retries. Raises RuntimeError after the final failure."""
    last = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            r = session.get(url, stream=stream, timeout=timeout)
            if r.status_code == 200:
                return r
            last = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last = str(e)
        if attempt < retries:
            time.sleep(1.5 * attempt)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last}")


def fetch_mirror_metadata(session, logger):
    """Return {video_id: label} from the mirror metadata.csv (test split)."""
    logger.info(f"Fetching mirror metadata: {MIRROR_METADATA_URL}")
    r = _get(session, MIRROR_METADATA_URL)
    text = r.content.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise RuntimeError("mirror metadata.csv is empty")
    logger.info(f"  mirror metadata header: {header}")

    clips = {}
    for row in reader:
        if len(row) < 2:
            continue
        file_name = row[0].strip()
        label = row[1].strip()
        parts = file_name.split("/")
        if len(parts) < 2 or not parts[1]:
            continue
        clips[parts[1]] = label
    if not clips:
        raise RuntimeError("mirror metadata.csv contained no clip rows")
    logger.info(f"  mirror clips: {len(clips)}")
    return clips


def fetch_official_test_ids(session, logger):
    """Return {video_id: label} for official test-split rows, or None on failure.

    Official CSV columns: youtube_id, start_seconds, label, split (no header,
    quoted fields).
    """
    logger.info(f"Fetching official CSV: {OFFICIAL_CSV_URL}")
    try:
        r = _get(session, OFFICIAL_CSV_URL)
    except Exception as e:
        logger.warning(f"  official CSV unavailable ({e}); using mirror list only")
        return None
    reader = csv.reader(io.StringIO(r.content.decode("utf-8", errors="replace")))
    test = {}
    for row in reader:
        if len(row) < 4:
            continue
        youtube_id, start_raw, label, split = row[0], row[1], row[2], row[3]
        if split.strip() != "test":
            continue
        try:
            start = int(float(start_raw))
        except ValueError:
            continue
        test[f"{youtube_id}_{start:06d}"] = label.strip()
    logger.info(f"  official test clips: {len(test)}")
    return test


def _ffprobe_ok(path: str, timeout: int = 120) -> bool:
    """True if ffprobe can read a duration from the file. No-op without ffprobe."""
    if shutil.which("ffprobe") is None:
        return True
    try:
        res = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=timeout,
        )
        return res.returncode == 0 and res.stdout.strip() not in ("", "N/A")
    except Exception:
        return False


def download_file(session, url, dest, retries=3, timeout=300):
    """Download url -> dest idempotently. Returns 'skipped' or 'downloaded'."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skipped"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    last = "unknown error"
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            if os.path.getsize(tmp) == 0:
                raise RuntimeError("downloaded 0 bytes")
            os.replace(tmp, dest)
            return "downloaded"
        except Exception as e:
            last = e
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            if attempt < retries:
                time.sleep(1.0 * attempt)
    raise RuntimeError(f"failed after {retries} attempts: {last}")


def process_clip(task, out_dir, check_ffprobe):
    """Download video+audio for one clip. Returns a result dict."""
    index, video_id, label = task
    clip_dir = os.path.join(out_dir, "clips", video_id)
    vpath = os.path.join(clip_dir, "video.mp4")
    apath = os.path.join(clip_dir, "audio.wav")
    try:
        session = _thread_session()
        download_file(
            session, f"{HF_BASE}/video/{video_id}/video.mp4", vpath
        )
        download_file(
            session, f"{HF_BASE}/audio/{video_id}/audio.wav", apath
        )
    except Exception as e:
        return {"index": index, "video_id": video_id, "label": label,
                "ok": False, "error": f"download: {e}"}

    if check_ffprobe:
        for p in (vpath, apath):
            if not _ffprobe_ok(p):
                # Remove the corrupt pair so a re-run retries it.
                for q in (vpath, apath):
                    try:
                        os.remove(q)
                    except OSError:
                        pass
                return {"index": index, "video_id": video_id, "label": label,
                        "ok": False, "error": f"ffprobe failed: {os.path.basename(p)}"}

    return {"index": index, "video_id": video_id, "label": label, "ok": True}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download a class-stratified subset of the VGGSound test split."
    )
    p.add_argument("--classes", type=int, default=40,
                   help="Number of most-frequent classes to include (default: 40).")
    p.add_argument("--per-class", type=int, default=15,
                   help="Target number of successfully downloaded clips per class "
                        "(default: 15).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for clip sampling (default: 42).")
    p.add_argument("--out-dir", type=str, default="data/vggsound_subset/",
                   help="Output directory (default: data/vggsound_subset/).")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel download workers (default: 8).")
    p.add_argument("--skip-ffprobe", action="store_true",
                   help="Skip the post-download ffprobe decodability check.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger = setup_logging()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": _USER_AGENT})

    # --- metadata ---------------------------------------------------------
    mirror = fetch_mirror_metadata(session, logger)
    official_test = fetch_official_test_ids(session, logger)

    if official_test is not None:
        # Keep only clips present in BOTH the mirror file list and the official
        # test split. Use the official label (authoritative).
        candidates = {
            vid: official_test[vid]
            for vid in mirror
            if vid in official_test
        }
        logger.info(
            f"Intersected mirror ({len(mirror)}) with official test "
            f"({len(official_test)}): {len(candidates)} usable test clips"
        )
    else:
        # Mirror is test-only; fall back to its labels.
        candidates = dict(mirror)
        logger.warning(
            f"Using mirror-only list ({len(candidates)} clips, test split assumed)"
        )

    if not candidates:
        logger.error("No candidate clips found; aborting.")
        return 1

    # --- class selection (deterministic) ---------------------------------
    counts = Counter(candidates.values())
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    n_classes = min(args.classes, len(ranked))
    if n_classes < args.classes:
        logger.warning(
            f"Requested {args.classes} classes but only {len(ranked)} exist; "
            f"using {n_classes}."
        )
    chosen = [label for label, _ in ranked[:n_classes]]
    logger.info(f"Selected {len(chosen)} classes (most frequent first): {chosen}")

    # 25% extra clips per class as a buffer against download failures.
    n_buffer = args.per_class + math.ceil(args.per_class * 0.25)
    logger.info(
        f"Target {args.per_class} clips/class; sampling {n_buffer}/class "
        f"(25% buffer)"
    )

    by_class = defaultdict(list)
    for vid, label in candidates.items():
        by_class[label].append(vid)

    rng = random.Random(args.seed)
    tasks = []
    for label in chosen:
        vids = sorted(by_class[label])
        rng.shuffle(vids)
        for vid in vids[:n_buffer]:
            tasks.append((len(tasks), vid, label))

    logger.info(f"Queued {len(tasks)} clip downloads ({len(chosen)} classes)")

    # --- parallel download ------------------------------------------------
    check_ffprobe = not args.skip_ffprobe
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(process_clip, t, out_dir, check_ffprobe): t
            for t in tasks
        }
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 25 == 0 or done == len(futures):
                logger.info(f"  {done}/{len(futures)} attempted")

    # Deterministic order, then cap to per-class successes (buffer fills gaps).
    results.sort(key=lambda r: r["index"])
    per_class_count = Counter()
    manifest = []
    failures = []
    skipped = 0
    for r in results:
        if not r["ok"]:
            failures.append({"video_id": r["video_id"], "label": r["label"],
                             "error": r["error"]})
            continue
        label = r["label"]
        if per_class_count[label] >= args.per_class:
            skipped += 1
            continue
        per_class_count[label] += 1
        manifest.append({
            "video_id": r["video_id"],
            "label": label,
            "video_path": os.path.join("clips", r["video_id"], "video.mp4"),
            "audio_path": os.path.join("clips", r["video_id"], "audio.wav"),
        })

    manifest_path = os.path.join(out_dir, "clips_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    failures_path = os.path.join(out_dir, "download_failures.json")
    with open(failures_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    logger.info("=" * 60)
    logger.info(f"Classes:            {len(chosen)}")
    logger.info(f"Requested/class:    {args.per_class}")
    logger.info(f"Downloaded clips:   {len(manifest)}")
    logger.info(f"Extra (capped):     {skipped}")
    logger.info(f"Failures:           {len(failures)}")
    logger.info(f"Manifest:           {manifest_path}")
    logger.info(f"Failures log:       {failures_path}")
    missing = [c for c in chosen if per_class_count[c] < args.per_class]
    if missing:
        logger.warning(
            f"{len(missing)} classes below target: "
            f"{[(c, per_class_count[c]) for c in missing]}"
        )
    logger.info("=" * 60)

    return 0 if manifest else 1


if __name__ == "__main__":
    sys.exit(main())
