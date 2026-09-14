#!/usr/bin/env python3
"""Download the QC pilot films from Internet Archive.

Reads a plan JSON produced on the devbox: a list of
{"video_id", "identifier", "filename", "video_path"} entries and downloads
https://archive.org/download/<identifier>/<filename> to the location the
training manifest expects (<PROJECT_ROOT>/data/out/<video_id>.mp4).

Idempotent: existing non-empty files are skipped.
"""

import argparse
import json
import logging
import os
import sys
import urllib.parse

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("download_qc_films")

IA_BASE = "https://archive.org/download"


def download_one(entry: dict, project_root: str, retries: int = 3) -> bool:
    """Download one film file. Returns True on success."""
    rel = entry["video_path"]  # e.g. data/out/<video_id>.mp4
    dest = os.path.join(project_root, rel)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        logger.info("SKIP (exists): %s", rel)
        return True

    url = "{}/{}/{}".format(
        IA_BASE,
        urllib.parse.quote(entry["identifier"]),
        urllib.parse.quote(entry["filename"]),
    )
    tmp = dest + ".part"
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            os.replace(tmp, dest)
            size_mb = os.path.getsize(dest) / 1e6
            logger.info("OK: %s (%.0f MB)", rel, size_mb)
            return True
        except Exception as e:
            logger.warning("attempt %d failed for %s: %s", attempt, url, e)
    if os.path.exists(tmp):
        os.remove(tmp)
    return False


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", default="data/qc_pilot_downloads.json")
    args = p.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(args.plan, "r", encoding="utf-8") as f:
        plan = json.load(f)

    logger.info("Downloading %d pilot films", len(plan))
    ok = sum(download_one(e, project_root) for e in plan)
    logger.info("Done: %d/%d succeeded", ok, len(plan))
    sys.exit(0 if ok == len(plan) else 1)


if __name__ == "__main__":
    main()
