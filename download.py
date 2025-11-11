from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional

from internetarchive import get_session


# Preferred formats in descending order (first found wins)
PREFERRED_FORMATS: List[str] = ["h.264", "512Kb MPEG4", "256Kb MPEG4"]


def _configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_manifest(json_path: Path) -> List[Dict[str, Any]]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Manifest JSON must be a list of film objects.")
    return data


def _iter_files_by_preference(item, formats: Iterable[str]):
    """
    Yield the first non-empty list of files matching the given formats preference.
    """
    for fmt in formats:
        try:
            files = list(item.get_files(formats=[fmt]))
        except Exception as e:
            logging.warning("Failed to enumerate files for format '%s': %s", fmt, e)
            continue
        if files:
            logging.info("Selected format: %s (%d files found)", fmt, len(files))
            return files
    return []


def _safe_filename(obj) -> Optional[str]:
    # internetarchive file objects typically have a `.name` attribute
    return getattr(obj, "name", None)


def download(
    shuffle: bool = False,
    manifest_json: str = "data/in/cinedantan_movies.json",
    out_dir: str = "data/out/cinedantan",
    formats: Iterable[str] = PREFERRED_FORMATS,
    max_items: Optional[int] = None,
    delay_seconds: float = 0.0,
    dry_run: bool = False,
    log_level: int = logging.INFO,
) -> None:
    """
    Download Internet Archive film files using a preferred format order.

    Parameters
    ----------
    shuffle : bool
        If True, randomly permute the manifest entries before downloading.
    manifest_json : str
        Path to a JSON file containing a list of dicts; each dict must have an 'identifier'.
    out_dir : str
        Base directory where films will be stored as {out_dir}/{identifier}/.
    formats : Iterable[str]
        Ordered list of preferred formats; the first non-empty match is used per item.
    max_items : Optional[int]
        If set, limit processing to the first N items (after shuffling if enabled).
    delay_seconds : float
        Optional delay between items (politeness / rate-limiting).
    dry_run : bool
        If True, do not download—only log what would happen.
    log_level : int
        Logging verbosity (e.g., logging.INFO).
    """
    _configure_logging(log_level)

    manifest_path = Path(manifest_json)
    base_out = Path(out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    data = _load_manifest(manifest_path)

    if shuffle:
        random.shuffle(data)

    if max_items is not None:
        data = data[:max_items]

    session = get_session()
    processed = 0
    skipped = 0
    downloaded = 0

    for film in data:
        identifier = film.get("identifier")
        if not identifier:
            logging.warning("Skipping entry without 'identifier': %r", film)
            skipped += 1
            continue

        dest_dir = base_out / identifier
        dest_dir.mkdir(parents=True, exist_ok=True)

        logging.info("Processing %s", identifier)

        try:
            item = session.get_item(identifier)
        except Exception as e:
            logging.error("Error fetching item '%s': %s", identifier, e)
            skipped += 1
            continue

        files = _iter_files_by_preference(item, formats)
        if not files:
            logging.info("No files found in preferred formats for %s", identifier)
            skipped += 1
            continue

        # Download each file if not already present
        item_downloaded = 0
        for f in files:
            fname = _safe_filename(f)
            if not fname:
                logging.warning("Unknown file name for an entry in %s; skipping.", identifier)
                continue

            target_path = dest_dir / Path(fname).name
            if target_path.exists():
                logging.info("Already exists, skipping: %s", target_path)
                continue

            logging.info("Downloading -> %s", target_path)
            if not dry_run:
                try:
                    f.download(destdir=str(dest_dir))
                    item_downloaded += 1
                    downloaded += 1
                except KeyboardInterrupt:
                    logging.warning("Interrupted by user.")
                    return
                except Exception as e:
                    logging.error("Failed to download %s (%s): %s", identifier, fname, e)

        if item_downloaded == 0:
            logging.info("No new files downloaded for %s (all present or failed).", identifier)

        processed += 1

        if delay_seconds > 0:
            time.sleep(delay_seconds)

    logging.info("Done. Processed: %d | Downloaded: %d files | Skipped: %d", processed, downloaded, skipped)

if __name__ == '__main__':
    download(max_items=None, dry_run=False)
    
