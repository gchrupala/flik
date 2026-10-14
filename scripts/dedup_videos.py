#!/usr/bin/env python
"""De-duplicate the cinedantan film corpus at the *movie* level.

Why this exists
---------------
The train/held-out split is per ``video_id``. If one movie appears in the
corpus under several ids (a ``_512kb`` derivative, a DVD ``VTS_*`` part, a
compilation-disc part, an ``_x264`` re-encode), those ids can land on opposite
sides of the split and leak across it. Cosmetic de-duplication is not enough;
the canonical key must group at the movie level.

Two distinct artifacts
----------------------
The family grouping and the training manifest answer different questions and
must not be confused:

- **Split safety** — "which ids must never be split across train/val?" This is
  the *family* grouping ``id -> movie_key``. It deliberately over-groups: a
  multi-part serial, an anthology item, or a compilation item collapses to one
  movie key because its pieces are related content. Emitted as
  ``data/movie_map.json`` (a flat ``{id: movie_key}`` object).
- **Training data** — "which ids are redundant copies of the same file?" This
  is the *training manifest*. It drops only genuine duplicate copies
  (``duplicate_derivative`` and DVD/compilation artefacts such as ``VTS_*``,
  ``M4V*``, ``NOIR_DISC_*`` that duplicate the item's main film) and KEEPS
  distinct pieces of content (serial chapters, ``*of*`` split parts,
  anthologies). Selected with ``--drop-policy`` (default ``genuine``), emitted
  as ``data/manifest_deduped.json``. The family-collapsed list is kept
  separately as ``data/manifest_deduped_family.json``.

Two passes
----------
1. **Metadata pass (always runs).** Maps every used id (a filename stem) to
   its parent index item and to a canonical movie key:
   - strip derivative suffixes (``_512kb``/``_256kb``/``_64kb``/``_ipod_*``)
     and the video extension;
   - detect DVD/compilation artefacts (``VTS_*``, ``M4V*``, ``NOIR_DISC_*``)
     and split/chapter artefacts (``part*``, ``*of*``, ``chNN``);
   - rank derivatives by quality (h.264 > HiRes MPEG4 > MPEG4 > 512Kb >
     256Kb > 64Kb) and select the best copy per movie.
2. **Content backstop (opt-in, windowed).** Cross-item duplicates cannot be
   found from metadata. With ``--phash`` the script samples ~8 frames per film,
   computes a pHash, and (with ``--audio``) an audio fingerprint, then clusters
   films whose frames/audio match. Full-corpus runs are slow; use
   ``--films-from-file`` to run a window on a machine that holds the videos.
   Where metadata cannot decide (for example Green Hornet episodes named under
   two different items), this pass is the arbiter and it requires the film
   files.

Outputs
-------
- ``data/dedup_report.json``  — clusters, per-policy dropped ids, kept id,
  reason, and the unresolved cross-item candidates.
- ``data/movie_map.json`` — split-safety map, flat ``{id: movie_key}``.
- ``data/manifest_deduped.json`` — training manifest for ``--drop-policy``
  (default ``genuine``), same schema as the input.
- ``data/manifest_deduped_family.json`` — family-collapsed manifest (drop all
  non-kept members).
- ``data/manifest_deduped_genuine.json`` — genuine-duplicates-only manifest.

Usage
-----
    # Metadata only (fast; works without the film files)
    uv run --extra cpu python -m scripts.dedup_videos

    # Content pass over the whole corpus (slow; needs the videos)
    uv run --extra cu128 python -m scripts.dedup_videos --phash --audio

    # Windowed content pass over a list of films
    uv run --extra cu128 python -m scripts.dedup_videos --phash --audio \
        --films-from-file data/dedup_window.json

    # Unit self-test
    uv run --extra cpu python -m scripts.dedup_videos --self-test
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from urllib.parse import unquote

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.CONSTANTS import PROJECT_ROOT
from src.utils.paths import resolve_path

# Lazy optional imports: cv2/subprocess are only needed for the content pass.
cv2 = None
np = None


# ===========================================================================
# Constants
# ===========================================================================

DEFAULT_INPUT = "data/filtered_manifest.json"
DEFAULT_INDEX = "data/in/cinedantan_movies.json"
DEFAULT_REPORT = "data/dedup_report.json"
DEFAULT_OUTPUT = "data/manifest_deduped.json"
DEFAULT_MOVIE_MAP = "data/movie_map.json"
DEFAULT_OUTPUT_FAMILY = "data/manifest_deduped_family.json"
DEFAULT_OUTPUT_GENUINE = "data/manifest_deduped_genuine.json"

# Genuine duplicate copies (safe to drop for training). DVD/compilation parts
# duplicate the item's main film.
GENUINE_PART_REASONS = {"dvd_vts", "dvd_video_ts", "dvd_m4v", "compilation_disc"}
# Reasons that mark a distinct piece of content (must be kept for training).
CONTENT_PIECE_REASONS = {"split_part", "disc_part", "chapter", "same_item_alternate"}
# All reasons that mark a genuine duplicate.
GENUINE_REASONS = {"duplicate_derivative", "content_duplicate"} | GENUINE_PART_REASONS
# Valid values for --drop-policy.
DROP_POLICIES = ("family", "genuine")

# A used id that maps to no index item becomes its own singleton group.
SINGLETON_PREFIX = "__singleton__:"
# A film passed via --films-from-file that is not in the manifest.
EXTERNAL_PREFIX = "__external__:"

VIDEO_EXTENSIONS = (
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v",
    ".mpg", ".mpeg", ".flv", ".ogv", ".wmv",
)
_EXT_RE = re.compile(r"\.(" + "|".join(e.lstrip(".") for e in VIDEO_EXTENSIONS) + r")$", re.IGNORECASE)

# Derivative suffixes observed in the corpus (case-insensitive). Applied in a
# loop so combined suffixes such as ``_ipod_512kb`` fully collapse.
_DERIVATIVE_SUFFIX_RE = re.compile(
    r"(?:_(?:ipod|512kb|256kb|64kb|256k|64k|hires|hi-res|hi_res))+$", re.IGNORECASE
)

# Quality ordering. Lower rank = better. Mirrors the task's required order.
_FORMAT_RANK = {
    "h.264": 0,
    "h.264 MPEG4": 0,
    "HiRes MPEG4": 1,
    "MPEG4": 2,
    "MPEG2": 2,
    "MPEG-2": 2,
    "WebM": 2,
    "Matroska": 2,
    "Quicktime": 2,
    "Flash Video": 3,
    "Flash": 3,
    "512Kb MPEG4": 4,
    "256Kb MPEG4": 5,
    "64Kb MPEG4": 6,
}
_RANK_LABEL = {
    0: "h.264", 1: "HiRes MPEG4", 2: "MPEG4", 3: "Flash",
    4: "512Kb", 5: "256Kb", 6: "64Kb",
}

# Non-video formats that must never be treated as a movie derivative.
_NON_VIDEO_FORMATS = {
    "Archive BitTorrent", "BitTorrent", "BitTorrentContents", "Thumbnail",
    "Metadata", "Text", "MP3", "PNG", "Animated GIF",
}

# Part / artefact detection patterns, checked in order.
_PART_PATTERNS = [
    ("dvd_vts", re.compile(r"^vts_\d+_\d+$", re.IGNORECASE)),
    ("dvd_video_ts", re.compile(r"^video_ts", re.IGNORECASE)),
    ("dvd_m4v", re.compile(r"^m4v\d+", re.IGNORECASE)),
    ("compilation_disc", re.compile(r"^(?:noir_disc|disc|cd)\d*", re.IGNORECASE)),
    ("disc_part", re.compile(r"^(?:part|pt)\d+", re.IGNORECASE)),
    ("chapter", re.compile(r"(?:^|_)(?:ch|chapter)0*\d+", re.IGNORECASE)),
    ("split_part", re.compile(r"^.*[-_]\d+of\d+(?:_|$)", re.IGNORECASE)),
]

# Families used to break ties when a stem is shared by several index items.
_VTS_FAMILY_RE = re.compile(r"^(vts_\d+)_\d+$", re.IGNORECASE)
_PART_FAMILY_RE = re.compile(r"^(.+?)_(?:ch|chapter|part|pt)0*\d+$", re.IGNORECASE)
_SPLIT_FAMILY_RE = re.compile(r"^(.+?)[-_]\d+of\d+$", re.IGNORECASE)


# ===========================================================================
# Logging
# ===========================================================================

def setup_logging(verbose: bool = False) -> logging.Logger:
    log_dir = os.path.join(PROJECT_ROOT, "logdir")
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("dedup_videos")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(os.path.join(log_dir, "dedup_videos.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


# ===========================================================================
# 1. Stem normalisation, part detection, quality
# ===========================================================================

def normalize_stem(name: str) -> str:
    """Return the canonical movie stem for a filename or used id.

    Strips the path, URL-encoding, video extension, and any stacked derivative
    suffixes (``_512kb``/``_256kb``/``_64kb``/``_ipod_*``), then lowercases.

    Examples:
        ``/Constantine_and_the_Cross_256kb.mp4`` -> ``constantine_and_the_cross``
        ``First_Spaceship_on_Venus_ipod_512kb``  -> ``first_spaceship_on_venus``
        ``M4V10005``                             -> ``m4v10005``
    """
    if not name:
        return ""
    s = unquote(str(name))
    s = os.path.basename(s.replace("\\", "/"))
    s = _EXT_RE.sub("", s)
    s = s.strip().lower()
    prev = None
    while prev != s:
        prev = s
        s = _DERIVATIVE_SUFFIX_RE.sub("", s)
    return s


def classify_part(stem: str) -> str | None:
    """Return the artefact kind if ``stem`` is a DVD/compilation part, else None.

    Examples:
        ``VTS_01_2``      -> ``dvd_vts``
        ``M4V10005``      -> ``dvd_m4v``
        ``NOIR_DISC_11``  -> ``compilation_disc``
    """
    s = normalize_stem(stem)
    if not s:
        return None
    for kind, pattern in _PART_PATTERNS:
        if pattern.match(s):
            return kind
    return None


def part_family(stem: str) -> str:
    """Return the family key shared by parts of the same work.

    ``VTS_01_1`` and ``VTS_01_2`` -> ``vts_01``; a serial chapter
    ``AdventuresOfCaptainMarvel_ch03`` -> ``adventuresofcaptainmarvel``.
    """
    s = normalize_stem(stem)
    m = _VTS_FAMILY_RE.match(s)
    if m:
        return m.group(1)
    m = _PART_FAMILY_RE.match(s)
    if m:
        return m.group(1)
    m = _SPLIT_FAMILY_RE.match(s)
    if m:
        return m.group(1)
    return s


def _format_rank(fmt: str | None) -> int | None:
    if not fmt:
        return None
    return _FORMAT_RANK.get(fmt)


def infer_quality(used_id: str, fmt_hint: str | None = None) -> tuple[int, str]:
    """Rank a used id's copy quality. Returns ``(rank, label)``; lower is better.

    Prefers the explicit ``aoFiles`` format when given; otherwise infers the
    rank from the id's suffix. Unknown copies are treated as ``MPEG4``.
    """
    rank = _format_rank(fmt_hint)
    label = fmt_hint
    if rank is None:
        low = used_id.lower()
        if low.endswith(("_h264", "_x264")):
            rank, label = 0, "h.264 (suffix)"
        elif low.endswith("_512kb"):
            rank, label = 4, "512Kb MPEG4 (suffix)"
        elif low.endswith(("_256kb", "_256k")):
            rank, label = 5, "256Kb MPEG4 (suffix)"
        elif low.endswith(("_64kb", "_64k")):
            rank, label = 6, "64Kb MPEG4 (suffix)"
        elif "ipod" in low:
            rank, label = 2, "MPEG4 (ipod)"
        else:
            rank, label = 2, "MPEG4 (assumed)"
    return rank, (label or _RANK_LABEL.get(rank, "?"))


# ===========================================================================
# 2. Index loading and id -> item mapping
# ===========================================================================

def _raw_stem(url: str) -> str:
    """Basename without extension, lowercased, URL-decoded (no suffix strip)."""
    if not url:
        return ""
    s = unquote(str(url))
    s = os.path.basename(s.replace("\\", "/"))
    s = _EXT_RE.sub("", s)
    return s.strip().lower()


class Index:
    """The cinedantan source index, pre-processed for fast lookups."""

    def __init__(self, items: list):
        self.items = items
        self.identifiers = set()
        self.item_files = collections.defaultdict(set)   # identifier -> {norm stem}
        self.stem_to_items = collections.defaultdict(set)  # norm stem -> {identifier}
        self.format_by_raw = {}  # raw lower stem -> best format label

        for it in items:
            ident = it.get("identifier")
            if not ident:
                continue
            self.identifiers.add(ident)
            for f in it.get("aoFiles") or []:
                if not isinstance(f, dict):
                    continue
                url = f.get("url")
                if not url:
                    continue
                fmt = f.get("format")
                if fmt in _NON_VIDEO_FORMATS:
                    continue  # torrents, thumbnails, metadata are not movies
                ns = normalize_stem(url)
                if ns:
                    self.item_files[ident].add(ns)
                    self.stem_to_items[ns].add(ident)
                raw = _raw_stem(url)
                if not raw:
                    continue
                cand = self.format_by_raw.get(raw)
                new_rank = _format_rank(fmt)
                old_rank = _format_rank(cand)
                if new_rank is not None and (old_rank is None or new_rank < old_rank):
                    self.format_by_raw[raw] = fmt

    @classmethod
    def load(cls, path: str) -> Index:
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        if isinstance(items, dict):
            items = list(items.values())
        return cls(items)

    def candidates(self, used_id: str) -> set:
        """Index items that could own ``used_id``."""
        c = set(self.stem_to_items.get(normalize_stem(used_id), ()))
        if used_id in self.identifiers:
            c.add(used_id)
        return c

    def fmt_hint(self, used_id: str) -> str | None:
        return self.format_by_raw.get(used_id.lower())


def build_mapping(
    used_ids: list,
    index: Index,
) -> tuple[dict, list]:
    """Map every used id to a movie-level group key (parent item identifier).

    A stem shared by several items (e.g. ``VTS_01_1`` appears under both
    ``Martha_Ivers`` and ``Man_Who_Cheated_Himself``) is resolved by family
    support: the item that contributes the most other parts of the same DVD
    family wins; remaining ties break on identifier. Returns
    ``(mapping, ambiguities)``.
    """
    cands = {i: index.candidates(i) for i in used_ids}

    # Family support: how many *unambiguously* owned ids each item has in a
    # family. Used only to break ties for ambiguous ids.
    family_support = collections.defaultdict(collections.Counter)
    for i, c in cands.items():
        if len(c) == 1:
            family_support[part_family(i)][next(iter(c))] += 1

    mapping = {}
    ambiguities = []
    for i in used_ids:
        c = cands[i]
        if not c:
            mapping[i] = SINGLETON_PREFIX + i
        elif len(c) == 1:
            mapping[i] = next(iter(c))
        else:
            fam = part_family(i)
            best = sorted(c, key=lambda it: (-family_support[fam][it], it))[0]
            mapping[i] = best
            ambiguities.append({
                "id": i,
                "candidates": sorted(c),
                "chosen": best,
                "family": fam,
            })
    return mapping, ambiguities


# ===========================================================================
# 3. Grouping and selection
# ===========================================================================

def select_kept(members: list, index: Index) -> tuple[str, int, str]:
    """Pick the best copy from a movie group.

    Order: lower quality rank, then non-part stem, then shorter id, then id.
    Returns ``(kept_id, rank, format_label)``.
    """
    scored = []
    for mid in members:
        rank, label = infer_quality(mid, index.fmt_hint(mid))
        is_part = 1 if classify_part(mid) else 0
        scored.append((rank, is_part, len(mid), mid, label))
    scored.sort()
    rank, _is_part, _len, kept, label = scored[0]
    return kept, rank, label


def member_reason(member: str, kept: str) -> str:
    """Classify why ``member`` is not the kept copy of its group.

    Returns the specific artefact kind (``dvd_vts``, ``dvd_m4v``, ``split_part``,
    ``chapter``, ...), ``duplicate_derivative`` for a same-stem copy at a
    different bitrate, or ``same_item_alternate`` for any other in-item variant.
    """
    if member == kept:
        return "kept"
    part = classify_part(member)
    if part:
        return part
    if normalize_stem(member) == normalize_stem(kept):
        return "duplicate_derivative"
    return "same_item_alternate"


def is_genuine_drop(reason: str, origin: str, has_main: bool) -> bool:
    """True if a member is a redundant copy (drop for training), else content.

    ``has_main`` is True when the group contains at least one non-artefact
    member, i.e. the full film that the DVD/compilation parts duplicate. If the
    group has no main film (for example a film split into ``*of4`` parts), the
    parts are the content and must NOT be dropped.
    """
    if origin == "content":
        return True  # the content pass proved the members are the same film
    if reason == "duplicate_derivative":
        return True
    if reason in GENUINE_PART_REASONS:
        return has_main
    return False


def make_group(key: str, members: list, origin: str, index: Index) -> dict:
    """Build one movie group with per-policy drop decisions."""
    kept, rank, label = select_kept(members, index)
    reasons = {m: member_reason(m, kept) for m in members}
    has_main = any(classify_part(m) is None for m in members)

    dropped_family = [m for m in members if m != kept]
    dropped_genuine = [
        m for m in members
        if m != kept and is_genuine_drop(reasons[m], origin, has_main)
    ]
    return {
        "key": key,
        "kept_id": kept,
        "kept_rank": rank,
        "kept_format": label,
        "n_members": len(members),
        "members": list(members),
        "reasons": reasons,
        "has_main": has_main,
        "dropped_family": dropped_family,
        "dropped_genuine": dropped_genuine,
        "origin": origin,
    }


def build_metadata_groups(used_ids: list, index: Index, origin: str = "metadata"):
    """Return ``(groups, mapping, ambiguities)`` for the metadata-only pass."""
    mapping, ambiguities = build_mapping(used_ids, index)

    by_key = collections.defaultdict(list)
    for i in used_ids:
        by_key[mapping[i]].append(i)

    groups = {key: make_group(key, members, origin, index)
              for key, members in by_key.items()}
    return groups, mapping, ambiguities


# ===========================================================================
# 4. Content backstop: pHash + audio fingerprint
# ===========================================================================

def _require_np_cv2():
    global cv2, np
    if np is None:
        try:
            import numpy as _np
            np = _np
        except Exception as e:  # pragma: no cover
            raise RuntimeError("numpy is required for the content pass") from e
    if cv2 is None:
        try:
            import cv2 as _cv2  # type: ignore
            cv2 = _cv2
        except Exception as e:  # pragma: no cover
            raise RuntimeError("opencv-python (cv2) is required for --phash") from e
    return np, cv2


def sample_frames(path: str, n: int) -> list:
    """Decode ``n`` evenly spaced frames (BGR arrays)."""
    np_mod, cv2_mod = _require_np_cv2()
    cap = cv2_mod.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    total = int(cap.get(cv2_mod.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    if total > 0:
        targets = np_mod.linspace(0, max(total - 1, 0), n).astype(int)
        for idx in targets:
            cap.set(cv2_mod.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
    cap.release()
    if not frames:
        # Fallback: sequential read for containers that lie about frame count.
        cap = cv2_mod.VideoCapture(path)
        got = 0
        while got < n:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            got += 1
        cap.release()
    return frames


def phash_bits(gray) -> object:
    """64-bit pHash: 32x32 gray, DCT, top-left 8x8, threshold at the median."""
    np_mod, cv2_mod = _require_np_cv2()
    g = cv2_mod.resize(gray, (32, 32), interpolation=cv2_mod.INTER_AREA)
    g = g.astype(np_mod.float32)
    d = cv2_mod.dct(g)[:8, :8]
    med = np_mod.median(d[1:])  # exclude DC term
    return (d > med).flatten()


def frame_hashes(path: str, n: int = 8) -> list:
    """Return up to ``n`` pHash bit-vectors for a video."""
    out = []
    for frame in sample_frames(path, n):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out.append(phash_bits(gray))
    return out


def _popcount_pair_matrix(a, b):
    """Hamming distances between two arrays of 64-bit hashes (uint64)."""
    np_mod, _ = _require_np_cv2()
    if a.size == 0 or b.size == 0:
        return np_mod.zeros((a.size, b.size), dtype=np_mod.int32)
    x = np_mod.bitwise_xor(a[:, None], b[None, :])
    bits = np_mod.unpackbits(x.view(np_mod.uint8).reshape(-1, 8), axis=1)
    return bits.sum(axis=1).reshape(a.size, b.size).astype(np_mod.int32)


def decode_audio(path: str, sr: int = 8000):
    """Decode mono PCM with ffmpeg. Returns float32 in [-1, 1]."""
    np_mod, _ = _require_np_cv2()
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-f", "s16le", "-ac", "1", "-ar", str(sr), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(
            f"ffmpeg audio decode failed for {path}: "
            f"{proc.stderr.decode('utf-8', 'ignore')[:200]}"
        )
    return np_mod.frombuffer(proc.stdout, dtype=np_mod.int16).astype(np_mod.float32) / 32768.0


def audio_signature(path: str, sr: int = 8000, n_bands: int = 24, bins: int = 16):
    """Coarse audio fingerprint: log band energies, averaged into fixed time bins.

    Robust to lossy re-encoding and level changes; returns a ``(bins, n_bands)``
    float array (z-normalised per band).
    """
    np_mod, _ = _require_np_cv2()
    x = decode_audio(path, sr=sr)
    frame, hop = 4096, 4096
    n_frames = max(1, 1 + (len(x) - frame) // hop)
    idx = np_mod.arange(frame)[None, :] + hop * np_mod.arange(n_frames)[:, None]
    spec = np_mod.fft.rfft(x[idx] * np_mod.hanning(frame), axis=1)
    power = np_mod.abs(spec) ** 2
    freqs = np_mod.fft.rfftfreq(frame, 1.0 / sr)
    edges = np_mod.unique(np_mod.geomspace(60, sr / 2, n_bands + 1).astype(int))
    band = np_mod.zeros((n_frames, len(edges) - 1), dtype=np_mod.float32)
    for i in range(len(edges) - 1):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        if mask.any():
            band[:, i] = power[:, mask].mean(axis=1)
    band = np_mod.log1p(band)
    # Average into fixed time bins so signatures are length-independent.
    edges_t = np_mod.linspace(0, n_frames, bins + 1).astype(int)
    out = np_mod.zeros((bins, band.shape[1]), dtype=np_mod.float32)
    for i in range(bins):
        lo, hi = edges_t[i], max(edges_t[i] + 1, edges_t[i + 1])
        out[i] = band[lo:hi].mean(axis=0)
    out = (out - out.mean(axis=0, keepdims=True)) / (out.std(axis=0, keepdims=True) + 1e-6)
    return out


def audio_similarity(sig_a, sig_b) -> float:
    """Cosine similarity of two flattened audio signatures."""
    np_mod, _ = _require_np_cv2()
    a = np_mod.asarray(sig_a, dtype=np_mod.float32).ravel()
    b = np_mod.asarray(sig_b, dtype=np_mod.float32).ravel()
    denom = float(np_mod.linalg.norm(a) * np_mod.linalg.norm(b)) + 1e-9
    return float(np_mod.dot(a, b) / denom)


def _cache_path(cache_dir: str, film_id: str, path: str) -> str:
    try:
        st = os.stat(path)
        key = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|{film_id}"
    except OSError:
        key = f"{os.path.abspath(path)}|{film_id}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_dir, digest + ".json")


def compute_film_hashes(
    film: dict,
    do_phash: bool,
    do_audio: bool,
    phash_sample: int,
    cache_dir: str | None,
) -> dict:
    """Compute pHash frame hashes and/or an audio signature for one film."""
    path = film["abs_path"]
    result = {"id": film["id"], "path": path, "phash": None, "audio": None, "error": None}

    if cache_dir and os.path.exists(_cache_path(cache_dir, film["id"], path)):
        try:
            with open(_cache_path(cache_dir, film["id"], path), "r", encoding="utf-8") as f:
                cached = json.load(f)
            result["phash"] = cached.get("phash")
            result["audio"] = cached.get("audio")
            return result
        except Exception:
            pass

    try:
        if do_phash:
            hashes = frame_hashes(path, phash_sample)
            # Store as 16-char hex strings (64 bits) for a compact, portable report.
            result["phash"] = [
                "".join(str(int(b)) for b in bits) for bits in hashes
            ]
        if do_audio:
            sig = audio_signature(path)
            result["audio"] = [round(float(v), 4) for v in sig.ravel()]
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    if cache_dir and result["error"] is None:
        os.makedirs(cache_dir, exist_ok=True)
        try:
            with open(_cache_path(cache_dir, film["id"], path), "w", encoding="utf-8") as f:
                json.dump({"phash": result["phash"], "audio": result["audio"]}, f)
        except Exception:
            pass
    return result


def _bits_to_uint64(hex_or_bits):
    """Accept a 64-char '0'/'1' string, hex string, or bit sequence."""
    np_mod, _ = _require_np_cv2()
    if isinstance(hex_or_bits, str):
        if len(hex_or_bits) == 64 and set(hex_or_bits) <= {"0", "1"}:
            val = int(hex_or_bits, 2)
        else:
            val = int(hex_or_bits, 16)
    else:
        val = 0
        for b in hex_or_bits:
            val = (val << 1) | (int(b) & 1)
    return np_mod.uint64(val)


def phash_similarity(a_bits: list, b_bits: list, threshold: int):
    """Return ``(is_match, min_dist, n_matches)`` for two films' frame hashes."""
    np_mod, _ = _require_np_cv2()
    a = np_mod.array([_bits_to_uint64(h) for h in a_bits], dtype=np_mod.uint64)
    b = np_mod.array([_bits_to_uint64(h) for h in b_bits], dtype=np_mod.uint64)
    dist = _popcount_pair_matrix(a, b)
    if dist.size == 0:
        return False, -1, 0
    n_match = int((dist <= threshold).sum())
    return n_match > 0, int(dist.min()), n_match


def cluster_by_content(
    films: list,
    do_phash: bool,
    do_audio: bool,
    phash_threshold: int,
    phash_min_matches: int,
    audio_threshold: float,
    cache_dir: str | None,
    phash_sample: int,
    logger: logging.Logger,
):
    """Union-find merge films whose frames/audio match across metadata groups.

    Returns ``(clusters, merged_pairs, failures, n_hashed)``.
    """
    np_mod, _ = _require_np_cv2()

    hashed = []
    failures = []
    for i, film in enumerate(films, 1):
        res = compute_film_hashes(film, do_phash, do_audio, phash_sample, cache_dir)
        if res["error"]:
            failures.append({"id": film["id"], "path": film["abs_path"], "error": res["error"]})
        else:
            hashed.append((film, res))
        if i % 25 == 0 or i == len(films):
            logger.info(f"  hashed {i}/{len(films)} films ({len(failures)} failed)")

    n = len(hashed)
    if n < 2:
        return [], [], failures, n

    # pHash: chunked all-pairs so memory stays bounded on the full corpus.
    phash_hits = collections.defaultdict(dict)  # (i,j) -> stats
    if do_phash:
        per_film = []
        for film, res in hashed:
            hs = res["phash"] or []
            per_film.append([_bits_to_uint64(h) for h in hs])
        chunk = 64
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            for i in range(start, end):
                a = np_mod.array(per_film[i], dtype=np_mod.uint64)
                for j in range(i + 1, n):
                    b = np_mod.array(per_film[j], dtype=np_mod.uint64)
                    if a.size == 0 or b.size == 0:
                        continue
                    dist = _popcount_pair_matrix(a, b)
                    n_match = int((dist <= phash_threshold).sum())
                    if n_match >= phash_min_matches:
                        phash_hits[(i, j)] = {
                            "min": int(dist.min()),
                            "matches": n_match,
                        }

    # Audio: full similarity matrix from compact signatures.
    audio_hits = {}
    if do_audio:
        sigs = [np_mod.asarray((res["audio"] or [0.0]), dtype=np_mod.float32).ravel()
                for _, res in hashed]
        dim = max((s.size for s in sigs), default=1)
        mat = np_mod.zeros((n, dim), dtype=np_mod.float32)
        for i, s in enumerate(sigs):
            mat[i, : s.size] = s
        norm = np_mod.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        mat_n = mat / norm
        sim = mat_n @ mat_n.T
        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= audio_threshold:
                    audio_hits[(i, j)] = float(sim[i, j])

    # Union-find.
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    merged_pairs = []
    for (i, j), stats in phash_hits.items():
        union(i, j)
    for (i, j), s in audio_hits.items():
        union(i, j)

    # Collect pairs (report every cross-group match) and merge within groups.
    for i in range(n):
        for j in range(i + 1, n):
            ph = phash_hits.get((i, j))
            au = audio_hits.get((i, j))
            if ph or au:
                merged_pairs.append({
                    "a": hashed[i][0]["id"],
                    "b": hashed[j][0]["id"],
                    "phash_min_distance": ph["min"] if ph else None,
                    "phash_matches": ph["matches"] if ph else None,
                    "audio_similarity": round(au, 4) if au is not None else None,
                    "merged": find(i) == find(j),
                })

    # Regroup by root.
    roots = collections.defaultdict(list)
    for i in range(n):
        roots[find(i)].append(i)
    clusters = []
    for root, idxs in roots.items():
        if len(idxs) > 1:
            clusters.append([hashed[i][0] for i in idxs])
    return clusters, merged_pairs, failures, n


# ===========================================================================
# 5. Report + manifest
# ===========================================================================

def apply_content_clusters(groups: dict, clusters: list, index: Index) -> dict:
    """Merge metadata groups that the content pass found to be the same film.

    A content merge makes every non-kept member a genuine duplicate: the content
    pass proved the members are the same film, so the training manifest drops
    them. The merged group is rebuilt with ``origin="content"``.
    """
    id_to_key = {}
    for key, g in groups.items():
        for m in g["members"]:
            id_to_key[m] = key

    changed = True
    while changed:
        changed = False
        for cluster in clusters:
            keys = sorted({id_to_key[f["id"]] for f in cluster if f["id"] in id_to_key})
            if len(keys) > 1:
                primary = keys[0]
                members = list(groups[primary]["members"])
                for k in keys[1:]:
                    if k in groups:
                        members.extend(groups.pop(k)["members"])
                for m in members:
                    id_to_key[m] = primary
                groups[primary] = make_group(primary, members, "content", index)
                changed = True
    return groups


def flat_dropped(groups: dict, policy: str) -> list:
    """Flatten a policy's dropped members into ``{id, kept_id, movie_key, reason}``."""
    field = "dropped_family" if policy == "family" else "dropped_genuine"
    out = []
    for key, g in groups.items():
        for m in g[field]:
            out.append({
                "id": m,
                "kept_id": g["kept_id"],
                "movie_key": key,
                "reason": g["reasons"][m],
            })
    return out


def _name_tokens(stem: str) -> list:
    return [t for t in re.split(r"[^a-z0-9]+", normalize_stem(stem).lower()) if t]


# Common words that do not identify a work.
_NAME_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with",
    "at", "by", "from", "part", "pt", "disc", "dvd", "vcd", "cd", "movie",
    "film", "chapter", "ch", "vol", "volume", "no", "de", "la", "el", "le",
    "der", "die", "das", "les", "vil",
    # archive.org identifier scaffolding, not a work title
    "gov", "archives", "arc", "archive", "publicdomain", "public", "domain",
    "feature", "features", "full", "restored",
}


def cross_item_candidates(groups: dict, max_pairs: int = 25) -> list:
    """Flag movie keys that look like the same work under two namings.

    Cross-item duplicates cannot be confirmed from metadata. This finds pairs of
    distinct movie keys whose member names share a distinctive word bigram (for
    example ``the green hornet`` under both ``quigs`` and
    ``The.Green.Hornet.Strikes.Again``). The windowed content pass is the arbiter.
    """
    bigram_keys = collections.defaultdict(set)
    bigram_examples = collections.defaultdict(lambda: collections.defaultdict(list))

    for key, g in groups.items():
        for m in g["members"]:
            toks = _name_tokens(m)
            for a, b in zip(toks, toks[1:]):
                if a.isdigit() or b.isdigit():
                    continue
                # Keep only bigrams of two content words: this drops generic
                # phrases such as "the night", "attack of", "santa fe".
                if a in _NAME_STOPWORDS or b in _NAME_STOPWORDS:
                    continue
                if max(len(a), len(b)) < 4:
                    continue
                bigram_keys[(a, b)].add(key)
                bigram_examples[(a, b)][key].append(m)

    pair_bigrams = collections.defaultdict(set)
    for bg, keys in bigram_keys.items():
        if 1 < len(keys) <= 6:  # distinctive, but not a ubiquitous phrase
            ks = sorted(keys)
            for i in range(len(ks)):
                for j in range(i + 1, len(ks)):
                    pair_bigrams[(ks[i], ks[j])].add(bg)

    out = []
    for (k1, k2), bgs in sorted(
        pair_bigrams.items(), key=lambda kv: (-len(kv[1]), kv[0])
    )[:max_pairs]:
        samples = []
        for bg in sorted(bgs)[:3]:
            ex = bigram_examples[bg]
            samples.append({
                "shared_bigram": " ".join(bg),
                "a_example": sorted(ex[k1])[0],
                "b_example": sorted(ex[k2])[0],
            })
        out.append({"movie_key_a": k1, "movie_key_b": k2,
                    "n_shared_bigrams": len(bgs), "samples": samples})
    return out


def build_report(
    used_ids: list,
    groups: dict,
    mapping: dict,
    ambiguities: list,
    index: Index,
    n_index_items: int,
    input_path: str,
    index_path: str,
    content_info: dict | None,
    selected_policy: str = "genuine",
) -> dict:
    dropped_family = flat_dropped(groups, "family")
    dropped_genuine = flat_dropped(groups, "genuine")
    selected_dropped = dropped_family if selected_policy == "family" else dropped_genuine

    multi = {k: g for k, g in groups.items() if g["n_members"] > 1}
    n_films_in_multi = sum(g["n_members"] for g in multi.values())

    fam_kept = len(groups)
    gen_kept = len(used_ids) - len(dropped_genuine)

    policies = {
        "family": {
            "description": (
                "Split safety: one kept id per movie group. Drops every non-kept "
                "member, including distinct serial/anthology pieces. Use this map "
                "to keep a work out of both train and val."
            ),
            "kept": fam_kept,
            "dropped": len(dropped_family),
            "manifest": DEFAULT_OUTPUT_FAMILY,
        },
        "genuine": {
            "description": (
                "Training data: drops only genuine duplicate copies "
                "(duplicate_derivative and DVD/compilation artefacts that "
                "duplicate a main film) and keeps distinct pieces of content "
                "(serial chapters, split parts, anthologies)."
            ),
            "kept": gen_kept,
            "dropped": len(dropped_genuine),
            "manifest": DEFAULT_OUTPUT_GENUINE,
        },
        "selected": selected_policy,
    }

    report = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "task": "movie-level deduplication of the cinedantan corpus",
        "inputs": {"manifest": input_path, "index": index_path},
        "artifact_policy": {
            "movie_map": {
                "file": DEFAULT_MOVIE_MAP,
                "question": "which ids must never be split across train/val?",
                "purpose": (
                    "Family grouping id -> movie_key. Deliberately over-groups "
                    "multi-part serials, anthologies and compilation items so "
                    "their pieces cannot leak across a per-video_id split."
                ),
                "n_keys": len(mapping),
                "n_groups": len(groups),
            },
            "training_manifest": {
                "file": DEFAULT_OUTPUT,
                "question": "which ids are redundant copies of the same file?",
                "purpose": (
                    "Drops genuine duplicates only; keeps distinct content. "
                    "Selected by --drop-policy."
                ),
                "selected_policy": selected_policy,
            },
            "note": (
                "The family grouping is NOT a training manifest. Collapsing a "
                "12-chapter serial to one film would discard 11 distinct films."
            ),
        },
        "counts": {
            "used_ids": len(used_ids),
            "index_items": n_index_items,
            "movie_groups": len(groups),
            "items_contributing_multiple": len(multi),
            "films_in_multi_items": n_films_in_multi,
            "family_kept": fam_kept,
            "family_dropped": len(dropped_family),
            "genuine_kept": gen_kept,
            "genuine_dropped": len(dropped_genuine),
        },
        "policies": policies,
        "reconciliation_with_task_estimate": {
            "task_claimed": {
                "multi_items": 32,
                "films_in_multi_items": 131,
                "redundant": 99,
                "movie_groups": 1329,
            },
            "observed": {
                "multi_items": len(multi),
                "films_in_multi_items": n_films_in_multi,
                "redundant_family": len(dropped_family),
                "movie_groups": len(groups),
            },
            "note": (
                "The task's 32/131/99 double-counts ids whose stem is shared by "
                "several items (VTS_01_1 under both Martha_Ivers and "
                "Man_Who_Cheated_Himself; VTS_03_1..3 under Man_Who_Cheated_Himself, "
                "The_Scar_1948 and Strange_Illusion; ep1_512kb under three items). "
                "A partition assigns each id to exactly one movie, so the observed "
                "film count is lower and the group count higher. Groups here use "
                "the parent index item as the movie key."
            ),
        },
        "largest_groups": [
            {
                "movie_key": k,
                "n_members": g["n_members"],
                "kept_id": g["kept_id"],
                "genuine_dropped": len(g["dropped_genuine"]),
            }
            for k, g in sorted(
                groups.items(), key=lambda kv: -kv[1]["n_members"]
            )[:12]
        ],
        "id_to_movie_key": dict(mapping),
        "mapping_ambiguities": ambiguities,
        "groups": [
            {
                "movie_key": k,
                "kept_id": g["kept_id"],
                "kept_format": g["kept_format"],
                "kept_rank": g["kept_rank"],
                "n_members": g["n_members"],
                "members": g["members"],
                "reasons": g["reasons"],
                "has_main": g["has_main"],
                "dropped_family": g["dropped_family"],
                "dropped_genuine": g["dropped_genuine"],
                "origin": g["origin"],
            }
            for k, g in groups.items()
        ],
        "dropped": selected_dropped,
        "dropped_family": dropped_family,
        "dropped_genuine": dropped_genuine,
        "cross_item_unresolved": {
            "note": (
                "Cross-item duplicates cannot be confirmed from metadata. For "
                "example, Green Hornet episodes appear under both 'quigs' "
                "(The_Green_Hornet_-_NN_-_..._512kb) and "
                "'The.Green.Hornet.Strikes.Again' (..._NN_Title_512kb), which may "
                "be the same episodes under two naming conventions. The windowed "
                "content pass is the arbiter (--phash --audio --films-from-file); "
                "it requires the film files, which are not on this machine, so "
                "these candidates stay unresolved here."
            ),
            "candidates": cross_item_candidates(groups),
        },
    }

    # Extra context: 512Kb ids that could be upgraded to a better sibling copy
    # present in the index but not downloaded/used.
    upgrades = []
    for key, g in groups.items():
        kept = g["kept_id"]
        rank, _ = infer_quality(kept, index.fmt_hint(kept))
        if rank >= 4:  # 512Kb or worse
            item = mapping.get(kept, "")
            better = [
                f.get("url")
                for f in _item_by_identifier(index, item).get("aoFiles", [])
                if _format_rank(f.get("format")) is not None
                and _format_rank(f.get("format")) < rank
            ]
            if better:
                upgrades.append({"id": kept, "movie_key": key, "better_derivatives": better})
    report["quality_upgrades_available"] = upgrades

    if content_info is not None:
        report["content_pass"] = content_info
    return report


def _item_by_identifier(index: Index, ident: str) -> dict:
    for it in index.items:
        if it.get("identifier") == ident:
            return it
    return {}


def policy_kept_ids(groups: dict, policy: str) -> list:
    """Return the kept ids under a drop policy (before manifest filtering)."""
    kept = []
    for g in groups.values():
        if policy == "family":
            kept.append(g["kept_id"])
        else:
            dropped = set(g["dropped_genuine"])
            kept.extend(m for m in g["members"] if m not in dropped)
    return kept


def write_manifest(
    used_entries: list,
    kept_ids: list,
    output_path: str,
    external_ids: set,
) -> list:
    """Write a canonical manifest: original entries for every kept id."""
    entry_by_id = {e["id"]: e for e in used_entries if "id" in e}
    order = {e["id"]: i for i, e in enumerate(used_entries) if "id" in e}
    seen = set()
    kept_entries = []
    for kid in kept_ids:
        if kid in seen:
            continue
        seen.add(kid)
        if kid in external_ids and kid not in entry_by_id:
            continue  # external film has no manifest entry to emit
        kept_entries.append(entry_by_id.get(kid, {"id": kid}))
    kept_entries.sort(key=lambda e: order.get(e.get("id"), len(order)))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kept_entries, f, indent=2, ensure_ascii=False)
    return kept_entries


def write_movie_map(mapping: dict, output_path: str) -> None:
    """Write the split-safety map as a byte-stable flat ``{id: movie_key}`` object.

    Byte-stable: keys are sorted and the indentation is fixed, so repeated runs
    produce identical bytes regardless of input order.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=0, ensure_ascii=False, sort_keys=True)


# ===========================================================================
# 6. Film list for the content pass
# ===========================================================================

def resolve_film_path(video_path: str) -> str:
    if not video_path:
        return video_path
    if os.path.isabs(video_path):
        return video_path
    return resolve_path(video_path)


def build_film_list(args, used_entries: list, mapping: dict) -> tuple[list, set]:
    """Return ``(films, external_ids)`` for the content pass.

    Without ``--films-from-file`` this is one kept representative per metadata
    group. With it, the given films are used verbatim (ids or paths).
    """
    entry_by_id = {e["id"]: e for e in used_entries if "id" in e}
    external = set()

    if args.films_from_file:
        with open(args.films_from_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        films = []
        for item in raw:
            if isinstance(item, str):
                if item in entry_by_id:
                    e = entry_by_id[item]
                    films.append({
                        "id": item,
                        "abs_path": resolve_film_path(e.get("video_path", "")),
                        "video_path": e.get("video_path", ""),
                    })
                else:
                    external.add(item)
                    films.append({"id": item, "abs_path": resolve_film_path(item), "video_path": item})
            elif isinstance(item, dict):
                fid = item.get("id") or normalize_stem(item.get("video_path", ""))
                vp = item.get("video_path", "")
                if fid not in entry_by_id:
                    external.add(fid)
                films.append({"id": fid, "abs_path": resolve_film_path(vp), "video_path": vp})
        return films, external

    # Default: one representative per metadata group (handled by the caller
    # through films_from_groups).
    return [], external


def films_from_groups(groups: dict, entry_by_id: dict) -> list:
    films = []
    for key, g in groups.items():
        kept = g["kept_id"]
        e = entry_by_id.get(kept, {})
        films.append({
            "id": kept,
            "abs_path": resolve_film_path(e.get("video_path", "")),
            "video_path": e.get("video_path", ""),
        })
    return films


# ===========================================================================
# 7. Summary table
# ===========================================================================

def format_summary(report: dict) -> str:
    c = report["counts"]
    lines = []
    lines.append("=" * 68)
    lines.append("DEDUP SUMMARY (movie level)")
    lines.append("=" * 68)
    lines.append(f"  input used ids:                  {c['used_ids']:>8d}")
    lines.append(f"  index items:                     {c['index_items']:>8d}")
    lines.append(f"  movie-level groups (split map):  {c['movie_groups']:>8d}")
    lines.append(f"  items contributing >1 film:      {c['items_contributing_multiple']:>8d}")
    lines.append(f"  films inside those items:        {c['films_in_multi_items']:>8d}")
    lines.append("-" * 68)
    lines.append(f"  family policy    kept:           {c['family_kept']:>8d}  dropped: {c['family_dropped']:>6d}")
    lines.append(f"  genuine policy   kept:           {c['genuine_kept']:>8d}  dropped: {c['genuine_dropped']:>6d}")
    lines.append(f"  selected policy: {report['policies']['selected']}")
    if "content_pass" in report:
        cp = report["content_pass"]
        lines.append("-" * 68)
        lines.append(f"  content pass hashed films:       {cp.get('n_hashed', 0):>8d}")
        lines.append(f"  content clusters merged:         {cp.get('n_clusters', 0):>8d}")
        lines.append(f"  films failed to hash:            {len(cp.get('failures', [])):>8d}")
    lines.append("=" * 68)
    return "\n".join(lines)


# ===========================================================================
# 8. Self-test
# ===========================================================================

def self_test(index_path: str, manifest_path: str, movie_map_path: str | None = None) -> int:
    failures = []

    def check(cond, msg):
        if cond:
            print(f"  PASS: {msg}")
        else:
            print(f"  FAIL: {msg}")
            failures.append(msg)

    print("normalize_stem:")
    check(normalize_stem("/Constantine_and_the_Cross_256kb.mp4") == "constantine_and_the_cross",
          "strips /path, _256kb, .mp4")
    check(normalize_stem("First_Spaceship_on_Venus_ipod_512kb") == "first_spaceship_on_venus",
          "strips _ipod_512kb")
    check(normalize_stem("Secret_Agent_512kb") == "secret_agent", "strips _512kb")
    check(normalize_stem("X_64kb.MP4") == "x", "strips _64kb and uppercase ext")
    check(normalize_stem("Man_Who_Cheated_Himself") == "man_who_cheated_himself", "plain stem")
    check(normalize_stem("M4V10005") == "m4v10005", "M4V left intact as a stem")

    print("classify_part:")
    check(classify_part("VTS_01_1") == "dvd_vts", "VTS_01_1 is a dvd_vts part")
    check(classify_part("VTS_01_2_512kb") == "dvd_vts", "VTS_01_2_512kb is a dvd_vts part")
    check(classify_part("M4V10005") == "dvd_m4v", "M4V10005 is a dvd_m4v part")
    check(classify_part("NOIR_DISC_11") == "compilation_disc", "NOIR_DISC_11 is a disc part")
    check(classify_part("phantom_thunderbolt") is None, "a normal film is not a part")

    print("quality order:")
    order = [
        infer_quality("x", "h.264")[0],
        infer_quality("x", "HiRes MPEG4")[0],
        infer_quality("x", "MPEG4")[0],
        infer_quality("x", "512Kb MPEG4")[0],
        infer_quality("x", "256Kb MPEG4")[0],
        infer_quality("x", "64Kb MPEG4")[0],
    ]
    check(order == sorted(order), "h.264 > HiRes > MPEG4 > 512Kb > 256Kb > 64Kb")

    if not (os.path.exists(index_path) and os.path.exists(manifest_path)):
        print("  SKIP: data files not found; grouping assertions not run")
    else:
        index = Index.load(index_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        used = [e["id"] for e in entries]
        groups, mapping, _amb = build_metadata_groups(used, index)

        martha = groups.get("Martha_Ivers")
        check(martha is not None, "Martha_Ivers group exists")
        if martha:
            expected = {"Martha_Ivers", "VTS_01_1", "VTS_01_2", "VTS_01_3", "VTS_01_4"}
            check(set(martha["members"]) == expected,
                  "Martha_Ivers + 4 VTS parts collapse to one group")
            check(martha["kept_id"] == "Martha_Ivers",
                  "Martha_Ivers (primary h.264) is the kept copy")
            check(len(martha["dropped_genuine"]) == 4,
                  "genuine drops the 4 Martha VTS parts")

        spaceship = groups.get("the_first_spaceship_on_venus")
        if spaceship:
            check(set(spaceship["members"]) == {
                "First_Spaceship_on_Venus_512kb", "First_Spaceship_on_Venus_ipod_512kb"},
                "spaceship derivatives collapse to one group")

        # Genuine policy: Man_Who_Cheated_Himself still collapses its VTS/NOIR parts.
        noir = groups.get("Man_Who_Cheated_Himself")
        if noir:
            m = set(noir["members"])
            check({"NOIR_DISC_11", "NOIR_DISC_13", "NOIR_DISC_14"} <= m,
                  "NOIR_DISC parts are in the Man_Who_Cheated_Himself group")
            check({"NOIR_DISC_11", "NOIR_DISC_13", "NOIR_DISC_14"} <= set(noir["dropped_genuine"]),
                  "genuine drops the Man_Who NOIR_DISC parts")
            kept_ids = [x for x in noir["members"] if x not in set(noir["dropped_genuine"])]
            check(kept_ids == [noir["kept_id"]],
                  "genuine keeps exactly one Man_Who_Cheated_Himself film")

        # Genuine policy: serial/anthology pieces are distinct content and kept.
        musk = groups.get("TheThreeMuskateers")
        if musk:
            check(len(musk["dropped_genuine"]) == 0,
                  "genuine keeps all 12 TheThreeMuskateers chapters")
            check(musk["n_members"] == 12, "TheThreeMuskateers has 12 members")

        toon = groups.get("Cartoontheater1930sAnd1950s1")
        if toon:
            check(len(toon["dropped_genuine"]) == 0,
                  "genuine keeps all 5 Cartoontheater cartoons")
            check(toon["n_members"] == 5, "Cartoontheater has 5 members")

        # Distinct split parts (*of*) are content, not duplicates.
        tff = groups.get("The_Fast_and_the_Furious")
        if tff:
            check(len(tff["dropped_genuine"]) == 0,
                  "genuine keeps all thefastandthefurious *of4 parts")

        # Every used id is covered exactly once.
        covered = [mid for g in groups.values() for mid in g["members"]]
        check(sorted(covered) == sorted(used), "every used id belongs to exactly one group")
        check(len(groups) == len({g["kept_id"] for g in groups.values()}),
              "one kept id per group")

        fam_drop = sum(len(g["dropped_family"]) for g in groups.values())
        gen_drop = sum(len(g["dropped_genuine"]) for g in groups.values())
        check(fam_drop == 92, f"family drops 92 (got {fam_drop})")
        print(f"  INFO: {len(used)} ids -> {len(groups)} movie groups; "
              f"family drops {fam_drop}, genuine drops {gen_drop}")

        # Split-safety map, if it exists, must match the freshly built mapping.
        if movie_map_path and os.path.exists(movie_map_path):
            with open(movie_map_path, "r", encoding="utf-8") as f:
                on_disk = json.load(f)
            check(on_disk == dict(mapping),
                  "data/movie_map.json matches id_to_movie_key")

    print()
    if failures:
        print(f"SELF-TEST FAILED ({len(failures)} failures)")
        return 1
    print("SELF-TEST PASSED")
    return 0


# ===========================================================================
# 9. Main
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Movie-level de-duplication of the cinedantan corpus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", default=DEFAULT_INPUT, help="input video manifest")
    p.add_argument("--index", default=DEFAULT_INDEX, help="cinedantan source index")
    p.add_argument("--report", default=DEFAULT_REPORT, help="output JSON report")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="output training manifest for --drop-policy (default: %(default)s)")
    p.add_argument("--movie-map", default=DEFAULT_MOVIE_MAP,
                   help="output split-safety map, flat {id: movie_key} (default: %(default)s)")
    p.add_argument("--output-family", default=DEFAULT_OUTPUT_FAMILY,
                   help="output family-collapsed manifest (default: %(default)s)")
    p.add_argument("--output-genuine", default=DEFAULT_OUTPUT_GENUINE,
                   help="output genuine-duplicates-only manifest (default: %(default)s)")
    p.add_argument("--drop-policy", choices=DROP_POLICIES, default="genuine",
                   help="family drops all non-kept members; genuine drops only "
                        "redundant copies (default: %(default)s)")
    p.add_argument("--phash", action="store_true", help="run the perceptual-hash pass")
    p.add_argument("--audio", action="store_true", help="run the audio-fingerprint pass")
    p.add_argument("--phash-sample", type=int, default=8,
                   help="frames sampled per film for pHash (default 8)")
    p.add_argument("--phash-threshold", type=int, default=10,
                   help="max Hamming distance for a frame match (default 10)")
    p.add_argument("--phash-min-matches", type=int, default=3,
                   help="matching frame pairs required to merge (default 3)")
    p.add_argument("--audio-threshold", type=float, default=0.90,
                   help="min audio cosine similarity to merge (default 0.90)")
    p.add_argument("--films-from-file", default=None,
                   help="JSON list of ids or {id, video_path} to hash (window)")
    p.add_argument("--max-films", type=int, default=0,
                   help="limit films hashed in the content pass (0 = all)")
    p.add_argument("--cache-dir", default=None,
                   help="cache film hashes here (re-run friendly)")
    p.add_argument("--dry-run", action="store_true", help="do not write outputs")
    p.add_argument("--self-test", action="store_true", help="run unit self-test and exit")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logger = setup_logging(args.verbose)

    input_path = os.path.join(PROJECT_ROOT, args.input)
    index_path = os.path.join(PROJECT_ROOT, args.index)
    report_path = os.path.join(PROJECT_ROOT, args.report)
    output_path = os.path.join(PROJECT_ROOT, args.output)
    movie_map_path = os.path.join(PROJECT_ROOT, args.movie_map)
    output_family_path = os.path.join(PROJECT_ROOT, args.output_family)
    output_genuine_path = os.path.join(PROJECT_ROOT, args.output_genuine)

    if args.self_test:
        return self_test(index_path, input_path, movie_map_path)

    logger.info("=" * 60)
    logger.info("MOVIE-LEVEL DE-DUPLICATION")
    logger.info("=" * 60)
    logger.info(f"Manifest:  {input_path}")
    logger.info(f"Index:     {index_path}")

    if not os.path.exists(input_path):
        logger.error(f"Input manifest not found: {input_path}")
        return 1
    if not os.path.exists(index_path):
        logger.error(f"Index not found: {index_path}")
        return 1

    t0 = time.time()
    index = Index.load(index_path)
    with open(input_path, "r", encoding="utf-8") as f:
        used_entries = json.load(f)
    used_ids = [e["id"] for e in used_entries if "id" in e]
    logger.info(f"Loaded {len(used_ids)} used ids and {len(index.items)} index items")

    # -- Pass 1: metadata --
    groups, mapping, ambiguities = build_metadata_groups(used_ids, index)
    logger.info(
        f"Metadata pass: {len(groups)} movie groups (split map); "
        f"family drops {sum(len(g['dropped_family']) for g in groups.values())}, "
        f"genuine drops {sum(len(g['dropped_genuine']) for g in groups.values())}"
    )

    # -- Pass 2: content backstop (opt-in) --
    content_info = None
    external_ids = set()
    do_phash = args.phash or bool(args.films_from_file)
    do_audio = args.audio
    if do_phash or do_audio:
        entry_by_id = {e["id"]: e for e in used_entries if "id" in e}
        if args.films_from_file:
            films, external_ids = build_film_list(args, used_entries, mapping)
        else:
            films = films_from_groups(groups, entry_by_id)
        if args.max_films:
            films = films[: args.max_films]
        logger.info(f"Content pass: hashing {len(films)} films "
                    f"(phash={do_phash}, audio={do_audio})")
        clusters, merged_pairs, failures, n_hashed = cluster_by_content(
            films, do_phash, do_audio, args.phash_threshold, args.phash_min_matches,
            args.audio_threshold, args.cache_dir, args.phash_sample, logger,
        )
        if clusters:
            groups = apply_content_clusters(groups, clusters, index)
        content_info = {
            "n_films_requested": len(films),
            "n_hashed": n_hashed,
            "n_clusters": len(clusters),
            "do_phash": do_phash,
            "do_audio": do_audio,
            "phash_threshold": args.phash_threshold,
            "phash_min_matches": args.phash_min_matches,
            "audio_threshold": args.audio_threshold,
            "clusters": [
                [{"id": f["id"], "path": f["abs_path"]} for f in cluster]
                for cluster in clusters
            ],
            "matched_pairs": merged_pairs,
            "failures": failures,
        }
        logger.info(f"Content pass: {len(clusters)} cross-group clusters, "
                    f"{len(failures)} films failed to hash")

    # -- Report --
    report = build_report(
        used_ids, groups, mapping, ambiguities, index, len(index.items),
        args.input, args.index, content_info, selected_policy=args.drop_policy,
    )

    # -- Artifacts --
    if args.dry_run:
        logger.info("Dry run: not writing outputs")
    else:
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        # Split-safety map: family grouping, byte-stable.
        write_movie_map(mapping, movie_map_path)
        # Training manifests: selected policy plus both named policies.
        fam_entries = write_manifest(
            used_entries, policy_kept_ids(groups, "family"), output_family_path, external_ids)
        gen_entries = write_manifest(
            used_entries, policy_kept_ids(groups, "genuine"), output_genuine_path, external_ids)
        selected_entries = (
            fam_entries if args.drop_policy == "family" else gen_entries)
        if os.path.abspath(output_path) not in (
            os.path.abspath(output_family_path), os.path.abspath(output_genuine_path)
        ):
            write_manifest(used_entries, policy_kept_ids(groups, args.drop_policy),
                           output_path, external_ids)
        logger.info(f"Wrote report:        {report_path}")
        logger.info(f"Wrote movie map:     {movie_map_path} ({len(mapping)} ids)")
        logger.info(f"Wrote manifest:      {output_path} "
                    f"({args.drop_policy}, {len(selected_entries)} films)")
        logger.info(f"Wrote family list:   {output_family_path} "
                    f"({len(fam_entries)} films)")
        logger.info(f"Wrote genuine list:  {output_genuine_path} "
                    f"({len(gen_entries)} films)")

    summary = format_summary(report)
    logger.info("\n" + summary)
    print(summary)
    logger.info(f"Total time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
