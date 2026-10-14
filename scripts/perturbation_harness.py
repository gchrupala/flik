#!/usr/bin/env python
"""Perturbation-ladder harness for image-text alignment instruments.

Why this exists
---------------
The pipeline's semantic alignment score (CLIP text <-> sampled video frames,
``src/preprocess/check_correspondance.py``) was never validated. A high mean
score on matched windows is not evidence that the score measures *window-level*
alignment: an instrument that only tracks *film topic* (all windows of one film
look alike) produces the same matched score while being useless for filtering
windows. This harness separates the two.

The ladder
----------
For each sampled window we hold the window's TRUE frames fixed and vary only the
text, so any score change is attributable to the text:

* ``P0 matched``        - the window's own transcript text.
* ``P1 within_film``    - text from another window of the SAME film, at least
                          ``P1_MIN_INDEX_DISTANCE`` windows away in time. This is
                          the make-or-break rung: it removes film topic and
                          leaves only window-level content. An instrument that
                          fails P1 is a topic detector, not an alignment metric.
* ``P2 cross_film``     - text from a DIFFERENT film.
* ``P3 paraphrase``     - (optional, ``--enable-p3``) a paraphrased version of the
                          true text. Skipped -- and reported as skipped -- unless
                          a paraphraser is supplied, rather than faked.
* ``P4 filler``         - generic filler text (``FILLER_TEXTS``). Defines the
                          floor.

Metrics (per instrument)
------------------------
* ``AUC(P0 vs P1)`` and ``AUC(P0 vs P2)`` (rank/Mann-Whitney, ties handled).
* Median separation of P0 from the P2 distribution, in std-dev of P2.
* Dynamic range ``P0 - P4`` (mean and median variants).
* Candidate threshold at 95% specificity against P2 (the 95th percentile of P2),
  and the video-level retention that threshold implies over the sampled films.

Acceptance gates (explicit pass/fail)
-------------------------------------
* ``G1`` - ``AUC(P0 vs P1) >= 0.75`` AND median separation ``>= 0.5`` sd of P2.
  If it fails, the instrument only detects film topic and MUST NOT be used to
  filter windows.
* ``G2`` - ``AUC(P0 vs P2) >= 0.95``.
* ``G3`` - separate in-domain sanity check on COCO (``--coco``). Reuses the
  existing COCO baseline path (``check_correspondance.py --mode coco``) so a
  broken instrument can be told apart from "archival footage is just hard".
  Pass criterion used here: ``AUC(matched vs shuffled) >= 0.95`` (documented
  chosen criterion; the task fixes no threshold for G3).
* ``G4`` - implied video-level retention between 60% and 95%. Outside that
  range the threshold is either vacuous or destructive.

Retention definition: a film is retained when the MEAN of its P0 window scores
is at or above the threshold. This mirrors the existing video-level check in
``check_correspondance.main()`` (average score vs a percentile threshold). The
alternative "at least 50% of windows pass" fraction is also reported.

Instruments
-----------
Scoring goes through one pluggable interface (``ImageTextInstrument``). The
default ``clip`` instrument reuses ``_score_pairs_batched`` from
``check_correspondance`` so its numbers are directly comparable to the existing
pipeline; ``auto`` is a generic HuggingFace image-text path. Several instruments
run through the same code path -- switch with ``--instrument`` / ``--model``.

Usage
-----
    # Build a small manifest from the two local films, then run the ladder.
    uv run --extra cpu python -m scripts.perturbation_harness \
        --build-manifest data/local_smoke_manifest.json \
        --n-windows 40 --n-films 2 --device cpu

    # Real run (GPU) over the pilot manifest.
    uv run --extra cu128 python -m scripts.perturbation_harness \
        --manifest data/qc_pilot_manifest.json --n-windows 2000 --n-films 200

    # In-domain sanity check on COCO (reuses the existing path).
    uv run --extra cu128 python -m scripts.perturbation_harness --coco
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root so `src.*` imports work under `python -m`.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import numpy as np
import torch

from src.CONSTANTS import PROJECT_ROOT, TARGET_FPS, CLIP_BATCH_SIZE, VIDEO_EXTENSIONS
from src.utils.paths import resolve_path, to_relative
from src.utils.segments import merge_segments

# Reuse the existing decode + CLIP scoring path so results are comparable to
# the production correspondence stages. Importing this module is side-effect
# free except for constructing its logger (it writes to logdir/).
from src.preprocess.check_correspondance import (
    load_video_frames,
    _score_pairs_batched as score_pairs_batched,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_MANIFEST = "data/qc_pilot_manifest.json"
DEFAULT_OUT_DIR = "perturbation_results"

# P1 donors must be at least this many positions away in the film's time-ordered
# selected-window list. Guarantees the donor is a genuinely different moment.
P1_MIN_INDEX_DISTANCE = 3

FILLER_TEXTS = ["[music plays]", "silence"]

# Window builder defaults (match the training manifest builder).
BUILD_MIN_DURATION = 3.0
BUILD_MAX_DURATION = 10.0
BUILD_MAX_GAP = 1.0

# Candidate threshold specificity against P2.
THRESHOLD_SPECIFICITY = 0.95

# Gate thresholds.
G1_AUC_MIN = 0.75
G1_SEP_MIN = 0.5
G2_AUC_MIN = 0.95
G3_AUC_MIN = 0.95
G4_RETENTION_MIN = 0.60
G4_RETENTION_MAX = 0.95


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("perturbation_harness")


def setup_logging():
    log_dir = os.path.join(PROJECT_ROOT, "logdir")
    os.makedirs(log_dir, exist_ok=True)
    logger.setLevel(logging.INFO)
    # check_correspondance calls logging.basicConfig at import time; do not
    # let our records reach the root handlers as well.
    logger.propagate = False
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(os.path.join(log_dir, "perturbation_harness.log"))
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(fh)
        logger.addHandler(sh)


# ---------------------------------------------------------------------------
# Instrument interface
# ---------------------------------------------------------------------------

INSTRUMENT_REGISTRY = {}


def register_instrument(name):
    def _deco(cls):
        INSTRUMENT_REGISTRY[name] = cls
        return cls

    return _deco


class ImageTextInstrument:
    """Base class: score (text, frames) items to a single scalar per item.

    Subclasses implement ``load`` and ``_score_pairs``. ``score_items``
    aggregates frames per item with max (matching the production pipeline's
    "best frame-text alignment" rule).
    """

    name = "base"

    def __init__(self, model_id, device, batch_size, use_fp16=True):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.use_fp16 = use_fp16
        self.model = None
        self.processor = None

    def load(self):
        raise NotImplementedError

    def _score_pairs(self, texts, images):
        """Return a list of scores, one per (text, image) pair, aligned by index."""
        raise NotImplementedError

    def score_items(self, items):
        """Score ``items`` = list of ``(text, [PIL.Image, ...])``.

        Returns a list of floats (max frame score per item). Items with no
        frames score ``nan``.
        """
        flat_texts, flat_images, owners = [], [], []
        for i, (text, frames) in enumerate(items):
            for frame in frames:
                flat_texts.append(text)
                flat_images.append(frame)
                owners.append(i)

        best = [float("-inf")] * len(items)
        for s in range(0, len(flat_texts), self.batch_size):
            chunk_texts = flat_texts[s:s + self.batch_size]
            chunk_images = flat_images[s:s + self.batch_size]
            chunk_scores = self._score_pairs(chunk_texts, chunk_images)
            for k, sc in enumerate(chunk_scores):
                owner = owners[s + k]
                if sc > best[owner]:
                    best[owner] = sc

        return [v if v != float("-inf") else float("nan") for v in best]


@register_instrument("clip")
class ClipInstrument(ImageTextInstrument):
    """CLIP-family instrument via CLIPModel/CLIPProcessor.

    Uses the production ``_score_pairs_batched`` (full text, tokenizer
    truncation, fp16 autocast on CUDA) so scores match the pipeline exactly.
    """

    name = "clip"

    def load(self):
        from transformers import CLIPModel, CLIPProcessor

        logger.info("Loading CLIP model %s on %s", self.model_id, self.device)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(self.model_id)

    def _score_pairs(self, texts, images):
        return score_pairs_batched(
            self.model, self.processor, texts, images, self.device, self.use_fp16
        )


@register_instrument("auto")
class AutoImageTextInstrument(ImageTextInstrument):
    """Generic HuggingFace image-text instrument (cosine of normalised features).

    Works for any ``AutoModel`` exposing ``get_text_features``/
    ``get_image_features`` (for example SigLIP). Texts are de-duplicated per
    batch before encoding.
    """

    name = "auto"

    def load(self):
        from transformers import AutoModel, AutoProcessor

        logger.info("Loading AutoModel %s on %s", self.model_id, self.device)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def _score_pairs(self, texts, images):
        uniq_texts = list(dict.fromkeys(texts))
        text_index = {t: i for i, t in enumerate(uniq_texts)}

        text_inputs = self.processor(
            text=uniq_texts, return_tensors="pt", padding=True, truncation=True
        )
        image_inputs = self.processor(images=images, return_tensors="pt")
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}

        with torch.no_grad():
            text_feats = self.model.get_text_features(**text_inputs)
            image_feats = self.model.get_image_features(**image_inputs)
        if not torch.is_tensor(text_feats):
            text_feats = text_feats[0]
        if not torch.is_tensor(image_feats):
            image_feats = image_feats[0]

        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        image_feats = image_feats / image_feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        order = torch.tensor([text_index[t] for t in texts], device=text_feats.device)
        sims = (text_feats[order] * image_feats).sum(dim=-1)
        return sims.detach().cpu().tolist()


def build_instrument(name, model_id, device, batch_size, use_fp16):
    if name not in INSTRUMENT_REGISTRY:
        raise ValueError(
            f"Unknown instrument {name!r}. Known: {sorted(INSTRUMENT_REGISTRY)}"
        )
    instrument = INSTRUMENT_REGISTRY[name](model_id, device, batch_size, use_fp16)
    instrument.load()
    return instrument


# ---------------------------------------------------------------------------
# Manifest loading / window selection
# ---------------------------------------------------------------------------

def film_id_of(window):
    return str(window.get("video_id") or window.get("id") or "unknown")


def window_id_of(window):
    wid = window.get("id")
    if wid:
        return str(wid)
    return "{}_{:.2f}_{:.2f}".format(
        film_id_of(window),
        float(window.get("start_sec", 0.0)),
        float(window.get("end_sec", 0.0)),
    )


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("windows", data.get("segments", []))
    return data


def select_windows(manifest, n_windows, n_films, seed):
    """Pick up to ``n_films`` films and up to ``n_windows`` windows total.

    Windows are allocated across the chosen films with a floor of
    ``P1_MIN_INDEX_DISTANCE + 1`` per film when the budget allows, so P1 donors
    exist. Each film's selected windows are returned ordered by start time,
    which defines the "distance" for P1.
    """
    rng = random.Random(seed)
    groups = defaultdict(list)
    for w in manifest:
        groups[film_id_of(w)].append(w)

    film_ids = sorted(groups.keys())
    if n_films > 0 and len(film_ids) > n_films:
        film_ids = sorted(rng.sample(film_ids, n_films))

    avail = {f: len(groups[f]) for f in film_ids}
    total_avail = sum(avail.values())
    if total_avail == 0 or n_windows <= 0:
        return [], film_ids

    n_films_selected = len(film_ids)
    target = min(n_windows, total_avail)
    floor_each = P1_MIN_INDEX_DISTANCE + 1

    counts = {f: 0 for f in film_ids}
    if target >= floor_each * n_films_selected:
        for f in film_ids:
            counts[f] = min(floor_each, avail[f])
    else:
        for f in film_ids:
            counts[f] = min(1, avail[f])

    remaining = target - sum(counts.values())
    if remaining > 0:
        quota = {f: remaining * avail[f] / total_avail for f in film_ids}
        floors = {f: int(quota[f]) for f in film_ids}
        for f in film_ids:
            counts[f] = min(avail[f], counts[f] + floors[f])
        leftover = target - sum(counts.values())
        order = sorted(film_ids, key=lambda f: quota[f] - floors[f], reverse=True)
        i = 0
        guard = 0
        while leftover > 0 and order and guard < 10 * len(order) + target:
            f = order[i % len(order)]
            if counts[f] < avail[f]:
                counts[f] += 1
                leftover -= 1
            i += 1
            guard += 1

    selected = []
    for f in film_ids:
        k = min(counts[f], avail[f])
        if k <= 0:
            continue
        film_windows = rng.sample(groups[f], k)
        film_windows.sort(key=lambda w: float(w.get("start_sec", 0.0)))
        selected.extend(film_windows)
    return selected, film_ids


# ---------------------------------------------------------------------------
# Frame decoding
# ---------------------------------------------------------------------------

def decode_frames(windows, frames_per_window, workers):
    """Decode and evenly subsample frames for each window. Returns id -> frames."""
    cache = {}

    def _one(window):
        path = resolve_path(window.get("video_path", ""))
        frames = load_video_frames(
            path,
            float(window["start_sec"]),
            float(window["end_sec"]),
            TARGET_FPS,
        )
        return frames

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, w): w for w in windows}
        for future in as_completed(futures):
            window = futures[future]
            wid = window_id_of(window)
            try:
                frames = future.result()
            except Exception as exc:  # noqa: BLE001 - bad media must not kill the run
                logger.warning("Decode failed for %s: %s", wid, exc)
                continue
            if not frames:
                logger.warning("No frames decoded for %s", wid)
                continue
            if len(frames) > frames_per_window:
                idx = np.linspace(0, len(frames) - 1, frames_per_window)
                idx = sorted({int(round(i)) for i in idx})
                frames = [frames[i] for i in idx]
            cache[wid] = frames

    return cache


# ---------------------------------------------------------------------------
# Ladder construction
# ---------------------------------------------------------------------------

def build_ladder(windows, film_ids, rng, enable_p3):
    """Return rung -> list of (window, text). Frames are attached later."""
    ladder = {"P0": [], "P1": [], "P2": [], "P4": []}

    film_windows = defaultdict(list)
    for w in windows:
        film_windows[film_id_of(w)].append(w)
    for f in film_windows:
        film_windows[f].sort(key=lambda w: float(w.get("start_sec", 0.0)))

    for w in windows:
        ladder["P0"].append((w, str(w.get("text") or "")))
        ladder["P4"].append((w, rng.choice(FILLER_TEXTS)))

    # P1: within-film donor, >= P1_MIN_INDEX_DISTANCE positions away.
    for f, ws in film_windows.items():
        for i, w in enumerate(ws):
            candidates = [j for j in range(len(ws)) if abs(j - i) >= P1_MIN_INDEX_DISTANCE]
            if not candidates:
                continue
            j = rng.choice(candidates)
            ladder["P1"].append((w, str(ws[j].get("text") or "")))

    # P2: donor from a different film.
    other_films_present = [f for f in film_ids if film_windows.get(f)]
    if len(other_films_present) >= 2:
        for w in windows:
            own = film_id_of(w)
            choices = [f for f in other_films_present if f != own]
            if not choices:
                continue
            donor_film = rng.choice(choices)
            donor = rng.choice(film_windows[donor_film])
            ladder["P2"].append((w, str(donor.get("text") or "")))

    ladder["P3"] = []
    if enable_p3:
        logger.info("P3 requested; requires --paraphraser to actually run.")

    return ladder, film_windows


def paraphrase_texts(texts, paraphraser_id, device, batch_size):
    """Generate paraphrases with a HuggingFace seq2seq model.

    Only called when ``--paraphraser`` is given. This is a real local model
    route; when no paraphraser is configured, P3 is reported as skipped instead
    of being fabricated.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    logger.info("Loading paraphraser %s on %s", paraphraser_id, device)
    tokenizer = AutoTokenizer.from_pretrained(paraphraser_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(paraphraser_id).to(device)
    model.eval()

    out = []
    prefix = "paraphrase: "
    for s in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[s:s + batch_size]]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs, max_new_tokens=64, num_beams=4, do_sample=False
            )
        out.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def auc_score(pos, neg):
    """Rank-based AUC = P(pos > neg) + 0.5 * P(pos == neg). Ties average-ranked."""
    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)
    pos = pos[~np.isnan(pos)]
    neg = neg[~np.isnan(neg)]
    if pos.size == 0 or neg.size == 0:
        return float("nan")

    values = np.concatenate([pos, neg])
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)

    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1

    pos_ranks = np.empty(values.size, dtype=np.float64)
    pos_ranks[order] = ranks
    r_pos = pos_ranks[:pos.size].sum()
    u = r_pos - pos.size * (pos.size + 1) / 2.0
    return float(u / (pos.size * neg.size))


def distribution_stats(values):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
    }


def compute_metrics(rung_scores, p0_by_film):
    """Compute the four headline metrics, threshold and retention."""
    p0 = np.asarray(rung_scores.get("P0", []), dtype=np.float64)
    p1 = np.asarray(rung_scores.get("P1", []), dtype=np.float64)
    p2 = np.asarray(rung_scores.get("P2", []), dtype=np.float64)
    p4 = np.asarray(rung_scores.get("P4", []), dtype=np.float64)

    p2 = p2[~np.isnan(p2)]
    threshold = (
        float(np.percentile(p2, THRESHOLD_SPECIFICITY * 100.0))
        if p2.size
        else float("nan")
    )

    std_p2 = float(p2.std()) if p2.size else float("nan")
    median_sep = (
        float(np.median(p0[~np.isnan(p0)]) - np.median(p2)) / std_p2
        if p2.size and std_p2 > 0
        else float("nan")
    )

    metrics = {
        "auc_p0_p1": auc_score(p0, p1),
        "auc_p0_p2": auc_score(p0, p2),
        "median_sep_p0_p2_sd_p2": median_sep,
        "dynamic_range_mean_p0_p4": (
            float(np.nanmean(p0) - np.nanmean(p4))
            if p0.size and p4.size
            else float("nan")
        ),
        "dynamic_range_median_p0_p4": (
            float(np.nanmedian(p0) - np.nanmedian(p4))
            if p0.size and p4.size
            else float("nan")
        ),
        "threshold_specificity": THRESHOLD_SPECIFICITY,
        "threshold_at_95_spec_vs_p2": threshold,
        "p2_std": std_p2,
        "p2_p95": threshold,
    }

    if p0_by_film and not np.isnan(threshold):
        retained = [
            f for f, scores in p0_by_film.items()
            if np.mean([s for s in scores if not np.isnan(s)]) >= threshold
        ]
        metrics["video_retention_mean_rule"] = len(retained) / len(p0_by_film)
        metrics["video_retention_mean_rule_counts"] = [len(retained), len(p0_by_film)]
        frac_rule = []
        for f, scores in p0_by_film.items():
            clean = [s for s in scores if not np.isnan(s)]
            if clean and (np.mean(np.asarray(clean) >= threshold) >= 0.5):
                frac_rule.append(f)
        metrics["video_retention_fraction_rule"] = len(frac_rule) / len(p0_by_film)
    else:
        metrics["video_retention_mean_rule"] = float("nan")
        metrics["video_retention_mean_rule_counts"] = [0, len(p0_by_film)]
        metrics["video_retention_fraction_rule"] = float("nan")

    return metrics


def evaluate_gates(metrics, coco_result):
    """Evaluate G1-G4. Returns {gate: {criterion, passed, observed}}."""
    def _ok(value):
        return bool(np.isfinite(value))

    auc_p1 = metrics["auc_p0_p1"]
    auc_p2 = metrics["auc_p0_p2"]
    sep = metrics["median_sep_p0_p2_sd_p2"]
    retention = metrics["video_retention_mean_rule"]

    g1_passed = _ok(auc_p1) and _ok(sep) and auc_p1 >= G1_AUC_MIN and sep >= G1_SEP_MIN
    g2_passed = _ok(auc_p2) and auc_p2 >= G2_AUC_MIN
    g4_passed = _ok(retention) and G4_RETENTION_MIN <= retention <= G4_RETENTION_MAX

    gates = {
        "G1": {
            "criterion": (
                f"AUC(P0 vs P1) >= {G1_AUC_MIN} AND median separation >= {G1_SEP_MIN} sd(P2)"
            ),
            "observed": f"AUC={fmt(auc_p1)}, sep={fmt(sep)}",
            "passed": bool(g1_passed),
        },
        "G2": {
            "criterion": f"AUC(P0 vs P2) >= {G2_AUC_MIN}",
            "observed": f"AUC={fmt(auc_p2)}",
            "passed": bool(g2_passed),
        },
        "G4": {
            "criterion": (
                f"{G4_RETENTION_MIN} <= video-level retention <= {G4_RETENTION_MAX}"
            ),
            "observed": f"retention={fmt(retention)}",
            "passed": bool(g4_passed),
        },
    }

    if coco_result and coco_result.get("status") == "ok":
        coco_auc = coco_result["auc_matched_vs_random"]
        gates["G3"] = {
            "criterion": f"COCO AUC(matched vs shuffled) >= {G3_AUC_MIN}",
            "observed": f"AUC={coco_auc:.4f}",
            "passed": bool(np.isfinite(coco_auc) and coco_auc >= G3_AUC_MIN),
        }
    else:
        gates["G3"] = {
            "criterion": f"COCO AUC(matched vs shuffled) >= {G3_AUC_MIN}",
            "observed": coco_result.get("status", "not_run") if coco_result else "not_run",
            "passed": None,
        }
    return gates


def make_verdict(gates):
    if gates["G1"]["passed"] is not True:
        return (
            "FAIL G1: instrument tracks film topic, not window-level alignment; "
            "it must not be used to filter windows."
        )
    if gates["G2"]["passed"] is not True:
        return (
            "FAIL G2: cross-film separation too weak; window-level signal not "
            "demonstrated."
        )
    if gates["G4"]["passed"] is not True:
        return (
            "FAIL G4: implied video-level retention is outside 60-95%; the "
            "threshold is vacuous or destructive."
        )
    return (
        "PASS: window-level alignment is detectable (G1), cross-film separation "
        "is strong (G2), and the 95%-specificity threshold retains a sane share "
        "of films (G4)."
    )


# ---------------------------------------------------------------------------
# COCO sanity check (G3) -- reuse the existing production path
# ---------------------------------------------------------------------------

def run_coco_sanity(model_id, batch_size, use_fp16, device, seed, annotations,
                    images, limit):
    """Run ``check_correspondance.check_coco_clip_scores`` and summarise it.

    The scoring itself is NOT reimplemented: the existing function is invoked
    with its module globals overridden, and its saved output is summarised.
    """
    from src.preprocess import check_correspondance as cc

    cc.CLIP_MODEL = model_id
    cc._BATCH_SIZE = batch_size
    cc._USE_FP16 = use_fp16
    cc.DEVICE = device
    cc.COCO_ANNOTATIONS = annotations
    cc.COCO_IMAGES = images
    cc.COCO_LIMIT = limit

    random.seed(seed)
    cc.check_coco_clip_scores()

    out_path = os.path.join(PROJECT_ROOT, "data/clip_baseline_scores_coco.json")
    if not os.path.exists(out_path):
        return {"status": "no_output", "path": out_path}

    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    matched = np.asarray(data.get("matched", []), dtype=np.float64)
    rand = np.asarray(data.get("random", []), dtype=np.float64)
    if matched.size == 0 or rand.size == 0:
        return {"status": "empty_output", "path": out_path}

    std_rand = float(rand.std())
    return {
        "status": "ok",
        "path": out_path,
        "n_matched": int(matched.size),
        "n_random": int(rand.size),
        "matched_mean": float(matched.mean()),
        "random_mean": float(rand.mean()),
        "auc_matched_vs_random": auc_score(matched, rand),
        "median_sep_matched_vs_random_sd_random": (
            float(np.median(matched) - np.median(rand)) / std_rand
            if std_rand > 0
            else float("nan")
        ),
        "matched_pass_at_random_p95": float(
            np.mean(matched >= np.percentile(rand, 95))
        ),
    }


# ---------------------------------------------------------------------------
# Local manifest builder ("--build-manifest")
# ---------------------------------------------------------------------------

def build_local_manifest(transcripts_dir, videos_dir, out_path):
    """Build a window manifest from local transcripts + videos.

    Maps each transcript (by stem) to a video with the same stem, merges
    consecutive transcript segments into [3, 10]s windows via the shared
    ``merge_segments`` helper, and writes the manifest in the shape used by
    ``data/qc_pilot_manifest.json``.
    """
    transcripts_dir = resolve_path(transcripts_dir)
    videos_dir = resolve_path(videos_dir)

    video_index = {}
    for name in sorted(os.listdir(videos_dir)):
        stem, ext = os.path.splitext(name)
        if ext.lower() in VIDEO_EXTENSIONS:
            video_index[stem] = os.path.join(videos_dir, name)

    windows = []
    for name in sorted(os.listdir(transcripts_dir)):
        if not name.endswith(".json"):
            continue
        stem = name[: -len(".json")]
        video_path = video_index.get(stem)
        if video_path is None:
            logger.warning("No local video found for transcript %s", name)
            continue

        with open(os.path.join(transcripts_dir, name), "r", encoding="utf-8") as f:
            transcript = json.load(f)

        merged = merge_segments(
            transcript,
            min_duration=BUILD_MIN_DURATION,
            max_duration=BUILD_MAX_DURATION,
            max_gap=BUILD_MAX_GAP,
        )
        rel_video_path = to_relative(video_path)
        for w in merged:
            wid = "{}_{:.2f}_{:.2f}".format(stem, w["start_sec"], w["end_sec"])
            windows.append(
                {
                    "id": wid,
                    "video_id": stem,
                    "video_path": rel_video_path,
                    "start_sec": w["start_sec"],
                    "end_sec": w["end_sec"],
                    "text": w["text"],
                    "n_segments": w["n_segments"],
                }
            )

    out_abs = resolve_path(out_path)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as f:
        json.dump(windows, f, indent=2, ensure_ascii=False)

    logger.info(
        "Built local manifest: %d windows from %d films -> %s",
        len(windows),
        len({w["video_id"] for w in windows}),
        out_path,
    )
    return out_abs


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def fmt(value, nd=4):
    if value is None:
        return "n/a"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(f):
        return "n/a"
    return f"{f:.{nd}f}"


def render_markdown(report):
    m = report["metrics"]
    gates = report["gates"]
    lines = []
    lines.append(f"# Perturbation harness - {report['instrument']} / `{report['model']}`")
    lines.append("")
    lines.append(report["description"])
    lines.append("")
    lines.append(
        "Profile: n_windows={n_windows}, n_films={n_films}, frames={frames}, "
        "seed={seed}, device={device}, batch_size={batch_size}".format(**report["params"])
    )
    lines.append("")

    lines.append("## Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| AUC(P0 vs P1) | {fmt(m.get('auc_p0_p1'))} |")
    lines.append(f"| AUC(P0 vs P2) | {fmt(m.get('auc_p0_p2'))} |")
    lines.append(
        f"| Median separation P0-P2 (sd of P2) | "
        f"{fmt(m.get('median_sep_p0_p2_sd_p2'))} |"
    )
    lines.append(
        f"| Dynamic range P0-P4 (mean) | "
        f"{fmt(m.get('dynamic_range_mean_p0_p4'))} |"
    )
    lines.append(
        f"| Dynamic range P0-P4 (median) | "
        f"{fmt(m.get('dynamic_range_median_p0_p4'))} |"
    )
    lines.append(
        f"| Threshold @ {fmt(m.get('threshold_specificity'), 2)} specificity vs P2 | "
        f"{fmt(m.get('threshold_at_95_spec_vs_p2'))} |"
    )
    lines.append(
        f"| Video-level retention (mean rule) | "
        f"{fmt(m.get('video_retention_mean_rule'))} "
        f"{m.get('video_retention_mean_rule_counts')} |"
    )
    lines.append(
        f"| Video-level retention (fraction rule) | "
        f"{fmt(m.get('video_retention_fraction_rule'))} |"
    )
    lines.append("")

    lines.append("## Gates")
    lines.append("")
    lines.append("| Gate | Criterion | Observed | Result |")
    lines.append("|---|---|---|---|")
    for gate in ["G1", "G2", "G3", "G4"]:
        g = gates.get(gate, {})
        passed = g.get("passed")
        result = "PASS" if passed is True else ("FAIL" if passed is False else "SKIPPED")
        lines.append(
            f"| {gate} | {g.get('criterion', '')} | {g.get('observed', '')} | {result} |"
        )
    lines.append("")

    lines.append("## Per-rung score distributions")
    lines.append("")
    lines.append("| Rung | n | mean | median | std | p5 | p95 |")
    lines.append("|---|---|---|---|---|---|---|")
    for rung in ["P0", "P1", "P2", "P3", "P4"]:
        d = report["rung_stats"].get(rung)
        if not d or d.get("count", 0) == 0:
            lines.append(f"| {rung} | 0 | - | - | - | - | - |")
            continue
        lines.append(
            f"| {rung} | {d['count']} | {fmt(d['mean'])} | {fmt(d['median'])} | "
            f"{fmt(d['std'])} | {fmt(d['p5'])} | {fmt(d['p95'])} |"
        )
    lines.append("")

    if report.get("coco"):
        coco = report["coco"]
        lines.append("## COCO sanity check (G3)")
        lines.append("")
        if coco.get("status") == "ok":
            lines.append(
                f"Matched={fmt(coco['matched_mean'])} random={fmt(coco['random_mean'])} "
                f"AUC={fmt(coco['auc_matched_vs_random'])} "
                f"(n={coco['n_matched']}/{coco['n_random']})."
            )
        else:
            lines.append(f"Status: {coco.get('status')}.")
        lines.append("")

    lines.append("## Verdict")
    lines.append("")
    lines.append(report["verdict"])
    lines.append("")
    return "\n".join(lines)


def slugify_model(model_id):
    return model_id.replace("/", "__").replace(" ", "_")


def default_out_path(instrument, model_id):
    return os.path.join(
        DEFAULT_OUT_DIR, f"{instrument}_{slugify_model(model_id)}.json"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Perturbation-ladder validation for image-text alignment instruments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help=(
            "Window manifest (default: "
            f"{DEFAULT_MANIFEST}, used unless --coco is given alone)"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"HuggingFace image-text model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="clip",
        choices=sorted(INSTRUMENT_REGISTRY.keys()),
        help="Instrument implementation (default: clip)",
    )
    parser.add_argument(
        "--n-windows",
        type=int,
        default=2000,
        help="Number of windows to sample (default: 2000)",
    )
    parser.add_argument(
        "--n-films",
        type=int,
        default=200,
        help="Number of films to sample windows from (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for sampling and shuffles (default: 42)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=8,
        help="Frames sampled per window (default: 8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=CLIP_BATCH_SIZE,
        help=f"(text, frame) pairs per forward pass (default: {CLIP_BATCH_SIZE})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device (default: auto-detect)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path (default: perturbation_results/<instrument>_<model>.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Threads for parallel video decode (default: 8)",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        default=True,
        help="Disable fp16 autocast on CUDA",
    )
    parser.add_argument(
        "--enable-p3",
        action="store_true",
        help="Enable the paraphrase rung (requires --paraphraser)",
    )
    parser.add_argument(
        "--paraphraser",
        type=str,
        default=None,
        help="HuggingFace seq2seq id for P3 (e.g. a T5 paraphrase model)",
    )
    parser.add_argument(
        "--coco",
        action="store_true",
        help="Run the COCO matched-vs-shuffled sanity check (G3)",
    )
    parser.add_argument(
        "--coco-annotations",
        type=str,
        default="~/corpora/MSCOCO-2017/captions_val2017.json",
        help="MSCOCO captions JSON for --coco",
    )
    parser.add_argument(
        "--coco-images",
        type=str,
        default="~/corpora/MSCOCO-2017/val2017/",
        help="MSCOCO val image root for --coco",
    )
    parser.add_argument(
        "--coco-limit",
        type=int,
        default=0,
        help="Cap COCO images scored (0 = all)",
    )
    parser.add_argument(
        "--build-manifest",
        type=str,
        default=None,
        help="Build a window manifest from local transcripts/videos at this path, "
             "then run",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build the manifest and exit",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=str,
        default="data/transcripts",
        help="Transcript dir for --build-manifest",
    )
    parser.add_argument(
        "--videos-dir",
        type=str,
        default="data/videos",
        help="Video dir for --build-manifest",
    )
    return parser.parse_args()


def pick_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; using CPU")
            return torch.device("cpu")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    setup_logging()
    args = parse_args()
    random.seed(args.seed)
    rng = random.Random(args.seed)
    device = pick_device(args.device)
    logger.info("Device: %s", device)

    manifest_explicit = args.manifest is not None
    manifest_path = resolve_path(args.manifest or DEFAULT_MANIFEST)
    if args.build_manifest:
        manifest_path = build_local_manifest(
            args.transcripts_dir, args.videos_dir, args.build_manifest
        )
        manifest_explicit = True
        if args.build_only:
            print(f"wrote {args.build_manifest}")
            return

    # `--coco` alone is a standalone sanity check: do not silently run the
    # ladder on the default manifest too.
    run_ladder = (
        args.n_windows > 0
        and os.path.exists(manifest_path)
        and (not args.coco or manifest_explicit)
    )
    if not run_ladder and not args.coco:
        logger.error(
            "No usable manifest at %s and --coco not set. Nothing to do.",
            manifest_path,
        )
        sys.exit(1)

    coco_result = None
    if args.coco:
        logger.info("Running COCO sanity check via the existing baseline path...")
        coco_result = run_coco_sanity(
            args.model, args.batch_size, args.fp16, device, args.seed,
            args.coco_annotations, args.coco_images, args.coco_limit,
        )
        logger.info("COCO sanity: %s", coco_result)

    if not run_ladder:
        report = {
            "instrument": args.instrument,
            "model": args.model,
            "description": (
                "COCO-only sanity check: no manifest was available, so the "
                "perturbation ladder did not run."
            ),
            "params": {
                "n_windows": 0, "n_films": 0, "frames": args.frames,
                "seed": args.seed, "device": str(device),
                "batch_size": args.batch_size,
            },
            "metrics": {},
            "rung_stats": {},
            "gates": evaluate_gates(
                {
                    "auc_p0_p1": float("nan"),
                    "auc_p0_p2": float("nan"),
                    "median_sep_p0_p2_sd_p2": float("nan"),
                    "video_retention_mean_rule": float("nan"),
                },
                coco_result,
            ),
            "coco": coco_result,
        }
        if coco_result and coco_result.get("status") == "ok":
            g3 = report["gates"]["G3"]
            report["verdict"] = (
                "COCO sanity "
                + ("PASS" if g3["passed"] else "FAIL")
                + f" (G3: {g3['observed']}). No manifest was available, so the "
                "window ladder did not run."
            )
        else:
            report["verdict"] = (
                "COCO sanity check could not run (status="
                f"{coco_result.get('status') if coco_result else 'not_run'}). "
                "No manifest was available, so the window ladder did not run."
            )
        write_reports(report, args.out)
        return

    logger.info("Loading manifest %s", manifest_path)
    manifest = load_manifest(manifest_path)
    windows, film_ids = select_windows(
        manifest, args.n_windows, args.n_films, args.seed
    )
    logger.info(
        "Selected %d windows across %d films", len(windows), len(film_ids)
    )
    if not windows:
        logger.error("No windows selected; aborting.")
        sys.exit(1)

    t0 = time.time()
    frames_cache = decode_frames(windows, args.frames, args.workers)
    logger.info(
        "Decoded frames for %d/%d windows in %.1fs",
        len(frames_cache), len(windows), time.time() - t0,
    )

    windows = [w for w in windows if window_id_of(w) in frames_cache]
    film_ids = [f for f in film_ids if any(film_id_of(w) == f for w in windows)]
    if not windows:
        logger.error("No windows decoded successfully; aborting.")
        sys.exit(1)

    ladder, film_windows = build_ladder(windows, film_ids, rng, args.enable_p3)

    # P3: paraphrase the true text (real model or explicit skip).
    p3_status = {"status": "disabled"}
    if args.enable_p3:
        if not args.paraphraser:
            p3_status = {
                "status": "skipped",
                "reason": (
                    "no local paraphrasing route configured; pass --paraphraser "
                    "with a HuggingFace seq2seq model id. Not faked."
                ),
            }
            logger.warning("P3 skipped: %s", p3_status["reason"])
        else:
            unique_texts = list(dict.fromkeys(t for _, t in ladder["P0"]))
            paraphrases = paraphrase_texts(
                unique_texts, args.paraphraser, device, args.batch_size
            )
            para_by_text = dict(zip(unique_texts, paraphrases))
            for w, text in ladder["P0"]:
                ladder["P3"].append((w, para_by_text.get(text, text)))
            p3_status = {"status": "ok", "paraphraser": args.paraphraser}

    instrument = build_instrument(
        args.instrument, args.model, device, args.batch_size, args.fp16
    )

    rung_scores = {}
    window_scores = []
    for rung in ["P0", "P1", "P2", "P3", "P4"]:
        items = ladder.get(rung, [])
        if not items:
            rung_scores[rung] = []
            continue
        score_items = [(text, frames_cache[window_id_of(w)]) for w, text in items]
        scores = instrument.score_items(score_items)
        rung_scores[rung] = scores
        for (w, text), sc in zip(items, scores):
            window_scores.append(
                {"window": window_id_of(w), "film": film_id_of(w), "rung": rung,
                 "text": text, "score": sc}
            )
        logger.info("Scored rung %s: %d items", rung, len(scores))

    p0_by_film = defaultdict(list)
    for entry in window_scores:
        if entry["rung"] == "P0":
            p0_by_film[entry["film"]].append(entry["score"])

    metrics = compute_metrics(rung_scores, p0_by_film)
    gates = evaluate_gates(metrics, coco_result)
    verdict = make_verdict(gates)

    rung_stats = {
        rung: distribution_stats(rung_scores.get(rung, []))
        for rung in ["P0", "P1", "P2", "P3", "P4"]
    }

    report = {
        "instrument": args.instrument,
        "model": args.model,
        "description": (
            "Perturbation ladder over sampled windows. P0=matched, P1=within-film "
            "shuffle, P2=cross-film shuffle, P3=paraphrase, P4=filler."
        ),
        "params": {
            "manifest": args.manifest,
            "n_windows": len(windows),
            "n_windows_requested": args.n_windows,
            "n_films": len(film_ids),
            "n_films_requested": args.n_films,
            "frames": args.frames,
            "seed": args.seed,
            "device": str(device),
            "batch_size": args.batch_size,
            "p1_min_index_distance": P1_MIN_INDEX_DISTANCE,
            "filler_texts": FILLER_TEXTS,
        },
        "p3_status": p3_status,
        "metrics": metrics,
        "rung_stats": rung_stats,
        "gates": gates,
        "coco": coco_result,
        "verdict": verdict,
        "window_scores": window_scores,
    }
    write_reports(report, args.out)


def write_reports(report, out_arg):
    out_path = resolve_path(out_arg) if out_arg else os.path.join(
        PROJECT_ROOT, default_out_path(report["instrument"], report["model"])
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_path = os.path.splitext(out_path)[0] + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(report))

    logger.info("Wrote %s", out_path)
    logger.info("Wrote %s", md_path)
    print(f"wrote {out_path}")
    print(f"wrote {md_path}")
    print("")
    print(report["verdict"])


if __name__ == "__main__":
    main()
