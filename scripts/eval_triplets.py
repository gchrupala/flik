#!/usr/bin/env python
"""Controlled two-alternative forced-choice (2AFC) evaluation for FlikModel.

One item is ``(audio anchor, positive video, negative video)``. Accuracy is the
mean of ``sign(cos(anchor, positive) - cos(anchor, negative))``; ties count 0.5,
so chance is 0.5. The reverse direction (video anchor against audio candidates)
runs on the same items. This is a discriminative minimal-pair / forced-choice
probe: it asks whether the shared space recovers the *right* pairing rather than
whether retrieval looks good on average.

Design source (replicated, not invented):

    Nikolaus, Alishahi & Chrupała (2022), "Learning English with Peppa Pig",
    Transactions of the ACL 10:293-307. doi:10.1162/tacl_a_00498.
    Reference implementation: https://github.com/gchrupala/peppa
    Pinned revision: c92719d5e5abf5a29e776a09c42a2a2f6c74a10c
    License: Apache-2.0 (that repository's LICENSE).

    The reference code is NOT a runtime dependency: the logic below is
    reimplemented, so this script runs without it. To diff against the
    original locally:

        git clone https://github.com/gchrupala/peppa vendor/peppa
        git -C vendor/peppa checkout c92719d5e5abf5a29e776a09c42a2a2f6c74a10c

    ``vendor/`` is gitignored, so that checkout is optional and never
    committed.

Reused reference logic (mapped to functions below):

  * ``pig.metrics.triplet_accuracy`` -- cosine-difference sign rule.
    -> ``triplet_accuracy`` below.
  * ``pig.triplet._triplets`` / ``pairs`` / ``pig.util.grouped`` -- group clips
    by duration, pair them at random, use each clip at most once.
    -> ``build_triplets`` / ``_pair_*`` below (grouping is by duration BIN so
    the corpus is not restricted to one exact clip length).
  * ``evaluation_targeted_triplets.get_bootstrapped_scores`` -- bootstrap
    resampling of per-sample accuracies. -> ``cluster_bootstrap_ci`` below,
    extended so the resampling unit is the FILM (segments inside one film are
    correlated, so the film is the cluster).
  * ``pig.targeted_triplets.PeppaTargetedTripletCachedDataset(scrambled_video=)``
    and ``hparams_static.yaml`` -- temporal ablations.
    -> ``--scramble-frames`` and ``--static-frames`` below.

Nested distractor conditions (the scientific core):

  1. ``cross_film``       -- negative from a different film. Baseline; still
                             leaks domain / era / cast / quality cues.
  2. ``same_film``        -- negative from the same film. Film-level cues are
                             matched, so only content can decide.
  3. ``speaker_matched``  -- negative from the same speaker as the anchor.
                             Runs only when speaker labels exist (manifest
                             ``speaker`` field, ``--speaker-map``, or
                             ``--diarize-mfcc``). Otherwise it is OMITTED and
                             the reason is reported; it is never faked.

Held-out films only. Pass ``--holdout-films`` (JSON list or one id per line) or
let the script reuse ``scripts.train_hydra._split_manifest_by_video``. A negative
never shares the anchor's movie identity (``--movie-map``).

Usage:
    uv run --extra cu128 python -m scripts.eval_triplets \
        --checkpoint aug13_best:checkpoints/checkpoint_best.pth \
        --random-init --n-trials 4000

    # CPU smoke run on the two local films (no checkpoint):
    uv run --extra cpu python -m scripts.eval_triplets \
        --films Black_Brigade TheStilwellRoad \
        --random-init --device cpu --n-trials 8 --workers 1
"""

import argparse
import bisect
import json
import logging
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Match scripts/train_hydra.py and scripts/eval_zeroshot_vggsound.py: HF cache
# lives in the checkout, and the .pth dicts are not plain state dicts.
os.environ.setdefault("HF_HOME", os.path.join(_REPO_ROOT, "cache"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "true")

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

CONDITIONS = ("cross_film", "same_film", "speaker_matched")
DIRECTIONS = ("a2v", "v2a")
RANDOM_INIT_NAME = "random_init"

# 80% power, two-sided alpha=0.05, a balanced test, a small effect (p0 near 0.5):
#   n ~= 7.85 * p0 * (1 - p0) / delta^2
MDE_CONSTANT = 7.85


# ---------------------------------------------------------------------------
# Reference logic from github.com/gchrupala/peppa (Apache-2.0)
# ---------------------------------------------------------------------------

def triplet_accuracy(anchor: torch.Tensor, positive: torch.Tensor,
                     negative: torch.Tensor, dim: int = -1) -> float:
    """Sign rule of ``pig.metrics.triplet_accuracy`` for one item.

    Returns 1.0 when the positive candidate is closer to the anchor, 0.0 when
    the negative is closer and 0.5 on an exact tie (``torch.sign(0) == 0``),
    which is the random-guess value. ``pig.metrics`` does the same without the
    explicit tie branch.
    """
    sim_pos = F.cosine_similarity(anchor, positive, dim=dim)
    sim_neg = F.cosine_similarity(anchor, negative, dim=dim)
    diff = float(sim_pos - sim_neg)
    if diff > 0:
        return 1.0
    if diff < 0:
        return 0.0
    return 0.5


def min_detectable_effect(n: int, p0: float = 0.5) -> float:
    """Smallest detectable accuracy shift over chance at 80% power.

    Inverts ``n ~= 7.85 * p0 * (1 - p0) / delta^2`` (the formula requested for
    this eval). Chance is p0 = 0.5, so the return value is a fraction (multiply
    by 100 for percentage points).
    """
    if n <= 0:
        return float("nan")
    return math.sqrt(MDE_CONSTANT * p0 * (1.0 - p0) / float(n))


def cluster_bootstrap_ci(values: Sequence[float], clusters: Sequence[str],
                         n_iter: int = 2000, seed: int = 42,
                         alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI with the FILM as the resampling unit.

    Mirrors ``qc_significance.cluster_bootstrap`` (same resample-the-group,
    pool, take percentiles pattern) and generalises
    ``evaluation_targeted_triplets.get_bootstrapped_scores``, which resamples
    individual items. Resampling items would understate the interval whenever
    failures or successes cluster inside a film; resampling films is honest.
    """
    groups: Dict[str, List[float]] = defaultdict(list)
    for v, c in zip(values, clusters):
        groups[c].append(v)
    keys = sorted(groups.keys())
    if not keys:
        return (float("nan"), float("nan"))
    if len(keys) == 1:
        # One cluster: between-film variance is unidentifiable. Every resample
        # returns the same pooled value, so the interval collapses to a point.
        m = sum(groups[keys[0]]) / len(groups[keys[0]])
        return (m, m)

    rng = random.Random(seed)
    means = []
    for _ in range(n_iter):
        pooled = []
        for _ in range(len(keys)):
            k = keys[rng.randrange(len(keys))]
            pooled.extend(groups[k])
        means.append(sum(pooled) / len(pooled) if pooled else float("nan"))
    means.sort()
    lo = means[int((alpha / 2.0) * n_iter)]
    hi = means[min(n_iter - 1, int((1.0 - alpha / 2.0) * n_iter))]
    return (lo, hi)


# ---------------------------------------------------------------------------
# Triplet construction (duration-bin matched, each clip used at most once)
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """One decodable manifest segment (audio and video share the file)."""
    segment_id: str
    video_id: str
    video_path: str
    start_sec: float
    end_sec: float
    movie_id: str = ""
    speaker: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec


@dataclass
class Triplet:
    positive: Segment
    negative: Segment
    condition: str
    bin_index: int


@dataclass
class TripletSet:
    """Built items plus the information needed to audit the construction."""
    condition: str
    items: List[Triplet]
    bin_edges: List[float] = field(default_factory=list)
    bin_of: Dict[str, int] = field(default_factory=dict)


def assign_duration_bins(segments: Sequence[Segment], n_bins: int
                         ) -> Tuple[Dict[str, int], List[float]]:
    """Quantile duration bins; returns (segment_id -> bin index, bin edges).

    Quantile edges give every bin the same number of candidate clips, so a
    scarce condition can still find a matched negative. ``n_bins <= 1`` puts
    all segments in bin 0.
    """
    if not segments:
        return {}, []
    if n_bins <= 1:
        return {s.segment_id: 0 for s in segments}, []

    durations = sorted(s.duration for s in segments)
    edges = []
    for i in range(1, n_bins):
        # Nearest-rank quantile; deterministic and dependency-free.
        pos = int(round(i * (len(durations) - 1) / n_bins))
        edges.append(durations[pos])
    # Deduplicate while preserving order (ties can repeat an edge).
    deduped = []
    for e in edges:
        if not deduped or e > deduped[-1]:
            deduped.append(e)
    edges = deduped

    bin_of = {}
    for s in segments:
        idx = bisect.bisect_right(edges, s.duration)
        bin_of[s.segment_id] = idx
    return bin_of, edges


def _alternate(a: Segment, b: Segment, rng: random.Random
               ) -> Tuple[Segment, Segment]:
    """Randomly decide which member of a pair is the positive."""
    return (b, a) if rng.random() < 0.5 else (a, b)


def _adjacent_pairs(group: Sequence[Segment], rng: random.Random
                    ) -> List[Tuple[Segment, Segment]]:
    """Peppa ``pairs(shuffled(xs))``: shuffle then pair adjacent elements.

    Each clip lands in at most one pair, so it can be used by at most one item.
    """
    pool = list(group)
    rng.shuffle(pool)
    out = []
    for i in range(0, len(pool) - 1, 2):
        out.append(_alternate(pool[i], pool[i + 1], rng))
    return out


def _pair_same_film(group: Sequence[Segment], rng: random.Random
                    ) -> List[Tuple[Segment, Segment]]:
    """Pair only inside one film: both candidates share film-level cues."""
    by_film: Dict[str, List[Segment]] = defaultdict(list)
    for s in group:
        by_film[s.video_id].append(s)
    out = []
    for film in sorted(by_film):
        out.extend(_adjacent_pairs(by_film[film], rng))
    return out


def _pair_cross_film(group: Sequence[Segment], rng: random.Random,
                     movie_of) -> List[Tuple[Segment, Segment]]:
    """Pair clips from DIFFERENT movie identities.

    Greedy matching: take the first clip, pair it with the first remaining clip
    whose movie identity differs, drop both. Clips with no eligible partner are
    discarded, so the constraint "no positive/negative share a movie identity"
    holds exactly.
    """
    pool = list(group)
    rng.shuffle(pool)
    out = []
    while len(pool) >= 2:
        a = pool.pop(0)
        for j, b in enumerate(pool):
            if movie_of(a) != movie_of(b):
                pool.pop(j)
                out.append(_alternate(a, b, rng))
                break
    return out


def _pair_speaker_matched(group: Sequence[Segment], rng: random.Random
                          ) -> List[Tuple[Segment, Segment]]:
    """Pair clips whose speaker label matches (same duration bin already).

    Segments without a speaker label are excluded. A speaker is almost always
    tied to one film, so this condition is usually a strict subset of
    ``same_film`` (same film AND same speaker); the composition is reported in
    the JSON.
    """
    by_speaker: Dict[str, List[Segment]] = defaultdict(list)
    for s in group:
        if s.speaker is not None:
            by_speaker[s.speaker].append(s)
    out = []
    for spk in sorted(by_speaker):
        out.extend(_adjacent_pairs(by_speaker[spk], rng))
    return out


def build_triplets(segments: Sequence[Segment], condition: str, n_trials: int,
                   n_bins: int = 10, seed: int = 42,
                   movie_of=None, speaker_of=None) -> TripletSet:
    """Build one condition's items from held-out segments.

    Positives and negatives always come from the same duration bin, and each
    clip appears in at most one item within the condition. The item list is
    shuffled with ``seed`` and truncated to ``n_trials`` (0 = use all).
    """
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    movie_of = movie_of or (lambda s: s.movie_id or s.video_id)

    ordered = sorted(segments, key=lambda s: (s.video_id, s.start_sec,
                                              s.segment_id))
    bin_of, edges = assign_duration_bins(ordered, n_bins)
    by_bin: Dict[int, List[Segment]] = defaultdict(list)
    for s in ordered:
        by_bin[bin_of[s.segment_id]].append(s)

    rng = random.Random(seed)
    items: List[Triplet] = []
    for bin_index in sorted(by_bin):
        group = by_bin[bin_index]
        if condition == "cross_film":
            pairs = _pair_cross_film(group, rng, movie_of)
        elif condition == "same_film":
            pairs = _pair_same_film(group, rng)
        else:
            pairs = _pair_speaker_matched(group, rng)
        for pos, neg in pairs:
            items.append(Triplet(pos, neg, condition, bin_index))

    rng.shuffle(items)
    if n_trials and len(items) > n_trials:
        items = items[:n_trials]
    return TripletSet(condition=condition, items=items, bin_edges=edges,
                      bin_of=bin_of)


def validate_triplets(triplet_set: TripletSet, movie_of=None) -> None:
    """Assert the construction invariants; raises AssertionError on violation.

    Used by the unit tests and defensively before scoring. Cheap enough to run
    on every build.
    """
    movie_of = movie_of or (lambda s: s.movie_id or s.video_id)
    seen = set()
    for t in triplet_set.items:
        assert t.bin_index == triplet_set.bin_of[t.positive.segment_id], t
        assert t.bin_index == triplet_set.bin_of[t.negative.segment_id], t
        if t.condition == "same_film":
            assert t.positive.video_id == t.negative.video_id, t
        elif t.condition == "cross_film":
            assert t.positive.video_id != t.negative.video_id, t
            assert movie_of(t.positive) != movie_of(t.negative), t
        elif t.condition == "speaker_matched":
            assert t.positive.speaker == t.negative.speaker, t
        for s in (t.positive, t.negative):
            assert s.segment_id not in seen, f"clip reused: {s.segment_id}"
            seen.add(s.segment_id)


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def _movie_key(video_id: str, movie_map: Dict[str, str]) -> str:
    return movie_map.get(video_id, video_id)


def load_movie_map(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--movie-map must be a JSON object {video_id: movie_id}")
    return {str(k): str(v) for k, v in data.items()}


def load_segments_from_manifest(path: str, movie_map: Dict[str, str]
                                ) -> List[Segment]:
    """Load a segment-level manifest (start_sec/end_sec present)."""
    from src.utils.paths import resolve_path

    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    segments = []
    for it in items:
        if "start_sec" not in it or "end_sec" not in it:
            raise ValueError(
                "manifest is not segment-level (missing start_sec/end_sec); "
                "run scripts/expand_and_validate_manifest first")
        video_path = resolve_path(it.get("video_path") or it.get("audio_path"))
        video_id = it.get("video_id") or os.path.splitext(
            os.path.basename(video_path))[0]
        segments.append(Segment(
            segment_id=it.get("id") or f"{video_id}_{it['start_sec']}_{it['end_sec']}",
            video_id=video_id,
            video_path=video_path,
            start_sec=float(it["start_sec"]),
            end_sec=float(it["end_sec"]),
            movie_id=_movie_key(video_id, movie_map),
            speaker=it.get("speaker"),
        ))
    return segments


def load_segments_from_transcripts(films: Sequence[str], videos_dir: str,
                                   transcripts_dir: str, min_duration: float,
                                   max_duration: float, max_gap: float,
                                   movie_map: Dict[str, str]) -> List[Segment]:
    """Build segments for the given films with the TRAINING segment definition.

    Reuses ``src.utils.segments.merge_segments`` (the same window logic as
    ``scripts/expand_and_validate_manifest.py`` and the dataset), so smoke runs
    and shortlisted-film runs use exactly the training segmentation.
    """
    from src.utils.segments import merge_segments

    segments = []
    for film in films:
        transcript = os.path.join(transcripts_dir, f"{film}.json")
        if not os.path.exists(transcript):
            raise FileNotFoundError(f"transcript not found: {transcript}")
        video_path = os.path.join(videos_dir, f"{film}.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"video not found: {video_path}")
        with open(transcript, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for w in merge_segments(raw, min_duration=min_duration,
                                max_duration=max_duration, max_gap=max_gap):
            segments.append(Segment(
                segment_id=f"{film}_{w['start_sec']:.2f}_{w['end_sec']:.2f}",
                video_id=film,
                video_path=video_path,
                start_sec=w["start_sec"],
                end_sec=w["end_sec"],
                movie_id=_movie_key(film, movie_map),
            ))
    return segments


def load_speaker_map(path: Optional[str]) -> Dict[str, str]:
    """Load a precomputed segment -> speaker assignment.

    Accepts a JSON object whose keys are either segment ids or
    ``video_id|start_sec|end_sec`` strings. Produced externally, e.g. by
    pyannote diarization or MFCC/i-vector clustering.
    """
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("--speaker-map must be a JSON object")
    return {str(k): str(v) for k, v in data.items()}


def attach_speakers(segments: Sequence[Segment], speaker_map: Dict[str, str]
                    ) -> int:
    """Attach speakers from a map; returns the number of segments labelled."""
    n = 0
    for s in segments:
        if s.speaker is not None:
            n += 1
            continue
        key = f"{s.video_id}|{s.start_sec}|{s.end_sec}"
        spk = speaker_map.get(s.segment_id) or speaker_map.get(key)
        if spk is not None:
            s.speaker = spk
            n += 1
    return n


def diarize_mfcc(segments: Sequence[Segment], sample_rate: int = 16000,
                 n_clusters: int = 4, align_to_start: bool = True) -> None:
    """Pseudo-speaker labels by KMeans over mean MFCCs, fitted per film.

    This is a heuristic (no ground truth), used only to make the
    ``speaker_matched`` condition runnable when no diarization is available.
    Labelled ``mfcc_kmeans`` in the output so it cannot be mistaken for a
    real speaker representation.

    ``align_to_start`` is forwarded to ``load_audio_from_video`` so the MFCC
    features are computed on the same (aligned) audio windows as the triplet
    encoding.
    """
    from sklearn.cluster import KMeans
    import torchaudio

    from src.utils.audio import load_audio_from_video

    mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=20)
    by_film: Dict[str, List[Segment]] = defaultdict(list)
    for s in segments:
        by_film[s.video_id].append(s)

    for film, segs in by_film.items():
        feats = []
        for s in segs:
            wav = load_audio_from_video(s.video_path, s.start_sec, s.end_sec,
                                        target_sample_rate=sample_rate,
                                        align_to_start=align_to_start)
            feats.append(mfcc(wav).mean(dim=2).squeeze(0).numpy())
        k = max(1, min(n_clusters, len(segs)))
        if k < 2:
            labels = [0] * len(segs)
        else:
            labels = KMeans(n_clusters=k, n_init=10,
                            random_state=0).fit(feats).labels_
        for s, lab in zip(segs, labels):
            s.speaker = f"{film}:spk{int(lab)}"


def resolve_holdout(segments: List[Segment], holdout_path: Optional[str],
                    val_ratio: float, seed: int, logger) -> List[Segment]:
    """Restrict to held-out films.

    ``--holdout-films`` (JSON list or newline file) wins. Otherwise reuse
    ``scripts.train_hydra._split_manifest_by_video`` so the held-out films match
    the training split semantics (whole movies, deterministic seed).
    """
    if holdout_path:
        if holdout_path.endswith(".json"):
            with open(holdout_path, "r", encoding="utf-8") as f:
                films = json.load(f)
        else:
            with open(holdout_path, "r", encoding="utf-8") as f:
                films = [ln.strip() for ln in f if ln.strip()]
        allowed = set(str(x) for x in films)
        kept = [s for s in segments if s.video_id in allowed]
        logger.info("Holdout file: %d films requested, %d segments kept "
                    "(%d/%d film ids present)", len(allowed), len(kept),
                    len({s.video_id for s in kept}),
                    len({s.video_id for s in segments}))
        return kept

    from scripts.train_hydra import _split_manifest_by_video

    raw = [{"video_id": s.video_id, "start_sec": s.start_sec,
            "end_sec": s.end_sec} for s in segments]
    _, val_items = _split_manifest_by_video(raw, val_ratio, seed)
    val_keys = {(it["video_id"], it["start_sec"], it["end_sec"])
                for it in (val_items or [])}
    key = lambda s: (s.video_id, s.start_sec, s.end_sec)
    if not val_keys:  # val_ratio <= 0 or a single film: fall back to all films
        logger.warning("Split returned no validation films; using all films")
        return list(segments)
    kept = [s for s in segments if key(s) in val_keys]
    logger.info("Train-split holdout: %d films, %d segments",
                len({s.video_id for s in kept}), len(kept))
    return kept


# ---------------------------------------------------------------------------
# Decode + encode
# ---------------------------------------------------------------------------

def _apply_video_ablation(video: torch.Tensor, mode: Optional[str],
                          seed: int) -> torch.Tensor:
    """Temporal ablation on an (F, 3, H, W) clip.

    ``scramble`` shuffles frame order; ``static`` repeats the middle frame.
    Both destroy temporal structure while keeping the same tensor shape, which
    is the Peppa ``scrambled_video`` / ``hparams_static`` manipulation. The
    transform is applied to EVERY decoded video tensor: the two candidates in
    the a2v direction and the anchor plus the two candidates in the v2a
    direction.
    """
    if not mode:
        return video
    if mode == "static":
        mid = video.shape[0] // 2
        return video[mid:mid + 1].repeat(video.shape[0], 1, 1, 1)
    rng = random.Random(seed)
    order = list(range(video.shape[0]))
    rng.shuffle(order)
    return video[torch.tensor(order, dtype=torch.long)]


def _decode_segment_worker(index: int, video_path: str, start_sec: float,
                           end_sec: float, num_frames: int, sample_rate: int,
                           video_ablation: Optional[str],
                           align_to_start: bool = True):
    """Decode one segment in a worker process (top-level for pickling)."""
    try:
        from src.utils.video import video_to_tensor
        from src.utils.audio import audio_to_tensor

        video = video_to_tensor(video_path, start_sec, end_sec,
                                num_frames=num_frames)
        video = _apply_video_ablation(video, video_ablation, seed=index)
        audio = audio_to_tensor(video_path, start_sec, end_sec,
                                sample_rate=sample_rate,
                                align_to_start=align_to_start)
        return index, audio, video, None
    except Exception as e:  # noqa: BLE001 - report per-clip failures
        return index, None, None, str(e)


def _iter_decoded(segments: Sequence[Segment], batch_size: int, workers: int,
                  num_frames: int, sample_rate: int,
                  video_ablation: Optional[str],
                  align_to_start: bool = True):
    """Yield batches of ``(index, audio, video, error)`` in segment order.

    Same prefetch-bounded strategy as scripts/eval_zeroshot_vggsound.py, so the
    ``--workers 1`` and ``--workers N`` paths produce identical results.
    """
    starts = range(0, len(segments), batch_size)
    args = [(i, segments[i].video_path, segments[i].start_sec,
             segments[i].end_sec, num_frames, sample_rate, video_ablation,
             align_to_start)
            for i in range(len(segments))]

    if workers <= 1:
        for start in starts:
            yield [_decode_segment_worker(*args[i]) for i in range(
                start, min(start + batch_size, len(segments)))]
        return

    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    mp_context = multiprocessing.get_context("spawn")
    prefetch = 3 * batch_size
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp_context) as ex:
        pending = []
        next_index = 0

        def submit(i):
            return ex.submit(_decode_segment_worker, *args[i])

        while next_index < len(args) and len(pending) < prefetch:
            pending.append(submit(next_index))
            next_index += 1
        for start in starts:
            batch = []
            for _ in range(start, min(start + batch_size, len(segments))):
                batch.append(pending.pop(0).result())
                if next_index < len(args):
                    pending.append(submit(next_index))
                    next_index += 1
            yield batch


def _pad_audio(audio_list: List[torch.Tensor]):
    """Pad (1, T) waveforms to (B, 1, T_max) and a padded-position mask."""
    max_len = max(a.shape[1] for a in audio_list)
    audio = torch.zeros(len(audio_list), 1, max_len)
    mask = torch.ones(len(audio_list), max_len, dtype=torch.bool)
    for i, a in enumerate(audio_list):
        audio[i, :, :a.shape[1]] = a
        mask[i, :a.shape[1]] = False
    return audio, mask


def encode_segments(models: Dict[str, torch.nn.Module], segments: Sequence[Segment],
                    device, batch_size: int, workers: int, num_frames: int,
                    sample_rate: int, video_ablation: Optional[str], logger,
                    align_to_start: bool = True
                    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]],
                               Dict[str, Dict[str, torch.Tensor]], List[dict]]:
    """Decode every segment once and encode it with every model.

    Returns ``(audio_embs, video_embs, failures)`` where each embedding map is
    ``{model_name: {segment_id: 1-D float tensor}}``.
    """
    names = list(models.keys())
    audio_embs: Dict[str, Dict[str, torch.Tensor]] = {n: {} for n in names}
    video_embs: Dict[str, Dict[str, torch.Tensor]] = {n: {} for n in names}
    failures: List[dict] = []
    done = 0
    t0 = time.time()

    for decoded in _iter_decoded(segments, batch_size, workers, num_frames,
                                 sample_rate, video_ablation, align_to_start):
        audio_list, video_list, ids = [], [], []
        for index, a, v, error in decoded:
            if error is not None:
                failures.append({"segment_id": segments[index].segment_id,
                                 "error": error})
                continue
            audio_list.append(a)
            video_list.append(v)
            ids.append(segments[index].segment_id)
        if not audio_list:
            continue

        audio_padded, audio_mask = _pad_audio(audio_list)
        audio_padded = audio_padded.to(device)
        audio_mask = audio_mask.to(device)
        video_stacked = torch.stack(video_list, dim=0).to(device)

        with torch.no_grad():
            for name in names:
                ae = models[name].encode_audio(audio_padded, audio_mask)
                ve = models[name].encode_video(video_stacked)
                ae = ae.detach().float().cpu()
                ve = ve.detach().float().cpu()
                for i, sid in enumerate(ids):
                    audio_embs[name][sid] = ae[i]
                    video_embs[name][sid] = ve[i]

        done += len(audio_list)
        if done % (batch_size * 10) == 0 or done >= len(segments):
            logger.info("  encoded %d/%d segments (%d failures) [%.0fs]",
                        done, len(segments), len(failures), time.time() - t0)

    return audio_embs, video_embs, failures


def score_condition(triplet_set: TripletSet, direction: str,
                    audio_embs: Dict[str, torch.Tensor],
                    video_embs: Dict[str, torch.Tensor]
                    ) -> Tuple[List[float], List[str], int]:
    """Score every item in one direction; returns (values, film clusters, n_skipped)."""
    values: List[float] = []
    clusters: List[str] = []
    skipped = 0
    for t in triplet_set.items:
        p, n = t.positive, t.negative
        if p.segment_id not in audio_embs or n.segment_id not in audio_embs:
            skipped += 1
            continue
        if direction == "a2v":
            acc = triplet_accuracy(audio_embs[p.segment_id],
                                   video_embs[p.segment_id],
                                   video_embs[n.segment_id])
        else:
            acc = triplet_accuracy(video_embs[p.segment_id],
                                   audio_embs[p.segment_id],
                                   audio_embs[n.segment_id])
        values.append(acc)
        clusters.append(p.video_id)
    return values, clusters, skipped


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_ci(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


def build_markdown(results: dict) -> str:
    lines = []
    lines.append("# Flik triplet (2AFC) evaluation")
    lines.append("")
    lines.append(f"Chance = 0.5. n requested = {results['n_trials_requested']}. "
                 f"MDE formula: n ~= 7.85 * p0(1-p0) / delta^2 "
                 f"(80% power, p0 = 0.5).")
    lines.append("")
    lines.append("| condition | n items | MDE (pp) | model | direction | "
                 "accuracy | 95% CI (film bootstrap) |")
    lines.append("|---|---:|---:|---|---|---:|---|")
    for cond in results["conditions"]:
        entry = results["conditions"][cond]
        if not entry.get("available", True):
            lines.append(f"| {cond} | 0 | - | - | - | omitted | "
                         f"{entry.get('reason', '')} |")
            continue
        for model in entry["models"]:
            m = entry["models"][model]
            for direction in DIRECTIONS:
                d = m.get(direction)
                if not d:
                    continue
                tag = " (floor)" if model == RANDOM_INIT_NAME else ""
                lines.append(
                    f"| {cond} | {entry['n_items']} | "
                    f"{100.0 * entry['mde']:.2f} | {model}{tag} | {direction} | "
                    f"{d['accuracy']:.3f} | {_fmt_ci(d['ci'][0], d['ci'][1])} |")
    for model, per_dir in results.get("pooled", {}).items():
        tag = " (floor)" if model == RANDOM_INIT_NAME else ""
        for direction in DIRECTIONS:
            d = per_dir.get(direction)
            if not d:
                continue
            lines.append(
                f"| POOLED | {d['n_items']} | "
                f"{100.0 * min_detectable_effect(d['n_items']):.2f} | "
                f"{model}{tag} | {direction} | {d['accuracy']:.3f} | "
                f"{_fmt_ci(d['ci'][0], d['ci'][1])} |")
    lines.append("")
    lines.append("Random-init floor per condition/direction is the row tagged "
                 "`(floor)`; it is the same model for all conditions.")
    lines.append("")
    lines.append("`POOLED` pools all scored items across the available "
                 "conditions. It is descriptive: the conditions share films "
                 "and clips, so it is not an independent estimate and its "
                 "cluster bootstrap is not a test of a condition difference. "
                 "Compare conditions, not the pooled row to a condition row.")
    return "\n".join(lines) + "\n"


def format_table(results: dict) -> str:
    rows = []
    header = ["condition", "n", "MDE pp", "model", "dir", "acc", "95% CI"]
    rows.append(" | ".join(header))
    rows.append("-" * len(rows[0]))
    for cond in results["conditions"]:
        entry = results["conditions"][cond]
        if not entry.get("available", True):
            rows.append(f"{cond} | - | - | OMITTED: {entry.get('reason', '')}")
            continue
        for model in entry["models"]:
            m = entry["models"][model]
            for direction in DIRECTIONS:
                d = m.get(direction)
                if not d:
                    continue
                rows.append(" | ".join([
                    cond, str(entry["n_items"]),
                    f"{100.0 * entry['mde']:.2f}", model, direction,
                    f"{d['accuracy']:.3f}",
                    _fmt_ci(d["ci"][0], d["ci"][1]),
                ]))
    for model, per_dir in results.get("pooled", {}).items():
        for direction in DIRECTIONS:
            d = per_dir.get(direction)
            if not d:
                continue
            rows.append(" | ".join([
                "POOLED", str(d["n_items"]),
                f"{100.0 * min_detectable_effect(d['n_items']):.2f}",
                model, direction, f"{d['accuracy']:.3f}",
                _fmt_ci(d["ci"][0], d["ci"][1]),
            ]))
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="2AFC triplet/minimal-pair evaluation for FlikModel "
                    "(Peppa Pig design).")
    p.add_argument("--checkpoint", action="append", default=[],
                   metavar="NAME:PATH", help="Checkpoint to evaluate (repeatable).")
    p.add_argument("--random-init", action="store_true",
                   help="Add a randomly initialized model as the floor.")
    p.add_argument("--config", type=str, default=None,
                   help="Model YAML config (default: src/configs/default.yaml).")
    p.add_argument("--device", type=str, default=None, help="cuda|cpu")

    p.add_argument("--manifest", type=str,
                   default="data/expanded_manifest_segments.json",
                   help="Segment-level manifest (default: expanded manifest).")
    p.add_argument("--films", nargs="+", default=None,
                   help="Build segments from these film ids instead of a "
                        "manifest (uses the training merge_segments logic).")
    p.add_argument("--videos-dir", type=str, default="data/videos")
    p.add_argument("--transcripts-dir", type=str, default="data/transcripts")
    p.add_argument("--holdout-films", type=str, default=None,
                   help="JSON list or newline file of held-out film ids. "
                        "Default: reuse the train split helper.")
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--movie-map", type=str, default=None,
                   help="JSON {video_id: movie_id}; blocks positives and "
                        "negatives that share a movie identity.")
    p.add_argument("--min-duration", type=float, default=3.0)
    p.add_argument("--max-duration", type=float, default=10.0)
    p.add_argument("--max-gap", type=float, default=1.0)

    p.add_argument("--n-trials", type=int, default=4000)
    p.add_argument("--duration-bins", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-iterations", type=int, default=2000)
    p.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                   choices=list(CONDITIONS))

    p.add_argument("--speaker-map", type=str, default=None,
                   help="JSON segment/pair -> speaker id (precomputed).")
    p.add_argument("--diarize-mfcc", action="store_true",
                   help="Derive heuristic pseudo-speakers by MFCC KMeans "
                        "(per film); off by default.")
    p.add_argument("--speaker-clusters", type=int, default=4)

    p.add_argument("--scramble-frames", action="store_true",
                   help="Ablation: shuffle candidate frame order.")
    p.add_argument("--static-frames", action="store_true",
                   help="Ablation: repeat each video's middle frame.")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--audio-align-to-start",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Trim the audio windows to start_sec by decoded PTS "
                        "(default: true; matches dataset.audio_align_to_start). "
                        "--no-audio-align-to-start keeps the historical "
                        "keyframe pre-roll, for reproducing runs logged before "
                        "2026-09-16.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel decode processes (default 1).")
    p.add_argument("--out", type=str, default="triplet_eval_results.json")
    p.add_argument("--markdown-out", type=str, default=None)
    return p.parse_args(argv)


def setup_logging():
    logger = logging.getLogger("eval_triplets")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(sh)
    return logger


def main(argv=None):
    args = parse_args(argv)
    logger = setup_logging()
    logger.info(
        "Audio window alignment: "
        + (
            "timestamp-accurate, starts at start_sec (audio_align_to_start=true)"
            if args.audio_align_to_start
            else "legacy keyframe pre-roll retained (audio_align_to_start=false, bug-compatible)"
        )
    )

    if args.scramble_frames and args.static_frames:
        logger.error("--scramble-frames and --static-frames are mutually exclusive.")
        return 2
    if not args.checkpoint and not args.random_init:
        logger.error("Nothing to evaluate: pass --checkpoint and/or --random-init.")
        return 2

    # Model helpers live in the zero-shot eval script; importing lazily keeps
    # the unit tests free of transformers/model imports.
    from scripts.eval_zeroshot_vggsound import (
        RANDOM_INIT_SEED as ZS_SEED, build_model, load_checkpoint_into,
        load_model_config, randomize_model,
    )

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    model_cfg, _ = load_model_config(args.config)
    movie_map = load_movie_map(args.movie_map)

    # --- data ------------------------------------------------------------
    if args.films:
        segments = load_segments_from_transcripts(
            args.films, args.videos_dir, args.transcripts_dir,
            args.min_duration, args.max_duration, args.max_gap, movie_map)
    else:
        segments = load_segments_from_manifest(args.manifest, movie_map)
    logger.info("Loaded %d segments from %d films",
                len(segments), len({s.video_id for s in segments}))
    if not segments:
        logger.error("No segments loaded.")
        return 1

    if args.films and not args.holdout_films:
        # --films is an explicit selection; treat it as the held-out set.
        logger.warning("--films given without --holdout-films: assuming the "
                       "listed films are the held-out set (%d films)",
                       len({s.video_id for s in segments}))
    else:
        segments = resolve_holdout(segments, args.holdout_films, args.val_ratio,
                                  args.split_seed, logger)
    if not segments:
        logger.error("No held-out segments; check --holdout-films/--val-ratio.")
        return 1

    # --- speakers --------------------------------------------------------
    speaker_source = None
    if any(s.speaker is not None for s in segments):
        speaker_source = "manifest"
    speaker_map = load_speaker_map(args.speaker_map)
    if speaker_map:
        n_labelled = attach_speakers(segments, speaker_map)
        speaker_source = speaker_source or "speaker_map"
        logger.info("Speaker map: %d/%d segments labelled", n_labelled, len(segments))
    if args.diarize_mfcc:
        logger.info("Diarizing with MFCC KMeans (heuristic; %d clusters/film)",
                    args.speaker_clusters)
        diarize_mfcc(segments, args.sample_rate, args.speaker_clusters,
                     align_to_start=args.audio_align_to_start)
        speaker_source = "mfcc_kmeans"

    if speaker_source is None and "speaker_matched" in args.conditions:
        reason = ("no speaker labels available (supply --speaker-map, "
                  "manifest 'speaker' fields, or --diarize-mfcc); "
                  "condition omitted rather than faked")
        logger.warning("speaker_matched: %s", reason)

    # --- triplets --------------------------------------------------------
    movie_of = lambda s: s.movie_id or s.video_id  # noqa: E731
    triplet_sets: Dict[str, TripletSet] = {}
    for cond in args.conditions:
        if cond == "speaker_matched" and speaker_source is None:
            continue
        ts = build_triplets(segments, cond, args.n_trials,
                            n_bins=args.duration_bins, seed=args.seed,
                            movie_of=movie_of)
        validate_triplets(ts, movie_of=movie_of)
        logger.info("Condition %-16s: %d items (requested %d)",
                    cond, len(ts.items), args.n_trials)
        triplet_sets[cond] = ts

    # MDE for the requested n, so the number is interpretable up front.
    logger.info("MDE at n=%d: %.2f pp (%.3f) over chance", args.n_trials,
                100.0 * min_detectable_effect(args.n_trials),
                min_detectable_effect(args.n_trials))

    # --- unique segments to encode --------------------------------------
    unique: Dict[str, Segment] = {}
    for ts in triplet_sets.values():
        for t in ts.items:
            unique[t.positive.segment_id] = t.positive
            unique[t.negative.segment_id] = t.negative
    unique_list = [unique[k] for k in sorted(unique)]
    logger.info("Encoding %d unique clips", len(unique_list))

    # --- models ----------------------------------------------------------
    specs = []
    for spec in args.checkpoint:
        if ":" not in spec:
            logger.error("--checkpoint must be NAME:PATH, got: %s", spec)
            return 2
        name, path = spec.split(":", 1)
        specs.append((name, path))
    if args.random_init:
        specs.append((RANDOM_INIT_NAME, None))
    if not specs:
        logger.error("No models to evaluate.")
        return 2

    models = {}
    for name, path in specs:
        t0 = time.time()
        logger.info("[%s] building FlikModel (%s)", name,
                    "random init" if path is None else path)
        model = build_model(model_cfg, device)
        if path is None:
            randomize_model(model, ZS_SEED)
        else:
            load_checkpoint_into(model, path, logger)
        model.eval()
        models[name] = model
        logger.info("[%s] ready in %.1fs", name, time.time() - t0)

    video_ablation = ("scramble" if args.scramble_frames else
                      "static" if args.static_frames else None)
    if video_ablation:
        logger.info("Video temporal ablation: %s", video_ablation)

    audio_embs, video_embs, failures = encode_segments(
        models, unique_list, device, args.batch_size, args.workers,
        args.num_frames, args.sample_rate, video_ablation, logger,
        align_to_start=args.audio_align_to_start)

    # --- score -----------------------------------------------------------
    results = {
        "design": "2AFC triplet (Peppa Pig, doi:10.1162/tacl_a_00498)",
        "reference": "https://github.com/gchrupala/peppa (Apache-2.0)",
        "chance": 0.5,
        "n_trials_requested": args.n_trials,
        "duration_bins": args.duration_bins,
        "bootstrap_iterations": args.bootstrap_iterations,
        "video_ablation": video_ablation,
        "audio_align_to_start": args.audio_align_to_start,
        "speaker_condition": {
            "available": "speaker_matched" in triplet_sets,
            "source": speaker_source,
            "reason": None if "speaker_matched" in triplet_sets else (
                "no speaker labels available (supply --speaker-map, manifest "
                "'speaker' fields, or --diarize-mfcc)"),
        },
        "n_held_out_films": len({s.video_id for s in segments}),
        "n_unique_clips": len(unique_list),
        "n_decode_failures": len(failures),
        "decode_failures": failures[:50],
        "conditions": {},
    }

    for cond in args.conditions:
        if cond not in triplet_sets:
            results["conditions"][cond] = {
                "available": False,
                "reason": ("no speaker labels available (supply --speaker-map, "
                           "manifest 'speaker' fields, or --diarize-mfcc)"),
            }
            continue
        ts = triplet_sets[cond]
        # Items skipped when a clip failed to decode define the realised n.
        scored_items = [
            t for t in ts.items
            if all(sid in next(iter(audio_embs.values()))
                   for sid in (t.positive.segment_id, t.negative.segment_id))
        ]
        n_items = len(scored_items)
        ts_scored = TripletSet(cond, scored_items, ts.bin_edges, ts.bin_of)
        entry = {
            "available": True,
            "n_items": n_items,
            "n_items_built": len(ts.items),
            "mde": min_detectable_effect(n_items),
            "mde_pp": 100.0 * min_detectable_effect(n_items),
            "models": {},
        }
        for name in models:
            per_dir = {}
            for direction in DIRECTIONS:
                vals, clusters, skipped = score_condition(
                    ts_scored, direction, audio_embs[name], video_embs[name])
                if not vals:
                    continue
                acc = sum(vals) / len(vals)
                lo, hi = cluster_bootstrap_ci(
                    vals, clusters, n_iter=args.bootstrap_iterations,
                    seed=args.seed)
                per_dir[direction] = {
                    "accuracy": acc,
                    "ci": [lo, hi],
                    "n_items": len(vals),
                    "n_skipped": skipped,
                    "n_films": len(set(clusters)),
                }
            entry["models"][name] = per_dir
        results["conditions"][cond] = entry

    # Pooled over conditions that were actually scored (never over fake items).
    pooled = {}
    for name in models:
        for direction in DIRECTIONS:
            vals, clusters = [], []
            for cond, entry in results["conditions"].items():
                if not entry.get("available"):
                    continue
                ts = triplet_sets[cond]
                v, c, _ = score_condition(ts, direction, audio_embs[name],
                                          video_embs[name])
                vals.extend(v)
                clusters.extend(c)
            if vals:
                pooled.setdefault(name, {})[direction] = {
                    "accuracy": sum(vals) / len(vals),
                    "ci": list(cluster_bootstrap_ci(
                        vals, clusters, n_iter=args.bootstrap_iterations,
                        seed=args.seed)),
                    "n_items": len(vals),
                }
    results["pooled"] = pooled

    # --- output ----------------------------------------------------------
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote %s", args.out)

    md_path = args.markdown_out or (os.path.splitext(args.out)[0] + ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(build_markdown(results))
    logger.info("Wrote %s", md_path)

    print()
    print("=" * 110)
    print(format_table(results))
    print("=" * 110)
    print(f"MDE at n={args.n_trials}: "
          f"{100.0 * min_detectable_effect(args.n_trials):.2f} pp over chance "
          f"(formula n ~= 7.85 * p0(1-p0) / delta^2, 80% power, p0=0.5)")
    if not results["speaker_condition"]["available"]:
        print("speaker_matched OMITTED: "
              + results["speaker_condition"]["reason"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
