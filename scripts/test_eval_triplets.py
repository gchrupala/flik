#!/usr/bin/env python
"""Unit tests for scripts/eval_triplets.py.

Pure synthetic data: no films, no checkpoints, no decoding. Covers the two
pieces that can silently produce a wrong number:

  * the scoring rule (positive closer -> 1.0, negative closer -> 0.0, ties ->
    ~0.5 with a CI that covers chance), and
  * triplet construction (duration-bin matching, no clip reused, same_film
    really shares a film, cross_film really does not, movie identity respected).

Run with:
    uv run --extra cpu python -m scripts.test_eval_triplets
    # or: uv run --extra cpu python -m pytest scripts/test_eval_triplets.py
"""

import os
import random
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from scripts.eval_triplets import (  # noqa: E402
    Segment, Triplet, TripletSet, _apply_video_ablation, build_triplets,
    cluster_bootstrap_ci, min_detectable_effect, score_condition,
    triplet_accuracy, validate_triplets,
)


def _segment(i, film="filmA", start=0.0, dur=5.0, movie=None, speaker=None):
    return Segment(
        segment_id=f"{film}_{i}",
        video_id=film,
        video_path=f"/tmp/{film}.mp4",
        start_sec=start,
        end_sec=start + dur,
        movie_id=movie if movie is not None else film,
        speaker=speaker,
    )


def _emb(token_vecs):
    """Build {segment_id: tensor} from {id: (list,)}."""
    return {k: torch.tensor(v, dtype=torch.float32).reshape(-1)
            for k, v in token_vecs.items()}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_positive_always_closer_is_accuracy_1():
    n = 200
    segs = [_segment(i, start=float(i)) for i in range(n)]
    pairs = []
    audio, video = {}, {}
    for i, s in enumerate(segs):
        audio[s.segment_id] = torch.tensor([1.0, 0.0])
        video[s.segment_id] = torch.tensor([1.0, 0.0])
    # Give each negative an orthogonal video so the positive is always closer.
    for i in range(0, n - 1, 2):
        a, b = segs[i], segs[i + 1]
        video[b.segment_id] = torch.tensor([0.0, 1.0])
        pairs.append(Triplet(a, b, "same_film", 0))
    ts = TripletSet("same_film", pairs)
    vals, clusters, skipped = score_condition(ts, "a2v", audio, video)
    assert skipped == 0
    assert all(v == 1.0 for v in vals), vals[:5]
    lo, hi = cluster_bootstrap_ci(vals, clusters, n_iter=200)
    assert lo == 1.0 and hi == 1.0


def test_negative_always_closer_is_accuracy_0():
    segs = [_segment(i, start=float(i)) for i in range(100)]
    audio = {s.segment_id: torch.tensor([1.0, 0.0]) for s in segs}
    video = {}
    pairs = []
    for i in range(0, 99, 2):
        a, b = segs[i], segs[i + 1]
        video[a.segment_id] = torch.tensor([0.0, 1.0])   # positive worse
        video[b.segment_id] = torch.tensor([1.0, 0.0])   # negative better
        pairs.append(Triplet(a, b, "cross_film", 0))
    ts = TripletSet("cross_film", pairs)
    vals, clusters, _ = score_condition(ts, "a2v", audio, video)
    assert all(v == 0.0 for v in vals), vals[:5]
    assert sum(vals) / len(vals) == 0.0


def test_ties_are_random_and_ci_covers_chance():
    # Exact ties: cos(anchor, pos) == cos(anchor, neg) for every item.
    segs = [_segment(i, start=float(i)) for i in range(200)]
    audio = {s.segment_id: torch.tensor([1.0, 0.0]) for s in segs}
    video = {s.segment_id: torch.tensor([1.0, 0.0]) for s in segs}
    pairs = []
    for i in range(0, 199, 2):
        pairs.append(Triplet(segs[i], segs[i + 1], "same_film", 0))
    ts = TripletSet("same_film", pairs)
    vals, clusters, _ = score_condition(ts, "a2v", audio, video)
    assert all(v == 0.5 for v in vals)
    acc = sum(vals) / len(vals)
    lo, hi = cluster_bootstrap_ci(vals, clusters, n_iter=500)
    assert abs(acc - 0.5) < 1e-9
    assert lo <= 0.5 <= hi

    # Random signs: mean is near chance and the CI covers 0.5.
    g = torch.Generator().manual_seed(0)
    rand_audio, rand_video = {}, {}
    pairs2 = []
    for j, i in enumerate(range(0, 999, 2)):
        # Distinct film per pair so the cluster bootstrap has >1 cluster.
        a = _segment(i, film=f"film{j}")
        b = _segment(i + 1, film=f"film{j}")
        rand_audio[a.segment_id] = torch.randn(16, generator=g)
        rand_audio[b.segment_id] = torch.randn(16, generator=g)
        rand_video[a.segment_id] = torch.randn(16, generator=g)
        rand_video[b.segment_id] = torch.randn(16, generator=g)
        pairs2.append(Triplet(a, b, "cross_film", 0))
    ts2 = TripletSet("cross_film", pairs2)
    vals2, clusters2, _ = score_condition(ts2, "a2v", rand_audio, rand_video)
    acc2 = sum(vals2) / len(vals2)
    lo2, hi2 = cluster_bootstrap_ci(vals2, clusters2, n_iter=500, seed=1)
    assert 0.4 < acc2 < 0.6, acc2
    assert lo2 <= 0.5 <= hi2, (lo2, hi2)
    # And the reverse direction on random embeddings is also at chance.
    vals3, _, _ = score_condition(ts2, "v2a", rand_audio, rand_video)
    assert 0.4 < sum(vals3) / len(vals3) < 0.6


def test_triplet_accuracy_all_directions_share_rule():
    a = torch.tensor([1.0, 0.0])
    pos = torch.tensor([0.9, 0.1])
    neg = torch.tensor([0.1, 0.9])
    assert triplet_accuracy(a, pos, neg) == 1.0
    assert triplet_accuracy(a, neg, pos) == 0.0
    tie = torch.tensor([0.0, 1.0])
    assert triplet_accuracy(a, tie, tie) == 0.5


# ---------------------------------------------------------------------------
# Triplet construction
# ---------------------------------------------------------------------------

def _synthetic_corpus(n_films=4, per_film=40, n_bins=8):
    rng = random.Random(7)
    segs = []
    for f in range(n_films):
        for i in range(per_film):
            dur = rng.uniform(3.0, 10.0)
            segs.append(_segment(i, film=f"film{f}", start=float(i * 12),
                                 dur=dur))
    return segs


def test_duration_bin_matching_and_no_reuse():
    segs = _synthetic_corpus()
    ts = build_triplets(segs, "cross_film", n_trials=0, n_bins=8, seed=1)
    assert ts.items, "expected items"
    validate_triplets(ts)  # raises on bin mismatch or clip reuse
    for t in ts.items:
        assert t.bin_index == ts.bin_of[t.positive.segment_id]
        assert t.bin_index == ts.bin_of[t.negative.segment_id]


def test_same_film_draws_both_candidates_from_one_film():
    segs = _synthetic_corpus()
    ts = build_triplets(segs, "same_film", n_trials=0, n_bins=8, seed=2)
    assert ts.items
    assert all(t.positive.video_id == t.negative.video_id for t in ts.items)
    validate_triplets(ts)


def test_cross_film_never_shares_a_film():
    segs = _synthetic_corpus()
    ts = build_triplets(segs, "cross_film", n_trials=0, n_bins=8, seed=3)
    assert ts.items
    assert all(t.positive.video_id != t.negative.video_id for t in ts.items)
    assert all(t.positive.movie_id != t.negative.movie_id for t in ts.items)
    validate_triplets(ts)


def test_cross_film_respects_movie_identity():
    # Two distinct video files that are the same movie must not be paired.
    segs = []
    for film, movie in [("partA", "movieX"), ("partB", "movieX"),
                        ("partC", "movieY")]:
        for i in range(20):
            segs.append(_segment(i, film=film, start=float(i * 12), dur=5.0,
                                 movie=movie))
    ts = build_triplets(segs, "cross_film", n_trials=0, n_bins=4, seed=4)
    movie_of = lambda s: s.movie_id
    validate_triplets(ts, movie_of=movie_of)
    for t in ts.items:
        assert t.positive.movie_id != t.negative.movie_id, t
    # partA and partB can never pair with each other.
    for t in ts.items:
        assert {t.positive.video_id, t.negative.video_id} != {"partA", "partB"}


def test_no_clip_reused_across_items():
    segs = _synthetic_corpus()
    for cond in ("cross_film", "same_film"):
        ts = build_triplets(segs, cond, n_trials=0, n_bins=8, seed=5)
        ids = []
        for t in ts.items:
            ids.extend([t.positive.segment_id, t.negative.segment_id])
        assert len(ids) == len(set(ids)), f"{cond}: clip reused"


def test_speaker_matched_pairs_by_speaker():
    segs = []
    counter = 0
    for film, spk in [("filmA", "s1"), ("filmA", "s2"),
                      ("filmB", "s1"), ("filmB", "s3")]:
        for i in range(10):
            counter += 1
            segs.append(_segment(counter, film=film, start=float(i * 12),
                                 dur=5.0, speaker=spk))
    ts = build_triplets(segs, "speaker_matched", n_trials=0, n_bins=4, seed=6)
    assert ts.items
    for t in ts.items:
        assert t.positive.speaker == t.negative.speaker
    validate_triplets(ts)


def test_speaker_matched_empty_without_labels():
    segs = _synthetic_corpus(n_films=3, per_film=10)
    ts = build_triplets(segs, "speaker_matched", n_trials=0, n_bins=4, seed=6)
    assert ts.items == []


def test_n_trials_truncates():
    segs = _synthetic_corpus(n_films=6, per_film=60)
    ts = build_triplets(segs, "cross_film", n_trials=20, n_bins=8, seed=8)
    assert len(ts.items) == 20


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def test_mde_formula():
    # n = 7.85 * p0(1-p0)/delta^2 with p0=0.5 -> delta = sqrt(7.85*0.25/n)
    assert abs(min_detectable_effect(4000) - (7.85 * 0.25 / 4000) ** 0.5) < 1e-12
    assert abs(min_detectable_effect(4000) - 0.02215) < 1e-4
    # More items -> smaller effect, and the relation is exactly 1/sqrt(n).
    assert min_detectable_effect(16000) < min_detectable_effect(4000)
    assert abs(min_detectable_effect(16000) * 2 - min_detectable_effect(4000)) < 1e-12


def test_cluster_bootstrap_single_cluster_is_degenerate():
    vals = [1.0, 0.0, 1.0, 1.0]
    lo, hi = cluster_bootstrap_ci(vals, ["f"] * 4, n_iter=200)
    assert lo == hi == 0.75


def test_cluster_bootstrap_wider_than_item_bootstrap():
    # Two films: one always correct, one always wrong. Resampling films gives a
    # wide interval; per-item resampling would look narrow.
    vals = [1.0] * 50 + [0.0] * 50
    clusters = ["f1"] * 50 + ["f2"] * 50
    lo_c, hi_c = cluster_bootstrap_ci(vals, clusters, n_iter=2000, seed=2)
    lo_i, hi_i = cluster_bootstrap_ci(vals, ["x"] * 100, n_iter=2000, seed=2)
    assert hi_c - lo_c > hi_i - lo_i
    assert lo_c <= 0.5 <= hi_c


def test_attach_speakers_enables_speaker_condition(tmp_path=None):
    # A speaker map keyed by segment_id and by film|start|end.
    import tempfile

    from scripts.eval_triplets import attach_speakers, load_speaker_map

    segs = _synthetic_corpus(n_films=2, per_film=12)
    smap = {segs[0].segment_id: "spkA",
            f"{segs[1].video_id}|{segs[1].start_sec}|{segs[1].end_sec}": "spkA"}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "speakers.json")
        with open(path, "w", encoding="utf-8") as f:
            import json as _json
            _json.dump(smap, f)
        loaded = load_speaker_map(path)
    n = attach_speakers(segs, loaded)
    assert n == 2
    assert segs[0].speaker == "spkA"
    assert segs[1].speaker == "spkA"
    ts = build_triplets(segs, "speaker_matched", n_trials=0, n_bins=4, seed=1)
    for t in ts.items:
        assert t.positive.speaker == t.negative.speaker == "spkA"


def test_video_ablations():
    video = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1).repeat(1, 3, 2, 2)
    # static: every frame equals the middle frame.
    static = _apply_video_ablation(video, "static", seed=0)
    assert torch.equal(static, video[2:3].repeat(4, 1, 1, 1))
    # scramble: a permutation of the frames (multiset unchanged).
    scrambled = _apply_video_ablation(video, "scramble", seed=0)
    assert scrambled.shape == video.shape
    assert sorted(scrambled[:, 0, 0, 0].tolist()) == sorted(video[:, 0, 0, 0].tolist())
    # no ablation is the identity.
    assert torch.equal(_apply_video_ablation(video, None, seed=0), video)


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
