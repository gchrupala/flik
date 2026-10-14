#!/usr/bin/env python
"""Tests for the frozen retrieval-pool protocol.

Covers:
  * retrieval_metrics hit counts on synthetic embeddings with constructed
    rank-1 matches, and word-for-word parity of the legacy wrapper.
  * StridedSampler exact partition for world_size 1..4.
  * make_val_pool movie-disjointness and sha256 determinism on a small manifest.

Runnable directly (``python tests/test_frozen_retrieval.py``) or via pytest.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from src.eval.retrieval import retrieval_metrics, retrieval_recall_at_k  # noqa: E402
from scripts.train_hydra import (  # noqa: E402
    StridedSampler,
    _frozen_live_accept,
    _group_key,
    _load_movie_key_map,
    _selection_metric_value,
    _split_manifest_by_video,
    _strided_gather_order,
    _trailing_mean,
)


def test_retrieval_metrics_known_hits():
    """Construct exactly ``m`` rank-1 matches for a2v and assert hit counts."""
    n, m = 12, 5
    # A = I. sim = V^T, so a2v query i ranks the columns of V. Column i is set
    # to e_i for hit queries (i < m) and to e_{(i+1) % n} for miss queries, so
    # exactly m queries have their true match at rank 1.
    V = torch.zeros(n, n)
    for i in range(n):
        if i < m:
            V[i, i] = 1.0
        else:
            V[(i + 1) % n, i] = 1.0
    A = torch.eye(n)

    metrics = retrieval_metrics(A, V, ks=(1, 5, 10))

    assert metrics["n_pool"] == n
    assert metrics["a2v_hits@1"] == m, metrics["a2v_hits@1"]
    assert abs(metrics["a2v_rate@1"] - m / n) < 1e-6
    assert abs(metrics["a2v_lift@1"] - (m / n) * n / 1) < 1e-6

    vec = metrics["a2v_hit_vec@1"]
    assert len(vec) == n
    assert sum(bool(x) for x in vec) == m
    # The known hit/miss pattern.
    assert [bool(x) for x in vec] == [i < m for i in range(n)]

    # Hit vectors exist for every direction/k and are boolean lists.
    for k in (1, 5, 10):
        for d in ("a2v", "v2a"):
            hv = metrics[f"{d}_hit_vec@{k}"]
            assert len(hv) == n and all(isinstance(b, bool) for b in hv)
            assert sum(bool(x) for x in hv) == metrics[f"{d}_hits@{k}"]
            assert abs(metrics[f"{d}_rate@{k}"] - metrics[f"{d}_hits@{k}"] / n) < 1e-6


def test_retrieval_recall_wrapper_parity():
    """retrieval_recall_at_k must match the original top-k formula exactly."""
    torch.manual_seed(0)
    for n in (3, 16):
        a = torch.nn.functional.normalize(torch.randn(n, 8), dim=1)
        v = torch.nn.functional.normalize(torch.randn(n, 8), dim=1)
        sim = a @ v.t()
        labels = torch.arange(n)
        expected = {}
        for k in (1, 5, 10):
            k_eff = min(k, n)
            a2v = (sim.topk(k_eff, dim=1).indices == labels.unsqueeze(1)).any(1).float().mean().item()
            v2a = (sim.t().topk(k_eff, dim=1).indices == labels.unsqueeze(1)).any(1).float().mean().item()
            expected[f"a2v_r@{k_eff}"] = a2v
            expected[f"v2a_r@{k_eff}"] = v2a
        assert retrieval_recall_at_k(a, v, ks=(1, 5, 10)) == expected


def test_strided_sampler_partition():
    """Union of shards == all indices exactly once, for world_size 1..4."""
    n = 10
    dataset = list(range(n))
    for world_size in range(1, 5):
        shards = []
        for r in range(world_size):
            sampler = StridedSampler(dataset, rank=r, world_size=world_size)
            idx = list(iter(sampler))
            assert len(sampler) == len(idx)
            # Exact stride pattern.
            assert idx == list(range(r, n, world_size))
            shards.extend(idx)
        assert sorted(shards) == list(range(n)), (world_size, sorted(shards))
        assert len(shards) == len(set(shards)) == n, world_size


def test_strided_gather_order_restores_dataset_order():
    """Simulate the DDP gather: rank-ordered concat -> dataset order."""
    for n in (1, 2, 7, 10, 11):
        for world_size in range(1, 5):
            counts = [len(range(r, n, world_size)) for r in range(world_size)]
            assert sum(counts) == n
            # Rank-ordered concatenation of the global indices each rank holds.
            gathered = []
            for r in range(world_size):
                gathered.extend(range(r, n, world_size))
            order = _strided_gather_order(counts, world_size)
            restored = [gathered[i] for i in order]
            assert restored == list(range(n)), (n, world_size, restored)


def _dedup_manifest(n_movies=6, segments_per_id=2):
    """Manifest where each film has two video_ids that map to one movie key."""
    manifest = []
    mapping = {}
    for i in range(n_movies):
        movie = f"film{i:02d}"
        for suffix in ("", "_512kb"):
            vid = movie + suffix
            mapping[vid] = movie
            for j in range(segments_per_id):
                start = j * 4.0
                manifest.append({
                    "id": f"{vid}_{start:.2f}_{start + 4.0:.2f}",
                    "video_id": vid,
                    "video_path": f"data/out/{vid}.mp4",
                    "json_path": f"data/out/{vid}.json",
                    "start_sec": start,
                    "end_sec": start + 4.0,
                    "text": f"{vid} {j}",
                })
    return manifest, {"id_to_movie_key": mapping}


def test_make_val_pool_movie_grouping():
    """Dedup mapping keeps all ids of one film on the same side of the split."""
    from scripts.make_val_pool import build_val_pool, _serialize

    manifest, report = _dedup_manifest()
    movie_key_map = _load_movie_key_map_dict(report)

    # The hold-out decision is movie-level, so no movie key is in both splits.
    def groups(items):
        return {_group_key(i, movie_key_map) for i in items}

    train_items, val_items = _split_manifest_by_video(
        manifest, 0.5, 7, movie_key_map
    )
    train_keys, val_keys = groups(train_items), groups(val_items)
    assert train_keys.isdisjoint(val_keys)
    assert train_keys | val_keys == {movie_key_map[i["video_id"]] for i in manifest}

    pool = build_val_pool(manifest, val_ratio=0.5, seed=7,
                          per_movie_cap=50, total_cap=20000, movie_key_map=movie_key_map)
    assert pool
    pool_keys = {movie_key_map[it["video_id"]] for it in pool}
    assert pool_keys <= val_keys and pool_keys.isdisjoint(train_keys)

    # No sibling id of a held-out film leaks into the training split.
    train_ids = {i["video_id"] for i in train_items}
    for movie in pool_keys:
        siblings = {k for k, v in movie_key_map.items() if v == movie}
        assert siblings.isdisjoint(train_ids), f"leak: {siblings & train_ids}"
    # Every segment of a held-out film is present in the pool (video_id dedup).
    pool_seg_ids = {(p["video_id"], p["start_sec"], p["end_sec"]) for p in pool}
    for it in manifest:
        if movie_key_map[it["video_id"]] in pool_keys:
            assert (it["video_id"], it["start_sec"], it["end_sec"]) in pool_seg_ids

    # Deterministic across runs.
    sha1 = hashlib.sha256(_serialize(build_val_pool(
        manifest, val_ratio=0.5, seed=7, movie_key_map=movie_key_map))).hexdigest()
    sha2 = hashlib.sha256(_serialize(build_val_pool(
        manifest, val_ratio=0.5, seed=7, movie_key_map=movie_key_map))).hexdigest()
    assert sha1 == sha2


def _load_movie_key_map_dict(report):
    """Build the id->movie_key map directly from a report dict (no file I/O)."""
    mapping = report["id_to_movie_key"]
    return {str(k): str(v) for k, v in mapping.items()}


def test_load_movie_key_map_full_and_plain(tmp_path=None):
    """_load_movie_key_map accepts both a full report and a plain mapping."""
    import tempfile
    manifest, report = _dedup_manifest()
    with tempfile.TemporaryDirectory() as td:
        full = os.path.join(td, "dedup_report.json")
        with open(full, "w") as f:
            json.dump(report, f)
        assert _load_movie_key_map(full) == report["id_to_movie_key"]

        plain = os.path.join(td, "map.json")
        with open(plain, "w") as f:
            json.dump(report["id_to_movie_key"], f)
        assert _load_movie_key_map(plain) == report["id_to_movie_key"]


def test_make_val_pool_dedup_cli():
    """--dedup-report is accepted and produces a movie-disjoint pool."""
    manifest, report = _dedup_manifest()
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "manifest.json")
        rep_path = os.path.join(td, "dedup_report.json")
        with open(in_path, "w") as f:
            json.dump(manifest, f)
        with open(rep_path, "w") as f:
            json.dump(report, f)
        out = os.path.join(td, "pool.json")
        sha_out = os.path.join(td, "pool.sha256")
        proc = subprocess.run(
            [sys.executable, "-m", "scripts.make_val_pool",
             "--input", in_path, "--output", out, "--sha-output", sha_out,
             "--dedup-report", rep_path, "--val-ratio", "0.5", "--seed", "7"],
            cwd=_REPO_ROOT, check=True, capture_output=True, text=True,
        )
        assert "dedup_report=" in proc.stdout or "movie_key" in proc.stdout
        with open(out) as f:
            pool = json.load(f)
        with open(sha_out) as f:
            assert hashlib.sha256(open(out, "rb").read()).hexdigest() == f.read().strip()
        mapping = report["id_to_movie_key"]
        train_items, val_items = _split_manifest_by_video(manifest, 0.5, 7, mapping)
        train_ids = {i["video_id"] for i in train_items}
        for it in pool:
            movie = mapping[it["video_id"]]
            siblings = {k for k, v in mapping.items() if v == movie}
            assert siblings.isdisjoint(train_ids), f"siblings of {it['video_id']} leaked"


def test_evaluate_retrieval_single_process():
    """evaluate_retrieval returns n_pool == len(dataset) and aligned hit vectors."""
    import logging
    from torch.utils.data import DataLoader
    from omegaconf import OmegaConf
    from scripts.train_hydra import evaluate_retrieval

    class TinyDataset:
        def __len__(self):
            return 7

        def __getitem__(self, i):
            return {
                "audio": torch.zeros(1, 160),
                "video": torch.zeros(2, 3, 4, 4),
                "audio_padding_mask": torch.zeros(1, 160, dtype=torch.bool),
            }

    def collate(batch):
        return {
            "audio": torch.stack([b["audio"] for b in batch]),
            "video": torch.stack([b["video"] for b in batch]),
            "audio_padding_mask": torch.stack([b["audio_padding_mask"] for b in batch]),
        }

    class TinyModel:
        def eval(self):
            return self

        def encode_audio(self, audio, mask):
            return torch.ones(audio.shape[0], 4)

        def encode_video(self, video):
            return torch.ones(video.shape[0], 4)

    cfg = OmegaConf.create({
        "training": {"mixed_precision": False},
        "validation": {"retrieval_top_k": [1, 5, 10], "dump_embeddings": False},
    })
    loader = DataLoader(TinyDataset(), batch_size=3, shuffle=False,
                        collate_fn=collate, drop_last=False)
    m = evaluate_retrieval(cfg, TinyModel(), loader, torch.device("cpu"),
                           logging.getLogger("test"), step=0, is_ddp=False)
    assert m["n_pool"] == 7
    assert len(m["a2v_hit_vec@1"]) == 7
    assert len(m["v2a_hit_vec@10"]) == 7


def test_selection_rules():
    """Frozen selection rules: metric mapping, sigma gate, trailing-3 winner."""
    metrics = {"a2v_hits@5": 30, "v2a_hits@5": 20, "a2v_r@1": 0.2, "v2a_r@1": 0.4}
    assert _selection_metric_value(metrics, "hits5_sum") == 50.0
    assert abs(_selection_metric_value(metrics, "mean_r1") - 0.3) < 1e-9

    # Trailing-3 window is inclusive of the current epoch and bounded above.
    tr = [(1, 10.0), (2, 20.0), (3, 30.0), (4, 0.0)]
    assert _trailing_mean(tr, 3, window=3) == 20.0
    assert abs(_trailing_mean(tr, 4, window=3) - (20.0 + 30.0 + 0.0) / 3) < 1e-9

    # Sigma gate: first epoch always accepted; a +5 fluke over a 100-hit
    # incumbent (n=1000, p=0.05, sigma=6.89) is rejected; a +15 real gain passes.
    assert _frozen_live_accept(5, float("-inf"), -1, 1000, 2.0) is True
    assert _frozen_live_accept(105, 100.0, 1, 1000, 2.0) is False
    assert _frozen_live_accept(115, 100.0, 1, 1000, 2.0) is True
    assert _frozen_live_accept(90, 100.0, 1, 1000, 2.0) is False

    # Post-hoc winner = max trailing-3 mean; strict > keeps the earlier tie.
    def winner(series):
        best, w, trailing = float("-inf"), -1, []
        for epoch, h in enumerate(series, start=1):
            trailing.append((epoch, float(h)))
            t = _trailing_mean(trailing, epoch, window=3)
            if t > best:
                best, w = t, epoch
        return w, best

    assert winner([10, 10, 10, 20, 20, 20, 5, 5, 5]) == (6, 20.0)
    assert winner([5, 5, 5, 5])[0] == 1  # ties -> earlier epoch


def _small_manifest(n_movies=20, segs_per_movie=3):
    manifest = []
    for i in range(n_movies):
        for j in range(segs_per_movie):
            vid = f"movie{i:02d}"
            start = j * 4.0
            manifest.append({
                "id": f"{vid}_{start:.2f}_{start + 4.0:.2f}",
                "video_id": vid,
                "video_path": f"data/out/{vid}.mp4",
                "json_path": f"data/out/{vid}.json",
                "start_sec": start,
                "end_sec": start + 4.0,
                "text": f"segment {i} {j}",
            })
    return manifest


def test_make_val_pool_disjoint_and_deterministic():
    sys.path.insert(0, _REPO_ROOT)
    from scripts.make_val_pool import build_val_pool, _serialize

    manifest = _small_manifest()
    pool = build_val_pool(manifest, val_ratio=0.25, seed=42, per_movie_cap=50, total_cap=20000)
    assert pool, "pool should not be empty"

    pool_keys = {(it["video_id"], it["start_sec"], it["end_sec"]) for it in pool}
    train_items = [it for it in manifest
                   if (it["video_id"], it["start_sec"], it["end_sec"]) not in pool_keys]
    pool_movies = {it["video_id"] for it in pool}
    train_movies = {it["video_id"] for it in train_items}
    assert pool_movies.isdisjoint(train_movies), (pool_movies & train_movies)
    assert pool_movies | train_movies == {it["video_id"] for it in manifest}

    # Determinism: two builds serialize to the same bytes/sha256.
    sha1 = hashlib.sha256(_serialize(build_val_pool(manifest))).hexdigest()
    sha2 = hashlib.sha256(_serialize(build_val_pool(manifest))).hexdigest()
    assert sha1 == sha2

    # Exercise the CLI/IO path twice and compare the written sha file.
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "manifest.json")
        with open(in_path, "w") as f:
            json.dump(manifest, f)
        shas = []
        for run in range(2):
            out = os.path.join(td, f"pool_{run}.json")
            sha_out = os.path.join(td, f"pool_{run}.sha256")
            subprocess.run(
                [sys.executable, "-m", "scripts.make_val_pool",
                 "--input", in_path, "--output", out, "--sha-output", sha_out,
                 "--val-ratio", "0.25", "--seed", "42"],
                cwd=_REPO_ROOT, check=True, capture_output=True, text=True,
            )
            with open(sha_out) as f:
                shas.append(f.read().strip())
            with open(out, "rb") as f:
                body = f.read()
            assert hashlib.sha256(body).hexdigest() == shas[-1]
        assert shas[0] == shas[1]


def test_clustered_mcnemar():
    """Movie clustering must widen the variance vs treating segments independent."""
    from scripts.cross_eval import clustered_mcnemar

    # No discordance -> no evidence.
    assert clustered_mcnemar([True, True], [True, True], ["a", "b"])["p"] == 1.0

    # A wins on 4 segments. Clustering them in one movie is weaker evidence
    # (one independent cluster) than spreading them across four movies.
    a = [True, True, True, True]
    b = [False, False, False, False]
    together = clustered_mcnemar(a, b, ["m1", "m1", "m1", "m1"])
    separate = clustered_mcnemar(a, b, ["m1", "m2", "m3", "m4"])
    assert together["delta"] == separate["delta"] == 4
    assert together["z"] < separate["z"]
    assert together["p"] > separate["p"]


def test_eval_zeroshot_imports():
    """The zero-shot eval script must still import and expose its own path."""
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.eval_zeroshot_vggsound", "--help"],
        cwd=_REPO_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "checkpoint" in proc.stdout


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    raise SystemExit(1 if failures else 0)
