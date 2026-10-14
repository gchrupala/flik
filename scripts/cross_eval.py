#!/usr/bin/env python
"""One-off 2x2 diagnostic: two checkpoints x two validation pools.

For every cell (checkpoint, pool) this computes the full retrieval metric set
(including per-query hit vectors) and writes one JSON. It then reports a
summary table of hits@1/5/10 and lift, and -- within each pool -- a
movie-clustered McNemar test between the two checkpoints. Segments from the
same movie are correlated, so the test clusters discordant segments by movie.

This is a diagnostic only. It is NOT wired into the training loop.

Usage:
    python -m scripts.cross_eval \
        --checkpoint aug9:checkpoints/aug9/checkpoint_best.pth \
        --checkpoint aug13:checkpoints/aug13/checkpoint_best.pth \
        --pool v1:data/val_pool.json \
        --pool v2:data/val_pool_v2.json \
        --out-dir logdir/cross_eval --device cpu
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(_REPO_ROOT, "cache"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "true")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.models.flik_model import FlikModel  # noqa: E402
from src.datasets.video_audio_dataset import VideoAudioDataset  # noqa: E402
from src.eval.retrieval import retrieval_metrics  # noqa: E402

KS = (1, 5, 10)


def _parse_named(specs, what):
    """Parse repeated ``name:path`` arguments into an ordered dict."""
    if len(specs) != 2:
        raise SystemExit(f"expected exactly 2 --{what} name:path args, got {len(specs)}")
    out = {}
    for spec in specs:
        if ":" not in spec:
            raise SystemExit(f"--{what} must be name:path, got {spec!r}")
        name, path = spec.split(":", 1)
        out[name] = path
    return out


def _load_model_config(checkpoint_path=None, config_path=None):
    """Prefer the config stored inside the checkpoint, else the YAML config."""
    if checkpoint_path is not None:
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
                return OmegaConf.create(ckpt["config"])
        except Exception:
            pass
    if config_path is None:
        config_path = os.path.join(_REPO_ROOT, "src", "configs", "default.yaml")
    return OmegaConf.load(config_path)


def build_model(cfg, device):
    m = cfg.model
    return FlikModel(
        audio_model_name=m.audio_model_name,
        video_model_name=m.video_model_name,
        hidden_dim=m.hidden_dim,
        audio_feature_layer=m.audio_feature_layer,
        temporal_layers=m.temporal_layers,
        cross_attention_layers=m.cross_attention_layers,
        num_heads=m.num_heads,
        ff_dim=m.ff_dim,
        dropout=m.dropout,
        freeze_audio=m.freeze_audio,
        freeze_video=m.freeze_video,
        use_grounded_masked_prediction=m.use_grounded_masked_prediction,
        mlm_mask_prob=m.mlm_mask_prob,
        mlm_mask_length=m.mlm_mask_length,
        num_codebook_entries=m.num_codebook_entries,
    ).to(device)


def load_checkpoint_into(model, path, logger):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    cleaned = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger(f"{path}: {len(missing)} missing keys")
    if unexpected:
        logger(f"{path}: {len(unexpected)} unexpected keys")
    return model


def load_pool(path):
    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return items


@torch.no_grad()
def encode_pool(model, dataset, device, batch_size, num_workers):
    """Return (audio_embs, video_embs) in dataset order (CPU float32)."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=VideoAudioDataset.collate_fn,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    model.eval()
    audio_list, video_list = [], []
    for batch in loader:
        audio = batch["audio"].to(device)
        video = batch["video"].to(device)
        mask = batch["audio_padding_mask"].to(device)
        audio_list.append(model.encode_audio(audio, mask).detach().float().cpu())
        video_list.append(model.encode_video(video).detach().float().cpu())
    return torch.cat(audio_list, 0), torch.cat(video_list, 0)


def dataset_clusters(dataset):
    """Movie cluster id per segment, derived from the resolved video path."""
    return [
        os.path.splitext(os.path.basename(s["video_path"]))[0]
        for s in dataset.segments
    ]


def clustered_mcnemar(hits_a, hits_b, clusters):
    """Cluster-robust McNemar test on paired per-query hits.

    d_i = hit_a_i - hit_b_i in {-1, 0, 1}. The score U = sum_i d_i has
    cluster-robust variance sum_g (sum_{i in g} d_i)^2 under H0, because
    segments within a movie are correlated. Returns dict with the raw
    difference (delta = hits_a - hits_b), z and two-sided p.
    """
    cluster_sums = defaultdict(int)
    for a, b, c in zip(hits_a, hits_b, clusters):
        cluster_sums[c] += int(bool(a)) - int(bool(b))
    delta = sum(cluster_sums.values())
    var = sum(s * s for s in cluster_sums.values())
    if var <= 0:
        return {"delta": int(delta), "z": 0.0, "p": 1.0}
    z = delta / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return {"delta": int(delta), "z": z, "p": p}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="name:path (exactly 2)")
    parser.add_argument("--pool", action="append", required=True,
                        help="name:path (exactly 2)")
    parser.add_argument("--config", default=None, help="Fallback YAML config")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", default="logdir/cross_eval")
    args = parser.parse_args(argv)

    checkpoints = _parse_named(args.checkpoint, "checkpoint")
    pools = _parse_named(args.pool, "pool")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    os.makedirs(args.out_dir, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    cells = {}
    hit_vecs = {}
    cluster_map = {}
    for ckpt_name, ckpt_path in checkpoints.items():
        cfg = _load_model_config(ckpt_path, args.config)
        model = load_checkpoint_into(build_model(cfg, device), ckpt_path, log)
        for pool_name, pool_path in pools.items():
            items = load_pool(pool_path)
            dataset = VideoAudioDataset(
                manifest_path=pool_path,
                sample_rate=cfg.dataset.sample_rate,
                num_frames=cfg.dataset.num_frames,
                min_duration=cfg.dataset.min_duration,
                max_duration=cfg.dataset.max_duration,
                dummy=False,
                manifest_items=items,
            )
            audio_embs, video_embs = encode_pool(
                model, dataset, device, args.batch_size, args.num_workers
            )
            metrics = retrieval_metrics(audio_embs, video_embs, ks=KS)
            cells[(ckpt_name, pool_name)] = metrics
            hit_vecs[(ckpt_name, pool_name)] = metrics
            cluster_map[pool_name] = dataset_clusters(dataset)
            cell_path = os.path.join(args.out_dir, f"cell_{ckpt_name}_{pool_name}.json")
            with open(cell_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, sort_keys=True)
            log(f"Wrote {cell_path} (n_pool={metrics['n_pool']})")
        del model

    # Summary table
    summary = []
    for (ckpt_name, pool_name), m in cells.items():
        row = {"checkpoint": ckpt_name, "pool": pool_name, "n_pool": m["n_pool"]}
        for direction in ("a2v", "v2a"):
            for k in KS:
                row[f"{direction}_hits@{k}"] = m[f"{direction}_hits@{k}"]
                row[f"{direction}_lift@{k}"] = m[f"{direction}_lift@{k}"]
        summary.append(row)

    # Movie-clustered McNemar between the two checkpoints, within each pool.
    ckpt_names = list(checkpoints.keys())
    mcnemar = {}
    for pool_name in pools:
        clusters = cluster_map[pool_name]
        mcnemar[pool_name] = {}
        for direction in ("a2v", "v2a"):
            mcnemar[pool_name][direction] = {}
            for k in KS:
                key = f"{direction}_hit_vec@{k}"
                a = hit_vecs[(ckpt_names[0], pool_name)][key]
                b = hit_vecs[(ckpt_names[1], pool_name)][key]
                mcnemar[pool_name][direction][f"@{k}"] = clustered_mcnemar(a, b, clusters)

    summary_obj = {
        "checkpoints": checkpoints,
        "pools": pools,
        "summary": summary,
        "mcnemar_clustered_by_movie": mcnemar,
    }
    summary_path = os.path.join(args.out_dir, "cross_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, indent=2, sort_keys=True)

    # Print a compact table.
    print("\nhits@k (lift over chance) per cell")
    header = f"{'checkpoint':<16} {'pool':<12} {'n':>7} " + " ".join(
        f"{d}@{k}" for d in ("a2v", "v2a") for k in KS
    )
    print(header)
    for row in summary:
        cols = " ".join(
            f"{row[f'{d}_hits@{k}']}({row[f'{d}_lift@{k}']:.1f})"
            for d in ("a2v", "v2a")
            for k in KS
        )
        print(f"{row['checkpoint']:<16} {row['pool']:<12} {row['n_pool']:>7} {cols}")

    print(f"\nmovie-clustered McNemar ({ckpt_names[0]} vs {ckpt_names[1]})")
    for pool_name in pools:
        for direction in ("a2v", "v2a"):
            for k in KS:
                r = mcnemar[pool_name][direction][f"@{k}"]
                print(
                    f"  pool={pool_name:<12} {direction}@{k}: "
                    f"delta={r['delta']:+d} z={r['z']:.2f} p={r['p']:.4g}"
                )
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
