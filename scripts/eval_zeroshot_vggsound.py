#!/usr/bin/env python
"""
Zero-shot audio<->video retrieval on a VGGSound subset.

Evaluates one or more FlikModel checkpoints (plus an optional randomly
initialized baseline) on the class-stratified VGGSound test subset produced by
``scripts/download_vggsound_subset.py``.

Per clip:
  * video: 16 frames from window [0, 10]s via ``video_to_tensor``
  * audio: 10s at 16 kHz via ``audio_to_tensor``
  * embeddings: ``FlikModel.encode_audio`` / ``encode_video`` (L2-normalized),
    computed in FP32 under ``torch.no_grad()`` (no autocast — matches the
    verified eval-side precision used during training).

Metrics over ALL successfully decoded clips:
  * paired retrieval R@1/5/10 (a2v and v2a) via ``retrieval_recall_at_k``
  * median rank (1-based) for a2v and v2a
  * per-class table: a2v top-1 class-match over the full candidate pool,
    plus paired a2v R@1 restricted to the class.

Usage:
    uv run --extra cu128 python -m scripts.eval_zeroshot_vggsound \
        --checkpoint aug11_final:checkpoints/checkpoint_epoch50.pth \
        --checkpoint aug13_best:checkpoints/checkpoint_best.pth \
        --random-init

    # CPU smoke test (no checkpoints needed):
    uv run --extra cpu python -m scripts.eval_zeroshot_vggsound \
        --clips-manifest /tmp/vgg_smoke/clips_manifest.json \
        --random-init --device cpu --batch-size 2 --out /tmp/vgg_smoke/results.json
"""

import argparse
import json
import logging
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# HuggingFace cache lives in the repo checkout (matches scripts/train_hydra.py
# and run_train_snellius.sh). Set before importing transformers/model code.
os.environ.setdefault("HF_HOME", os.path.join(_REPO_ROOT, "cache"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "true")

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from src.models.flik_model import FlikModel  # noqa: E402
from src.eval.retrieval import retrieval_recall_at_k  # noqa: E402
from src.utils.video import video_to_tensor  # noqa: E402
from src.utils.audio import audio_to_tensor  # noqa: E402

RANDOM_INIT_SEED = 1234


# ---------------------------------------------------------------------------
# Config / model
# ---------------------------------------------------------------------------

def setup_logging():
    logger = logging.getLogger("eval_zeroshot_vggsound")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


def load_model_config(config_path=None):
    """Read the training config so model hyperparameters stay in sync."""
    if config_path is None:
        config_path = os.path.join(_REPO_ROOT, "src", "configs", "default.yaml")
    cfg = OmegaConf.load(config_path)
    return cfg.model, cfg


def build_model(model_cfg, device):
    """Build FlikModel with the training config's architecture kwargs."""
    return FlikModel(
        audio_model_name=model_cfg.audio_model_name,
        video_model_name=model_cfg.video_model_name,
        hidden_dim=model_cfg.hidden_dim,
        audio_feature_layer=model_cfg.audio_feature_layer,
        temporal_layers=model_cfg.temporal_layers,
        cross_attention_layers=model_cfg.cross_attention_layers,
        num_heads=model_cfg.num_heads,
        ff_dim=model_cfg.ff_dim,
        dropout=model_cfg.dropout,
        freeze_audio=model_cfg.freeze_audio,
        freeze_video=model_cfg.freeze_video,
        use_grounded_masked_prediction=model_cfg.use_grounded_masked_prediction,
        mlm_mask_prob=model_cfg.mlm_mask_prob,
        mlm_mask_length=model_cfg.mlm_mask_length,
        num_codebook_entries=model_cfg.num_codebook_entries,
    ).to(device)


def randomize_model(model, seed=RANDOM_INIT_SEED):
    """Deterministically re-initialize all parameters of a model.

    Used for the random-init sanity baseline. The seed is fixed so the same
    weights are produced across processes (this lets the checkpoint-loading
    path be verified against the random-init row). Raw ``nn.Parameter``s that
    are not owned by a standard layer (e.g. MultiheadAttention.in_proj_weight,
    VideoMAE cls_token / frame_pos_embed) are also reset so no constructor-time
    random value leaks through.
    """
    handled = set()

    def _init(m):
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            nn.init.xavier_uniform_(m.weight)
            handled.add(id(m.weight))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                handled.add(id(m.bias))
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d,
                            nn.BatchNorm2d, nn.BatchNorm3d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
                handled.add(id(m.weight))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                handled.add(id(m.bias))
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            handled.add(id(m.weight))

    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    try:
        model.apply(_init)
        for _, p in model.named_parameters():
            if id(p) in handled:
                continue
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.zeros_(p)
    finally:
        torch.set_rng_state(rng_state)
    return model


def load_checkpoint_into(model, path, logger):
    """Load a training checkpoint's model_state_dict (strips DDP prefixes)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        logger.warning(
            f"{path}: no 'model_state_dict' key — treating the file as a raw "
            f"state dict"
        )
        state_dict = ckpt
    else:
        raise ValueError(f"{path}: unsupported checkpoint type {type(ckpt)}")

    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        logger.warning(
            f"{path}: {len(missing)} missing key(s): {list(missing)[:8]}"
            + (" ..." if len(missing) > 8 else "")
        )
    if unexpected:
        logger.warning(
            f"{path}: {len(unexpected)} unexpected key(s): {list(unexpected)[:8]}"
            + (" ..." if len(unexpected) > 8 else "")
        )
    if not missing and not unexpected:
        logger.info(f"{path}: all keys matched")
    return model


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_clips(manifest_path, logger):
    """Load the clips manifest and resolve paths against the manifest dir."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        clips = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    resolved = []
    for c in clips:
        vp = c["video_path"]
        ap = c["audio_path"]
        if not os.path.isabs(vp):
            vp = os.path.join(base_dir, vp)
        if not os.path.isabs(ap):
            ap = os.path.join(base_dir, ap)
        resolved.append({
            "video_id": c.get("video_id", os.path.basename(os.path.dirname(vp))),
            "label": c.get("label", ""),
            "video_path": vp,
            "audio_path": ap,
        })
    logger.info(f"Loaded {len(resolved)} clips from {manifest_path}")
    return resolved


def _decode_clip(clip, num_frames, sample_rate, start_sec, end_sec):
    video = video_to_tensor(clip["video_path"], start_sec, end_sec,
                            num_frames=num_frames)
    audio = audio_to_tensor(clip["audio_path"], start_sec, end_sec,
                            sample_rate=sample_rate)
    return audio, video


def _pad_audio(audio_list):
    """Pad a list of (1, T) waveforms; return (B,1,T_max) and mask (padded=True)."""
    max_len = max(a.shape[1] for a in audio_list)
    audio = torch.zeros(len(audio_list), 1, max_len)
    mask = torch.ones(len(audio_list), max_len, dtype=torch.bool)
    for i, a in enumerate(audio_list):
        audio[i, :, :a.shape[1]] = a
        mask[i, :a.shape[1]] = False
    return audio, mask


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(audio_embs, video_embs, labels, ks=(1, 5, 10)):
    """Paired retrieval + median rank + per-class table."""
    n = audio_embs.shape[0]
    # retrieval_recall_at_k keys by min(k, N); re-key to the requested ks so
    # small smoke-test subsets still report every requested column.
    raw = retrieval_recall_at_k(audio_embs, video_embs, ks=ks)
    metrics = {}
    for k in ks:
        for direction in ("a2v", "v2a"):
            metrics[f"{direction}_r@{k}"] = raw.get(f"{direction}_r@{min(int(k), n)}")

    sim = torch.mm(audio_embs, video_embs.t())
    idx = torch.arange(n)
    a2v_rank = (sim > sim[idx, idx].unsqueeze(1)).sum(dim=1) + 1
    v2a_sim = sim.t()
    v2a_rank = (v2a_sim > v2a_sim[idx, idx].unsqueeze(1)).sum(dim=1) + 1
    metrics["a2v_median_rank"] = float(a2v_rank.median())
    metrics["v2a_median_rank"] = float(v2a_rank.median())

    # Class-match top-1 over the FULL candidate pool.
    label_names = sorted(set(labels))
    label_to_id = {name: i for i, name in enumerate(label_names)}
    label_ids = torch.tensor([label_to_id[l] for l in labels])
    a2v_top1 = sim.argmax(dim=1)
    v2a_top1 = sim.t().argmax(dim=1)
    a2v_class_hit = (label_ids[a2v_top1] == label_ids).float()
    v2a_class_hit = (label_ids[v2a_top1] == label_ids).float()
    metrics["a2v_class_r@1"] = float(a2v_class_hit.mean())
    metrics["v2a_class_r@1"] = float(v2a_class_hit.mean())

    per_class = {}
    for name in label_names:
        members = [i for i, l in enumerate(labels) if l == name]
        member_t = torch.tensor(members)
        per_class[name] = {
            "n_clips": len(members),
            "a2v_r@1": float(a2v_class_hit[members].mean()),
            "v2a_r@1": float(v2a_class_hit[members].mean()),
            "paired_a2v_r@1": float((a2v_top1[members] == member_t).float().mean()),
        }
    metrics["per_class"] = per_class
    return metrics


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(models, clips, device, batch_size, num_frames, sample_rate,
             start_sec, end_sec, logger):
    """Encode all clips for every model and return {name: metrics} + failures."""
    names = list(models.keys())
    audio_embs = {n: [] for n in names}
    video_embs = {n: [] for n in names}
    labels = []
    failures = []
    used = 0
    t0 = time.time()

    for start in range(0, len(clips), batch_size):
        batch = clips[start:start + batch_size]
        audio_list, video_list = [], []
        for c in batch:
            try:
                a, v = _decode_clip(c, num_frames, sample_rate, start_sec, end_sec)
            except Exception as e:
                failures.append({"video_id": c["video_id"], "error": str(e)})
                continue
            audio_list.append(a)
            video_list.append(v)
            labels.append(c["label"])

        if not audio_list:
            continue

        audio_padded, audio_mask = _pad_audio(audio_list)
        audio_padded = audio_padded.to(device)
        audio_mask = audio_mask.to(device)
        video_stacked = torch.stack(video_list, dim=0).to(device)

        with torch.no_grad():
            for name in names:
                model = models[name]
                ae = model.encode_audio(audio_padded, audio_mask).detach().float().cpu()
                ve = model.encode_video(video_stacked).detach().float().cpu()
                audio_embs[name].append(ae)
                video_embs[name].append(ve)

        used += len(audio_list)
        if used % (batch_size * 10) == 0 or used >= len(clips) - 1:
            logger.info(
                f"  decoded {used}/{len(clips)} clips "
                f"({len(failures)} failures) [{time.time() - t0:.0f}s]"
            )

    results = {}
    for name in names:
        if not audio_embs[name]:
            logger.warning(f"{name}: no embeddings (all clips failed to decode)")
            results[name] = None
            continue
        a = torch.cat(audio_embs[name], dim=0)
        v = torch.cat(video_embs[name], dim=0)
        logger.info(f"{name}: embedding matrices {tuple(a.shape)} / {tuple(v.shape)}")
        results[name] = compute_metrics(a, v, labels)
    return results, failures, used


def format_table(rows, ks):
    headers = ["checkpoint", "N", "fail"]
    for k in ks:
        headers += [f"a2v R@{k}", f"v2a R@{k}"]
    headers += ["a2v mrank", "v2a mrank", "a2v cls R@1"]
    lines = [" | ".join(headers)]
    lines.append("-" * len(lines[0]))
    for name, m in rows:
        if m is None:
            lines.append(f"{name} | FAILED")
            continue
        cells = [name, str(m.get("n_clips", "?")), str(m.get("n_decode_failures", "?"))]
        for k in ks:
            cells += [f"{m.get(f'a2v_r@{k}', float('nan')):.3f}",
                      f"{m.get(f'v2a_r@{k}', float('nan')):.3f}"]
        cells += [f"{m.get('a2v_median_rank', float('nan')):.1f}",
                  f"{m.get('v2a_median_rank', float('nan')):.1f}",
                  f"{m.get('a2v_class_r@1', float('nan')):.3f}"]
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Zero-shot VGGSound retrieval eval for FlikModel checkpoints."
    )
    p.add_argument("--checkpoint", action="append", default=[],
                   metavar="NAME:PATH",
                   help="Checkpoint to evaluate, repeatable (e.g. "
                        "aug11_final:checkpoints/checkpoint_epoch50.pth).")
    p.add_argument("--clips-manifest", type=str,
                   default="data/vggsound_subset/clips_manifest.json",
                   help="Clips manifest from download_vggsound_subset.py.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", type=str, default=None,
                   help="cuda|cpu (default: cuda if available).")
    p.add_argument("--out", type=str, default="zeroshot_results.json",
                   help="Output JSON path (default: zeroshot_results.json).")
    p.add_argument("--random-init", action="store_true",
                   help="Also evaluate a randomly initialized model (sanity baseline).")
    p.add_argument("--config", type=str, default=None,
                   help="Optional path to the model YAML config.")
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--end-sec", type=float, default=10.0)
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--sample-rate", type=int, default=16000)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger = setup_logging()

    if not args.checkpoint and not args.random_init:
        logger.error("Nothing to evaluate: pass --checkpoint and/or --random-init.")
        return 2

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    model_cfg, cfg = load_model_config(args.config)
    logger.info(
        "Model config: hidden_dim=%s audio_feature_layer=%s temporal_layers=%s "
        "cross_attention_layers=%s heads=%s ff_dim=%s dropout=%s",
        model_cfg.hidden_dim, model_cfg.audio_feature_layer,
        model_cfg.temporal_layers, model_cfg.cross_attention_layers,
        model_cfg.num_heads, model_cfg.ff_dim, model_cfg.dropout,
    )

    clips = load_clips(args.clips_manifest, logger)
    if not clips:
        logger.error("Clips manifest is empty.")
        return 1

    # --- build + load models ---------------------------------------------
    specs = []
    for spec in args.checkpoint:
        if ":" not in spec:
            logger.error(f"--checkpoint must be NAME:PATH, got: {spec}")
            return 2
        name, path = spec.split(":", 1)
        specs.append((name, path))
    if args.random_init:
        specs.append(("random_init", None))

    models = {}
    for name, path in specs:
        t0 = time.time()
        logger.info(f"[{name}] building FlikModel ({'random init' if path is None else path})")
        model = build_model(model_cfg, device)
        if path is None:
            randomize_model(model, RANDOM_INIT_SEED)
        else:
            load_checkpoint_into(model, path, logger)
        model.eval()
        models[name] = model
        logger.info(f"[{name}] ready in {time.time() - t0:.1f}s")

    # --- run --------------------------------------------------------------
    ks = tuple(cfg.validation.get("retrieval_top_k", [1, 5, 10]))
    logger.info(f"Evaluating {len(models)} model(s) on {len(clips)} clips "
                f"(window [{args.start_sec}, {args.end_sec}]s, "
                f"batch_size={args.batch_size}, ks={list(ks)})")
    results, failures, used = evaluate(
        models, clips, device, args.batch_size, args.num_frames,
        args.sample_rate, args.start_sec, args.end_sec, logger,
    )

    # --- assemble output --------------------------------------------------
    checkpoints_out = {}
    for name, m in results.items():
        if m is None:
            checkpoints_out[name] = None
            continue
        m = dict(m)
        m["n_clips"] = used
        m["n_decode_failures"] = len(failures)
        checkpoints_out[name] = m

    output = {
        "clips_manifest": os.path.abspath(args.clips_manifest),
        "n_clips_manifest": len(clips),
        "n_clips_used": used,
        "n_decode_failures": len(failures),
        "window_sec": [args.start_sec, args.end_sec],
        "num_frames": args.num_frames,
        "sample_rate": args.sample_rate,
        "ks": list(ks),
        "checkpoints": checkpoints_out,
        "decode_failures": failures,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Wrote results to {args.out}")

    if failures:
        logger.warning(f"{len(failures)} clip(s) failed to decode and were skipped:")
        for fail in failures[:10]:
            logger.warning(f"  {fail['video_id']}: {fail['error']}")
        if len(failures) > 10:
            logger.warning(f"  ... and {len(failures) - 10} more (see JSON)")

    print()
    print("=" * 100)
    print(format_table(list(checkpoints_out.items()), list(ks)))
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
