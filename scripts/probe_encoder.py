#!/usr/bin/env python
"""Frozen-feature linear probe for the flik encoder towers.

Measures whether fine-tuning regressed the two pretrained towers by freezing
each tower, extracting pooled features for a fixed clip list, training a linear
classifier on a fixed stratified split, and comparing every checkpoint against
the pretrained (HF) baseline.

What is probed (tower output, BEFORE flik's projection heads):

* audio checkpoints: ``model.dual_encoder.audio_encoder(audio, padding_mask)``
  -> ``out["features"]`` pooled by ``DualEncoder._pool_audio_features``
  (masked mean over valid time steps). 768-d, pre ``audio_proj``.
* video checkpoints: ``model.dual_encoder.video_encoder(video)["cls_embedding"]``
  (default) or ``["frame_embeddings"].mean(dim=1)`` with
  ``--video-feature tower_mean``. 768-d, pre ``video_proj``.
* pretrained audio baseline: a fresh ``AudioEncoder(facebook/wav2vec2-base)``
  (``audio_feature_layer`` from the training config), same masked-mean pooling.
  No flik checkpoint and no flik projection head is touched.
* pretrained video baseline: a fresh ``VideoMAEModel(MCG-NJU/videomae-base)``;
  ``last_hidden_state`` is mean-pooled over tokens. VideoMAE-base has no CLS
  token, so mean-over-tokens is the canonical VideoMAE pooling.

The clip list and the stratified 80/20 split are computed once and applied to
every backbone. The split indices are saved in the output JSON.

Usage:
    python -m scripts.probe_encoder \
        --modality audio --dataset esc50 \
        --backbone pretrained:PRETRAINED \
        --backbone aug13_best:eval_assets/aug13_best.pth \
        --device cuda --out probe_results_esc50.json
"""

import argparse
import collections
import functools
import json
import logging
import multiprocessing
import os
import sys
import time
import wave
from concurrent.futures import ProcessPoolExecutor

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# HuggingFace cache lives in the repo checkout (matches scripts/train_hydra.py
# and scripts/eval_zeroshot_vggsound.py). Set before importing model code.
os.environ.setdefault("HF_HOME", os.path.join(_REPO_ROOT, "cache"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "true")

if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from transformers import VideoMAEModel  # noqa: E402

from src.models.audio_encoder import AudioEncoder  # noqa: E402
from src.models.dual_encoder import DualEncoder  # noqa: E402
from src.utils.audio import load_audio_from_wav  # noqa: E402
from src.utils.video import video_to_tensor  # noqa: E402

# Reuse the checkpoint/config helpers from the zero-shot eval so checkpoint
# loading and model construction stay identical to the retrieval pipeline.
from scripts.eval_zeroshot_vggsound import (  # noqa: E402
    build_model,
    load_checkpoint_into,
    load_model_config,
)

# Defaults per dataset (clips kept per class before the split).
DEFAULT_MAX_CLIPS_PER_CLASS = {"esc50": 40, "speech_commands": 100, "vggsound": 48}
AUDIO_DATASETS = ("esc50", "speech_commands")
VIDEO_DATASETS = ("vggsound",)
SPLIT_SEED = 42
TEST_FRACTION = 0.2

# Video window used by the zero-shot eval (and by the probe for VGGSound).
VIDEO_START_SEC = 0.0
VIDEO_END_SEC = 10.0
VIDEO_NUM_FRAMES = 16
AUDIO_SAMPLE_RATE = 16000

# Batches in flight when decoding with worker processes. A decoded video clip
# is ~10 MB, so keep the window bounded (matches eval_zeroshot_vggsound.py).
_DECODE_PREFETCH_BATCHES = 3


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging():
    logger = logging.getLogger("probe_encoder")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Clip lists
# ---------------------------------------------------------------------------

def _load_audio_clip_list(dataset, dataset_root, logger):
    if dataset == "esc50":
        root = os.path.join(dataset_root, "ESC-50-master")
        csv_path = os.path.join(root, "meta", "esc50.csv")
        audio_dir = os.path.join(root, "audio")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"ESC-50 metadata not found: {csv_path}")
        import csv as csvmod

        clips = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csvmod.DictReader(f):
                clips.append(
                    {
                        "id": row["filename"],
                        "path": os.path.join(audio_dir, row["filename"]),
                        "label": row["category"],
                    }
                )
        return clips

    if dataset == "speech_commands":
        root = os.path.join(dataset_root, "speech_commands_subset")
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Speech Commands dir not found: {root}")
        clips = []
        for label in sorted(os.listdir(root)):
            class_dir = os.path.join(root, label)
            if not os.path.isdir(class_dir):
                continue
            for fn in sorted(os.listdir(class_dir)):
                if fn.lower().endswith(".wav"):
                    clips.append(
                        {
                            "id": f"{label}/{fn}",
                            "path": os.path.join(class_dir, fn),
                            "label": label,
                        }
                    )
        return clips

    raise ValueError(f"{dataset!r} is not an audio dataset")


def _load_vggsound_clip_list(manifest_path, logger):
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"VGGSound manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    clips = []
    for c in manifest:
        vp = c["video_path"]
        if not os.path.isabs(vp):
            vp = os.path.join(base_dir, vp)
        clips.append(
            {
                "id": c.get("video_id", os.path.basename(os.path.dirname(vp))),
                "path": vp,
                "label": c["label"],
            }
        )
    return clips


def build_clip_list(dataset, dataset_root, max_per_class, vggsound_manifest, logger):
    """Build a deterministic clip list with at most ``max_per_class`` per class.

    Determinism: within a class, clips are sorted by path and the first
    ``max_per_class`` are kept; the final list is sorted by (label, path). This
    makes the list, and therefore the split, identical across backbones.
    """
    if dataset in AUDIO_DATASETS:
        clips = _load_audio_clip_list(dataset, dataset_root, logger)
    elif dataset in VIDEO_DATASETS:
        clips = _load_vggsound_clip_list(vggsound_manifest, logger)
    else:
        raise ValueError(f"unknown dataset: {dataset}")

    by_class = collections.defaultdict(list)
    for c in clips:
        by_class[c["label"]].append(c)

    selected = []
    for label in sorted(by_class):
        items = sorted(by_class[label], key=lambda c: c["path"])
        selected.extend(items[:max_per_class])
    selected.sort(key=lambda c: (c["label"], c["path"]))

    logger.info(
        "Clip list: %d clips, %d classes (max %d/class) from %s",
        len(selected), len(by_class), max_per_class, dataset,
    )
    return selected


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def stratified_split_seeded(labels, test_fraction, seed):
    """Deterministic per-class 80/20 split that tolerates singleton classes.

    ``sklearn.model_selection.StratifiedShuffleSplit`` refuses classes with
    fewer than 2 members. The VGGSound subset has singleton classes, so this
    helper performs the equivalent operation manually: shuffle each class with
    a single seeded RNG, take ``round(n * test_fraction)`` (clamped to
    ``[1, n-1]`` when ``n >= 2``) for test, keep the rest for train. Singleton
    classes stay in train.
    """
    rng = np.random.RandomState(seed)
    by_class = collections.defaultdict(list)
    for i, label in enumerate(labels):
        by_class[label].append(i)

    train_idx, test_idx = [], []
    for label in sorted(by_class, key=str):
        idx = np.array(sorted(by_class[label]), dtype=np.int64)
        rng.shuffle(idx)
        n = len(idx)
        if n >= 2:
            n_test = int(round(n * test_fraction))
            n_test = max(1, min(n - 1, n_test))
        else:
            n_test = 0
        test_idx.extend(idx[:n_test].tolist())
        train_idx.extend(idx[n_test:].tolist())

    return sorted(int(i) for i in train_idx), sorted(int(i) for i in test_idx)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def _wav_duration_sec(path):
    """Read a WAV header for its duration. Falls back to PyAV on failure."""
    try:
        with wave.open(path, "rb") as w:
            n = w.getnframes()
            rate = w.getframerate()
            if rate > 0:
                return n / float(rate)
    except Exception:
        pass

    import av

    av.logging.set_level(None)
    container = av.open(path)
    try:
        stream = next(s for s in container.streams if s.type == "audio")
        if stream.duration is not None:
            return float(stream.duration * stream.time_base)
        return float(container.duration) / av.time_base
    finally:
        container.close()


def _decode_audio_worker(index, clip, sample_rate):
    """Decode one WAV clip with the training-time normalization.

    ``load_audio_from_wav`` -> ``_load_audio_segment`` resamples to 16 kHz mono,
    trims/pads to the segment length, and normalizes to zero mean / unit
    variance. This is the exact training-time preprocessing; do not reimplement.
    """
    try:
        duration = _wav_duration_sec(clip["path"])
        if duration <= 0:
            raise RuntimeError(f"non-positive duration: {duration}")
        audio = load_audio_from_wav(
            clip["path"], 0.0, duration, target_sample_rate=sample_rate
        )
        return index, audio, None
    except Exception as e:  # noqa: BLE001 - report per-clip failure
        return index, None, f"{type(e).__name__}: {e}"


def _decode_video_worker(index, clip, num_frames, start_sec, end_sec):
    """Decode one video clip with ``video_to_tensor`` (matches zero-shot eval)."""
    try:
        video = video_to_tensor(
            clip["path"], start_sec, end_sec, num_frames=num_frames
        )
        return index, video, None
    except Exception as e:  # noqa: BLE001 - report per-clip failure
        return index, None, f"{type(e).__name__}: {e}"


def _pad_audio(audio_list):
    """Pad a list of (1, T) waveforms; mask True = padded. Matches training."""
    max_len = max(a.shape[1] for a in audio_list)
    audio = torch.zeros(len(audio_list), 1, max_len)
    mask = torch.ones(len(audio_list), max_len, dtype=torch.bool)
    for i, a in enumerate(audio_list):
        audio[i, :, : a.shape[1]] = a
        mask[i, : a.shape[1]] = False
    return audio, mask


def _iter_decoded_batches(clips, batch_size, workers, decode_fn):
    """Yield decoded batches in clip-list order.

    ``decode_fn`` is a picklable top-level callable ``(index, clip) ->
    (index, tensor_or_None, error_or_None)``. With ``workers <= 1`` decoding is
    in-process and sequential; otherwise a spawn-based ProcessPoolExecutor is
    used (spawn avoids the fork-after-CUDA/OpenCV deadlock the eval script
    documents).
    """
    starts = range(0, len(clips), batch_size)

    if workers <= 1:
        for start in starts:
            yield [
                decode_fn(i, clips[i])
                for i in range(start, min(start + batch_size, len(clips)))
            ]
        return

    mp_context = multiprocessing.get_context("spawn")
    window = max(workers, _DECODE_PREFETCH_BATCHES * batch_size)
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp_context
    ) as executor:
        pending = collections.deque()

        def _submit(i):
            return executor.submit(decode_fn, i, clips[i])

        next_index = 0
        while next_index < len(clips) and len(pending) < window:
            pending.append(_submit(next_index))
            next_index += 1

        for start in starts:
            batch = []
            for _ in range(start, min(start + batch_size, len(clips))):
                batch.append(pending.popleft().result())
                if next_index < len(clips):
                    pending.append(_submit(next_index))
                    next_index += 1
            yield batch


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

class AudioTowerExtractor:
    """Fine-tuned flik audio tower, pooled pre-projection."""

    tensor = (
        "FlikModel.dual_encoder.audio_encoder(waveform, padding_mask)['features'] "
        "pooled by DualEncoder._pool_audio_features (masked mean over time); "
        "768-d, before dual_encoder.audio_proj"
    )

    def __init__(self, model, device):
        self.encoder = model.dual_encoder.audio_encoder
        self.encoder.eval()
        self.device = device

    def extract(self, batch):
        audio = batch["audio"].to(self.device, non_blocking=True)
        mask = batch["audio_padding_mask"].to(self.device, non_blocking=True)
        out = self.encoder(audio, mask)
        return DualEncoder._pool_audio_features(
            out["features"], out.get("padding_mask")
        )


class VideoTowerExtractor:
    """Fine-tuned flik video tower, pre-projection."""

    def __init__(self, model, device, video_feature="cls"):
        self.encoder = model.dual_encoder.video_encoder
        self.encoder.eval()
        self.device = device
        self.video_feature = video_feature
        if video_feature == "cls":
            self.tensor = (
                "FlikModel.dual_encoder.video_encoder(video)['cls_embedding'] "
                "(post flik temporal head); 768-d, before dual_encoder.video_proj"
            )
        else:
            self.tensor = (
                "FlikModel.dual_encoder.video_encoder(video)['frame_embeddings'] "
                "mean over temporal tokens (raw VideoMAE tower path); 768-d, "
                "before dual_encoder.video_proj"
            )

    def extract(self, batch):
        video = batch["video"].to(self.device, non_blocking=True)
        out = self.encoder(video)
        if self.video_feature == "tower_mean":
            return out["frame_embeddings"].mean(dim=1)
        return out["cls_embedding"]


class PretrainedAudioExtractor:
    """Fresh HF wav2vec2-base. No flik checkpoint, no flik projection head."""

    tensor = (
        "AudioEncoder(facebook/wav2vec2-base, feature_layer=<config>).features "
        "pooled by DualEncoder._pool_audio_features (masked mean over time); "
        "768-d. Fresh HF weights loaded with from_pretrained; no flik weights."
    )

    def __init__(self, model_name, feature_layer, hidden_dim, device):
        self.encoder = (
            AudioEncoder(
                model_name=model_name,
                feature_layer=feature_layer,
                hidden_dim=hidden_dim,
                pretrained=True,
            )
            .to(device)
            .eval()
        )
        self.device = device

    def extract(self, batch):
        audio = batch["audio"].to(self.device, non_blocking=True)
        mask = batch["audio_padding_mask"].to(self.device, non_blocking=True)
        out = self.encoder(audio, mask)
        return DualEncoder._pool_audio_features(
            out["features"], out.get("padding_mask")
        )


class PretrainedVideoExtractor:
    """Fresh HF VideoMAE-base. No flik checkpoint, no flik temporal head."""

    tensor = (
        "VideoMAEModel(MCG-NJU/videomae-base).last_hidden_state mean over tokens "
        "(VideoMAE-base has no CLS token); 768-d. Fresh HF weights loaded with "
        "from_pretrained; no flik weights."
    )

    def __init__(self, model_name, device, video_feature="cls"):
        self.model = VideoMAEModel.from_pretrained(model_name).to(device).eval()
        self.device = device
        self.video_feature = video_feature

    def extract(self, batch):
        video = batch["video"].to(self.device, non_blocking=True)
        out = self.model(video)
        return out.last_hidden_state.mean(dim=1)


# ---------------------------------------------------------------------------
# Extraction loop
# ---------------------------------------------------------------------------

def extract_features(clips, extractors, modality, args, hidden_dim, logger):
    n = len(clips)
    feats = {name: np.zeros((n, hidden_dim), dtype=np.float32) for name in extractors}
    ok = np.zeros(n, dtype=bool)
    failures = []
    logged_shapes = set()

    if modality == "audio":
        decode_fn = functools.partial(
            _decode_audio_worker, sample_rate=args.sample_rate
        )
    else:
        decode_fn = functools.partial(
            _decode_video_worker,
            num_frames=args.num_frames,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
        )

    used = 0
    t0 = time.time()
    for decoded in _iter_decoded_batches(
        clips, args.batch_size, args.workers, decode_fn
    ):
        idxs, tensors = [], []
        for i, tensor, err in decoded:
            if err is not None:
                failures.append(
                    {"index": int(i), "path": clips[i]["path"], "error": err}
                )
                continue
            idxs.append(i)
            tensors.append(tensor)
        if not idxs:
            continue

        if modality == "audio":
            audio, mask = _pad_audio(tensors)
            batch = {"audio": audio, "audio_padding_mask": mask}
        else:
            batch = {"video": torch.stack(tensors, dim=0)}

        with torch.no_grad():
            for name, extractor in extractors.items():
                f = extractor.extract(batch).detach().float().cpu().numpy()
                feats[name][idxs] = f
                if name not in logged_shapes:
                    logger.info(
                        "  [%s] feature tensor: shape=%s dtype=%s",
                        name, tuple(f.shape), f.dtype,
                    )
                    logged_shapes.add(name)
        ok[idxs] = True

        used += len(idxs)
        if used >= n or used % max(1, args.batch_size * 10) < len(idxs):
            logger.info(
                "  extracted %d/%d usable clips (%d failures) [%.0fs]",
                used, n, len(failures), time.time() - t0,
            )

    return feats, ok, failures


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def probe_features(X, y, train_idx, test_idx, logger):
    scaler = StandardScaler().fit(X[train_idx])
    X_train = scaler.transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y[train_idx])
    pred = clf.predict(X_test)

    return {
        "top1_accuracy": float(accuracy_score(y[test_idx], pred)),
        "macro_f1": float(
            f1_score(y[test_idx], pred, average="macro", zero_division=0)
        ),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_classes_train": int(len(set(y[train_idx].tolist()))),
        "feature_dim": int(X.shape[1]),
        "feature_mean": float(X.mean()),
        "feature_std": float(X.std()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_backbone_spec(spec):
    """Return (name, path, is_pretrained). Handles 'name:path' and 'name'."""
    if ":" in spec:
        name, path = spec.split(":", 1)
    else:
        name, path = spec, "PRETRAINED"
    name = name.strip()
    path = path.strip()
    is_pretrained = name.lower() == "pretrained" or path.upper() == "PRETRAINED"
    return name, path, is_pretrained


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Frozen-feature linear probe for flik encoder towers."
    )
    p.add_argument("--modality", required=True, choices=["audio", "video"])
    p.add_argument("--dataset", required=True, choices=list(DEFAULT_MAX_CLIPS_PER_CLASS))
    p.add_argument(
        "--backbone",
        action="append",
        default=[],
        metavar="NAME:PATH_OR_PRETRAINED",
        help="Repeatable. Use pretrained:PRETRAINED for the HF baseline, "
        "otherwise NAME:/path/to/checkpoint.pth.",
    )
    p.add_argument("--dataset-root", default="data/probe")
    p.add_argument(
        "--vggsound-manifest",
        default="data/vggsound_full/clips_manifest.json",
        help="VGGSound clips manifest (video modality only).",
    )
    p.add_argument(
        "--max-clips-per-class",
        type=int,
        default=None,
        help="Defaults: esc50 40, speech_commands 100, vggsound 48.",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default=None, help="cuda|cpu (default: cuda if available).")
    p.add_argument("--out", default="probe_results.json")
    p.add_argument(
        "--video-feature",
        choices=["cls", "tower_mean"],
        default="cls",
        help="Video tensor to probe: 'cls' = video_encoder(...)['cls_embedding'] "
        "(spec default), 'tower_mean' = mean of frame_embeddings (apple-to-apple "
        "with the pretrained VideoMAE mean-over-tokens baseline).",
    )
    p.add_argument("--seed", type=int, default=SPLIT_SEED)
    p.add_argument("--test-fraction", type=float, default=TEST_FRACTION)
    p.add_argument("--num-frames", type=int, default=VIDEO_NUM_FRAMES)
    p.add_argument("--sample-rate", type=int, default=AUDIO_SAMPLE_RATE)
    p.add_argument("--start-sec", type=float, default=VIDEO_START_SEC)
    p.add_argument("--end-sec", type=float, default=VIDEO_END_SEC)
    p.add_argument("--config", default=None, help="Optional model YAML config path.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logger = setup_logging()

    if not args.backbone:
        logger.error("No --backbone given.")
        return 2

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.error("--device cuda requested but CUDA is not available.")
        return 2
    logger.info("Device: %s", device)

    if args.dataset in AUDIO_DATASETS and args.modality != "audio":
        logger.error("%s is an audio dataset; use --modality audio.", args.dataset)
        return 2
    if args.dataset in VIDEO_DATASETS and args.modality != "video":
        logger.error("%s is a video dataset; use --modality video.", args.dataset)
        return 2

    model_cfg, _ = load_model_config(args.config)
    hidden_dim = int(model_cfg.hidden_dim)
    max_per_class = (
        args.max_clips_per_class
        if args.max_clips_per_class is not None
        else DEFAULT_MAX_CLIPS_PER_CLASS[args.dataset]
    )

    clips = build_clip_list(
        args.dataset,
        args.dataset_root,
        max_per_class,
        args.vggsound_manifest,
        logger,
    )
    if not clips:
        logger.error("Clip list is empty.")
        return 1

    labels = [c["label"] for c in clips]
    y = np.array(labels)
    n_clips = len(clips)
    n_classes = len(set(labels))

    train_idx, test_idx = stratified_split_seeded(
        labels, args.test_fraction, args.seed
    )
    logger.info(
        "Split: %d train / %d test (test_fraction=%.2f, seed=%d)",
        len(train_idx), len(test_idx), args.test_fraction, args.seed,
    )

    # --- build extractors -------------------------------------------------
    extractors = {}
    backbone_meta = {}
    for spec in args.backbone:
        name, path, is_pretrained = parse_backbone_spec(spec)
        if name in extractors:
            logger.error("duplicate backbone name: %s", name)
            return 2
        t0 = time.time()
        if is_pretrained:
            logger.info("[%s] pretrained baseline (no flik checkpoint)", name)
            if args.modality == "audio":
                extractor = PretrainedAudioExtractor(
                    model_cfg.audio_model_name,
                    int(model_cfg.audio_feature_layer),
                    hidden_dim,
                    device,
                )
            else:
                extractor = PretrainedVideoExtractor(
                    model_cfg.video_model_name, device, args.video_feature
                )
            backbone_meta[name] = {"type": "pretrained", "path": None}
        else:
            if not os.path.exists(path):
                logger.error("[%s] checkpoint not found: %s", name, path)
                return 2
            logger.info("[%s] FlikModel + %s", name, path)
            model = build_model(model_cfg, device)
            load_checkpoint_into(model, path, logger)
            model.eval()
            if args.modality == "audio":
                extractor = AudioTowerExtractor(model, device)
            else:
                extractor = VideoTowerExtractor(model, device, args.video_feature)
            backbone_meta[name] = {
                "type": "flik_checkpoint",
                "path": os.path.abspath(path),
            }
        extractors[name] = extractor
        logger.info("[%s] ready in %.1fs", name, time.time() - t0)

    # --- extract ----------------------------------------------------------
    logger.info(
        "Extracting %d clips for %d backbone(s) (batch=%d, workers=%d)",
        n_clips, len(extractors), args.batch_size, args.workers,
    )
    feats, ok, failures = extract_features(
        clips, extractors, args.modality, args, hidden_dim, logger
    )
    if not ok.any():
        logger.error("No clips decoded; cannot probe.")
        return 1
    logger.info("Usable clips: %d/%d (%d failures)", int(ok.sum()), n_clips, len(failures))

    # The split is fixed on the full label list; restrict to clips that decoded
    # for every backbone so all backbones share exactly the same train/test rows.
    train_use = [i for i in train_idx if ok[i]]
    test_use = [i for i in test_idx if ok[i]]
    if not train_use or not test_use:
        logger.error("Decoded clips are insufficient for train/test.")
        return 1

    # Majority-class baseline from the (raw) train split, scored on the test split.
    majority_label = collections.Counter(y[train_idx]).most_common(1)[0][0]
    majority_acc = float((y[test_idx] == majority_label).mean())

    results = {}
    for name in extractors:
        X = feats[name]
        res = probe_features(X, y, train_use, test_use, logger)
        res["tensor"] = extractors[name].tensor
        results[name] = res
        logger.info(
            "[%s] top1=%.4f macro_f1=%.4f n_train=%d n_test=%d "
            "feat_dim=%d feat_mean=%.3f feat_std=%.3f",
            name, res["top1_accuracy"], res["macro_f1"], res["n_train"],
            res["n_test"], res["feature_dim"], res["feature_mean"],
            res["feature_std"],
        )

    output = {
        "modality": args.modality,
        "dataset": args.dataset,
        "dataset_root": os.path.abspath(args.dataset_root),
        "vggsound_manifest": (
            os.path.abspath(args.vggsound_manifest)
            if args.dataset in VIDEO_DATASETS
            else None
        ),
        "max_clips_per_class": max_per_class,
        "n_clips": n_clips,
        "n_classes": n_classes,
        "n_clips_used": int(ok.sum()),
        "video_feature": args.video_feature if args.modality == "video" else None,
        "probe_protocol": {
            "split": "stratified per-class 80/20, computed once from the label list",
            "split_seed": args.seed,
            "test_fraction": args.test_fraction,
            "scaler": "sklearn.preprocessing.StandardScaler (fit on train only)",
            "classifier": "sklearn.linear_model.LogisticRegression(max_iter=1000)",
            "audio_preprocessing": (
                "src.utils.audio.load_audio_from_wav (16 kHz mono, zero-mean/"
                "unit-variance normalization) then zero-pad to batch max"
            ),
            "video_preprocessing": (
                f"src.utils.video.video_to_tensor(path, {args.start_sec}, "
                f"{args.end_sec}, num_frames={args.num_frames})"
            ),
        },
        "split": {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_idx_used": train_use,
            "test_idx_used": test_use,
        },
        "majority_class": {
            "label": majority_label,
            "accuracy_on_test": majority_acc,
        },
        "clip_list": clips,
        "backbone_meta": backbone_meta,
        "backbones": results,
        "failures": failures,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("Wrote results to %s", args.out)

    print()
    print("=" * 80)
    print(f"{args.modality} / {args.dataset}: {n_classes} classes, {n_clips} clips")
    print(f"majority-class baseline acc: {majority_acc:.4f} ({majority_label!r})")
    print(f"{'backbone':<24} {'top1':>8} {'macro_f1':>10} {'n_tr':>6} {'n_te':>6}")
    for name, res in results.items():
        print(
            f"{name:<24} {res['top1_accuracy']:>8.4f} {res['macro_f1']:>10.4f} "
            f"{res['n_train']:>6d} {res['n_test']:>6d}"
        )
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
