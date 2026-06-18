#!/usr/bin/env python
"""
Diagnostic script — validates data loading and embedding diversity on cluster.
Commit and run with: uv run --extra cu128 python -m scripts.debug_data
"""
import sys, os

os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.path.dirname(__file__), "..", "cache")
os.environ["HF_HOME"] = os.path.join(os.path.dirname(__file__), "..", "cache")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import torch
from collections import Counter

from src.datasets.video_audio_dataset import VideoAudioDataset
from src.utils.video import video_to_tensor
from src.utils.audio import audio_to_tensor
from src.models.flik_model import FlikModel
from src.models.losses import ContrastiveLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MANIFEST_PATH = "data/filtered_manifest_segments.json"

def check_manifest_paths():
    """Verify every path in the manifest exists."""
    print("=" * 60)
    print("1. MANIFEST PATH CHECK")
    print("=" * 60)

    if not os.path.exists(MANIFEST_PATH):
        print(f"  FAIL: manifest not found at {MANIFEST_PATH}")
        print(f"  CWD: {os.getcwd()}")
        return False

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    missing_videos = []
    missing_jsons = []
    for seg in manifest:
        vp = seg["video_path"]
        jp = seg.get("json_path", "")
        if not os.path.exists(vp):
            missing_videos.append(vp)
        if jp and not os.path.exists(jp):
            missing_jsons.append(jp)

    print(f"  Total segments: {len(manifest)}")
    print(f"  Missing video paths: {len(missing_videos)}")
    if missing_videos:
        print(f"    Example: {missing_videos[0]}")
        print(f"    Sample paths in manifest:")
        for seg in manifest[:2]:
            print(f"      video: {seg['video_path']}")
            print(f"      json:  {seg.get('json_path', 'N/A')}")
    print(f"  Missing json paths: {len(missing_jsons)}")
    return len(missing_videos) == 0


def check_single_load():
    """Try loading one segment's video and audio to verify I/O works."""
    print("\n" + "=" * 60)
    print("2. SINGLE SEGMENT LOAD TEST")
    print("=" * 60)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    seg = None
    for s in manifest:
        if os.path.exists(s["video_path"]):
            seg = s
            break

    if seg is None:
        print("  FAIL: no segment with a valid video path found")
        return False

    print(f"  Testing: {seg['id']}")
    print(f"  Video:  {seg['video_path']}")
    print(f"  Start:  {seg['start_sec']:.1f}s  End: {seg['end_sec']:.1f}s  Duration: {seg['end_sec'] - seg['start_sec']:.1f}s")
    print(f"  Text:   \"{seg.get('text', '')[:80]}\"")

    errors = []

    # Test video loading
    try:
        video = video_to_tensor(seg["video_path"], seg["start_sec"], seg["end_sec"], num_frames=16)
        print(f"  Video tensor: {video.shape}, dtype={video.dtype}, "
              f"min={video.min():.4f}, max={video.max():.4f}, "
              f"mean={video.mean():.4f}, nonzero_frac={video.sum(dim=(1,2,3)).abs().bool().float().mean().item():.3f}")
    except Exception as e:
        print(f"  Video FAIL: {e}")
        errors.append(f"video: {e}")

    # Test audio loading
    try:
        audio = audio_to_tensor(seg["video_path"], seg["start_sec"], seg["end_sec"], sample_rate=16000)
        energy = (audio ** 2).mean().item()
        print(f"  Audio tensor: {audio.shape}, dtype={audio.dtype}, "
              f"min={audio.min():.4f}, max={audio.max():.4f}, "
              f"energy={energy:.6f}")
        if energy < 1e-6:
            print("  WARNING: audio energy near zero (silence?)")
    except Exception as e:
        print(f"  Audio FAIL: {e}")
        errors.append(f"audio: {e}")

    return len(errors) == 0


def check_batch_loading():
    """Load a full batch through the dataset and count fallbacks."""
    print("\n" + "=" * 60)
    print("3. DATASET BATCH LOADING")
    print("=" * 60)

    dataset = VideoAudioDataset(
        manifest_path=MANIFEST_PATH,
        sample_rate=16000,
        num_frames=16,
        min_duration=3.0,
        max_duration=10.0,
        dummy=False,
    )
    print(f"  Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        print("  FAIL: dataset is empty (no segments within duration range)")
        return False

    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset, batch_size=min(8, len(dataset)),
        shuffle=True,
        collate_fn=VideoAudioDataset.collate_fn,
        num_workers=0,
    )
    batch = next(iter(loader))
    audio = batch["audio"]
    video = batch["video"]

    eps = 1e-6
    audio_flat = audio.any(dim=1).float()  # (B, T_max) — 1 where nonzero
    video_flat = (video.abs().sum(dim=(2,3,4)) > eps).float()  # (B, F) — 1 where frame has content

    print(f"  Batch size: {len(batch['segment_ids'])}")
    print(f"  Audio shape: {audio.shape}  (B=1, C=1, T_max)")
    for i in range(audio.shape[0]):
        nonzero = audio_flat[i].mean().item()
        print(f"    sample {i}: nonzero_frac={nonzero:.3f}  id={batch['segment_ids'][i][:60]}")

    print(f"  Video shape: {video.shape}  (B, F, C, H, W)")
    for i in range(video.shape[0]):
        nonzero = video_flat[i].mean().item()
        print(f"    sample {i}: nonzero_frames={nonzero:.3f} (expect ~1.0 for real video)")

    return True


def check_embedding_diversity():
    """Run a forward pass and check if embeddings are diverse."""
    print("\n" + "=" * 60)
    print("4. EMBEDDING DIVERSITY")
    print("=" * 60)

    model = FlikModel(use_grounded_masked_prediction=False).to(DEVICE)
    model.eval()

    dataset = VideoAudioDataset(
        manifest_path=MANIFEST_PATH,
        sample_rate=16000, num_frames=16,
        min_duration=3.0, max_duration=10.0,
        dummy=False,
    )

    if len(dataset) == 0:
        print("  SKIP: empty dataset")
        return

    from torch.utils.data import DataLoader
    loader = DataLoader(
        dataset, batch_size=min(8, len(dataset)),
        shuffle=True, collate_fn=VideoAudioDataset.collate_fn, num_workers=0,
    )
    batch = next(iter(loader))

    audio = batch["audio"].to(DEVICE)
    video = batch["video"].to(DEVICE)
    audio_padding_mask = batch["audio_padding_mask"].to(DEVICE)
    B = audio.shape[0]

    out = model(audio, video, audio_padding_mask)
    audio_emb = out["audio_embedding"]  # (B, D)
    video_emb = out["video_embedding"]  # (B, D)

    a_sim = torch.mm(audio_emb, audio_emb.t())
    v_sim = torch.mm(video_emb, video_emb.t())
    cross_sim = torch.mm(audio_emb, video_emb.t())

    eye = torch.eye(B, dtype=torch.bool, device=DEVICE)

    print(f"  Audio self-sim (off-diag): {a_sim[~eye].mean():.4f} ± {a_sim[~eye].std():.4f}")
    print(f"  Video self-sim (off-diag): {v_sim[~eye].mean():.4f} ± {v_sim[~eye].std():.4f}")
    print(f"  Cross-sim diag:    {cross_sim.diag().mean():.4f} ± {cross_sim.diag().std():.4f}")
    print(f"  Cross-sim off-diag: {cross_sim[~eye].mean():.4f} ± {cross_sim[~eye].std():.4f}")

    if a_sim[~eye].mean() > 0.9:
        print("  FAIL: audio embeddings nearly identical (collapsed)")
    elif a_sim[~eye].std() > 0.01:
        print("  OK: audio embeddings show diversity")
    else:
        print("  WARNING: audio diversity is marginal")

    if v_sim[~eye].mean() > 0.9:
        print("  FAIL: video embeddings nearly identical (collapsed)")
    elif v_sim[~eye].std() > 0.01:
        print("  OK: video embeddings show diversity")
    else:
        print("  WARNING: video diversity is marginal")


def check_gradient_flow():
    """Run one training step and check gradient statistics."""
    print("\n" + "=" * 60)
    print("5. GRADIENT FLOW")
    print("=" * 60)

    model = FlikModel(use_grounded_masked_prediction=False).to(DEVICE)
    model.train()
    loss_fn = ContrastiveLoss(temperature=0.07).to(DEVICE)

    # Use real data (single item, batched to 1)
    dataset = VideoAudioDataset(
        manifest_path=MANIFEST_PATH,
        sample_rate=16000, num_frames=16,
        min_duration=3.0, max_duration=10.0,
        dummy=False,
    )

    if len(dataset) < 2:
        print("  SKIP: need at least 2 segments for contrastive loss")
        return

    indices = torch.randperm(len(dataset))[:8].tolist()
    samples = [dataset[i] for i in indices]
    batch = VideoAudioDataset.collate_fn(samples)

    audio = batch["audio"].to(DEVICE)
    video = batch["video"].to(DEVICE)
    audio_padding_mask = batch["audio_padding_mask"].to(DEVICE)

    out = model(audio, video, audio_padding_mask)
    loss, metrics = loss_fn(out["audio_embedding"], out["video_embedding"])
    loss.backward()

    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            gnorm = p.grad.norm().item()
            if gnorm > 0:
                # Collapse to top-level module
                module = name.split(".")[0]
                grad_norms[module] = grad_norms.get(module, 0) + gnorm ** 2

    print(f"  Loss: {loss.item():.4f}  Acc: {metrics.get('contrastive_acc', 0):.3f}")
    print(f"  Gradient norms (sqrt sum of squared gradients per module):")
    for k, v in sorted(grad_norms.items(), key=lambda x: -x[1]):
        print(f"    {k}: {v**0.5:.4f}")

    if not grad_norms:
        print("  FAIL: all gradients are zero!")
    else:
        print("  OK: gradients are flowing")


def check_param_update():
    """Verify parameters actually change between optimizer steps."""
    print("\n" + "=" * 60)
    print("6. PARAMETER UPDATE TEST")
    print("=" * 60)

    from torch.optim import AdamW

    model = FlikModel(use_grounded_masked_prediction=False).to(DEVICE)
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    loss_fn = ContrastiveLoss(temperature=0.5).to(DEVICE)

    dataset = VideoAudioDataset(
        manifest_path=MANIFEST_PATH, sample_rate=16000, num_frames=16,
        min_duration=3.0, max_duration=10.0, dummy=False,
    )

    if len(dataset) < 2:
        print("  SKIP: need at least 2 segments")
        return

    # Snap pre-step params
    pre_params = {}
    for name, p in model.named_parameters():
        pre_params[name] = p.data.clone()

    # One step
    indices = torch.randperm(len(dataset))[:4].tolist()
    samples = [dataset[i] for i in indices]
    batch = VideoAudioDataset.collate_fn(samples)

    audio = batch["audio"].to(DEVICE)
    video = batch["video"].to(DEVICE)
    audio_padding_mask = batch["audio_padding_mask"].to(DEVICE)

    out = model(audio, video, audio_padding_mask)
    loss, metrics = loss_fn(out["audio_embedding"], out["video_embedding"])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Check which params changed
    changed = 0
    unchanged = 0
    for name, p in model.named_parameters():
        diff = (p.data - pre_params[name]).abs().max().item()
        if diff > 1e-9:
            changed += 1
        else:
            unchanged += 1

    print(f"  Parameters changed: {changed}")
    print(f"  Parameters unchanged: {unchanged}")
    if unchanged > 0:
        print(f"  WARNING: {unchanged} parameters did not update (frozen?)")


if __name__ == "__main__":
    print(f"Running on: {DEVICE}")
    print(f"CWD: {os.getcwd()}")
    print()

    ok = True
    ok &= check_manifest_paths()
    ok &= check_single_load()
    ok &= check_batch_loading()
    check_embedding_diversity()
    check_gradient_flow()
    check_param_update()

    print("\n" + "=" * 60)
    if ok:
        print("SUMMARY: All basic checks passed. Data loading is healthy.")
    else:
        print("SUMMARY: ISSUES DETECTED — check the FAIL lines above.")
    print("=" * 60)
