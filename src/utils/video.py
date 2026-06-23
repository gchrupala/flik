import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple

# VideoMAE preprocessing: resize 224x224, ImageNet normalization
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

VIDEOMAE_MEAN = [0.485, 0.456, 0.406]
VIDEOMAE_STD = [0.229, 0.224, 0.225]

# Precompute normalization as tensors for fast batch normalization
# These are in [0, 1] range (designed for tensors already divided by 255)
_MEAN_TENSOR = torch.tensor(VIDEOMAE_MEAN).view(1, 1, 3)
_STD_TENSOR = torch.tensor(VIDEOMAE_STD).view(1, 1, 3)

video_preprocess = Compose(
    [
        Resize(224),  # Resize shorter side to 224
        CenterCrop(224),  # Crop to 224x224
        ToTensor(),
        Normalize(mean=VIDEOMAE_MEAN, std=VIDEOMAE_STD),
    ]
)


def _resize_crop_normalize_np(
    frames: np.ndarray, target_size: int = 224
) -> torch.Tensor:
    """
    Resize, center-crop, and normalize a batch of frames using numpy/cv2.
    Much faster than PIL + torchvision transforms for individual frames.

    Args:
        frames: (N, H, W, 3) uint8 RGB array
        target_size: output spatial size

    Returns:
        (N, 3, target_size, target_size) float32 tensor, ImageNet-normalized
    """
    n, h, w, _ = frames.shape

    # Resize shorter side to target_size using cv2 (fast)
    if h < w:
        new_h, new_w = target_size, int(w * target_size / h)
    else:
        new_h, new_w = int(h * target_size / w), target_size

    resized = np.empty((n, new_h, new_w, 3), dtype=np.uint8)
    for i in range(n):
        resized[i] = cv2.resize(frames[i], (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center crop
    top = (new_h - target_size) // 2
    left = (new_w - target_size) // 2
    cropped = resized[:, top : top + target_size, left : left + target_size, :]

    # Convert to float tensor and normalize (vectorized)
    tensor = torch.from_numpy(cropped).float() / 255.0
    tensor = (tensor - _MEAN_TENSOR) / _STD_TENSOR  # (N, H, W, 3)

    # CHW format
    tensor = tensor.permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W)
    return tensor


def load_video_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    target_fps: int = 3,
) -> List[Image.Image]:
    """
    Load frames from video segment using OpenCV.
    Returns list of PIL Images.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0  # fallback

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)
    stride = max(1, int(round(fps / target_fps)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_pos = start_frame

    while current_pos < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (current_pos - start_frame) % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

        current_pos += 1

    cap.release()
    return frames


def sample_frames_uniform(
    video_path: str,
    start_sec: float,
    end_sec: float,
    num_frames: int = 16,
    preprocess: bool = True,
) -> torch.Tensor:
    """
    Sample exactly `num_frames` frames uniformly from the segment.
    If the segment contains fewer frames than requested, repeat the last frame.
    Returns tensor of shape (num_frames, 3, H, W) (normalized if preprocess=True).

    Optimized: reads frames sequentially in a single pass (no per-frame seeking),
    then picks the uniformly sampled indices. This is 5-10x faster than seeking
    to each frame individually because OpenCV seeking decodes from the nearest
    keyframe, making per-frame seeks extremely expensive.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 24.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_sec * fps)
    end_frame = min(int(end_sec * fps), total_frames - 1)

    if start_frame >= end_frame:
        cap.release()
        raise ValueError(
            f"Invalid segment: start_frame={start_frame}, end_frame={end_frame}"
        )

    # Compute which frame indices we want (relative to start_frame)
    available_frames = end_frame - start_frame
    if available_frames >= num_frames:
        indices = np.linspace(0, available_frames - 1, num_frames, dtype=int)
    else:
        indices = np.array(list(range(available_frames)) + [available_frames - 1] * (num_frames - available_frames))

    indices_set = set(int(i) for i in indices)
    max_idx = int(indices[-1])

    # Single sequential read: seek to start, then read forward
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    raw_frames = []
    frame_idx = 0
    while frame_idx <= max_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices_set:
            # Convert BGR → RGB
            raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1

    cap.release()

    # Handle case where video ended before we got all frames
    while len(raw_frames) < num_frames:
        if raw_frames:
            raw_frames.append(raw_frames[-1].copy())
        else:
            raw_frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

    raw_frames = raw_frames[:num_frames]

    if preprocess:
        # Batch resize/crop/normalize (much faster than per-frame PIL transforms)
        frames_np = np.stack(raw_frames, axis=0)  # (N, H, W, 3)
        return _resize_crop_normalize_np(frames_np, target_size=224)
    else:
        return torch.stack([torch.from_numpy(f).permute(2, 0, 1) for f in raw_frames])


def video_to_tensor(
    video_path: str,
    start_sec: float,
    end_sec: float,
    num_frames: int = 16,
) -> torch.Tensor:
    """Convenience wrapper: sample frames and preprocess."""
    return sample_frames_uniform(
        video_path, start_sec, end_sec, num_frames, preprocess=True
    )


if __name__ == "__main__":
    # Quick test with a dummy path
    try:
        tensor = video_to_tensor("/tmp/test.mp4", 0.0, 5.0, num_frames=16)
        print(f"Video tensor shape: {tensor.shape}")
        print(f"Range: {tensor.min():.3f} ~ {tensor.max():.3f}")
    except Exception as e:
        print(f"Test failed (expected): {e}")
