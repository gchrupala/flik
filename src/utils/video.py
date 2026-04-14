import cv2
import torch
import numpy as np
from PIL import Image
from typing import List, Optional, Tuple

# VideoMAE preprocessing: resize 224x224, ImageNet normalization
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

VIDEOMAE_MEAN = [0.485, 0.456, 0.406]
VIDEOMAE_STD = [0.229, 0.224, 0.225]

video_preprocess = Compose(
    [
        Resize(224),  # Resize shorter side to 224
        CenterCrop(224),  # Crop to 224x224
        ToTensor(),
        Normalize(mean=VIDEOMAE_MEAN, std=VIDEOMAE_STD),
    ]
)


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
    """
    # Get all frames (using original load_video_frames with high fps)
    # For efficiency, we could directly sample using frame indices.
    # Simpler: compute frame indices we want and seek to each.
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
        # Segment too short, pad with zeros later
        cap.release()
        raise ValueError(
            f"Invalid segment: start_frame={start_frame}, end_frame={end_frame}"
        )

    # Uniformly sample frame indices
    available_frames = end_frame - start_frame
    if available_frames >= num_frames:
        indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int)
    else:
        # Not enough frames: sample all available and repeat last
        indices = list(range(start_frame, end_frame))
        while len(indices) < num_frames:
            indices.append(end_frame - 1)
        indices = np.array(indices[:num_frames])

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            # If seek fails, use last successful frame or zero
            if frames:
                frames.append(frames[-1].copy())
            else:
                # Create black frame
                frame = np.zeros(
                    (
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        3,
                    ),
                    dtype=np.uint8,
                )
                frames.append(Image.fromarray(frame))
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()

    # Preprocess each frame
    if preprocess:
        frames = [video_preprocess(frame) for frame in frames]

    # Stack frames: (num_frames, C, H, W)
    return torch.stack(frames, dim=0)


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
