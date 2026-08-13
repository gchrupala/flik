import json
import random
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any
import os

# Local imports
from ..utils.video import video_to_tensor
from ..utils.audio import audio_to_tensor
from ..utils.segments import in_duration_range
from ..utils.paths import resolve_path
from ..utils.segments import merge_segments


class VideoAudioDataset(Dataset):
    """
    Dataset that loads video‑audio pairs from filtered manifest.
    Each item corresponds to a segment (start_sec, end_sec) sampled from a transcript.
    """

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 16000,
        num_frames: int = 16,
        min_duration: float = 3.0,
        max_duration: float = 10.0,
        max_gap: float = 1.0,
        dummy: bool = False,
        manifest_items: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Args:
            manifest_path: Path to filtered manifest JSON.
            sample_rate: Audio sample rate (Hz).
            num_frames: Number of video frames per segment.
            min_duration: Minimum segment duration in seconds.
            max_duration: Maximum segment duration in seconds.
            dummy: If True, generate random tensors (for testing).
        """
        self.sample_rate = sample_rate
        self.num_frames = num_frames
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.max_gap = max_gap
        self.dummy = dummy

        if manifest_items is not None:
            self.manifest = manifest_items
            self.segments = self._build_segment_list()
        elif not dummy:
            with open(manifest_path, "r", encoding="utf-8") as f:
                self.manifest = json.load(f)
            # Precompute segment candidates
            self.segments = self._build_segment_list()
        else:
            # Dummy data: generate random entries
            self.manifest = [
                {
                    "id": f"dummy_{i}",
                    "video_path": f"/tmp/dummy_{i}.mp4",
                    "json_path": f"/tmp/dummy_{i}.json",
                }
                for i in range(100)
            ]
            self.segments = self._build_segment_list()

    def _build_segment_list(self) -> List[Dict[str, Any]]:
        """Flatten manifest into list of segment candidates."""
        segments = []
        for item in self.manifest:
            if not self.dummy and "start_sec" in item and "end_sec" in item:
                # Segment-level manifest item
                dur = item["end_sec"] - item["start_sec"]
                if not in_duration_range(dur, self.min_duration, self.max_duration):
                    continue
                segments.append(
                    {
                        "id": item.get(
                            "id",
                            f"{item.get('video_id', 'segment')}_{item['start_sec']:.2f}_{item['end_sec']:.2f}",
                        ),
                        "video_path": resolve_path(item["video_path"]),
                        "json_path": resolve_path(item.get("json_path", "")),
                        "start_sec": item["start_sec"],
                        "end_sec": item["end_sec"],
                        "text": item.get("text", ""),
                    }
                )
                continue

            if self.dummy:
                # For dummy data, create random segments
                for seg_idx in range(5):  # 5 segments per video
                    start = random.uniform(0, 30)
                    end = start + random.uniform(self.min_duration, self.max_duration)
                    segments.append(
                        {
                            "id": f"{item['id']}_seg{seg_idx}",
                            "video_path": item["video_path"],
                            "json_path": item["json_path"],
                            "start_sec": start,
                            "end_sec": end,
                            "text": "dummy transcript",
                        }
                    )
            else:
                # Load actual segments from WhisperX JSON
                try:
                    with open(resolve_path(item["json_path"]), "r", encoding="utf-8") as f:
                        transcript = json.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load {item['json_path']}: {e}")
                    continue

                # Merge consecutive segments into [min,max] windows (rescues the
                # sub-minimum utterance fragments Whisper produces in dialogue).
                # Same logic as scripts/expand_and_validate_manifest.py.
                for w in merge_segments(
                    transcript,
                    min_duration=self.min_duration,
                    max_duration=self.max_duration,
                    max_gap=self.max_gap,
                ):
                    segments.append(
                        {
                            "id": f"{item['id']}_{w['start_sec']:.2f}_{w['end_sec']:.2f}",
                            "video_path": resolve_path(item["video_path"]),
                            "json_path": resolve_path(item["json_path"]),
                            "start_sec": w["start_sec"],
                            "end_sec": w["end_sec"],
                            "text": w["text"],
                            "n_segments": w["n_segments"],
                        }
                    )
        return segments

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seg = self.segments[idx]

        if self.dummy:
            # Generate random tensors (no I/O) — dummy mode is fine
            duration = seg["end_sec"] - seg["start_sec"]
            audio_len = int(duration * self.sample_rate)
            audio = torch.randn(1, audio_len)
            video = torch.randn(self.num_frames, 3, 224, 224)
            return {
                "audio": audio,
                "video": video,
                "text": seg["text"],
                "segment_id": seg["id"],
                "audio_len": audio_len,
            }

        # Real loading with retry on failure
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try_idx = idx if attempt == 0 else random.randint(0, len(self.segments) - 1)
            seg = self.segments[try_idx]
            try:
                video = video_to_tensor(
                    seg["video_path"],
                    seg["start_sec"],
                    seg["end_sec"],
                    num_frames=self.num_frames,
                )
                audio = audio_to_tensor(
                    seg["video_path"],
                    seg["start_sec"],
                    seg["end_sec"],
                    sample_rate=self.sample_rate,
                )
                return {
                    "audio": audio,
                    "video": video,
                    "text": seg["text"],
                    "segment_id": seg["id"],
                    "audio_len": audio.shape[1],
                }
            except Exception as e:
                last_error = e
                print(f"Warning: Failed to load segment {seg['id']} (attempt {attempt + 1}/{max_retries}): {e}")
                continue

        # All retries failed — raise instead of returning noise
        raise RuntimeError(
            f"Failed to load any segment after {max_retries} attempts. Last error: {last_error}"
        )

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Basic collate: pad audio sequences to max length in batch.
        Returns dict with keys: audio, video, audio_padding_mask, audio_lengths.
        """
        audio_list = [item["audio"] for item in batch]  # each (1, T)
        video_list = [item["video"] for item in batch]  # each (F, C, H, W)
        audio_lengths = torch.tensor([a.shape[1] for a in audio_list])

        # Pad audio
        max_audio_len = max(a.shape[1] for a in audio_list)
        audio_padded = torch.zeros(len(audio_list), 1, max_audio_len)
        # True means padded position.
        audio_padding_mask = torch.ones(
            len(audio_list), max_audio_len, dtype=torch.bool
        )
        for i, a in enumerate(audio_list):
            audio_padded[i, :, : a.shape[1]] = a
            audio_padding_mask[i, : a.shape[1]] = False

        # Stack video (already same shape)
        video_stacked = torch.stack(video_list, dim=0)

        return {
            "audio": audio_padded,  # (B, 1, T_max)
            "video": video_stacked,  # (B, F, C, H, W)
            "audio_padding_mask": audio_padding_mask,  # (B, T_max), True = padded
            "audio_lengths": audio_lengths,  # (B,)
            "segment_ids": [item["segment_id"] for item in batch],
            "texts": [item["text"] for item in batch],
        }


if __name__ == "__main__":
    # Test with dummy data
    dataset = VideoAudioDataset("", dummy=True)
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Audio shape: {sample['audio'].shape}")
    print(f"Video shape: {sample['video'].shape}")

    # Test collate
    batch = [dataset[i] for i in range(4)]
    collated = VideoAudioDataset.collate_fn(batch)
    print(f"Collated audio shape: {collated['audio'].shape}")
    print(f"Collated video shape: {collated['video'].shape}")
    print(f"Audio padding mask shape: {collated['audio_padding_mask'].shape}")
