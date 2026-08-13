"""Segment validation helpers.

Canonical home for validate_segment(), previously duplicated between
scripts/validate_manifest.py and scripts/expand_and_validate_manifest.py
(the latter kept a verbatim fallback copy). Importing here pulls in the
video/audio tensor loaders, which require torch + torchvision.
"""

from src.utils.video import video_to_tensor
from src.utils.audio import audio_to_tensor


def validate_segment(seg: dict, num_frames: int = 16, sample_rate: int = 16000) -> tuple:
    """
    Try to load a single segment's video and audio.
    Returns (segment_id, success, error_message).
    """
    seg_id = seg.get("id", f"{seg.get('video_path', '?')}_{seg['start_sec']:.1f}")
    try:
        video = video_to_tensor(
            seg["video_path"],
            seg["start_sec"],
            seg["end_sec"],
            num_frames=num_frames,
        )
        audio = audio_to_tensor(
            seg["video_path"],
            seg["start_sec"],
            seg["end_sec"],
            sample_rate=sample_rate,
        )
        # Basic sanity checks
        if video.shape[0] != num_frames:
            return (seg_id, False, f"Wrong frame count: {video.shape[0]}")
        if audio.shape[1] < sample_rate:  # less than 1 second
            return (seg_id, False, f"Audio too short: {audio.shape[1]} samples")
        return (seg_id, True, None)
    except Exception as e:
        return (seg_id, False, str(e))
