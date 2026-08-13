"""Segment merging utilities for manifest building.

Whisper (via WhisperX) emits utterance-level segments that are frequently
shorter than the training minimum duration (3s) — in dialogue-heavy films
well over half of the segments can be sub-3s. Merging consecutive segments
into longer windows rescues that content (measured +40-46% usable duration on
two sample films; up to ~2x on dialogue-heavy titles) and follows standard
practice in video-language/AV pretraining
(e.g. VideoCLIP, FrozenBiLM, HowTo100M-style pipelines), where a window may
span multiple ASR segments / speaker turns. Audio and video in a merged
window come from the same file at the same timestamps, so the AV
correspondence signal is preserved; merging across speaker turns is intended.
"""

from typing import Dict, List


def in_duration_range(duration: float, min_duration: float, max_duration: float) -> bool:
    """Return True if ``duration`` falls within [min_duration, max_duration].

    Single source of truth for the duration filter used across manifest
    building (merge_segments) and the training dataset. Keeps the filter
    consistent everywhere instead of re-implementing ``a <= d <= b`` inline.
    """
    return min_duration <= duration <= max_duration


def merge_segments(
    segments: List[Dict],
    min_duration: float = 3.0,
    max_duration: float = 10.0,
    max_gap: float = 1.0,
    concat_sep: str = " ",
) -> List[Dict]:
    """Pack consecutive transcript segments into training windows.

    Greedily accumulates segments in timestamp order into a window, stopping
    (and starting a new window) when:
      - the gap between consecutive segments exceeds ``max_gap`` (scene
        change / music / long silence), or
      - adding the next segment would make the window exceed ``max_duration``.
    A window is emitted only if it reaches ``min_duration``. Single segments
    already in [min_duration, max_duration] pass through as 1-segment windows.
    Segments longer than ``max_duration`` alone are dropped.

    Args:
        segments: transcript segments with start/end/text (list, or a dict
            with a "segments" key, matching WhisperX output).
        min_duration: minimum window duration in seconds.
        max_duration: maximum window duration in seconds.
        max_gap: maximum allowed gap (seconds) between merged segments.
        concat_sep: separator used to join segment texts.

    Returns:
        List of window dicts: {start_sec, end_sec, text, n_segments}.
    """
    if isinstance(segments, dict):
        segments = segments.get("segments", [])
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: s.get("start", 0.0))
    windows: List[List[Dict]] = []
    window: List[Dict] = []

    for seg in ordered:
        if not window:
            window = [seg]
            continue
        last = window[-1]
        gap = seg.get("start", 0.0) - last.get("end", 0.0)
        # Anchor on the merged window's true end. Overlapping segments
        # (seg.start < last.end) leave the window end at last.end; the merged
        # end is max(seg.end, last.end). Using seg.end alone would overstate
        # duration on overlap and split the window early, dropping content.
        merged_end = max(seg.get("end", 0.0), last.get("end", 0.0))
        proposed_dur = merged_end - window[0].get("start", 0.0)
        if gap > max_gap or proposed_dur > max_duration:
            windows.append(window)
            window = [seg]
        else:
            window.append(seg)
    if window:
        windows.append(window)

    out = []
    for w in windows:
        start = w[0].get("start", 0.0)
        # True window end is the max end across segments: with start-sorted
        # overlapping segments the latest end is not always w[-1] (e.g.
        # [0-8],[2-4] -> true end 8, not 4).
        end = max(s.get("end", 0.0) for s in w)
        dur = end - start
        if not in_duration_range(dur, min_duration, max_duration):
            continue  # still too short after merging, or a single oversized seg
        text = concat_sep.join(
            str(s.get("text") or "").strip() for s in w if s.get("text")
        ).strip()
        out.append(
            {
                "start_sec": start,
                "end_sec": end,
                "text": text,
                "n_segments": len(w),
            }
        )
    return out
