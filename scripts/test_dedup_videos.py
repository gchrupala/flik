#!/usr/bin/env python
"""Unit tests for scripts/dedup_videos.py.

Run with either:
    uv run --extra cpu python -m scripts.test_dedup_videos
    uv run --extra cpu python -m pytest scripts/test_dedup_videos.py -q

The content (pHash/audio) test needs ffmpeg and the two local films and is
gated behind RUN_SLOW_DEDUP_TESTS=1 because it shells out to ffmpeg.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from scripts.dedup_videos import (
    Index,
    audio_signature,
    audio_similarity,
    build_metadata_groups,
    classify_part,
    frame_hashes,
    infer_quality,
    normalize_stem,
    part_family,
    phash_similarity,
)

INDEX_PATH = os.path.join(_root, "data/in/cinedantan_movies.json")
MANIFEST_PATH = os.path.join(_root, "data/filtered_manifest.json")


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------

def test_normalize_stem():
    cases = {
        "/Constantine_and_the_Cross_256kb.mp4": "constantine_and_the_cross",
        "First_Spaceship_on_Venus_ipod_512kb": "first_spaceship_on_venus",
        "Secret_Agent_512kb": "secret_agent",
        "X_64kb.MP4": "x",
        "Man_Who_Cheated_Himself": "man_who_cheated_himself",
        "M4V10005": "m4v10005",
        "FLIP_FROG-FIDDLESTICKS_64k_512kb": "flip_frog-fiddlesticks",
        "http://x/y/The_Film_512Kb.mp4": "the_film",
    }
    for raw, expected in cases.items():
        assert normalize_stem(raw) == expected, (raw, normalize_stem(raw), expected)


def test_classify_part():
    assert classify_part("VTS_01_1") == "dvd_vts"
    assert classify_part("VTS_01_2_512kb") == "dvd_vts"
    assert classify_part("M4V10005") == "dvd_m4v"
    assert classify_part("NOIR_DISC_11") == "compilation_disc"
    assert classify_part("thefastandthefurious-1of4") == "split_part", classify_part("thefastandthefurious-1of4")
    assert classify_part("phantom_thunderbolt") is None, classify_part("phantom_thunderbolt")


def test_part_family():
    assert part_family("VTS_01_1") == part_family("VTS_01_4") == "vts_01"
    assert part_family("VTS_02_1") == "vts_02"
    assert part_family("plain_film") == "plain_film"


def test_quality_order():
    ranks = [
        infer_quality("x", "h.264")[0],
        infer_quality("x", "HiRes MPEG4")[0],
        infer_quality("x", "MPEG4")[0],
        infer_quality("x", "512Kb MPEG4")[0],
        infer_quality("x", "256Kb MPEG4")[0],
        infer_quality("x", "64Kb MPEG4")[0],
    ]
    assert ranks == sorted(ranks)
    assert infer_quality("Film_512kb")[0] == infer_quality("x", "512Kb MPEG4")[0]


# ---------------------------------------------------------------------------
# Data-based grouping tests
# ---------------------------------------------------------------------------

def _load_groups():
    if not (os.path.exists(INDEX_PATH) and os.path.exists(MANIFEST_PATH)):
        return None, None
    index = Index.load(INDEX_PATH)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)
    used = [e["id"] for e in entries]
    groups, _mapping, _ambiguities = build_metadata_groups(used, index)
    return used, groups


def test_martha_ivers_collapses():
    _used, groups = _load_groups()
    if groups is None:
        return  # data not available
    assert "Martha_Ivers" in groups
    g = groups["Martha_Ivers"]
    assert set(g["members"]) == {
        "Martha_Ivers", "VTS_01_1", "VTS_01_2", "VTS_01_3", "VTS_01_4"
    }
    assert g["kept_id"] == "Martha_Ivers"


def test_grouping_covers_all_ids_once():
    used, groups = _load_groups()
    if groups is None:
        return
    covered = [m for g in groups.values() for m in g["members"]]
    assert sorted(covered) == sorted(used)
    assert len(groups) == len({g["kept_id"] for g in groups.values()})


def test_movie_level_counts():
    _used, groups = _load_groups()
    if groups is None:
        return
    assert len(groups) == 1336, len(groups)
    family_drop = sum(len(g["dropped_family"]) for g in groups.values())
    genuine_drop = sum(len(g["dropped_genuine"]) for g in groups.values())
    assert family_drop == 92, family_drop
    assert genuine_drop == 22, genuine_drop


def test_genuine_keeps_serial_and_anthology_pieces():
    """Genuine drops only redundant copies: serial/anthology pieces stay."""
    _used, groups = _load_groups()
    if groups is None:
        return
    musk = groups["TheThreeMuskateers"]
    assert musk["n_members"] == 12
    assert musk["dropped_genuine"] == [], musk["dropped_genuine"]

    toon = groups["Cartoontheater1930sAnd1950s1"]
    assert toon["n_members"] == 5
    assert toon["dropped_genuine"] == [], toon["dropped_genuine"]

    # A film split into *of4 parts has no main file; all parts are content.
    tff = groups["The_Fast_and_the_Furious"]
    assert tff["dropped_genuine"] == [], tff["dropped_genuine"]


def test_genuine_collapses_dvd_parts():
    """Genuine still drops VTS/NOIR/M4V parts that duplicate a main film."""
    _used, groups = _load_groups()
    if groups is None:
        return
    noir = groups["Man_Who_Cheated_Himself"]
    dropped = set(noir["dropped_genuine"])
    assert {"NOIR_DISC_11", "NOIR_DISC_13", "NOIR_DISC_14"} <= dropped
    assert {"VTS_01_0", "VTS_02_3", "VTS_03_1", "VTS_04_1"} <= dropped
    kept = [m for m in noir["members"] if m not in dropped]
    assert kept == [noir["kept_id"]]

    martha = groups["Martha_Ivers"]
    assert set(martha["dropped_genuine"]) == {
        "VTS_01_1", "VTS_01_2", "VTS_01_3", "VTS_01_4"
    }


def test_movie_map_matches_report():
    """data/movie_map.json must equal the report's id_to_movie_key field."""
    report_path = os.path.join(_root, "data/dedup_report.json")
    map_path = os.path.join(_root, "data/movie_map.json")
    if not (os.path.exists(report_path) and os.path.exists(map_path)):
        return  # artifacts not generated yet
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    with open(map_path, "r", encoding="utf-8") as f:
        movie_map = json.load(f)
    assert movie_map == report["id_to_movie_key"]
    # The map is the split-safety artifact: one key per used id.
    assert len(movie_map) == report["counts"]["used_ids"]


def test_report_policy_counts():
    report_path = os.path.join(_root, "data/dedup_report.json")
    if not os.path.exists(report_path):
        return
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)
    c = report["counts"]
    assert c["family_kept"] == 1336
    assert c["family_dropped"] == 92
    assert c["genuine_kept"] == 1406
    assert c["genuine_dropped"] == 22
    assert report["policies"]["selected"] == "genuine"


# ---------------------------------------------------------------------------
# Slow content test (ffmpeg + local films)
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def _make_clip(src, dst, t, start=60, quality=10):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-ss", str(start), "-i", src, "-t", str(t),
         "-c:v", "mpeg4", "-q:v", str(quality), "-c:a", "libmp3lame", "-b:a", "64k", dst],
        check=True,
    )


def test_content_clusters_reencode_only():
    if os.environ.get("RUN_SLOW_DEDUP_TESTS") != "1":
        return  # gated: shells out to ffmpeg
    if not _ffmpeg_available():
        return
    bb = os.path.join(_root, "data/videos/Black_Brigade.mp4")
    sr = os.path.join(_root, "data/videos/TheStilwellRoad.mp4")
    if not (os.path.exists(bb) and os.path.exists(sr)):
        return

    tmp = "/tmp/opencode/dedup_probe/test"
    os.makedirs(tmp, exist_ok=True)
    clip_hi = os.path.join(tmp, "bb_hi.mp4")
    clip_lo = os.path.join(tmp, "bb_lo.mp4")
    clip_sr = os.path.join(tmp, "sr_hi.mp4")
    _make_clip(bb, clip_hi, 20, quality=2)
    _make_clip(bb, clip_lo, 20, quality=12)
    _make_clip(sr, clip_sr, 20, quality=2)

    h_hi = frame_hashes(clip_hi, 8)
    h_lo = frame_hashes(clip_lo, 8)
    h_sr = frame_hashes(clip_sr, 8)

    match, _min_d, n = phash_similarity(h_hi, h_lo, threshold=10)
    assert match and n >= 3, (match, n)
    match2, _min_d2, n2 = phash_similarity(h_hi, h_sr, threshold=10)
    assert not match2, (match2, n2)

    s_hi = audio_signature(clip_hi)
    s_lo = audio_signature(clip_lo)
    s_sr = audio_signature(clip_sr)
    assert audio_similarity(s_hi, s_lo) >= 0.9
    assert audio_similarity(s_hi, s_sr) < 0.9


# ---------------------------------------------------------------------------
# Minimal runner (no pytest dependency required)
# ---------------------------------------------------------------------------

def _main() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS: {fn.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {fn.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR: {fn.__name__}: {type(e).__name__}: {e}")
    print()
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_main())
