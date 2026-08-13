#!/usr/bin/env python
"""Rewrite absolute paths in manifest JSONs to PROJECT_ROOT-relative paths.

Backward compatibility: the pipeline now resolves relative paths against
PROJECT_ROOT at load time and passes absolute paths through unchanged, so
migrating is optional — but recommended so manifests survive machine moves.

Handles stale machine prefixes (e.g. /home/techsword/work_dir/flik/data/...
left over from a previous cluster): any absolute path containing "/data/" is
re-anchored to "data/..." relative to PROJECT_ROOT.

Usage:
    uv run --extra cpu python -m scripts.migrate_manifest_paths data/batch_manifest.json
    uv run --extra cpu python -m scripts.migrate_manifest_paths data/*.json --check
    uv run --extra cpu python -m scripts.migrate_manifest_paths file.json --dry-run

Options:
    --check    verify that each rewritten path exists under PROJECT_ROOT
    --dry-run  report what would change without writing
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.CONSTANTS import PROJECT_ROOT

PATH_KEYS = ("video_path", "json_path", "srt_path")
DATA_MARKER = "/data/"


def reanchor(path: str) -> tuple:
    """Return (new_path, status). status in: relative, migrated, reanchored, external."""
    if not path or not os.path.isabs(path):
        return path, "relative"
    rel = os.path.relpath(path, PROJECT_ROOT)
    if not rel.startswith(".."):
        return rel, "migrated"
    if DATA_MARKER in path:
        return "data/" + path.split(DATA_MARKER, 1)[1], "reanchored"
    return path, "external"


def migrate_file(path: str, check: bool, dry_run: bool) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    stats = {"relative": 0, "migrated": 0, "reanchored": 0, "external": 0, "missing": 0}
    external_paths = set()

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for key in PATH_KEYS:
            old = entry.get(key)
            if not isinstance(old, str) or not old:
                continue
            new, status = reanchor(old)
            stats[status] += 1
            if status == "external":
                external_paths.add(old)
            entry[key] = new
            if check and status in ("migrated", "reanchored"):
                if not os.path.exists(os.path.join(PROJECT_ROOT, new)):
                    stats["missing"] += 1

    print(f"{path}: {len(entries)} entries")
    print(f"  already relative: {stats['relative']}")
    print(f"  migrated (under PROJECT_ROOT): {stats['migrated']}")
    print(f"  reanchored (stale prefix -> data/): {stats['reanchored']}")
    if stats["external"]:
        print(f"  LEFT ABSOLUTE (outside project, no /data/ anchor): {stats['external']}")
        for p in sorted(external_paths)[:5]:
            print(f"    - {p}")
    if check:
        print(f"  rewritten paths missing on disk: {stats['missing']}")

    changed = stats["migrated"] + stats["reanchored"]
    if changed and not dry_run:
        backup = path + ".bak"
        if not os.path.exists(backup):
            os.replace(path, backup)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=2, ensure_ascii=False)
        print(f"  written (backup: {backup})")
    elif changed:
        print("  [dry-run] no changes written")
    else:
        print("  nothing to do")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifests", nargs="+", help="Manifest JSON files to migrate")
    parser.add_argument("--check", action="store_true",
                        help="Verify rewritten paths exist under PROJECT_ROOT")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report changes without writing")
    args = parser.parse_args()

    for path in args.manifests:
        if not os.path.exists(path):
            print(f"ERROR: not found: {path}")
            continue
        migrate_file(path, args.check, args.dry_run)


if __name__ == "__main__":
    main()
