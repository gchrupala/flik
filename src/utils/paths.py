"""Path helpers for portable manifests.

Manifests store paths RELATIVE to PROJECT_ROOT so they stay valid when the
repo moves between machines (devbox, Snellius, Purpureus). resolve_path()
is backward compatible: absolute paths pass through unchanged (old manifests
keep working), relative paths are anchored at PROJECT_ROOT (data/ lives
inside the repo checkout).
"""

import os

from src.CONSTANTS import PROJECT_ROOT


def resolve_path(path: str) -> str:
    """Return an absolute path. Absolute input passes through unchanged;
    relative input is joined to PROJECT_ROOT."""
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def to_relative(path: str) -> str:
    """Convert an absolute path under PROJECT_ROOT to a relative one.
    Relative paths (and absolute paths outside PROJECT_ROOT) pass through
    unchanged."""
    if not path:
        return path
    if not os.path.isabs(path):
        return path
    rel = os.path.relpath(path, PROJECT_ROOT)
    if rel.startswith(".."):
        return path  # outside PROJECT_ROOT — keep absolute
    return rel
