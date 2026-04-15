"""Filesystem helpers used by data, reports, and tests."""

from pathlib import Path


def ensure_directory(path: Path) -> Path:
    """Create ``path`` and return it.

    The helper keeps directory creation explicit at call sites and avoids
    repeated ``mkdir`` boilerplate across demos.
    """

    path.mkdir(parents=True, exist_ok=True)
    return path
