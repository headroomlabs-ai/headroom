"""Package version metadata."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

UNKNOWN_VERSION = "unknown"


def _source_root() -> Path | None:
    """Return the repository root when imported from a git checkout."""
    root = Path(__file__).resolve().parents[1]
    if (root / ".git").exists() and (root / "pyproject.toml").exists():
        return root
    return None


def _source_tree_version(root: Path) -> str | None:
    """Compute the version release automation would assign to this checkout."""
    try:
        from headroom.release_version import (
            compute_release_version,
            determine_bump_level,
            get_canonical_version,
            list_release_commits,
            list_release_tags,
        )

        tags = list_release_tags(root)
        previous_tag = compute_release_version(
            canonical_version=get_canonical_version(root),
            level="patch",
            tags=tags,
        ).previous_tag
        commits = list_release_commits(root, previous_tag)
        level = determine_bump_level(commits)
        return compute_release_version(
            canonical_version=get_canonical_version(root),
            level=level,
            tags=tags,
        ).version
    except Exception:
        return None


def get_version() -> str:
    """Return Headroom's runtime version."""
    root = _source_root()
    if root is not None:
        source_version = _source_tree_version(root)
        if source_version:
            return source_version

    try:
        return version("headroom-ai")
    except PackageNotFoundError:
        return UNKNOWN_VERSION


__version__ = get_version()
