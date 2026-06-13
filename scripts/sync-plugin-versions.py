"""Sync plugin manifest versions to the repo's computed release semver.

Branch-aware: by default this script is a NO-OP on feature branches.
Pre-this-fix it ran on every commit and bumped the manifests to the
PREDICTED next release version, which polluted every PR with version-
bump noise (the prediction advanced as commits landed; each PR ended
up carrying the bump as collateral).

Sync now only runs when EITHER:
  * We're on the ``main`` branch, OR
  * ``HEADROOM_SYNC_VERSIONS=1`` is set explicitly (release workflow)

Result: feature-branch PRs no longer carry manifest bumps; the
release workflow still gets a canonical sync at publish time.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from headroom.release_version import (  # noqa: E402
        compute_release_version,
        determine_bump_level,
        find_latest_release_tag,
        get_canonical_version,
        list_release_commits,
        list_release_tags,
    )
except ImportError:
    print("skip: headroom deps not installed (run from a dev venv to enable)")
    sys.exit(0)


def compute_repo_semver(root: Path) -> str:
    """Return the npm-style semver for the repo's next release."""
    tags = list_release_tags(root)
    previous_tag = find_latest_release_tag(tags) or ""
    level = determine_bump_level(list_release_commits(root, previous_tag))
    info = compute_release_version(
        canonical_version=get_canonical_version(root),
        level=level,
        tags=tags,
    )
    return info.npm_version


def _current_branch(root: Path) -> str | None:
    """Return the current git branch name, or None if git isn't usable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except (FileNotFoundError, OSError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _should_sync(root: Path) -> bool:
    """Decide whether to actually run the sync.

    Release workflow opts in via ``HEADROOM_SYNC_VERSIONS=1``; otherwise
    we only sync on ``main`` (where the next-release prediction
    legitimately lives). On feature branches we no-op — the prediction
    would just create PR-level noise.
    """
    if os.environ.get("HEADROOM_SYNC_VERSIONS") == "1":
        return True
    branch = _current_branch(root)
    if branch is None:
        # Git unavailable or detached HEAD — safest default is no-op.
        return False
    return branch == "main"


def main() -> None:
    root = ROOT
    if not _should_sync(root):
        # Quiet no-op on feature branches. Print a single line so
        # pre-commit users see the reason if they look.
        branch = _current_branch(root) or "<unknown>"
        print(
            f"sync-plugin-versions: skipping on branch '{branch}' (set HEADROOM_SYNC_VERSIONS=1 to force)"
        )
        return
    version = compute_repo_semver(root)
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "version-sync.py"),
            "--root",
            str(root),
            "--version",
            version,
            "--plugin-manifests-only",
        ],
        cwd=root,
        check=True,
    )


if __name__ == "__main__":
    main()
