"""scripts/install-git-hooks.sh: install into the checkout that owns the script.

The script runs for real against temporary repositories, with fake
``pre-commit`` and ``npx`` on PATH, so ordinary checkouts, linked worktrees,
a source copy nested inside another repository, and a directory outside git
are exercised end to end without touching real hooks.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "install-git-hooks.sh"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not (shutil.which("bash") and shutil.which("git")),
    reason="needs bash and git",
)

FAKE_PRE_COMMIT = """#!/usr/bin/env bash
hooks="$(git rev-parse --git-path hooks)"
case "$*" in
  install) echo "#!/bin/sh" > "$hooks/pre-commit" ;;
  "install --hook-type commit-msg") echo "#!/bin/sh" > "$hooks/commit-msg" ;;
esac
"""


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _checkout(path: Path) -> Path:
    """A repository containing scripts/install-git-hooks.sh and one commit."""
    (path / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, path / "scripts" / "install-git-hooks.sh")
    _git(path, "init", "-q")
    _git(path, "add", ".")
    _git(path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    return path


def _env(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir(exist_ok=True)
    for name, body in (("pre-commit", FAKE_PRE_COMMIT), ("npx", "#!/bin/sh\n")):
        tool = bin_dir / name
        tool.write_text(body)
        tool.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.pop("VIRTUAL_ENV", None)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["GIT_CEILING_DIRECTORIES"] = str(tmp_path)
    return env


def _run(script_root: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script_root / "scripts" / "install-git-hooks.sh")],
        cwd=script_root,
        env=env,
        capture_output=True,
        text=True,
    )


def _hooks(common_dir: Path) -> set[str]:
    return {
        p.name
        for p in (common_dir / "hooks").iterdir()
        if p.name in {"pre-push", "pre-commit", "commit-msg"}
    }


def test_installs_hooks_in_an_ordinary_checkout(tmp_path: Path) -> None:
    repo = _checkout(tmp_path / "repo")

    result = _run(repo, _env(tmp_path))

    assert result.returncode == 0, result.stderr
    assert _hooks(repo / ".git") == {"pre-push", "pre-commit", "commit-msg"}


def test_installs_shared_hooks_from_a_linked_worktree(tmp_path: Path) -> None:
    repo = _checkout(tmp_path / "repo")
    worktree = tmp_path / "worktree"
    _git(repo, "worktree", "add", "-q", str(worktree))
    assert (worktree / ".git").is_file()

    result = _run(worktree, _env(tmp_path))

    assert result.returncode == 0, result.stderr
    assert _hooks(repo / ".git") == {"pre-push", "pre-commit", "commit-msg"}


def test_refuses_a_source_copy_nested_in_another_repository(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    _git(parent, "init", "-q")
    nested = parent / "vendor" / "headroom"
    (nested / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, nested / "scripts" / "install-git-hooks.sh")

    result = _run(nested, _env(tmp_path))

    assert result.returncode == 1
    assert "not the root of a git checkout" in result.stderr
    assert _hooks(parent / ".git") == set()


def _push_setup(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    """A checkout with the hooks installed, a bare remote, and a fake ``make``
    that logs each call instead of running ci-precheck."""
    repo = _checkout(tmp_path / "repo")
    env = _env(tmp_path)
    assert _run(repo, env).returncode == 0
    make_log = tmp_path / "make.log"
    fake_make = tmp_path / "fake-bin" / "make"
    fake_make.write_text(f'#!/bin/sh\necho "$*" >> "{make_log}"\n')
    fake_make.chmod(0o755)
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "-q", "--bare", str(remote))
    _git(repo, "remote", "add", "origin", str(remote))
    return repo, env, make_log


def _push(repo: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "push", "-q", "origin", *args],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
    )


def _make_calls(make_log: Path) -> list[str]:
    return make_log.read_text().splitlines() if make_log.exists() else []


def test_pre_push_runs_ci_precheck_for_a_branch_update(tmp_path: Path) -> None:
    repo, env, make_log = _push_setup(tmp_path)

    result = _push(repo, env, "HEAD:refs/heads/feature")

    assert result.returncode == 0, result.stderr
    assert _make_calls(make_log) == ["ci-precheck"]


def test_pre_push_skips_ci_precheck_for_a_delete_only_push(tmp_path: Path) -> None:
    repo, env, make_log = _push_setup(tmp_path)
    _git(repo, "push", "-q", "--no-verify", "origin", "HEAD:refs/heads/feature")

    result = _push(repo, env, "--delete", "feature")

    assert result.returncode == 0, result.stderr
    assert _make_calls(make_log) == []
    assert _git(repo, "ls-remote", "--heads", "origin", "feature") == ""


def test_pre_push_runs_ci_precheck_when_a_push_mixes_a_delete_and_an_update(
    tmp_path: Path,
) -> None:
    repo, env, make_log = _push_setup(tmp_path)
    _git(repo, "push", "-q", "--no-verify", "origin", "HEAD:refs/heads/old")

    result = _push(repo, env, ":old", "HEAD:refs/heads/new")

    assert result.returncode == 0, result.stderr
    assert _make_calls(make_log) == ["ci-precheck"]


def test_refuses_a_directory_outside_git(tmp_path: Path) -> None:
    loose = tmp_path / "loose"
    (loose / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, loose / "scripts" / "install-git-hooks.sh")

    result = _run(loose, _env(tmp_path))

    assert result.returncode == 1
    assert "not the root of a git checkout" in result.stderr
