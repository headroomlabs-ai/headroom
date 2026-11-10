"""codemode_ab.main() must refuse a --worktree it would wipe with reset/clean."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import codemode_ab


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def run_main(tmp_path, monkeypatch, capsys):
    # Stand-in for the real `claude` run so no test can reach the CLI.
    task = ("t", "prompt", None)
    monkeypatch.setattr(codemode_ab, "TASKS", [task])
    monkeypatch.setattr(codemode_ab, "run_one", lambda *a, **k: {"cost_usd": 0, "num_turns": 1})

    def run(worktree: Path):
        argv = [
            "codemode_ab.py",
            "--worktree",
            str(worktree),
            "--plugin-dir",
            str(tmp_path),
            "--out",
            str(tmp_path / "out" / "r.json"),
        ]
        monkeypatch.setattr(sys, "argv", argv)
        rc = codemode_ab.main()
        return rc, capsys.readouterr().err

    return run


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    (r / "tracked.txt").write_text("committed")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "init")
    return r


def test_dirty_worktree_is_refused_and_left_intact(repo, run_main):
    (repo / "tracked.txt").write_text("my edit")
    (repo / "notes.md").write_text("untracked work")
    rc, err = run_main(repo)
    assert rc == 2
    assert "uncommitted changes" in err
    assert (repo / "tracked.txt").read_text() == "my edit"
    assert (repo / "notes.md").read_text() == "untracked work"


def test_untracked_work_is_refused_even_when_git_hides_untracked_files(repo, run_main, monkeypatch):
    # status.showUntrackedFiles=no makes a bare `git status --porcelain` blind to
    # untracked files, while the cleanup's `git clean -fd` still deletes them.
    _git(repo, "config", "status.showUntrackedFiles", "no")
    (repo / "precious-notes.txt").write_text("untracked work")

    def never_called(*a, **k):
        raise AssertionError("the benchmark must not start on a worktree with untracked work")

    monkeypatch.setattr(codemode_ab, "run_one", never_called)
    rc, err = run_main(repo)
    assert rc == 2
    assert "uncommitted changes" in err
    assert (repo / "precious-notes.txt").read_text() == "untracked work"


def test_non_git_directory_is_refused(tmp_path, run_main):
    plain = tmp_path / "plain"
    plain.mkdir()
    rc, err = run_main(plain)
    assert rc == 2
    assert "not inside a git work tree" in err


def test_clean_worktree_passes_the_guard(repo, run_main):
    rc, _ = run_main(repo)
    assert rc == 0
