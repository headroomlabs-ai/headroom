"""Serena "boost" wrap-time helpers: prefer-Serena instruction injection,
repo-language scoping of ``.serena/project.yml``, and symbol-cache pre-indexing.

All Serena subprocess calls are mocked — these tests never invoke real ``uvx``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from headroom.cli import wrap as wrap_cli

# ---------------------------------------------------------------------------
# _inject_serena_instructions
# ---------------------------------------------------------------------------


def _opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Enable the opt-in gate so injection actually writes.

    Instruction injection rewrites the user's CLAUDE.md/AGENTS.md, so it is
    off by default. Tests that exercise the write path must opt in via
    ``HEADROOM_SERENA_INSTRUCTIONS``.
    """
    monkeypatch.setenv("HEADROOM_SERENA_INSTRUCTIONS", "1")


def test_inject_creates_file_and_mentions_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _opt_in(monkeypatch)
    target = tmp_path / "AGENTS.md"
    assert wrap_cli._inject_serena_instructions(target) is True

    content = target.read_text()
    assert wrap_cli._SERENA_MARKER in content
    # The whole point is steering the agent toward Serena's symbol tools.
    for tool in ("get_symbols_overview", "find_symbol", "find_referencing_symbols"):
        assert tool in content, f"{tool} missing from injected guidance"


def test_inject_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _opt_in(monkeypatch)
    target = tmp_path / "AGENTS.md"
    wrap_cli._inject_serena_instructions(target)
    wrap_cli._inject_serena_instructions(target)  # second call is a no-op

    content = target.read_text()
    assert content.count(wrap_cli._SERENA_MARKER) == 1


def test_inject_appends_to_existing_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _opt_in(monkeypatch)
    target = tmp_path / "CLAUDE.md"
    target.write_text("# Project notes\n\nkeep me\n")
    wrap_cli._inject_serena_instructions(target)

    content = target.read_text()
    assert "keep me" in content  # existing content preserved
    assert wrap_cli._SERENA_MARKER in content


def test_inject_off_by_default_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Without opting in, injection is a no-op: returns False and never touches
    # the user's hint file (the default, so the two OpenCode AGENTS.md tests pass).
    monkeypatch.delenv("HEADROOM_SERENA_INSTRUCTIONS", raising=False)

    missing = tmp_path / "AGENTS.md"
    assert wrap_cli._inject_serena_instructions(missing) is False
    assert not missing.exists()  # nothing created

    existing = tmp_path / "CLAUDE.md"
    existing.write_text("# Project notes\n\nkeep me\n")
    assert wrap_cli._inject_serena_instructions(existing) is False
    assert existing.read_text() == "# Project notes\n\nkeep me\n"  # untouched
    assert wrap_cli._SERENA_MARKER not in existing.read_text()


def test_instruction_file_target_per_agent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    class _Reg:
        def __init__(self, name: str) -> None:
            self.name = name

    assert wrap_cli._serena_instruction_file(_Reg("claude")).name == "CLAUDE.md"
    assert wrap_cli._serena_instruction_file(_Reg("codex")).name == "AGENTS.md"
    assert wrap_cli._serena_instruction_file(_Reg("grok")).name == "AGENTS.md"


# ---------------------------------------------------------------------------
# _index_serena_project — best-effort, timeout-guarded pre-index
# ---------------------------------------------------------------------------


def _stub_uvx(monkeypatch: pytest.MonkeyPatch, present: bool = True) -> None:
    monkeypatch.setattr(
        wrap_cli.shutil,
        "which",
        lambda name, *a, **k: "/usr/bin/uvx" if (present and name == "uvx") else None,
    )


def test_preindex_runs_serena_in_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    mock_run = Mock(
        return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    )
    monkeypatch.setattr(wrap_cli, "run", mock_run)

    wrap_cli._index_serena_project()

    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    cmd = args[0]
    assert cmd[0] == "uvx"
    assert cmd[-3:] == ["serena", "project", "index"]
    assert "git+https://github.com/oraios/serena" in cmd
    assert kwargs["cwd"] == str(tmp_path)  # invoked in the project cwd
    assert "timeout" in kwargs  # timeout-guarded


def test_preindex_skips_without_uvx(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_uvx(monkeypatch, present=False)
    mock_run = Mock(side_effect=AssertionError("run must not be called without uvx"))
    monkeypatch.setattr(wrap_cli, "run", mock_run)

    wrap_cli._index_serena_project()  # no exception

    mock_run.assert_not_called()


def test_preindex_timeout_is_non_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_uvx(monkeypatch)
    monkeypatch.setattr(
        wrap_cli,
        "run",
        Mock(side_effect=subprocess.TimeoutExpired(cmd="serena", timeout=1)),
    )
    # Must not propagate.
    wrap_cli._index_serena_project(verbose=True)


def test_preindex_generic_error_is_non_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_uvx(monkeypatch)
    monkeypatch.setattr(wrap_cli, "run", Mock(side_effect=RuntimeError("boom")))
    # Must not propagate.
    wrap_cli._index_serena_project(verbose=True)


# ---------------------------------------------------------------------------
# _serena_project_skip_reason — keep per-project setup off non-project roots
# ---------------------------------------------------------------------------


def test_skip_reason_none_for_ordinary_project(tmp_path: Path) -> None:
    assert wrap_cli._serena_project_skip_reason(tmp_path) is None


def test_skip_reason_none_for_normal_checkout(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()  # real checkout: .git is a directory

    assert wrap_cli._serena_project_skip_reason(tmp_path) is None


def test_skip_reason_flags_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))

    assert wrap_cli._serena_project_skip_reason(tmp_path) == "$HOME is not a project"


def test_skip_reason_flags_linked_worktree(tmp_path: Path) -> None:
    (tmp_path / ".git").write_text("gitdir: /repo/.git/worktrees/wt\n")

    assert wrap_cli._serena_project_skip_reason(tmp_path) == "linked git worktree"


def test_skip_reason_survives_unresolvable_root(tmp_path: Path) -> None:
    assert wrap_cli._serena_project_skip_reason(tmp_path / "gone") is None
