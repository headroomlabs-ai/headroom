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


class _FakeProc:
    """Minimal stand-in for the ``serena project index`` child process."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stderr: str = "",
        communicate_error: BaseException | None = None,
    ) -> None:
        self.pid = 4242
        self.returncode = returncode
        self.stdin = None
        self.stdout = None
        self.stderr = None
        self._stderr = stderr
        self._communicate_error = communicate_error
        self.killed = False
        self.waited = False

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        if self._communicate_error is not None:
            raise self._communicate_error
        return "", self._stderr

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        self.waited = True
        return self.returncode


def _stub_popen(monkeypatch: pytest.MonkeyPatch, proc: _FakeProc) -> Mock:
    mock_popen = Mock(return_value=proc)
    monkeypatch.setattr(wrap_cli.subprocess, "Popen", mock_popen)
    return mock_popen


def test_preindex_runs_serena_in_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    mock_popen = _stub_popen(monkeypatch, _FakeProc())

    wrap_cli._index_serena_project()

    mock_popen.assert_called_once()
    args, kwargs = mock_popen.call_args
    cmd = args[0]
    assert cmd[0] == "uvx"
    assert cmd[-3:] == ["serena", "project", "index"]
    # PyPI package with prebuilt wheels, not the git source (#2871).
    assert "serena-agent" in cmd
    assert "git+https://github.com/oraios/serena" not in cmd
    assert kwargs["cwd"] == str(tmp_path)  # invoked in the project cwd


def test_preindex_is_timeout_guarded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    proc = _FakeProc()
    _stub_popen(monkeypatch, proc)
    seen: list[float | None] = []
    monkeypatch.setattr(
        proc, "communicate", lambda timeout=None: (seen.append(timeout), ("", ""))[1]
    )

    wrap_cli._index_serena_project()

    assert seen == [wrap_cli._SERENA_INDEX_TIMEOUT]


def test_preindex_never_inherits_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serena prompts ``[y/N]`` behind a captured stdout; stdin must be EOF (#2938)."""
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    mock_popen = _stub_popen(monkeypatch, _FakeProc())

    wrap_cli._index_serena_project()

    assert mock_popen.call_args.kwargs["stdin"] == subprocess.DEVNULL


def test_preindex_child_gets_its_own_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this the ``uvx`` grandchild survives the timeout kill (#2938)."""
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    mock_popen = _stub_popen(monkeypatch, _FakeProc())

    wrap_cli._index_serena_project()

    kwargs = mock_popen.call_args.kwargs
    if wrap_cli.sys.platform == "win32":
        assert kwargs["creationflags"] & subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        assert kwargs["start_new_session"] is True


def test_preindex_skips_without_uvx(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_uvx(monkeypatch, present=False)
    mock_popen = Mock(side_effect=AssertionError("Popen must not be called without uvx"))
    monkeypatch.setattr(wrap_cli.subprocess, "Popen", mock_popen)

    wrap_cli._index_serena_project()  # no exception

    mock_popen.assert_not_called()


def test_preindex_timeout_kills_the_tree_and_is_non_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    proc = _FakeProc(communicate_error=subprocess.TimeoutExpired(cmd="serena", timeout=1))
    _stub_popen(monkeypatch, proc)
    killed: list[object] = []
    monkeypatch.setattr(wrap_cli, "_kill_serena_index_tree", killed.append)

    wrap_cli._index_serena_project(verbose=True)  # must not propagate

    assert killed == [proc]


def test_preindex_generic_error_kills_the_tree_and_is_non_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    proc = _FakeProc(communicate_error=RuntimeError("boom"))
    _stub_popen(monkeypatch, proc)
    killed: list[object] = []
    monkeypatch.setattr(wrap_cli, "_kill_serena_index_tree", killed.append)

    wrap_cli._index_serena_project(verbose=True)  # must not propagate

    assert killed == [proc]


def test_preindex_spawn_failure_is_non_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    monkeypatch.setattr(wrap_cli.subprocess, "Popen", Mock(side_effect=OSError("no exec")))

    wrap_cli._index_serena_project(verbose=True)  # must not propagate


# ---------------------------------------------------------------------------
# HEADROOM_SERENA_INDEX_TIMEOUT — the stall budget is tunable (#3093)
# ---------------------------------------------------------------------------


def _clear_index_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve from a known-empty environment, not the developer's shell."""
    monkeypatch.delenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, raising=False)


def test_index_timeout_defaults_when_env_is_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_index_timeout(monkeypatch)

    assert wrap_cli._resolve_serena_index_timeout_seconds() == wrap_cli._SERENA_INDEX_TIMEOUT


def test_index_timeout_reads_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, "45")

    assert wrap_cli._resolve_serena_index_timeout_seconds() == 45


@pytest.mark.parametrize("raw", ["  30  ", "\t30\n"])
def test_index_timeout_tolerates_surrounding_whitespace(
    raw: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An env var exported from a shell heredoc keeps its padding."""
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, raw)

    assert wrap_cli._resolve_serena_index_timeout_seconds() == 30


@pytest.mark.parametrize("raw", ["", "   "])
def test_index_timeout_treats_a_blank_value_as_unset(
    raw: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``export HEADROOM_SERENA_INDEX_TIMEOUT=`` is not a misconfiguration."""
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, raw)

    assert wrap_cli._resolve_serena_index_timeout_seconds() == wrap_cli._SERENA_INDEX_TIMEOUT
    assert capsys.readouterr().out == ""  # no warning noise on the default path


def test_index_timeout_accepts_the_smallest_useful_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, "1")

    assert wrap_cli._resolve_serena_index_timeout_seconds() == 1


def test_index_timeout_accepts_a_budget_longer_than_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A deliberately huge monorepo may want *more* than 300s, not less."""
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, "86400")

    assert wrap_cli._resolve_serena_index_timeout_seconds() == 86400


@pytest.mark.parametrize("raw", ["0", "-1", "abc", "30s", "1.5", "1e3", "0x10", "None"])
def test_index_timeout_falls_back_on_an_unusable_value(
    raw: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Best-effort means degrade to the default, never abort the launch."""
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, raw)

    assert wrap_cli._resolve_serena_index_timeout_seconds() == wrap_cli._SERENA_INDEX_TIMEOUT

    # A silently-ignored knob is the bug being fixed, so say so — and quote the
    # value back, since the usual cause is a unit suffix the parser rejects.
    out = capsys.readouterr().out
    assert wrap_cli._SERENA_INDEX_TIMEOUT_ENV in out
    assert repr(raw) in out


def test_index_timeout_survives_a_float_unrepresentable_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``int()`` accepts integers ``float()`` cannot hold; resolving must not raise.

    ``communicate`` would go on to raise ``OverflowError`` adding that to a
    monotonic clock, which the caller's generic handler already absorbs as a
    non-fatal skip — so the budget is honoured as given rather than clamped.
    """
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, "1" + "0" * 400)

    assert wrap_cli._resolve_serena_index_timeout_seconds() == 10**400


def test_preindex_uses_the_env_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The resolved budget reaches ``communicate`` — the point of #3093."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, "5")
    _stub_uvx(monkeypatch)
    proc = _FakeProc()
    _stub_popen(monkeypatch, proc)
    seen: list[float | None] = []
    monkeypatch.setattr(
        proc, "communicate", lambda timeout=None: (seen.append(timeout), ("", ""))[1]
    )

    wrap_cli._index_serena_project()

    assert seen == [5]


def test_preindex_still_runs_on_an_unusable_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad value must not skip the pre-index or propagate out of it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, "not-a-number")
    _stub_uvx(monkeypatch)
    proc = _FakeProc()
    mock_popen = _stub_popen(monkeypatch, proc)
    seen: list[float | None] = []
    monkeypatch.setattr(
        proc, "communicate", lambda timeout=None: (seen.append(timeout), ("", ""))[1]
    )

    wrap_cli._index_serena_project()  # must not propagate

    mock_popen.assert_called_once()
    assert seen == [wrap_cli._SERENA_INDEX_TIMEOUT]


# ---------------------------------------------------------------------------
# _kill_serena_index_tree — no orphaned `serena project index` on timeout
# ---------------------------------------------------------------------------


def test_kill_tree_signals_the_group_on_posix(monkeypatch: pytest.MonkeyPatch) -> None:
    if wrap_cli.sys.platform == "win32":
        pytest.skip("POSIX process groups")
    proc = _FakeProc()
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(wrap_cli.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(wrap_cli.os, "killpg", lambda pgid, sig: signalled.append((pgid, sig)))

    wrap_cli._kill_serena_index_tree(proc)  # type: ignore[arg-type]

    assert signalled == [(proc.pid, wrap_cli.signal.SIGKILL)]
    assert proc.killed and proc.waited  # backstop still runs


def test_kill_tree_walks_the_tree_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    if wrap_cli.sys.platform != "win32":
        pytest.skip("Windows taskkill")
    proc = _FakeProc()
    calls: list[list[str]] = []
    monkeypatch.setattr(
        wrap_cli.subprocess,
        "run",
        lambda cmd, **kw: calls.append(cmd),
    )

    wrap_cli._kill_serena_index_tree(proc)  # type: ignore[arg-type]

    assert calls == [["taskkill", "/F", "/T", "/PID", str(proc.pid)]]
    assert proc.killed and proc.waited


def test_kill_tree_survives_a_dead_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cleanup is best-effort: a child that already exited must not raise."""
    proc = _FakeProc()
    monkeypatch.setattr(proc, "kill", Mock(side_effect=ProcessLookupError()))
    if wrap_cli.sys.platform == "win32":
        monkeypatch.setattr(wrap_cli.subprocess, "run", Mock(side_effect=OSError("gone")))
    else:
        monkeypatch.setattr(wrap_cli.os, "getpgid", Mock(side_effect=ProcessLookupError()))

    wrap_cli._kill_serena_index_tree(proc)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _serena_project_skip_reason — keep per-project setup off non-project roots
# ---------------------------------------------------------------------------

_NO_PROJECT_YML = "no .serena/project.yml yet — Serena will create it and index on demand"


def _with_project_yml(root: Path) -> Path:
    """Give *root* the ``.serena/project.yml`` Serena writes on first MCP start."""
    (root / ".serena").mkdir(parents=True, exist_ok=True)
    (root / ".serena" / "project.yml").write_text("project_name: demo\n")
    return root


def test_skip_reason_none_for_ordinary_project(tmp_path: Path) -> None:
    _with_project_yml(tmp_path)

    assert wrap_cli._serena_project_skip_reason(tmp_path) is None


def test_skip_reason_none_for_normal_checkout(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()  # real checkout: .git is a directory
    _with_project_yml(tmp_path)

    assert wrap_cli._serena_project_skip_reason(tmp_path) is None


def test_skip_reason_flags_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    _with_project_yml(tmp_path)

    assert wrap_cli._serena_project_skip_reason(tmp_path) == "$HOME is not a project"


def test_skip_reason_flags_linked_worktree(tmp_path: Path) -> None:
    (tmp_path / ".git").write_text("gitdir: /repo/.git/worktrees/wt\n")
    _with_project_yml(tmp_path)

    assert wrap_cli._serena_project_skip_reason(tmp_path) == "linked git worktree"


def test_skip_reason_flags_missing_project_yml(tmp_path: Path) -> None:
    """The pre-index cannot succeed here — Serena would stop on a hidden prompt (#2938)."""
    assert wrap_cli._serena_project_skip_reason(tmp_path) == _NO_PROJECT_YML


def test_skip_reason_flags_serena_dir_without_project_yml(tmp_path: Path) -> None:
    (tmp_path / ".serena").mkdir()  # cache dir exists, config does not

    assert wrap_cli._serena_project_skip_reason(tmp_path) == _NO_PROJECT_YML


def test_skip_reason_survives_unresolvable_root(tmp_path: Path) -> None:
    # Missing directory: resolves fine (non-strict), no config, no exception.
    assert wrap_cli._serena_project_skip_reason(tmp_path / "gone") == _NO_PROJECT_YML


# ---------------------------------------------------------------------------
# _setup_serena_mcp wiring — the launch path must not wait on a doomed index
# ---------------------------------------------------------------------------


class _FakeRegistrar:
    """Just enough registrar for the post-registration branch of the setup."""

    name = "claude"
    display_name = "Claude"

    def detect(self) -> bool:
        return True

    def get_server(self, server_name: str) -> None:
        return None

    def register_server(self, spec: object, *, force: bool = False) -> object:
        from headroom.mcp_registry.base import RegisterResult, RegisterStatus

        return RegisterResult(RegisterStatus.REGISTERED, "registered")


def _drive_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Run ``_setup_serena_mcp`` in *tmp_path*, returning pre-index call markers."""
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    monkeypatch.setattr(wrap_cli, "_inject_serena_instructions", lambda *a, **k: True)
    calls: list[str] = []
    monkeypatch.setattr(wrap_cli, "_index_serena_project", lambda **k: calls.append("indexed"))

    wrap_cli._setup_serena_mcp(_FakeRegistrar(), context="claude-code", verbose=True)
    return calls


def test_setup_does_not_preindex_a_project_without_serena_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """First wrap of a fresh project: launch immediately, do not wait out the timeout."""
    calls = _drive_setup(tmp_path, monkeypatch)

    assert calls == []
    assert "skipping pre-index" in capsys.readouterr().out


def test_setup_preindexes_once_serena_config_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Second wrap: Serena's MCP server has written project.yml, so the index can run."""
    _with_project_yml(tmp_path)

    assert _drive_setup(tmp_path, monkeypatch) == ["indexed"]
