"""Serena "boost" wrap-time helpers: prefer-Serena instruction injection,
repo-language scoping of ``.serena/project.yml``, and background symbol-cache
pre-indexing.

All Serena subprocess calls are mocked — these tests never invoke real ``uvx``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
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
# _index_serena_project — spawned in the background, never on the launch path
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
        poll_result: int | None = None,
        wait_error: BaseException | None = None,
    ) -> None:
        self.pid = 4242
        self.returncode = returncode
        self.stdin = None
        self.stdout = None
        self.stderr = None
        self._poll_result = poll_result
        self._wait_error = wait_error
        self.waits: list[float | None] = []
        self.killed = False
        self.waited = False

    def poll(self) -> int | None:
        return self._poll_result

    def wait(self, timeout: float | None = None) -> int:
        self.waits.append(timeout)
        self.waited = True
        if self._wait_error is not None:
            raise self._wait_error
        return self.returncode

    def kill(self) -> None:
        self.killed = True


def _stub_popen(monkeypatch: pytest.MonkeyPatch, proc: _FakeProc) -> Mock:
    mock_popen = Mock(return_value=proc)
    monkeypatch.setattr(wrap_cli.subprocess, "Popen", mock_popen)
    return mock_popen


@pytest.fixture(autouse=True)
def _no_leaked_index_child() -> Iterator[None]:
    """Never let a fake child stay parked in the module global.

    ``_index_serena_project`` stashes the background child so the atexit hook
    can stop it. Leaving a mock there would make an unrelated test's hook act
    on it — and would fire at interpreter shutdown.
    """
    yield
    wrap_cli._SERENA_INDEX_PROC = None


def _stub_atexit(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Collect atexit registrations instead of really queueing them."""
    registered: list[object] = []
    monkeypatch.setattr(wrap_cli.atexit, "register", registered.append)
    return registered


def test_preindex_runs_serena_in_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    _stub_atexit(monkeypatch)
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


def test_preindex_does_not_block_the_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point of #3436: spawn, say so, and let the agent start."""
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    registered = _stub_atexit(monkeypatch)
    proc = _FakeProc()
    _stub_popen(monkeypatch, proc)

    wrap_cli._index_serena_project()

    assert proc.waits == []  # never waited on
    assert wrap_cli._SERENA_INDEX_PROC is proc  # kept, so exit can stop it
    assert registered == [wrap_cli._stop_background_serena_index]
    assert "background" in capsys.readouterr().out


def test_preindex_discards_child_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pipes would wedge the child: nothing drains them once we stop waiting.

    ``serena project index`` writes a progress line per file, so a full pipe
    buffer would block it forever instead of letting the cache warm.
    """
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    _stub_atexit(monkeypatch)
    mock_popen = _stub_popen(monkeypatch, _FakeProc())

    wrap_cli._index_serena_project()

    kwargs = mock_popen.call_args.kwargs
    assert kwargs["stdout"] == subprocess.DEVNULL
    assert kwargs["stderr"] == subprocess.DEVNULL


def test_preindex_never_inherits_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Serena prompts ``[y/N]`` behind redirected output; stdin must be EOF (#2938)."""
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    _stub_atexit(monkeypatch)
    mock_popen = _stub_popen(monkeypatch, _FakeProc())

    wrap_cli._index_serena_project()

    assert mock_popen.call_args.kwargs["stdin"] == subprocess.DEVNULL


def test_preindex_child_gets_its_own_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keeps Ctrl-C off the indexer, and lets the exit hook take out the ``uvx``
    grandchild rather than orphaning it (#2938)."""
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    _stub_atexit(monkeypatch)
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


def test_preindex_spawn_failure_is_non_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    registered = _stub_atexit(monkeypatch)
    monkeypatch.setattr(wrap_cli.subprocess, "Popen", Mock(side_effect=OSError("no exec")))

    wrap_cli._index_serena_project(verbose=True)  # must not propagate

    assert registered == []  # nothing to stop later
    assert wrap_cli._SERENA_INDEX_PROC is None


# ---------------------------------------------------------------------------
# _stop_background_serena_index — the indexer does not outlive the session
# ---------------------------------------------------------------------------


def test_exit_stops_a_still_running_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Otherwise a 13-minute index grinds on with nobody left to want it."""
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    _stub_atexit(monkeypatch)
    proc = _FakeProc(poll_result=None)  # still running
    _stub_popen(monkeypatch, proc)
    killed: list[object] = []
    monkeypatch.setattr(wrap_cli, "_kill_serena_index_tree", killed.append)

    wrap_cli._index_serena_project()
    wrap_cli._stop_background_serena_index()

    assert killed == [proc]


def test_exit_leaves_a_finished_index_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    _stub_atexit(monkeypatch)
    _stub_popen(monkeypatch, _FakeProc(poll_result=0))  # already exited
    killed: list[object] = []
    monkeypatch.setattr(wrap_cli, "_kill_serena_index_tree", killed.append)

    wrap_cli._index_serena_project()
    wrap_cli._stop_background_serena_index()

    assert killed == []


def test_exit_hook_kills_once_and_is_safe_to_repeat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``atexit`` can hold several registrations from one process."""
    monkeypatch.chdir(tmp_path)
    _stub_uvx(monkeypatch)
    _stub_atexit(monkeypatch)
    proc = _FakeProc(poll_result=None)
    _stub_popen(monkeypatch, proc)
    killed: list[object] = []
    monkeypatch.setattr(wrap_cli, "_kill_serena_index_tree", killed.append)

    wrap_cli._index_serena_project()
    wrap_cli._stop_background_serena_index()
    wrap_cli._stop_background_serena_index()

    assert killed == [proc]


def test_exit_hook_without_a_child_is_a_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The hook also runs for wraps that never reached the pre-index."""
    monkeypatch.setattr(
        wrap_cli,
        "_kill_serena_index_tree",
        Mock(side_effect=AssertionError("nothing to kill")),
    )

    wrap_cli._stop_background_serena_index()  # no exception


def test_exit_hook_survives_an_unpollable_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shutdown is no place to raise: a broken handle just means no kill."""
    proc = Mock()
    proc.poll.side_effect = OSError("handle gone")
    wrap_cli._SERENA_INDEX_PROC = proc
    monkeypatch.setattr(
        wrap_cli,
        "_kill_serena_index_tree",
        Mock(side_effect=AssertionError("must not kill an unpollable child")),
    )

    wrap_cli._stop_background_serena_index()  # no exception


# ---------------------------------------------------------------------------
# HEADROOM_SERENA_INDEX_TIMEOUT — opting back into a blocking pre-index (#3436)
# ---------------------------------------------------------------------------


def _clear_index_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve from a known-empty environment, not the developer's shell."""
    monkeypatch.delenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, raising=False)


def test_index_wait_defaults_to_not_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset means background, which is what #3436 asked for."""
    _clear_index_timeout(monkeypatch)

    assert wrap_cli._resolve_serena_index_wait_seconds() == 0


def test_index_wait_reads_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, "45")

    assert wrap_cli._resolve_serena_index_wait_seconds() == 45


@pytest.mark.parametrize("raw", ["  30  ", "\t30\n"])
def test_index_wait_tolerates_surrounding_whitespace(
    raw: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An env var exported from a shell heredoc keeps its padding."""
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, raw)

    assert wrap_cli._resolve_serena_index_wait_seconds() == 30


@pytest.mark.parametrize("raw", ["", "   "])
def test_index_wait_treats_a_blank_value_as_unset(
    raw: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``export HEADROOM_SERENA_INDEX_TIMEOUT=`` is not a misconfiguration."""
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, raw)

    assert wrap_cli._resolve_serena_index_wait_seconds() == 0
    assert capsys.readouterr().out == ""  # no warning noise on the default path


def test_index_wait_accepts_the_smallest_useful_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, "1")

    assert wrap_cli._resolve_serena_index_wait_seconds() == 1


def test_index_wait_accepts_a_long_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deliberately huge monorepo may want to block for a long time."""
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, "86400")

    assert wrap_cli._resolve_serena_index_wait_seconds() == 86400


@pytest.mark.parametrize("raw", ["0", "-1", "abc", "30s", "1.5", "1e3", "0x10", "None"])
def test_index_wait_falls_back_to_the_background_on_an_unusable_value(
    raw: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Best-effort means degrade to the default, never abort the launch."""
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, raw)

    assert wrap_cli._resolve_serena_index_wait_seconds() == 0

    # A silently-ignored knob is the bug #3093 was about, so say so — and quote
    # the value back, since the usual cause is a unit suffix the parser rejects.
    out = capsys.readouterr().out
    assert wrap_cli._SERENA_INDEX_TIMEOUT_ENV in out
    assert repr(raw) in out


def test_index_wait_survives_a_float_unrepresentable_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``int()`` accepts integers ``float()`` cannot hold; resolving must not raise.

    ``wait`` would go on to raise ``OverflowError`` adding that to a monotonic
    clock, which the caller's generic handler already absorbs as a non-fatal
    skip — so the budget is honoured as given rather than clamped.
    """
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, "1" + "0" * 400)

    assert wrap_cli._resolve_serena_index_wait_seconds() == 10**400


def test_preindex_blocks_for_the_env_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setting the knob restores the pre-#3436 blocking wait."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, "5")
    _stub_uvx(monkeypatch)
    _stub_atexit(monkeypatch)
    proc = _FakeProc()
    _stub_popen(monkeypatch, proc)

    wrap_cli._index_serena_project()

    assert proc.waits == [5]
    assert wrap_cli._SERENA_INDEX_PROC is None  # waited for, nothing to stop later


def test_preindex_timeout_kills_the_tree_and_is_non_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, "5")
    _stub_uvx(monkeypatch)
    _stub_atexit(monkeypatch)
    proc = _FakeProc(wait_error=subprocess.TimeoutExpired(cmd="serena", timeout=5))
    _stub_popen(monkeypatch, proc)
    killed: list[object] = []
    monkeypatch.setattr(wrap_cli, "_kill_serena_index_tree", killed.append)

    wrap_cli._index_serena_project(verbose=True)  # must not propagate

    assert killed == [proc]


def test_preindex_generic_error_kills_the_tree_and_is_non_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, "5")
    _stub_uvx(monkeypatch)
    _stub_atexit(monkeypatch)
    proc = _FakeProc(wait_error=RuntimeError("boom"))
    _stub_popen(monkeypatch, proc)
    killed: list[object] = []
    monkeypatch.setattr(wrap_cli, "_kill_serena_index_tree", killed.append)

    wrap_cli._index_serena_project(verbose=True)  # must not propagate

    assert killed == [proc]


def test_preindex_backgrounds_on_an_unusable_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bad value must not skip the pre-index or start blocking on it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(wrap_cli._SERENA_INDEX_TIMEOUT_ENV, "not-a-number")
    _stub_uvx(monkeypatch)
    _stub_atexit(monkeypatch)
    proc = _FakeProc()
    mock_popen = _stub_popen(monkeypatch, proc)

    wrap_cli._index_serena_project()  # must not propagate

    mock_popen.assert_called_once()
    assert proc.waits == []


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
