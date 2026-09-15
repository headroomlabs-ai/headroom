"""Direct unit tests for the shared wrap-subcommand helpers.

These helpers (`_print_wrap_banner`, `_run_proxy_only_watcher`) were
extracted to remove ~150 LOC of
copy-pasted scaffolding across the wrap subcommands (cursor / cline /
continue / goose / openhands). The wrap-*.py subcommand tests exercise
them indirectly; these tests pin the contract directly so a future
refactor that breaks one of these helpers fails *here* — at the helper
unit boundary — instead of in five different subcommand suites at
once with confusing diffs.
"""

from __future__ import annotations

import errno
import json
import os
import signal
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import click
import pytest
from click.testing import CliRunner

from headroom import paths as paths_mod
from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main

# ---------------------------------------------------------------------------
# _print_wrap_banner — centering math + box drawing.
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _run_in_click_context(fn) -> str:  # type: ignore[no-untyped-def]
    """Invoke `fn` inside a Click `runner.invoke` so `click.echo` output is captured."""
    runner = CliRunner()

    @click.command()
    def _cmd() -> None:
        fn()

    result = runner.invoke(_cmd)
    assert result.exit_code == 0, result.output
    return result.output


@pytest.mark.parametrize(
    "agent",
    ["cline", "cursor", "continue", "goose", "openhands", "x", "a-very-long-agent-name"],
)
def test_print_wrap_banner_box_is_inner_width_chars_wide(agent: str) -> None:
    """The banner's horizontal rule should always be 47 chars between the `║` corners."""
    output = _run_in_click_context(lambda: wrap_mod._print_wrap_banner(agent))

    lines = [line for line in output.splitlines() if line.strip()]
    assert len(lines) == 3, f"banner should be 3 non-empty lines; got {lines!r}"
    top, title_line, bottom = lines

    # Top and bottom are the `╔════...╗` rules with 47 equals signs between the corners.
    assert top.endswith("╗")
    assert bottom.endswith("╝")
    assert top.count("═") == wrap_mod._WRAP_BANNER_INNER_WIDTH
    assert bottom.count("═") == wrap_mod._WRAP_BANNER_INNER_WIDTH

    # The middle line has the centered title.
    assert title_line.startswith("  ║")
    assert title_line.endswith("║")
    assert f"HEADROOM WRAP: {agent.upper()}" in title_line


def test_print_wrap_banner_title_is_centered_or_near_centered() -> None:
    """Centering: pad_left and pad_right may differ by at most 1 when total padding is odd."""
    output = _run_in_click_context(lambda: wrap_mod._print_wrap_banner("cline"))

    lines = [line for line in output.splitlines() if line.strip()]
    title_line = lines[1]

    # Strip the leading "  ║" and trailing "║" so we can measure spaces.
    inner = title_line[3:-1]
    assert len(inner) == wrap_mod._WRAP_BANNER_INNER_WIDTH

    title = "HEADROOM WRAP: CLINE"
    pad_left = len(inner) - len(inner.lstrip(" "))
    pad_right = len(inner) - len(inner.rstrip(" "))
    assert inner.strip() == title
    assert abs(pad_left - pad_right) <= 1, (
        f"banner not centered: pad_left={pad_left}, pad_right={pad_right}"
    )


# ---------------------------------------------------------------------------
# wrap claude argument passthrough.
# ---------------------------------------------------------------------------


def test_wrap_claude_allows_claude_print_short_flag_in_passthrough_args() -> None:
    """Claude owns -p/--print; wrap claude must not parse it as --port."""
    result = CliRunner().invoke(
        main,
        ["wrap", "claude", "--prepare-only", "-p", "Say only: hello"],
    )

    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# _apply_1m_to_claude_args — add the [1m] suffix to an explicit pass-through
# --model so it survives Claude Code's CLI-over-env precedence (#2915).
# ---------------------------------------------------------------------------
def test_apply_1m_rewrites_model_flag_value() -> None:
    args, rewritten = wrap_mod._apply_1m_to_claude_args(("--model", "opusplan"))
    assert args == ("--model", "opusplan[1m]")
    assert rewritten == "opusplan[1m]"


def test_apply_1m_rewrites_equals_model_flag() -> None:
    args, rewritten = wrap_mod._apply_1m_to_claude_args(("--model=opusplan",))
    assert args == ("--model=opusplan[1m]",)
    assert rewritten == "opusplan[1m]"


def test_apply_1m_is_idempotent_on_already_suffixed_model() -> None:
    args, rewritten = wrap_mod._apply_1m_to_claude_args(("--model", "opusplan[1m]"))
    assert args == ("--model", "opusplan[1m]")
    assert rewritten == "opusplan[1m]"


def test_apply_1m_noop_without_model_flag() -> None:
    original = ("--permission-mode", "auto", "--resume")
    args, rewritten = wrap_mod._apply_1m_to_claude_args(original)
    assert args == original
    assert rewritten is None


# ---------------------------------------------------------------------------
# _run_proxy_only_watcher — must print banner, call setup callback, install
# signal handlers, and clean up. Heavily mocked since the real watcher
# blocks on `time.sleep` indefinitely.
# ---------------------------------------------------------------------------


def test_run_proxy_only_watcher_calls_setup_lines_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The print_setup_lines callback runs after the proxy is ready."""
    # Fake proxy: a dummy object whose `.poll()` returns 0 after first iteration
    # so the watcher exits cleanly via the "proxy exited unexpectedly" branch.

    class _FakeProc:
        def __init__(self) -> None:
            self._polls = 0

        def poll(self) -> int | None:
            self._polls += 1
            return 0 if self._polls > 1 else None

    fake_proc = _FakeProc()

    callback_calls: list[None] = []

    def fake_setup(_port: int) -> None:
        callback_calls.append(None)

    monkeypatch.setattr(wrap_mod, "_ensure_proxy", lambda *a, **kw: (fake_proc, 8787))
    # Replace time.sleep with a no-op so the loop spins quickly.
    monkeypatch.setattr(wrap_mod.time, "sleep", lambda _s: None)
    # Replace _make_cleanup to avoid side-effects on real ports/files.
    monkeypatch.setattr(wrap_mod, "_make_cleanup", lambda holder, port: lambda *a, **kw: None)
    # Avoid touching real signal handlers in the test process.
    monkeypatch.setattr(wrap_mod.signal, "signal", lambda *a, **kw: None)

    runner = CliRunner()

    @click.command()
    def _cmd() -> None:
        wrap_mod._run_proxy_only_watcher(
            agent_label="cline",
            port=8787,
            no_proxy=False,
            learn=False,
            memory=False,
            agent_type="cline",
            print_setup_lines=fake_setup,
        )

    inv = runner.invoke(_cmd)
    # The watcher exits 1 when the proxy dies (our _FakeProc returns 0 on poll #2).
    assert inv.exit_code == 1
    assert callback_calls == [None]
    # Banner is part of the helper's contract.
    assert "HEADROOM WRAP: CLINE" in inv.output
    # The "proxy exited unexpectedly" message is the documented exit branch.
    assert "Proxy process exited unexpectedly." in inv.output


def test_run_proxy_only_watcher_keyboardinterrupt_shuts_down_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C during the watcher loop prints `Shutting down...` and exits 0."""

    class _FakeProc:
        def poll(self) -> int | None:
            return None  # Proxy is healthy; loop would run forever.

    sleep_calls = {"n": 0}

    def raising_sleep(_s: float) -> None:
        sleep_calls["n"] += 1
        if sleep_calls["n"] >= 1:
            raise KeyboardInterrupt

    monkeypatch.setattr(wrap_mod, "_ensure_proxy", lambda *a, **kw: (_FakeProc(), 8787))
    monkeypatch.setattr(wrap_mod.time, "sleep", raising_sleep)
    monkeypatch.setattr(wrap_mod, "_make_cleanup", lambda holder, port: lambda *a, **kw: None)
    monkeypatch.setattr(wrap_mod.signal, "signal", lambda *a, **kw: None)

    runner = CliRunner()

    @click.command()
    def _cmd() -> None:
        wrap_mod._run_proxy_only_watcher(
            agent_label="cursor",
            port=8787,
            no_proxy=False,
            learn=False,
            memory=False,
            agent_type="cursor",
            print_setup_lines=lambda _port: None,
        )

    inv = runner.invoke(_cmd)
    assert inv.exit_code == 0, inv.output
    assert "Shutting down..." in inv.output


def test_run_proxy_only_watcher_signal_handler_uses_clean_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Windows console stop handlers must use the clean shutdown path."""

    handlers: dict[int, Any] = {}
    cleanup_calls = {"n": 0}

    class _FakeProc:
        def poll(self) -> None:
            return None

    def capture_handler(sig: int, handler: Any) -> None:
        handlers[sig] = handler

    def trigger_sigint(_seconds: float) -> None:
        handlers[signal.SIGINT](signal.SIGINT, None)

    def cleanup(*_args: Any) -> None:
        cleanup_calls["n"] += 1

    monkeypatch.setattr(wrap_mod, "_ensure_proxy", lambda *a, **kw: (_FakeProc(), 8787))
    monkeypatch.setattr(wrap_mod.time, "sleep", trigger_sigint)
    monkeypatch.setattr(wrap_mod, "_make_cleanup", lambda holder, port: cleanup)
    monkeypatch.setattr(wrap_mod.signal, "signal", capture_handler)
    monkeypatch.setattr(wrap_mod.sys, "platform", "win32")
    sigbreak = 999
    monkeypatch.setattr(wrap_mod.signal, "SIGBREAK", sigbreak, raising=False)

    runner = CliRunner()

    @click.command()
    def _cmd() -> None:
        wrap_mod._run_proxy_only_watcher(
            agent_label="vscode copilot",
            port=8787,
            no_proxy=False,
            learn=False,
            memory=False,
            agent_type="copilot",
            print_setup_lines=lambda _port: None,
        )

    inv = runner.invoke(_cmd)
    assert inv.exit_code == 0, inv.output
    assert "Shutting down..." in inv.output
    assert "Proxy process exited unexpectedly" not in inv.output
    assert sigbreak in handlers
    assert cleanup_calls["n"] >= 2  # signal handler plus finally (idempotent)


def test_run_proxy_only_watcher_unexpected_exception_returns_exit_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unexpected exceptions in the body are caught and converted to SystemExit(1)."""

    def boom(*a: Any, **kw: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(wrap_mod, "_ensure_proxy", boom)
    monkeypatch.setattr(wrap_mod, "_make_cleanup", lambda holder, port: lambda *a, **kw: None)
    monkeypatch.setattr(wrap_mod.signal, "signal", lambda *a, **kw: None)

    runner = CliRunner()

    @click.command()
    def _cmd() -> None:
        wrap_mod._run_proxy_only_watcher(
            agent_label="cline",
            port=8787,
            no_proxy=False,
            learn=False,
            memory=False,
            agent_type="cline",
            print_setup_lines=lambda _port: None,
        )

    inv = runner.invoke(_cmd)
    assert inv.exit_code == 1
    assert "Error: boom" in inv.output


def test_run_proxy_only_watcher_calls_cleanup_on_finally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup callable is invoked in the `finally` block regardless of exit path."""

    cleanup_calls = {"n": 0}

    def fake_cleanup(*a: Any, **kw: Any) -> None:
        cleanup_calls["n"] += 1

    def boom(*a: Any, **kw: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(wrap_mod, "_ensure_proxy", boom)
    monkeypatch.setattr(wrap_mod, "_make_cleanup", lambda holder, port: fake_cleanup)
    monkeypatch.setattr(wrap_mod.signal, "signal", lambda *a, **kw: None)

    runner = CliRunner()

    @click.command()
    def _cmd() -> None:
        wrap_mod._run_proxy_only_watcher(
            agent_label="cline",
            port=8787,
            no_proxy=False,
            learn=False,
            memory=False,
            agent_type="cline",
            print_setup_lines=lambda _port: None,
        )

    inv = runner.invoke(_cmd)
    assert inv.exit_code == 1
    assert cleanup_calls["n"] >= 1, "cleanup must run via the finally block"


# ---------------------------------------------------------------------------
# _project_name_from_cwd / _apply_project_header_env — per-project savings
# header injection for `headroom wrap claude` (issue: per-project savings).
# ---------------------------------------------------------------------------


class TestApplyProjectHeaderEnv:
    """X-Headroom-Project injection into ANTHROPIC_CUSTOM_HEADERS."""

    def test_sets_header_from_cwd_basename(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "my-project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        env: dict[str, str] = {}
        wrap_mod._apply_project_header_env(env)

        assert env["ANTHROPIC_CUSTOM_HEADERS"] == "X-Headroom-Project: my-project"

    def test_appends_to_existing_custom_headers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        env = {"ANTHROPIC_CUSTOM_HEADERS": "X-Custom-Trace: abc123"}
        wrap_mod._apply_project_header_env(env)

        # User header preserved verbatim, ours appended on a new line.
        assert env["ANTHROPIC_CUSTOM_HEADERS"] == (
            "X-Custom-Trace: abc123\nX-Headroom-Project: proj"
        )

    @pytest.mark.parametrize(
        "user_value",
        [
            "X-Headroom-Project: their-name",
            "x-headroom-project: their-name",
            "X-HEADROOM-PROJECT: their-name",
            "X-Other: 1\nx-Headroom-Project: their-name",
        ],
    )
    def test_existing_project_header_wins_case_insensitive(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        user_value: str,
    ) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        env = {"ANTHROPIC_CUSTOM_HEADERS": user_value}
        wrap_mod._apply_project_header_env(env)

        # Untouched: no duplicate header, user override wins.
        assert env["ANTHROPIC_CUSTOM_HEADERS"] == user_value

    @pytest.mark.parametrize(
        "user_value",
        [
            "X-Headroom-Project-Id: other",
            "X-Trace: mentions x-headroom-project in the value",
        ],
    )
    def test_similar_header_names_do_not_suppress_injection(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        user_value: str,
    ) -> None:
        project_dir = tmp_path / "proj"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        env = {"ANTHROPIC_CUSTOM_HEADERS": user_value}
        wrap_mod._apply_project_header_env(env)

        # Only an exact header-name match counts as a user override.
        assert env["ANTHROPIC_CUSTOM_HEADERS"] == (f"{user_value}\nX-Headroom-Project: proj")

    def test_empty_cwd_name_sets_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A degenerate cwd (e.g. filesystem root → empty basename) is a no-op."""
        monkeypatch.setattr(wrap_mod.Path, "cwd", classmethod(lambda cls: Path("/")))

        env: dict[str, str] = {}
        wrap_mod._apply_project_header_env(env)

        assert "ANTHROPIC_CUSTOM_HEADERS" not in env

    def test_whitespace_only_cwd_name_sets_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(wrap_mod.Path, "cwd", classmethod(lambda cls: Path("/tmp/   ")))

        env: dict[str, str] = {}
        wrap_mod._apply_project_header_env(env)

        assert "ANTHROPIC_CUSTOM_HEADERS" not in env

    def test_project_name_from_cwd_returns_basename(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        project_dir = tmp_path / "vibe-headroom"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        assert wrap_mod._project_name_from_cwd() == "vibe-headroom"

    def test_non_ascii_cwd_name_is_percent_encoded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Non-ASCII directory names must be percent-encoded for HTTP headers."""
        project_dir = tmp_path / "第二大脑共享"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        result = wrap_mod._project_name_from_cwd()
        assert result is not None
        # Must be pure ASCII so it's safe in an HTTP header value.
        result.encode("ascii")
        # Must round-trip back to the original name via unquote.
        import urllib.parse

        assert urllib.parse.unquote(result) == "第二大脑共享"

    def test_non_ascii_cwd_header_is_ascii_safe(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """X-Headroom-Project header value must be ASCII when cwd has non-ASCII chars."""
        project_dir = tmp_path / "test-中文-项目"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        env: dict[str, str] = {}
        wrap_mod._apply_project_header_env(env)

        header_value = env["ANTHROPIC_CUSTOM_HEADERS"]
        assert header_value.startswith("X-Headroom-Project: ")
        header_value.encode("ascii")  # raises UnicodeEncodeError if non-ASCII


# ---------------------------------------------------------------------------
# Proxy-client reference counting
#
# The shared proxy must only be torn down by its owner once *no* other live
# wrap clients remain. Clients carry the proxy URL in ANTHROPIC_BASE_URL /
# OPENAI_BASE_URL (env, not argv), so the old `pgrep -f "127.0.0.1:<port>"`
# guard could neither see real clients nor reject unrelated processes that
# merely had the address in their command line. These tests pin the new
# marker-file contract: a per-PID file under paths.proxy_clients_dir(port).
# ---------------------------------------------------------------------------


class _FakeProxyProc:
    """Minimal stand-in for the proxy ``subprocess.Popen`` handle."""

    def __init__(self) -> None:
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return None  # alive

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        return 0

    def kill(self) -> None:
        self.killed = True


class TestProxyClientRefCounting:
    """Proxy lifecycle is reference-counted via marker files, not pgrep."""

    PORT = 8787

    @pytest.fixture
    def clients_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Redirect ``paths.proxy_clients_dir`` into a throwaway tmp tree."""
        base = tmp_path / "clients"
        monkeypatch.setattr(paths_mod, "proxy_clients_dir", lambda port: base / str(port))
        return base

    def _write_marker(
        self,
        clients_dir: Path,
        pid: int,
        *,
        identity: tuple[str, float] | None = None,
    ) -> Path:
        marker = clients_dir / str(self.PORT) / f"{pid}.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        rec: dict[str, Any] = {"pid": pid, "started_at": 0}
        if identity is not None:
            rec["start_src"], rec["start_time"] = identity
        marker.write_text(json.dumps(rec))
        return marker

    def test_cleanup_terminates_proxy_when_only_self_registered(self, clients_dir: Path) -> None:
        """The owner alone → no other clients → proxy is terminated on exit."""
        wrap_mod._register_proxy_client(self.PORT)
        proc = _FakeProxyProc()
        cleanup = wrap_mod._make_cleanup([proc], self.PORT)

        cleanup()

        assert proc.terminated is True
        # Our own marker is removed before we count.
        assert wrap_mod._live_proxy_clients(self.PORT, exclude_self=False) == []

    def test_cleanup_stops_detached_windows_serving_child(
        self, clients_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ctrl+C must stop the listener even when its launcher already exited."""
        wrap_mod._register_proxy_client(self.PORT)
        proc = _FakeProxyProc()
        proc.poll = lambda: 0  # type: ignore[method-assign]
        stopped: list[int] = []
        monkeypatch.setattr(wrap_mod.sys, "platform", "win32")
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: port == self.PORT)
        monkeypatch.setattr(wrap_mod, "_query_proxy_config", lambda port: {"pid": 123})
        monkeypatch.setattr(
            wrap_mod,
            "_stop_local_proxy_for_unwrap",
            lambda port: stopped.append(port) or "stopped",
        )

        wrap_mod._make_cleanup([proc], self.PORT)()

        assert not proc.terminated
        assert stopped == [self.PORT]

    def test_kill_proxy_uses_taskkill_tree_on_windows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Windows cleanup must terminate the native launcher's whole tree."""
        calls: list[tuple[list[str], dict[str, object]]] = []
        checks = iter([True, False])
        monkeypatch.setattr(wrap_mod.sys, "platform", "win32")
        monkeypatch.setattr(wrap_mod.time, "sleep", lambda _seconds: None)
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: next(checks))
        monkeypatch.setattr(
            wrap_mod.subprocess,
            "run",
            lambda command, **kwargs: calls.append((command, kwargs)),
        )

        assert wrap_mod._kill_proxy_by_pid(456, self.PORT)
        assert calls == [
            (
                ["taskkill", "/F", "/T", "/PID", "456"],
                {"capture_output": True, "timeout": 10, "check": False},
            )
        ]

    def test_cleanup_uses_pre_shutdown_pid_when_health_probe_races(
        self, clients_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A transient post-terminate /health miss must not orphan the listener."""
        wrap_mod._register_proxy_client(self.PORT)
        proc = _FakeProxyProc()
        killed: list[tuple[int, int]] = []
        monkeypatch.setattr(wrap_mod.sys, "platform", "win32")
        monkeypatch.setattr(wrap_mod, "_check_proxy", lambda port: port == self.PORT)
        monkeypatch.setattr(wrap_mod, "_query_proxy_config", lambda port: {"pid": 456})
        monkeypatch.setattr(wrap_mod, "_stop_local_proxy_for_unwrap", lambda port: "unidentified")
        monkeypatch.setattr(
            wrap_mod,
            "_kill_proxy_by_pid",
            lambda pid, port: killed.append((pid, port)) or True,
        )

        wrap_mod._make_cleanup([proc], self.PORT)()

        assert proc.terminated
        assert killed == [(456, self.PORT)]

    def test_cleanup_leaves_proxy_running_when_other_client_alive(self, clients_dir: Path) -> None:
        """A second live client (here: the test's parent) keeps the proxy up."""
        wrap_mod._register_proxy_client(self.PORT)
        other_pid = os.getppid()  # alive for the duration of the test run
        assert other_pid != os.getpid()
        self._write_marker(clients_dir, other_pid)

        proc = _FakeProxyProc()
        cleanup = wrap_mod._make_cleanup([proc], self.PORT)
        cleanup()

        assert proc.terminated is False

    def test_dead_client_marker_is_pruned_and_not_counted(
        self, clients_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A marker for a dead PID is pruned from disk and never counted."""
        dead_pid = 358784
        marker = self._write_marker(clients_dir, dead_pid)
        monkeypatch.setattr(wrap_mod, "_pid_alive", lambda pid: pid != dead_pid)

        live = wrap_mod._live_proxy_clients(self.PORT, exclude_self=True)

        assert dead_pid not in live
        assert not marker.exists()

    def test_reused_pid_with_mismatched_identity_is_pruned(
        self, clients_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A *live* PID that the original client no longer owns is pruned.

        Models the PID-reuse orphan path: a wrapper crashed, the OS later
        recycled its PID for an unrelated long-lived process. `os.kill(pid, 0)`
        succeeds, but the recorded start time no longer matches.
        """
        live_pid = os.getppid()  # alive, but not the process that "registered"
        marker = self._write_marker(clients_dir, live_pid, identity=("psutil", 1000.0))
        # The process currently holding that PID started much later → reuse.
        monkeypatch.setattr(wrap_mod, "_proc_identity", lambda p: ("psutil", 9000.0))

        live = wrap_mod._live_proxy_clients(self.PORT, exclude_self=True)

        assert live_pid not in live
        assert not marker.exists()

    def test_matching_identity_within_tolerance_is_kept(
        self, clients_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same process (start time within tolerance) is a real client — kept."""
        live_pid = os.getppid()
        marker = self._write_marker(clients_dir, live_pid, identity=("psutil", 1000.0))
        monkeypatch.setattr(wrap_mod, "_proc_identity", lambda p: ("psutil", 1000.4))

        live = wrap_mod._live_proxy_clients(self.PORT, exclude_self=True)

        assert live_pid in live
        assert marker.exists()

    def test_identity_check_skipped_when_source_unavailable(
        self, clients_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No reuse protection (e.g. macOS w/o psutil) → fall back to existence."""
        live_pid = os.getppid()
        self._write_marker(clients_dir, live_pid, identity=("psutil", 1000.0))
        # Start time unknowable for the live PID → must not prune a real client.
        monkeypatch.setattr(wrap_mod, "_proc_identity", lambda p: None)

        live = wrap_mod._live_proxy_clients(self.PORT, exclude_self=True)

        assert live_pid in live

    def test_non_marker_files_are_ignored(self, clients_dir: Path) -> None:
        """Stray non-numeric / non-json files don't crash or count as clients."""
        d = clients_dir / str(self.PORT)
        d.mkdir(parents=True, exist_ok=True)
        (d / "not-a-pid.json").write_text("{}")
        (d / "README.txt").write_text("ignore me")

        assert wrap_mod._live_proxy_clients(self.PORT, exclude_self=True) == []

    def test_cleanup_does_not_shell_out_to_pgrep(
        self, clients_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: liveness is never inferred from an argv scan.

        An unrelated process whose command line contains ``127.0.0.1:8787``
        used to be a false positive (orphaning the proxy). The new path never
        calls ``subprocess.run`` at all, so it can't be fooled by argv.
        """
        wrap_mod._register_proxy_client(self.PORT)

        def _no_subprocess(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("cleanup must not invoke subprocess.run (no pgrep)")

        monkeypatch.setattr(wrap_mod.subprocess, "run", _no_subprocess)

        proc = _FakeProxyProc()
        cleanup = wrap_mod._make_cleanup([proc], self.PORT)
        cleanup()  # must not raise

        assert proc.terminated is True

    def test_register_then_unregister_is_idempotent(self, clients_dir: Path) -> None:
        """Register adds exactly our marker; unregister removes it; re-call is safe."""
        wrap_mod._register_proxy_client(self.PORT)
        all_clients = wrap_mod._live_proxy_clients(self.PORT, exclude_self=False)
        assert all_clients == [os.getpid()]

        wrap_mod._unregister_proxy_client(self.PORT)
        assert wrap_mod._live_proxy_clients(self.PORT, exclude_self=False) == []

        # Second unregister is a no-op, not an error.
        wrap_mod._unregister_proxy_client(self.PORT)


# ---------------------------------------------------------------------------
# _ensure_proxy — dashboard URL is surfaced even when the proxy is already up.
# ---------------------------------------------------------------------------


def test_ensure_proxy_already_running_prints_dashboard_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a healthy proxy is already running, the dashboard URL is printed.

    Regression: the URL was only echoed on the start/restart path, so repeat
    wraps (the common case) never told the user where the dashboard lives.
    """
    port = 1234
    # A compatible running proxy exposes its config; proxies without a config
    # block are no longer reused (they cannot be compatibility-checked) and
    # are covered by test_ensure_proxy_never_reuses_configless_proxy.
    running_config = {"pid": 4321, "backend": "anthropic"}
    monkeypatch.setattr(wrap_mod, "_find_persistent_manifest", lambda _p: None)
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _p: True)
    monkeypatch.setattr(wrap_mod, "_query_proxy_health", lambda _p: {})
    monkeypatch.setattr(wrap_mod, "_proxy_needs_version_restart", lambda _h: False)
    monkeypatch.setattr(wrap_mod, "_proxy_health_config", lambda _h: running_config)
    monkeypatch.setattr(wrap_mod, "_live_proxy_clients", lambda *a, **kw: [])

    output = _run_in_click_context(lambda: wrap_mod._ensure_proxy(port, no_proxy=False))

    assert f"http://127.0.0.1:{port}/dashboard" in output


# ---------------------------------------------------------------------------
# _resolve_1m_model — 1M context window suffix logic (#1158).
# ---------------------------------------------------------------------------


def test_resolve_1m_model_appends_suffix_to_user_model() -> None:
    """A model the user already selected via ANTHROPIC_MODEL is preserved, with
    only the [1m] suffix appended so Claude Code requests the 1M window."""
    assert wrap_mod._resolve_1m_model("claude-opus-4-1-20250805") == (
        "claude-opus-4-1-20250805[1m]"
    )


def test_resolve_1m_model_is_idempotent() -> None:
    """A model that already carries [1m] is returned unchanged (no double suffix)."""
    assert wrap_mod._resolve_1m_model("claude-opus-4-8[1m]") == "claude-opus-4-8[1m]"


def test_resolve_1m_model_falls_back_to_default_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no model selected, fall back to the built-in default carrying [1m]."""
    monkeypatch.delenv("HEADROOM_1M_MODEL", raising=False)
    expected = f"{wrap_mod._DEFAULT_1M_MODEL}[1m]"
    assert wrap_mod._resolve_1m_model(None) == expected
    assert wrap_mod._resolve_1m_model("  ") == expected


def test_resolve_1m_model_env_overrides_builtin_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """HEADROOM_1M_MODEL overrides the built-in fallback so --1m can track new
    Opus releases without a code change or pinning ANTHROPIC_MODEL (#2937)."""
    monkeypatch.setenv("HEADROOM_1M_MODEL", "claude-opus-9")
    assert wrap_mod._resolve_1m_model(None) == "claude-opus-9[1m]"


def test_resolve_1m_model_current_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit ANTHROPIC_MODEL still wins; HEADROOM_1M_MODEL is only the
    fallback default when nothing else is selected."""
    monkeypatch.setenv("HEADROOM_1M_MODEL", "claude-opus-9")
    assert wrap_mod._resolve_1m_model("claude-sonnet-5") == "claude-sonnet-5[1m]"


def test_resolve_1m_model_env_idempotent_on_suffixed_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A HEADROOM_1M_MODEL that already carries [1m] is not double-suffixed."""
    monkeypatch.setenv("HEADROOM_1M_MODEL", "claude-opus-9[1m]")
    assert wrap_mod._resolve_1m_model(None) == "claude-opus-9[1m]"


def test_resolve_1m_model_blank_env_falls_back_to_builtin(monkeypatch: pytest.MonkeyPatch) -> None:
    """A blank/whitespace HEADROOM_1M_MODEL falls back to the built-in default."""
    monkeypatch.setenv("HEADROOM_1M_MODEL", "   ")
    assert wrap_mod._resolve_1m_model(None) == f"{wrap_mod._DEFAULT_1M_MODEL}[1m]"


def test_headroom_1m_model_is_documented_and_default_matches_code() -> None:
    """The HEADROOM_1M_MODEL knob must stay documented, and the documented
    default must track the code, so the supported configuration surface cannot
    silently drift or disappear (#2937).
    """
    docs = Path(__file__).resolve().parents[2] / "docs" / "content" / "docs" / "configuration.mdx"
    text = docs.read_text(encoding="utf-8")
    assert wrap_mod._1M_MODEL_ENV in text, f"{wrap_mod._1M_MODEL_ENV} is not documented"
    # The env-var catalog row must advertise the current built-in default.
    assert f"`{wrap_mod._DEFAULT_1M_MODEL}`" in text, (
        "documented HEADROOM_1M_MODEL default is out of sync with "
        f"_DEFAULT_1M_MODEL={wrap_mod._DEFAULT_1M_MODEL!r}"
    )


class TestFindAvailablePort:
    """Tests for _find_available_port (Vite-style port fallback)."""

    def test_port_free_returns_same(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When port is free, returns the same port."""
        monkeypatch.setattr(wrap_mod, "_port_bind_error", lambda port: None)
        assert wrap_mod._find_available_port(8787) == 8787

    def test_port_busy_finds_next(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When port is busy, returns the next free port."""

        def mock_bind(port: int) -> OSError | None:
            if port == 8787:
                return OSError(errno.EADDRINUSE, "Address in use")
            return None

        monkeypatch.setattr(wrap_mod, "_port_bind_error", mock_bind)
        assert wrap_mod._find_available_port(8787) == 8788

    def test_multiple_busy_ports(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When multiple consecutive ports are busy, skips all of them."""

        def mock_bind(port: int) -> OSError | None:
            if port in (8787, 8788, 8789):
                return OSError(errno.EADDRINUSE, "Address in use")
            return None

        monkeypatch.setattr(wrap_mod, "_port_bind_error", mock_bind)
        assert wrap_mod._find_available_port(8787) == 8790

    def test_propagates_unexpected_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Errors other than EADDRINUSE/EACCES (e.g. EADDRNOTAVAIL) propagate."""
        monkeypatch.setattr(
            wrap_mod,
            "_port_bind_error",
            lambda port: OSError(errno.EADDRNOTAVAIL, "Address not available"),
        )
        with pytest.raises(OSError, match="Address not available"):
            wrap_mod._find_available_port(8787)

    def test_propagates_eaddrinuse_with_eacces(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Both EADDRINUSE and EACCES are skipped (not propagated)."""

        def mock_bind(port: int) -> OSError | None:
            if port == 8787:
                return OSError(errno.EACCES, "Permission denied")
            return None

        monkeypatch.setattr(wrap_mod, "_port_bind_error", mock_bind)
        assert wrap_mod._find_available_port(8787) == 8788

    def test_exhausts_range(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When all ports in range are busy, raises RuntimeError."""
        monkeypatch.setattr(
            wrap_mod,
            "_port_bind_error",
            lambda port: OSError(errno.EADDRINUSE, "Address in use"),
        )
        with pytest.raises(RuntimeError, match="No available port found"):
            wrap_mod._find_available_port(8787, max_attempts=3)


def test_ensure_proxy_serializes_startup_per_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """A normal wrap must enter the per-port startup critical section."""
    events: list[object] = []

    @contextmanager
    def fake_lock(port: int):
        events.append(("lock-enter", port))
        try:
            yield
        finally:
            events.append(("lock-exit", port))

    monkeypatch.setattr(wrap_mod, "_proxy_start_lock", fake_lock)
    monkeypatch.setattr(
        wrap_mod,
        "_ensure_proxy_unlocked",
        lambda port, no_proxy, **kwargs: events.append(("ensure", port, no_proxy)) or (None, port),
    )

    assert wrap_mod._ensure_proxy(8787, False) == (None, 8787)
    assert events == [("lock-enter", 8787), ("ensure", 8787, False), ("lock-exit", 8787)]


def test_no_proxy_does_not_create_startup_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit --no-proxy reuses an existing service without taking the lock."""
    entered = False

    @contextmanager
    def fail_lock(port: int):
        nonlocal entered
        entered = True
        yield

    monkeypatch.setattr(wrap_mod, "_proxy_start_lock", fail_lock)
    monkeypatch.setattr(wrap_mod, "_ensure_proxy_unlocked", lambda *args, **kwargs: (None, 8787))

    assert wrap_mod._ensure_proxy(8787, True) == (None, 8787)
    assert entered is False
