"""SIGHUP must run wrap cleanup on every wrap path, not just ``claude``.

Closing a terminal (or ``tmux kill-session``) delivers SIGHUP, not SIGTERM.
``claude()`` learned to catch it in #1768/#3205, but the two shared paths that
every other wrapped tool goes through did not:

* ``_launch_tool`` -- Pattern-A (codex, aider, copilot, goose, openhands, ...)
* ``_run_proxy_only_watcher`` -- Pattern-B (cursor, cline, continue)

Unhandled SIGHUP kills the wrapper outright, so the ``finally: cleanup()`` never
runs. Since #3202 cleanup no longer terminates the proxy itself: it unregisters
this wrapper's client marker, and the wrap-owned proxy's orphan watchdog stops
the proxy once no live markers (and no traffic) remain. A wrapper that dies
without cleanup leaves a marker behind that only goes stale when its PID does,
so SIGHUP handling still decides how promptly the proxy is released.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from headroom.cli import wrap as wrap_cli

requires_sighup = pytest.mark.skipif(not hasattr(signal, "SIGHUP"), reason="SIGHUP is POSIX-only")


@pytest.fixture
def restore_signal_handlers() -> Iterator[None]:
    """Snapshot and restore the handlers the wrap paths install.

    These tests call the real registration code in-process, so without this the
    installed handlers would leak into the rest of the session.
    """
    names = [s for s in ("SIGINT", "SIGTERM", "SIGHUP") if hasattr(signal, s)]
    saved = {getattr(signal, n): signal.getsignal(getattr(signal, n)) for n in names}
    try:
        yield
    finally:
        for sig, handler in saved.items():
            signal.signal(sig, handler)


class _AlreadyExitedProxy:
    """Stand-in proxy that reports as dead, so the watcher loop exits at once."""

    def poll(self) -> int:
        return 0

    def terminate(self) -> None:  # pragma: no cover - not reached
        pass

    def wait(self, timeout: float | None = None) -> int:  # pragma: no cover
        return 0


@requires_sighup
def test_launch_tool_installs_sighup_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_signal_handlers: None
) -> None:
    """Pattern-A tools (codex et al.) share ``_launch_tool``'s handlers.

    Asserts the handler is actually installed by running the real registration,
    rather than pattern-matching the source.
    """
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setattr(wrap_cli, "_ensure_proxy", lambda *a, **k: (None, 18787))
    monkeypatch.setattr(wrap_cli, "_push_runtime_env", lambda *a, **k: None)

    observed: dict[str, Any] = {}

    def _capture_during_child_run(cmd: list[str], **kwargs: Any) -> Any:
        # Sampled where the real wrapper blocks on the child CLI -- the window
        # in which a terminal close actually arrives.
        observed["SIGHUP"] = signal.getsignal(signal.SIGHUP)
        observed["SIGTERM"] = signal.getsignal(signal.SIGTERM)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(wrap_cli.subprocess, "run", _capture_during_child_run)

    with pytest.raises(SystemExit):
        wrap_cli._launch_tool(
            binary=sys.executable,
            args=(),
            env={},
            port=18787,
            no_proxy=False,
            tool_label="sighup-test",
            env_vars_display=[],
        )

    assert observed["SIGHUP"] is wrap_cli._exit_on_signal
    assert observed["SIGTERM"] is wrap_cli._exit_on_signal


@requires_sighup
def test_proxy_only_watcher_installs_sighup_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_signal_handlers: None
) -> None:
    """Pattern-B tools (cursor et al.) share ``_run_proxy_only_watcher``.

    This path already special-cased Windows' SIGBREAK while leaving the POSIX
    terminal-close signal unhandled.
    """
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setattr(wrap_cli, "_ensure_proxy", lambda *a, **k: (_AlreadyExitedProxy(), 18787))
    monkeypatch.setattr(wrap_cli, "_push_runtime_env", lambda *a, **k: None)

    with pytest.raises(SystemExit):
        wrap_cli._run_proxy_only_watcher(
            agent_label="sighup-test",
            port=18787,
            no_proxy=False,
            learn=False,
            memory=False,
            agent_type="unknown",
            print_setup_lines=lambda _port: None,
        )

    # `_signal_shutdown` is a closure, so identity against a module attribute
    # is not available -- assert a real handler replaced the default instead.
    handler = signal.getsignal(signal.SIGHUP)
    assert handler not in (signal.SIG_DFL, signal.SIG_IGN, None)
    assert getattr(handler, "__name__", "") == "_signal_shutdown"
    assert signal.getsignal(signal.SIGTERM) is handler


# Harness driving the real `_launch_tool` under a real SIGHUP. Only
# `_ensure_proxy` is stubbed -- to a live child process standing in for the
# proxy -- so the signal handler, the `finally`, marker registration and
# `_make_cleanup` are all the shipping implementations.
_HARNESS = textwrap.dedent(
    """
    import os, subprocess, sys
    from headroom.cli import wrap

    port = 18787
    proxy = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    with open(sys.argv[1], "w") as fh:
        fh.write(str(proxy.pid))

    wrap._ensure_proxy = lambda *a, **k: (proxy, port)
    wrap._live_proxy_clients = lambda *a, **k: []   # no other clients
    wrap._push_runtime_env = lambda *a, **k: None

    with open(sys.argv[2], "w"):                    # ready
        pass
    wrap._launch_tool(
        binary=sys.executable,
        args=("-c", "import time; time.sleep(300)"),
        env=dict(os.environ),
        port=port,
        no_proxy=False,
        tool_label="sighup-test",
        env_vars_display=[],
    )
    """
)


def _wait_for(predicate: Any, timeout: float = 15.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@requires_sighup
def test_sighup_on_launch_tool_releases_the_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a real SIGHUP runs cleanup, releasing the wrapper's claim on the proxy.

    Cleanup unregisters the client marker, which is what lets the wrap-owned
    proxy's orphan watchdog stop the proxy (#3202). The wrapper itself must not
    kill the proxy: other clients, or in-flight markerless traffic, may still be
    using it.
    """
    from headroom import paths

    harness = tmp_path / "harness.py"
    harness.write_text(_HARNESS, encoding="utf-8")
    pid_file = tmp_path / "proxy.pid"
    ready = tmp_path / "ready"

    env = dict(os.environ)
    env["HEADROOM_WORKSPACE_DIR"] = str(tmp_path / "workspace")
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", env["HEADROOM_WORKSPACE_DIR"])
    marker_dir = paths.proxy_clients_dir(18787)

    # Own session/process group: the wrapper's children (stand-in proxy and
    # stand-in CLI) inherit it, so the cleanup below can reap the whole tree
    # without signalling the pytest process.
    wrapper = subprocess.Popen(
        [sys.executable, str(harness), str(pid_file), str(ready)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        assert _wait_for(ready.exists), "harness never reached _launch_tool"
        proxy_pid = int(pid_file.read_text())
        assert _pid_alive(proxy_pid), "stand-in proxy should be running"
        marker = marker_dir / f"{wrapper.pid}.json"
        assert _wait_for(marker.exists), "wrapper never registered as a proxy client"

        os.kill(wrapper.pid, signal.SIGHUP)

        assert _wait_for(lambda: wrapper.poll() is not None), "wrapper survived SIGHUP"
        assert not marker.exists(), (
            "SIGHUP skipped cleanup: the stale client marker keeps the wrap-owned "
            "proxy's orphan watchdog from ever releasing the proxy"
        )
        assert _pid_alive(proxy_pid), (
            "wrapper killed the proxy on exit; since #3202 the proxy's orphan "
            "watchdog owns shutdown so other clients and in-flight traffic survive"
        )
    finally:
        try:
            os.killpg(wrapper.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        wrapper.wait(timeout=10)
