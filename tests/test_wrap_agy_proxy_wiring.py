"""headroom-508.1: `wrap agy` ensures the shared 8787 proxy (drain -> dashboard).

These tests are FULLY ISOLATED from any real proxy: `_ensure_proxy`,
`_register_proxy_client`, `_make_cleanup`, `ensure_root_ca`,
`build_combined_bundle`, and `shutil.which` are all patched, and `agy()` is
short-circuited at the patched `_ensure_proxy` (before any MITM server or the
real agy launch). A throwaway ``--port`` is passed as a second safeguard so no
code path can contact port 8787. Nothing here starts, probes, or tears down a
real proxy.

Blocker regression guards (from the plan-review gate):
- BLOCKER 1: `_ensure_proxy` must run BEFORE `agy()` poisons `os.environ`
  (HEADROOM_AGY_INBOX_EMIT / HEADROOM_SAVINGS_PATH / HEADROOM_SAVINGS_EVENTS_PATH
  / HEADROOM_OTEL_METRICS_ENABLED), else a freshly-spawned shared proxy inherits
  those via `os.environ.copy()` and corrupts shared state.
- BLOCKER 2: teardown (`cleanup`) is wired into agy's `finally`. (The refcounted
  correctness of `_make_cleanup`/`_register_proxy_client` is agent-agnostic and
  covered in tests/test_cli/test_wrap_helpers.py; agy reuses them unchanged.)
"""

from __future__ import annotations

from typing import Any

import pytest
from click.testing import CliRunner

_POISON_VARS = (
    "HEADROOM_AGY_INBOX_EMIT",
    "HEADROOM_SAVINGS_PATH",
    "HEADROOM_SAVINGS_EVENTS_PATH",
    "HEADROOM_OTEL_METRICS_ENABLED",
)

_THROWAWAY_PORT = "59123"  # never 8787; also unreached because _ensure_proxy is faked


def _get_main() -> Any:
    from headroom.cli import main

    return main


class _StopBeforeLaunch(SystemExit):
    """Raised by the fake _ensure_proxy to short-circuit agy() cleanly."""


def _isolate(monkeypatch: pytest.MonkeyPatch, record: dict) -> None:
    """Patch every collaborator agy() reaches up to and including _ensure_proxy,
    so the command never touches a real proxy/port or the real ~/.headroom."""
    import os

    # agy binary present -> agy() does not bail early
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/agy" if name == "agy" else None)
    # CA setup (lazily imported from headroom.proxy.agy_ca inside agy())
    monkeypatch.setattr(
        "headroom.proxy.agy_ca.ensure_root_ca",
        lambda: (b"kkey", b"ccert", "/tmp/k.pem", "/tmp/c.pem"),
    )
    monkeypatch.setattr("headroom.proxy.agy_ca.build_combined_bundle", lambda: "/tmp/bundle.pem")
    # proxy lifecycle -> no real markers, no real proxy
    monkeypatch.setattr("headroom.cli.wrap._register_proxy_client", lambda port: None)
    monkeypatch.setattr(
        "headroom.cli.wrap._make_cleanup",
        lambda holder, port: record.setdefault("cleanup", _RecordingCleanup()),
    )

    def _fake_ensure_proxy(port: int, no_proxy: bool, **kwargs: Any) -> None:
        # Snapshot env at call time: agy() must not have set the poison vars YET.
        record["env_at_ensure"] = dict(os.environ)
        record["ensure_args"] = {"port": port, "no_proxy": no_proxy, "kwargs": kwargs}
        raise _StopBeforeLaunch(0)

    monkeypatch.setattr("headroom.cli.wrap._ensure_proxy", _fake_ensure_proxy)
    # Clean slate so any poison var in the snapshot can only come from agy().
    for var in _POISON_VARS:
        monkeypatch.delenv(var, raising=False)


def _invoke_agy(*extra_args: str) -> Any:
    """Run `wrap agy` under the isolation harness and assert it got there cleanly.

    ``_fake_ensure_proxy`` short-circuits with ``_StopBeforeLaunch(0)``, so a
    non-zero exit or any other exception means the command died somewhere else —
    in which case the recorded assertions below would be checking a run that
    never happened.
    """
    result = CliRunner().invoke(
        _get_main(), ["wrap", "agy", "--port", _THROWAWAY_PORT, *extra_args]
    )
    assert result.exit_code == 0, (
        f"wrap agy exited {result.exit_code} before the assertion point: "
        f"{result.exception!r}\n{result.output}"
    )
    if result.exception is not None:
        assert isinstance(result.exception, SystemExit), result.exception
    return result


class _RecordingCleanup:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *_a: Any, **_k: Any) -> None:
        self.calls += 1


class TestAgyEnsuresSharedProxy:
    def test_ensure_proxy_runs_before_env_poisoning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record: dict = {}
        _isolate(monkeypatch, record)
        _invoke_agy()

        assert "env_at_ensure" in record, "agy() never reached _ensure_proxy"
        leaked = [v for v in _POISON_VARS if v in record["env_at_ensure"]]
        assert leaked == [], (
            f"env poisoned before _ensure_proxy (would corrupt shared proxy): {leaked}"
        )

    def test_ensure_proxy_called_with_agy_agent_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record: dict = {}
        _isolate(monkeypatch, record)
        _invoke_agy()

        args = record.get("ensure_args", {})
        assert args.get("port") == int(_THROWAWAY_PORT)
        assert args.get("kwargs", {}).get("agent_type") == "agy"
        assert args.get("no_proxy") is False

    def test_no_proxy_flag_is_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record: dict = {}
        _isolate(monkeypatch, record)
        _invoke_agy("--no-proxy")

        assert record.get("ensure_args", {}).get("no_proxy") is True

    def test_code_graph_flag_forwards_to_proxy_watcher(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--code-graph drives the proxy-side watcher, not an agy MCP entry.

        Upstream repurposed ``--code-graph``: every wrap subcommand forwards it
        to ``_ensure_proxy``, which starts the live reindex watcher. agy must
        follow that contract instead of registering codebase-memory-mcp itself.
        """
        record: dict = {}
        _isolate(monkeypatch, record)
        _invoke_agy("--code-graph")

        assert record.get("ensure_args", {}).get("kwargs", {}).get("code_graph") is True

    def test_code_graph_defaults_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record: dict = {}
        _isolate(monkeypatch, record)
        _invoke_agy()

        assert record.get("ensure_args", {}).get("kwargs", {}).get("code_graph") is False

    def test_cleanup_runs_on_teardown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record: dict = {}
        _isolate(monkeypatch, record)
        _invoke_agy()

        cleanup = record.get("cleanup")
        assert cleanup is not None, "_make_cleanup was never built"
        # agy()'s finally must invoke cleanup() even though it short-circuited.
        assert cleanup.calls >= 1, "cleanup() not called on teardown (proxy would leak)"
