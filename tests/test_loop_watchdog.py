"""The event-loop watchdog must name a loop stall the hard watchdog cannot see.

``hard_watchdog`` only fires when native code *seizes* the GIL. #4967 is the
complementary case: the loop thread parked inside a GIL-*releasing* call, so
the heartbeat kept re-arming the hard timer (no dump, no exit) while every
asyncio timeout in the proxy was dead. Nothing was logged for 4m24s.

These tests pin the two properties that fix depends on:

* a loop blocked in a GIL-releasing call is reported in the log, with a stack
  dump that names the blocked thread, and the process keeps running, and
* a healthy loop is never reported.

The stall is simulated by blocking the loop thread in ``time.sleep`` -- a
stand-in for the GIL-releasing native initialization that parks in the report
(a ``PyDLL`` call would hold the GIL, which is the hard watchdog's case).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from headroom.proxy.loop_watchdog import DEFAULT_SECS, MIN_SECS, _resolve_secs, snapshot

_STALLED_LOOP = textwrap.dedent(
    """
    import asyncio, logging, sys, time
    from headroom.proxy.loop_watchdog import start_loop_watchdog, snapshot

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    async def main():
        assert start_loop_watchdog()
        time.sleep(6)  # blocks the loop thread for 6s, GIL released
        print("TICKED", flush=True)
        await asyncio.sleep(0.5)  # one heartbeat observes recovery
        print("SNAPSHOT", snapshot(), flush=True)

    asyncio.run(main())
    """
)

_HEALTHY_LOOP = textwrap.dedent(
    """
    import asyncio, logging, sys
    from headroom.proxy.loop_watchdog import start_loop_watchdog

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    async def main():
        assert start_loop_watchdog()
        await asyncio.sleep(2)  # well past the 1s threshold, loop always ready
        print("SURVIVED", flush=True)

    asyncio.run(main())
    """
)


def _subprocess_env(**extra: str) -> dict[str, str]:
    """Full environment plus overrides.

    A scrubbed env (PATH only, as the POSIX watchdog tests use) breaks
    ``import asyncio`` on Windows -- ``_overlapped`` needs SystemRoot -- so the
    child inherits the real environment and we override just the knob under
    test. ``python -c`` already puts the repo root on ``sys.path``.
    """
    env = dict(os.environ)
    env.update(extra)
    return env


def test_resolve_secs_defaults_and_clamps(monkeypatch):
    monkeypatch.delenv("HEADROOM_LOOP_WATCHDOG_SECS", raising=False)
    assert _resolve_secs() == DEFAULT_SECS

    monkeypatch.setenv("HEADROOM_LOOP_WATCHDOG_SECS", "0")
    assert _resolve_secs() == 0.0
    monkeypatch.setenv("HEADROOM_LOOP_WATCHDOG_SECS", "-3")
    assert _resolve_secs() == 0.0

    # A threshold near scheduler jitter would report healthy loops as stalls.
    monkeypatch.setenv("HEADROOM_LOOP_WATCHDOG_SECS", "0.2")
    assert _resolve_secs() == MIN_SECS

    monkeypatch.setenv("HEADROOM_LOOP_WATCHDOG_SECS", "not-a-number")
    assert _resolve_secs() == DEFAULT_SECS


def test_snapshot_is_inert_before_arming():
    # conftest disables the watchdog for every test; the payload must still be
    # a well-formed dict so /health never depends on the watchdog being on.
    payload = snapshot()
    assert payload["enabled"] is False
    assert payload["current_stall_seconds"] == 0.0
    assert payload["stalls_total"] == 0


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("HEADROOM_LOOP_WATCHDOG_SECS", "0")
    from headroom.proxy import loop_watchdog

    loop_watchdog._reset_for_test()

    async def arm() -> bool:
        return loop_watchdog.start_loop_watchdog()

    import asyncio

    try:
        assert asyncio.run(arm()) is False
        assert loop_watchdog.snapshot()["enabled"] is False
    finally:
        loop_watchdog._reset_for_test()


def test_blocked_loop_is_reported_and_survives():
    proc = subprocess.run(
        [sys.executable, "-c", _STALLED_LOOP],
        env=_subprocess_env(HEADROOM_LOOP_WATCHDOG_SECS="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    # The process must keep serving: this watchdog diagnoses, it never exits.
    assert proc.returncode == 0, proc.stderr
    assert "TICKED" in proc.stdout

    # The stall is named, with the counters /health exposes...
    assert "event=proxy_event_loop_stall" in proc.stderr
    assert "stall_seconds=" in proc.stderr
    # ...and the dump carries every thread's frames, including the loop
    # thread's blocked ``main`` -- which is the whole point: it names the
    # culprit frame the report could never see.
    assert "Thread 0x" in proc.stderr or "Current thread" in proc.stderr
    assert "asyncio" in proc.stderr
    assert "event=proxy_event_loop_stall_recovered" in proc.stderr
    assert "'stalls_total': 1" in proc.stdout
    assert "'current_stall_seconds': 0.0" in proc.stdout


def test_healthy_loop_is_never_reported():
    proc = subprocess.run(
        [sys.executable, "-c", _HEALTHY_LOOP],
        env=_subprocess_env(HEADROOM_LOOP_WATCHDOG_SECS="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "SURVIVED" in proc.stdout
    assert "proxy_event_loop_stall" not in proc.stderr


def test_health_reports_watchdog_and_arming_flips_enabled(monkeypatch):
    """``/health`` carries the counters, and arming makes them live.

    The report could not tell a wedged loop from a wedged worker; these fields
    are how the next one resolves that without guesswork (#4967).
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy import loop_watchdog
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import create_app

    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    monkeypatch.setenv("HEADROOM_LOOP_WATCHDOG_SECS", "5")

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
    )
    try:
        with TestClient(app) as client:
            runtime = client.get("/health").json()["runtime"]
            watchdog = runtime["event_loop_watchdog"]
            assert watchdog["enabled"] is True
            assert watchdog["stall_threshold_seconds"] == pytest.approx(5.0)
            assert watchdog["current_stall_seconds"] == 0.0
            assert watchdog["stalls_total"] == 0
    finally:
        loop_watchdog._reset_for_test()
