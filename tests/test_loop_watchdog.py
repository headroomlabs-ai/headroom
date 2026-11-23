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

import asyncio
import json
import logging
import math
import os
import subprocess
import sys
import textwrap
from typing import Any

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

# A logging handler whose lock is held by the blocked loop thread -- the shape of
# the proxy's synchronous rotating file handler with an unresponsive filesystem.
# The monitor must still reach its stack dump (#4967 review).
_STALLED_LOOP_WITH_HELD_LOG_LOCK = textwrap.dedent(
    """
    import asyncio, logging, os, sys, time, threading
    from headroom.proxy.loop_watchdog import start_loop_watchdog

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    class BlockingHandler(logging.Handler):
        def __init__(self):
            super().__init__()
            self.armed = False
            self.released = False

        def emit(self, record):
            # Only the loop thread's records block, and only once armed; the
            # reporter thread must not add a second confounder, and arming must
            # not deadlock start_loop_watchdog's own "armed" line.
            if self.armed and threading.current_thread().name == "MainThread":
                while not self.released:
                    time.sleep(0.05)

    handler = BlockingHandler()
    logging.getLogger().addHandler(handler)

    def mark_later(delay, text):
        # Timestamp a point in the stream without touching logging.
        time.sleep(delay)
        os.write(2, (text + "MARKLINE").encode())

    async def main():
        assert start_loop_watchdog()
        await asyncio.sleep(0.2)  # let the reporter thread start
        threading.Thread(target=mark_later, args=(3.5, "STALL_MARK"), daemon=True).start()
        handler.armed = True
        handler.acquire()  # loop thread holds the handler lock while it stalls
        time.sleep(6)
        handler.released = True
        handler.release()
        print("TICKED", flush=True)
        await asyncio.sleep(0.5)

    asyncio.run(main())
    """
)

# Start, stop, start again inside one interpreter: the second lifespan must
# still detect a blocked loop (review: the old start guard was terminal).
_RESTART_LIFESPAN = textwrap.dedent(
    """
    import asyncio, logging, sys, time
    from headroom.proxy.loop_watchdog import start_loop_watchdog, stop_loop_watchdog

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    async def lifespan(marker):
        assert start_loop_watchdog(), marker
        time.sleep(3)  # block the loop: this lifespan must report a stall
        await asyncio.sleep(0.3)
        print(marker, flush=True)
        stop_loop_watchdog()

    async def main():
        await lifespan("FIRST")
        await lifespan("SECOND")

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


def test_resolve_secs_rejects_non_finite(monkeypatch):
    """``nan``/``inf`` parse fine and must not be allowed to arm.

    A nan threshold makes every ``stalled_for >= threshold`` comparison false, so
    the watchdog arms and silently never reports; and the value reaches
    ``/health`` and ``/readyz``, where ``json.dumps`` emits a bare ``NaN`` token
    that strict parsers reject. Falling back to the finite default keeps both
    true.
    """
    for raw in ("nan", "NaN", "-nan", "inf", "-inf", "Infinity", "1e999"):
        monkeypatch.setenv("HEADROOM_LOOP_WATCHDOG_SECS", raw)
        assert math.isfinite(_resolve_secs()), raw
        assert _resolve_secs() == DEFAULT_SECS, raw


def test_snapshot_is_strict_json():
    """Nothing non-finite may reach the HTTP payload, even if state goes bad.

    ``/health`` and ``/readyz`` serve ``snapshot()`` verbatim and both are what
    orchestration probes parse. ``round(nan, 2)`` stays nan, so the guard has to
    be explicit.
    """
    from headroom.proxy import loop_watchdog

    assert json.dumps(snapshot(), allow_nan=False)
    loop_watchdog._reset_for_test()
    try:
        with loop_watchdog._stats_lock:
            loop_watchdog._threshold_seconds = float("nan")
            loop_watchdog._stall_seconds_max = float("nan")
            loop_watchdog._last_stall_seconds = float("inf")
        payload = snapshot()
        assert json.dumps(payload, allow_nan=False)  # must not raise
        for key in (
            "stall_threshold_seconds",
            "current_stall_seconds",
            "stall_seconds_max",
            "last_stall_seconds",
        ):
            assert math.isfinite(payload[key]), key
    finally:
        loop_watchdog._reset_for_test()


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


def test_stack_dump_survives_a_wedged_log_handler():
    """Logging must never gate the dump.

    The proxy installs a synchronous rotating file handler. If the loop thread
    blocks inside it, it holds the handler lock -- so a monitor that logs before
    dumping would wait on that same lock and produce nothing for the duration of
    the stall, which is exactly the outage this watchdog exists to explain. The
    dump therefore comes first and the record goes to a reporter thread.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _STALLED_LOOP_WITH_HELD_LOG_LOCK],
        env=_subprocess_env(HEADROOM_LOOP_WATCHDOG_SECS="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "TICKED" in proc.stdout

    # The discriminating assertion is the *order*: the dump must land while the
    # loop still holds the logging lock (marker written at 3.5s, lock released at
    # 6.2s). Logging-then-dumping writes it after the marker instead.
    stderr = proc.stderr
    dump_at = stderr.find("Thread 0x")
    assert dump_at >= 0, stderr
    assert "asyncio" in stderr[dump_at:]
    mark_at = stderr.find("STALL_MARK")
    assert mark_at >= 0, stderr
    assert dump_at < mark_at, (
        "stack dump did not precede the marker: the monitor waited for the "
        + "logging handler lock instead of dumping first\n"
        + stderr
    )


def test_watchdog_restarts_with_a_second_lifespan():
    """Stop then start inside one interpreter must re-arm, not no-op.

    An embedded app (or a test harness) can run two proxy lifespans in one
    process; a terminal start guard left the second one unmonitored.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _RESTART_LIFESPAN],
        env=_subprocess_env(HEADROOM_LOOP_WATCHDOG_SECS="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert "FIRST" in proc.stdout and "SECOND" in proc.stdout
    assert proc.stderr.count("event=proxy_event_loop_stall pid=") == 2, proc.stderr


def test_health_and_readyz_are_valid_json_with_nan_threshold(monkeypatch):
    """Both probes must stay parseable when the env var is non-finite."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy import loop_watchdog
    from headroom.proxy.models import ProxyConfig
    from headroom.proxy.server import create_app

    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    monkeypatch.setenv("HEADROOM_LOOP_WATCHDOG_SECS", "nan")
    loop_watchdog._reset_for_test()

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
            for path in ("/health", "/readyz"):
                response = client.get(path)
                # allow_nan=False is the parser contract orchestration uses.
                body = json.loads(json.dumps(response.json(), allow_nan=False))
                watchdog = body["runtime"]["event_loop_watchdog"]
                assert watchdog["stall_threshold_seconds"] == pytest.approx(DEFAULT_SECS)
                for key, value in watchdog.items():
                    if isinstance(value, float):
                        assert math.isfinite(value), (path, key)
    finally:
        loop_watchdog._reset_for_test()


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


def test_retry_after_startup_failure_rearms_on_the_new_loop(monkeypatch, caplog):
    """A lifespan that fails after arming must not poison the next one.

    The supervisor path (``_check_rust_core`` refusing, the observability gate,
    any startup check between arming and the lifespan's own try/finally) can
    raise with the watchdog already armed. The loop then closes, and a retry in
    the same process comes up on a *new* loop. Before this was handled,
    ``start_loop_watchdog()`` saw a live monitor, returned ``True``, and left
    ``_heartbeat_task`` on the closed loop: the heartbeat never ticked again, so
    the monitor read a frozen ``_last_tick``, reported a permanent stall, and
    would never have recovered it -- a fresh lifespan served with no usable
    diagnostics.
    """
    from headroom.proxy import loop_watchdog

    monkeypatch.setenv("HEADROOM_LOOP_WATCHDOG_SECS", "1")
    loop_watchdog._reset_for_test()

    async def failing_startup() -> None:
        assert loop_watchdog.start_loop_watchdog() is True
        # A startup check raising here leaves the arming in place, no cleanup.
        raise RuntimeError("observability check refused to start")

    try:
        with pytest.raises(RuntimeError, match="observability check refused"):
            asyncio.run(failing_startup())

        async def retry() -> dict[str, Any]:
            assert loop_watchdog.start_loop_watchdog() is True
            running = asyncio.get_running_loop()
            task = loop_watchdog._heartbeat_task
            assert task is not None
            assert task.get_loop() is running, "heartbeat still bound to the dead loop"
            # A healthy new loop must not be reported as stalled.
            with caplog.at_level(logging.ERROR, logger=loop_watchdog.logger.name):
                await asyncio.sleep(2.0)
            return loop_watchdog.snapshot()

        payload = asyncio.run(retry())
        assert "proxy_event_loop_stall" not in caplog.text
        assert payload["enabled"] is True
        assert payload["current_stall_seconds"] == 0.0
        assert payload["stalls_total"] == 0
    finally:
        loop_watchdog._reset_for_test()


def test_startup_failure_after_arming_stops_the_watchdog(monkeypatch):
    """A refused startup must not leave the monitor running.

    The observability gate and the rust-core check run *after* the watchdog is
    armed and, until this was fixed, outside the lifespan's cleanup. A refused
    lifespan therefore leaked a monitor thread plus a heartbeat whose loop was
    about to close.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy import loop_watchdog
    from headroom.proxy import server as proxy_server
    from headroom.proxy.models import ProxyConfig

    monkeypatch.setenv("HEADROOM_SKIP_UPSTREAM_CHECK", "1")
    monkeypatch.setenv("HEADROOM_LOOP_WATCHDOG_SECS", "5")
    loop_watchdog._reset_for_test()

    def refuse() -> None:
        raise SystemExit("observability gate refused to start")

    monkeypatch.setattr(proxy_server, "_configure_observability_or_refuse", refuse)

    app = proxy_server.create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
    )
    try:
        started = False
        try:
            with TestClient(app):
                started = True
        except BaseException:
            # The refusal surfaces as SystemExit, wrapped by the test harness.
            pass
        assert not started, "lifespan started despite a refused startup check"
        # Cleanup ran: no monitor left armed behind a closed loop.
        assert loop_watchdog.snapshot()["enabled"] is False
    finally:
        loop_watchdog._reset_for_test()
