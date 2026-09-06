"""Orphan watchdog for wrap-spawned proxies.

``headroom wrap`` starts its proxy detached (setsid on POSIX, breakaway
console flags on Windows) on purpose: a crashing wrapper must not tree-kill a
shared proxy out from under other clients. The flip side was a leak: once
every wrapper was gone, nothing ever stopped the proxy, so clientless proxies
accumulated for weeks and later wraps silently reused whatever they found on
the port (port-hijack incident, 2026-08-22). Cleanup on the wrapper side
cannot cover unclean exits, by definition nobody is alive to run it.

When the proxy was spawned by a wrap (``HEADROOM_WRAP_OWNED=1``, set in
``cli/wrap._start_proxy``), this module's background task stops the process
once no live wrap-client markers and no active sessions remain for a grace
period. Standalone ``headroom proxy`` instances and persistent installs never
get the marker env var and are never affected.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from headroom._subprocess import identity_mismatch, pid_alive
from headroom.paths import proxy_clients_dir

logger = logging.getLogger(__name__)

WRAP_OWNED_ENV = "HEADROOM_WRAP_OWNED"
GRACE_SECONDS_ENV = "HEADROOM_WRAP_ORPHAN_GRACE_SECONDS"
DEFAULT_GRACE_SECONDS = 900.0
# Floor so a typo ("5" for "5m") cannot make a proxy exit underneath a client
# that simply has not registered its marker yet.
MIN_GRACE_SECONDS = 60.0
DEFAULT_POLL_INTERVAL_SECONDS = 30.0


def orphan_watchdog_enabled(env: Mapping[str, str] | None = None) -> bool:
    """True when this proxy process was spawned by ``headroom wrap``."""
    env = os.environ if env is None else env
    return (env.get(WRAP_OWNED_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def orphan_grace_seconds(env: Mapping[str, str] | None = None) -> float:
    """Grace period with zero live clients before the watchdog exits the proxy."""
    env = os.environ if env is None else env
    raw = env.get(GRACE_SECONDS_ENV)
    if raw:
        try:
            return max(MIN_GRACE_SECONDS, float(raw))
        except ValueError:
            logger.warning(
                "event=orphan_watchdog_bad_grace value=%r default=%.0f", raw, DEFAULT_GRACE_SECONDS
            )
    return DEFAULT_GRACE_SECONDS


def _marker_pid_recycled(marker: Path, pid: int) -> bool:
    """True only if the live ``pid`` is provably a different process than the
    one that wrote ``marker`` (PID recycled after a crash)."""
    try:
        record = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(record, dict):
        return False
    return identity_mismatch(record.get("start_src"), record.get("start_time"), pid)


def live_client_pids(clients_dir: Path) -> list[int] | None:
    """Live wrap-client PIDs for a clients dir, pruning stale markers as we go.

    Mirrors ``cli.wrap._live_proxy_clients`` without the self-exclusion; the
    proxy is never its own client.
    """
    try:
        with os.scandir(clients_dir) as entries:
            markers = [
                Path(entry.path)
                for entry in entries
                if entry.name.endswith(".json") and entry.is_file()
            ]
    except FileNotFoundError:
        return []
    except OSError:
        logger.warning(
            "event=orphan_watchdog_client_scan_failed dir=%s", clients_dir, exc_info=True
        )
        return None

    live: list[int] = []
    for marker in markers:
        try:
            pid = int(marker.stem)
        except ValueError:
            continue
        alive = pid_alive(pid)
        if not alive or _marker_pid_recycled(marker, pid):
            try:
                marker.unlink(missing_ok=True)
            except OSError:
                pass
            continue
        live.append(pid)
    return live


def _active_session_count(proxy: Any) -> int | None:
    registry = getattr(proxy, "ws_sessions", None)
    if registry is None:
        return 0
    try:
        return int(registry.active_count())
    except Exception:
        logger.warning("event=orphan_watchdog_session_count_failed", exc_info=True)
        return None


def _active_request_count(proxy: Any) -> int | None:
    """Return active HTTP/WebSocket scope count, or None when unobservable."""
    try:
        count = proxy.active_request_count
    except Exception:
        logger.warning("event=orphan_watchdog_request_count_failed", exc_info=True)
        return None
    return count if isinstance(count, int) and count >= 0 else None


def _activity_generation(proxy: Any) -> int | None:
    """Return the monotonic ASGI activity generation, or None."""
    try:
        generation = proxy.activity_generation
    except Exception:
        logger.warning("event=orphan_watchdog_activity_generation_failed", exc_info=True)
        return None
    return generation if isinstance(generation, int) and generation >= 0 else None


def _default_stop() -> None:
    # signal.raise_signal (not os.kill): on Windows CPython maps
    # os.kill(pid, SIGTERM) to TerminateProcess, a hard kill that skips
    # uvicorn's handler and the lifespan teardown; raise_signal dispatches
    # through the emulated handler table, so uvicorn shuts down gracefully on
    # both platforms.
    signal.raise_signal(signal.SIGTERM)


async def orphan_watchdog_loop(
    proxy: Any,
    *,
    grace_seconds: float,
    interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    stop: Callable[[], None] = _default_stop,
    monotonic: Callable[[], float] = time.monotonic,
) -> None:
    """Poll wrap-client markers; ``stop()`` once clientless for ``grace_seconds``.

    The grace window covers the boot race (the wrapper registers its marker
    just before spawning the proxy, and re-registers on port fallback) and
    short client restarts. Active WebSocket relay sessions, in-flight HTTP
    requests, and any change to the ASGI activity generation suppress shutdown
    even with no live markers. The idle clock runs only while every supported
    activity signal is positively observable and idle.
    """
    port = proxy.config.port
    clients_dir = proxy_clients_dir(port)
    idle_since: float | None = None
    last_activity_generation = _activity_generation(proxy)
    while True:
        await asyncio.sleep(interval_seconds)
        clients = live_client_pids(clients_dir)
        active_sessions = _active_session_count(proxy)
        active_requests = _active_request_count(proxy)
        activity_generation = _activity_generation(proxy)
        activity_changed = (
            activity_generation is not None
            and last_activity_generation is not None
            and activity_generation != last_activity_generation
        )
        if activity_generation is not None:
            last_activity_generation = activity_generation
        activity_unknown = (
            clients is None
            or active_sessions is None
            or active_requests is None
            or activity_generation is None
        )
        if (
            activity_unknown
            or clients
            or (active_sessions is not None and active_sessions > 0)
            or (active_requests is not None and active_requests > 0)
            or activity_changed
        ):
            if idle_since is not None:
                logger.info(
                    "event=orphan_watchdog_reset port=%d live_clients=%s "
                    "active_sessions=%s active_requests=%s",
                    port,
                    clients,
                    active_sessions,
                    active_requests,
                )
            idle_since = None
            continue
        now = monotonic()
        if idle_since is None:
            idle_since = now
            continue
        if now - idle_since >= grace_seconds:
            logger.info(
                "event=orphan_watchdog_exit port=%d idle_seconds=%.0f reason=no_live_wrap_clients",
                port,
                now - idle_since,
            )
            stop()
            return
