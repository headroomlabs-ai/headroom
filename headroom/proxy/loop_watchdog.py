"""Event-loop liveness watchdog: name a stall the hard watchdog cannot see.

:mod:`headroom.proxy.hard_watchdog` bounds one failure mode: a native call that
*seizes the GIL*, which starves every Python thread at once. Its C timer fires
from a thread that never takes the GIL, dumps every stack, and exits.

Field report #4967 is the other mode. A 1.7 MB ``/v1/messages`` body arrived
next to a 4 KB one, the proxy logged nothing for 4m24s, and the process came
back with **no stack dump** -- which is proof the hard watchdog never fired:
its heartbeat kept re-arming the timer, so the interpreter was healthy and
every thread that wanted the GIL had it. What was blocked was the *event loop
thread*, inside a call that releases the GIL and never returns: a native
initialization parked on a lock, a ``threading.Lock.acquire()`` whose holder
went away, a blocking syscall. Nothing observes that today. All the proxy's
own bounds -- the pre-upstream acquire timeout, the 30s compression deadline,
the per-stage timers -- are asyncio timers, i.e. callbacks on the very loop
that stopped running, so they cannot fire and cannot log. The client waits, the
log is silent, and the report ends with "I'll attach the stack dump if you tell
me where it is".

This module closes that gap the cheap way: a heartbeat coroutine the loop
schedules, and a monitor thread that notices when the heartbeat stops. The
thread reads no loop state, sleeps on a plain ``time.sleep``, and on expiry logs
the stall length and dumps every thread's stack through ``faulthandler`` --
exactly the missing evidence, since the blocked loop thread names its own
culprit frame. It never exits: killing the process stays the hard watchdog's
job at its (much longer) deadline. The two are complementary and must stay that
way:

* GIL seized   -> none of our threads run -> hard watchdog dumps and exits.
* loop blocked -> our thread runs fine    -> this watchdog dumps and stays up.

The threshold is deliberately short (10s): the loop is allowed to be busy, but
it is never legitimately silent for ten seconds while a client is waiting. One
INFO line and one stack dump per stall occurrence is cheap next to an
undiagnosable 4.5-minute outage, and ``/health`` reports the same numbers so an
operator can tell a loop stall apart from a wedged compression worker (#4613).
"""

from __future__ import annotations

import asyncio
import faulthandler
import logging
import os
import sys
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

ENV_VAR = "HEADROOM_LOOP_WATCHDOG_SECS"
DEFAULT_SECS = 10.0
# Below this, a garbage-collection pause or a loaded machine reads as a stall
# and floods the log with dumps that name no real problem.
MIN_SECS = 1.0
# Heartbeat cadence as a fraction of the threshold: fine enough to measure a
# stall to a tenth, coarse enough to be free on an idle proxy.
_HEARTBEAT_DIVISOR = 10.0
_MIN_HEARTBEAT_SECS = 0.05
# While a stall lasts, re-dump at most once per threshold so a ten-minute wedge
# leaves a readable log instead of a megabyte of identical stacks.
_REDUMP_EVERY = 1.0

_started = threading.Event()
_stop = threading.Event()
# Strong reference to the heartbeat task. ``asyncio`` only keeps a weak one, so
# a fire-and-forget task can be collected mid-flight and the loop would look
# stalled forever -- the watchdog would then dump stacks at every proxy that
# armed it successfully.
_heartbeat_task: asyncio.Task[None] | None = None

# Stall bookkeeping shared with ``/health`` through :func:`snapshot`. Written by
# the monitor thread and the heartbeat task, read by the loop thread, so every
# read/write pair goes through ``_stats_lock``.
_stats_lock = threading.Lock()
_enabled = False
_threshold_seconds = 0.0
_last_tick = 0.0
# Monotonic time of the last heartbeat BEFORE the current stall; 0.0 when no
# stall is in progress. Recovery is declared by the heartbeat, not the monitor:
# only the loop knows when it started running again.
_stall_started_at = 0.0
_stall_seconds_max = 0.0
_last_stall_seconds = 0.0
_stalls_total = 0


def _resolve_secs() -> float:
    raw = os.environ.get(ENV_VAR, "")
    if not raw.strip():
        return DEFAULT_SECS
    try:
        secs = float(raw)
    except ValueError:
        logger.warning("Ignoring non-numeric %s=%r; using %ss", ENV_VAR, raw, DEFAULT_SECS)
        return DEFAULT_SECS
    if secs <= 0:
        return 0.0
    return max(secs, MIN_SECS)


def _stderr_fd() -> int | None:
    """Resolve the stderr fd once.

    Looking it up per dump would raise whenever something swaps ``sys.stderr``
    for a stream without ``fileno`` (pytest's capsys, click's CliRunner) -- the
    same trap ``hard_watchdog`` documents. Supervisors already redirect this fd
    into the proxy log, so a dump lands beside the surrounding request lines.
    """
    try:
        return sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        return None


def snapshot() -> dict[str, Any]:
    """Return the watchdog's current view for ``/health``.

    ``current_stall_seconds`` is live: a non-zero value means the event loop is
    blocked *right now*, and no asyncio-level timeout in the proxy can fire.
    """
    now = time.monotonic()
    with _stats_lock:
        threshold = _threshold_seconds
        last = _last_tick
        stalled_for = (now - last) if last > 0.0 else 0.0
        current_stall = stalled_for if threshold > 0.0 and stalled_for >= threshold else 0.0
        return {
            "enabled": _enabled,
            "stall_threshold_seconds": threshold,
            "current_stall_seconds": round(current_stall, 2),
            "stalls_total": _stalls_total,
            "stall_seconds_max": round(_stall_seconds_max, 2),
            "last_stall_seconds": round(_last_stall_seconds, 2),
        }


def _dump_stacks(fd: int | None) -> None:
    """Dump every thread's stack where the operator already looks.

    ``faulthandler.dump_traceback`` is safe from any thread and needs only the
    GIL -- precisely what a loop-blocked (but GIL-free) process offers. With no
    usable fd, say so rather than substituting an in-memory stream: the point
    of the dump is to outlive the process that produced it.
    """
    if fd is None:
        logger.warning(
            "event=proxy_event_loop_stall_dump_skipped reason=no_stderr_fd "
            "detail=stderr has no usable fileno (captured or replaced); "
            "stack dump unavailable"
        )
        return
    try:
        faulthandler.dump_traceback(file=fd, all_threads=True)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("event=proxy_event_loop_stall_dump_failed error=%r", exc)


def _begin_stall(last_tick: float, stalled_for: float, threshold: float, fd: int | None) -> None:
    global _stall_started_at, _stalls_total, _last_stall_seconds, _stall_seconds_max
    with _stats_lock:
        _stall_started_at = last_tick
        _stalls_total += 1
        _last_stall_seconds = stalled_for
        if stalled_for > _stall_seconds_max:
            _stall_seconds_max = stalled_for
    logger.error(
        "event=proxy_event_loop_stall pid=%d stall_seconds=%.1f "
        "threshold_seconds=%.1f detail=event loop has not run a callback; asyncio "
        "timeouts (compression deadline, pre-upstream acquire) cannot fire and "
        "in-flight requests are frozen; dumping all thread stacks to the proxy log",
        os.getpid(),
        stalled_for,
        threshold,
    )
    _dump_stacks(fd)


def _monitor(threshold: float, poll: float, fd: int | None) -> None:
    in_stall = False
    last_dump_at = 0.0
    while not _stop.wait(poll):
        now = time.monotonic()
        with _stats_lock:
            last_tick = _last_tick
        stalled_for = (now - last_tick) if last_tick > 0.0 else 0.0
        if stalled_for < threshold:
            in_stall = False
            continue
        if not in_stall:
            in_stall = True
            last_dump_at = now
            _begin_stall(last_tick, stalled_for, threshold, fd)
        elif now - last_dump_at >= threshold * _REDUMP_EVERY:
            last_dump_at = now
            logger.error(
                "event=proxy_event_loop_stall_still_wedged pid=%d stall_seconds=%.1f",
                os.getpid(),
                stalled_for,
            )
            _dump_stacks(fd)


async def _heartbeat(threshold: float) -> None:
    global _last_tick, _stall_started_at, _last_stall_seconds, _stall_seconds_max
    interval = max(threshold / _HEARTBEAT_DIVISOR, _MIN_HEARTBEAT_SECS)
    while True:
        await asyncio.sleep(interval)
        now = time.monotonic()
        # Recovery is declared here because only the loop knows it is running
        # again. The monitor records the stall; this closes it out.
        with _stats_lock:
            stall_started = _stall_started_at
            if stall_started > 0.0:
                _stall_started_at = 0.0
            _last_tick = now
        if stall_started > 0.0:
            duration = now - stall_started
            with _stats_lock:
                _last_stall_seconds = duration
                if duration > _stall_seconds_max:
                    _stall_seconds_max = duration
            logger.warning(
                "event=proxy_event_loop_stall_recovered pid=%d stall_seconds=%.1f "
                "detail=event loop resumed; the stack dump above covers the window "
                "the loop was frozen",
                os.getpid(),
                duration,
            )


def start_loop_watchdog() -> bool:
    """Arm the event-loop watchdog for this process. Returns True when armed.

    Safe to call more than once; the first call wins. Needs a running loop (it
    schedules the heartbeat on it). Never raises: a proxy that cannot arm its
    own diagnostic must still serve traffic.
    """
    global _enabled, _threshold_seconds, _last_tick
    if _started.is_set():
        return _enabled
    secs = _resolve_secs()
    if not secs:
        logger.info("Event-loop watchdog disabled (%s=0)", ENV_VAR)
        _started.set()
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        logger.debug("Event-loop watchdog not armed: no running event loop")
        return False

    _started.set()
    with _stats_lock:
        _enabled = True
        _threshold_seconds = secs
        _last_tick = time.monotonic()
    global _heartbeat_task
    _heartbeat_task = loop.create_task(_heartbeat(secs), name="headroom-loop-watchdog-heartbeat")
    threading.Thread(
        target=_monitor,
        args=(secs, max(secs / _HEARTBEAT_DIVISOR, _MIN_HEARTBEAT_SECS), _stderr_fd()),
        name="headroom-loop-watchdog",
        daemon=True,
    ).start()
    logger.info(
        "Event-loop watchdog armed: log and stack-dump if the loop runs no "
        "callback for %.0fs (%s to tune, 0 to disable)",
        secs,
        ENV_VAR,
    )
    return True


def stop_loop_watchdog() -> None:
    """Stop the monitor thread and the heartbeat. Idempotent; used by shutdown."""
    global _enabled, _last_tick, _stall_started_at, _heartbeat_task
    _stop.set()
    task = _heartbeat_task
    _heartbeat_task = None
    if task is not None and not task.done():
        task.cancel()
    with _stats_lock:
        _enabled = False
        _last_tick = 0.0
        _stall_started_at = 0.0


def _reset_for_test() -> None:
    """Reset module state so a test can arm the watchdog again in-process."""
    global _started, _stop, _heartbeat_task
    global _enabled, _threshold_seconds, _last_tick
    global _stall_started_at, _stall_seconds_max, _last_stall_seconds, _stalls_total
    stop_loop_watchdog()
    _heartbeat_task = None
    _started = threading.Event()
    _stop = threading.Event()
    with _stats_lock:
        _enabled = False
        _threshold_seconds = 0.0
        _last_tick = 0.0
        _stall_started_at = 0.0
        _stall_seconds_max = 0.0
        _last_stall_seconds = 0.0
        _stalls_total = 0
