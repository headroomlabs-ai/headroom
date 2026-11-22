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
import math
import os
import queue
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

# Per-lifespan state. The watchdog is process-global but its heartbeat belongs to
# one running proxy lifespan, so start/stop must be restartable: an embedded app
# that stops its proxy and starts another inside the same interpreter would
# otherwise keep a lifecycle guard from the first run and serve with no stall
# monitoring at all.
_lifecycle_lock = threading.Lock()
_stop: threading.Event | None = None
_monitor_thread: threading.Thread | None = None
_reporter_thread: threading.Thread | None = None
# Strong reference to the heartbeat task. ``asyncio`` only keeps a weak one, so
# a fire-and-forget task can be collected mid-flight and the loop would look
# stalled forever -- the watchdog would then dump stacks at every proxy that
# armed it successfully.
_heartbeat_task: asyncio.Task[None] | None = None

# Reporting never happens on the monitor thread directly. Python logging
# serialises emission per handler, and the proxy installs a *synchronous*
# rotating file handler, so a loop thread blocked in logging I/O holds that
# handler's lock -- exactly the situation worth dumping a stack for. A monitor
# that logs first would then wait on the same lock and reach its dump only after
# the loop unblocked, i.e. never while it mattered. So the monitor dumps first
# and hands the message to this thread, which absorbs the blocking.
_LOG_QUEUE_MAX = 64
_log_queue: queue.Queue[tuple[int, str, tuple[Any, ...]]] | None = None

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
    # ``nan``/``inf`` parse fine and then poison everything downstream: a nan
    # threshold makes every comparison false, so the watchdog arms and never
    # reports, and it reaches ``/health`` and ``/readyz`` where json.dumps emits
    # a bare ``NaN`` token that strict parsers reject. Fall back to the default.
    if not math.isfinite(secs):
        logger.warning("Ignoring non-finite %s=%r; using %ss", ENV_VAR, raw, DEFAULT_SECS)
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


def _finite(value: float) -> float:
    """Never let a non-finite number reach an HTTP payload.

    ``json.dumps`` renders ``nan``/``inf`` as the bare tokens ``NaN``/``Infinity``,
    which ``allow_nan=False`` rejects and which many clients refuse to parse, so a
    single bad number would break ``/health`` for the whole proxy.
    """
    return value if math.isfinite(value) else 0.0


def snapshot() -> dict[str, Any]:
    """Return the watchdog's current view for ``/health``.

    ``current_stall_seconds`` is live: a non-zero value means the event loop is
    blocked *right now*, and no asyncio-level timeout in the proxy can fire.
    """
    now = time.monotonic()
    with _stats_lock:
        threshold = _threshold_seconds
        last = _last_tick
        stalls_total = _stalls_total
        seconds_max = _stall_seconds_max
        last_stall = _last_stall_seconds
        enabled = _enabled
    stalled_for = (now - last) if last > 0.0 else 0.0
    current_stall = stalled_for if threshold > 0.0 and stalled_for >= threshold else 0.0
    return {
        "enabled": enabled,
        "stall_threshold_seconds": _finite(threshold),
        "current_stall_seconds": round(_finite(current_stall), 2),
        "stalls_total": stalls_total,
        "stall_seconds_max": round(_finite(seconds_max), 2),
        "last_stall_seconds": round(_finite(last_stall), 2),
    }


def _report(level: int, msg: str, *args: Any) -> None:
    """Hand a log record to the reporter thread; never block, never raise."""
    queue_ = _log_queue
    if queue_ is None:
        try:
            logger.log(level, msg, *args)
        except Exception:  # pragma: no cover - diagnostics must never raise
            pass
        return
    try:
        queue_.put_nowait((level, msg, args))
    except queue.Full:
        # Logging is wedged. The stack dump already written by the caller is the
        # diagnostic that matters; dropping the annotation beats a stuck monitor.
        pass


def _reporter(stop: threading.Event) -> None:
    """Emit queued records, absorbing any block in the logging handlers."""
    while not stop.is_set():
        queue_ = _log_queue
        if queue_ is None:
            return
        try:
            level, msg, args = queue_.get(timeout=0.2)
        except queue.Empty:
            continue
        try:
            logger.log(level, msg, *args)
        except Exception:  # pragma: no cover - diagnostics must never raise
            pass


def _dump_stacks(fd: int | None) -> None:
    """Dump every thread's stack where the operator already looks.

    ``faulthandler.dump_traceback`` is safe from any thread and needs only the
    GIL -- precisely what a loop-blocked (but GIL-free) process offers. With no
    usable fd, say so rather than substituting an in-memory stream: the point
    of the dump is to outlive the process that produced it.
    """
    if fd is None:
        _report(
            logging.WARNING,
            "event=proxy_event_loop_stall_dump_skipped reason=no_stderr_fd "
            "detail=stderr has no usable fileno (captured or replaced); "
            "stack dump unavailable",
        )
        return
    try:
        faulthandler.dump_traceback(file=fd, all_threads=True)
    except (OSError, ValueError, RuntimeError) as exc:
        _report(logging.WARNING, "event=proxy_event_loop_stall_dump_failed error=%r", exc)


def _begin_stall(last_tick: float, stalled_for: float, threshold: float, fd: int | None) -> None:
    global _stall_started_at, _stalls_total, _last_stall_seconds, _stall_seconds_max
    with _stats_lock:
        _stall_started_at = last_tick
        _stalls_total += 1
        _last_stall_seconds = stalled_for
        if stalled_for > _stall_seconds_max:
            _stall_seconds_max = stalled_for
    # Dump first, log second: the dump is the whole point and must not queue
    # behind a logging handler the blocked loop thread may be holding.
    _dump_stacks(fd)
    _report(
        logging.ERROR,
        "event=proxy_event_loop_stall pid=%d stall_seconds=%.1f "
        "threshold_seconds=%.1f detail=event loop has not run a callback; asyncio "
        "timeouts (compression deadline, pre-upstream acquire) cannot fire and "
        "in-flight requests are frozen; stack dump written above",
        os.getpid(),
        stalled_for,
        threshold,
    )


def _monitor(threshold: float, poll: float, fd: int | None, stop: threading.Event) -> None:
    in_stall = False
    last_dump_at = 0.0
    while not stop.wait(poll):
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
            _dump_stacks(fd)
            _report(
                logging.ERROR,
                "event=proxy_event_loop_stall_still_wedged pid=%d stall_seconds=%.1f",
                os.getpid(),
                stalled_for,
            )


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
            _report(
                logging.WARNING,
                "event=proxy_event_loop_stall_recovered pid=%d stall_seconds=%.1f "
                "detail=event loop resumed; the stack dump above covers the window "
                "the loop was frozen",
                os.getpid(),
                duration,
            )


def _stop_current_lifecycle() -> None:
    """Tear down the current monitor/reporter/heartbeat. Expects the lock."""
    global _enabled, _last_tick, _stall_started_at, _heartbeat_task
    global _stop, _monitor_thread, _reporter_thread, _log_queue
    stop = _stop
    _stop = None
    _monitor_thread = None
    _reporter_thread = None
    _log_queue = None
    if stop is not None:
        stop.set()
    task = _heartbeat_task
    _heartbeat_task = None
    if task is not None and not task.done():
        try:
            task.cancel()
        except RuntimeError:  # pragma: no cover - loop already closed
            pass
    with _stats_lock:
        _enabled = False
        _last_tick = 0.0
        _stall_started_at = 0.0


def start_loop_watchdog() -> bool:
    """Arm the event-loop watchdog for the current lifespan. True when armed.

    Restartable: a previous lifespan's monitor is stopped and a fresh one armed,
    so an embedded app that stops its proxy and starts another keeps its
    diagnostics. Re-arming while the monitor is already alive is a no-op. Needs
    a running loop (it schedules the heartbeat on it). Never raises: a proxy
    that cannot arm its own diagnostic must still serve traffic.
    """
    global _enabled, _threshold_seconds, _last_tick, _heartbeat_task
    global _stop, _monitor_thread, _reporter_thread, _log_queue
    with _lifecycle_lock:
        if _monitor_thread is not None and _monitor_thread.is_alive():
            return _enabled
        secs = _resolve_secs()
        if not secs:
            _stop_current_lifecycle()
            logger.info("Event-loop watchdog disabled (%s=0)", ENV_VAR)
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("Event-loop watchdog not armed: no running event loop")
            return False

        _stop_current_lifecycle()
        stop = threading.Event()
        _log_queue = queue.Queue(maxsize=_LOG_QUEUE_MAX)
        with _stats_lock:
            _enabled = True
            _threshold_seconds = secs
            _last_tick = time.monotonic()
        _stop = stop
        _reporter_thread = threading.Thread(
            target=_reporter,
            args=(stop,),
            name="headroom-loop-watchdog-reporter",
            daemon=True,
        )
        _reporter_thread.start()
        _monitor_thread = threading.Thread(
            target=_monitor,
            args=(secs, max(secs / _HEARTBEAT_DIVISOR, _MIN_HEARTBEAT_SECS), _stderr_fd(), stop),
            name="headroom-loop-watchdog",
            daemon=True,
        )
        _monitor_thread.start()
        _heartbeat_task = loop.create_task(
            _heartbeat(secs), name="headroom-loop-watchdog-heartbeat"
        )
    logger.info(
        "Event-loop watchdog armed: log and stack-dump if the loop runs no "
        "callback for %.0fs (%s to tune, 0 to disable)",
        secs,
        ENV_VAR,
    )
    return True


def stop_loop_watchdog() -> None:
    """Stop the monitor thread and the heartbeat. Idempotent; used by shutdown."""
    with _lifecycle_lock:
        _stop_current_lifecycle()


def _reset_for_test() -> None:
    """Reset module state so a test can arm the watchdog again in-process."""
    global _enabled, _threshold_seconds, _last_tick
    global _stall_started_at, _stall_seconds_max, _last_stall_seconds, _stalls_total
    stop_loop_watchdog()
    with _stats_lock:
        _enabled = False
        _threshold_seconds = 0.0
        _last_tick = 0.0
        _stall_started_at = 0.0
        _stall_seconds_max = 0.0
        _last_stall_seconds = 0.0
        _stalls_total = 0
