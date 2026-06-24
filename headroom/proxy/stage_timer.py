"""Stage-timing instrumentation for request handlers.

Provides a lightweight, synchronous+async context-manager utility for
measuring per-stage durations within a single request or WebSocket
session. Timings are collected into a single dict that can be emitted
on the structured log line for the request.

The design goals:
  1. Durations are captured even if the measured body raises (the
     ``finally`` clause records the partial duration before
     re-raising).
  2. A single ``StageTimer`` holds every stage for one request/session
     and produces a ``dict[str, float]`` of millisecond durations on
     :meth:`summary`.
  3. Concurrent :meth:`measure` calls are independent — each one owns
     its own entry in the dict, so overlapping stages do not collide.
  4. Uses ``time.perf_counter()`` for monotonic, high-resolution
     measurement.

This module is intentionally free of any external dependencies so it
can be safely imported from both handler code paths.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterable
from contextlib import AbstractAsyncContextManager, AbstractContextManager
from types import TracebackType
from typing import Any

logger = logging.getLogger("headroom.proxy")

__all__ = ["StageTimer", "StageMeasurement", "emit_stage_timings_log"]


class StageMeasurement(
    AbstractContextManager["StageMeasurement"],
    AbstractAsyncContextManager["StageMeasurement"],
):
    """Context manager that measures a single named stage.

    Acts as both a synchronous (``with timer.measure(...):``) and an
    asynchronous (``async with timer.measure(...):``) context manager.
    The body's duration is captured in a ``finally`` clause so it is
    recorded even if the body raises.
    """

    __slots__ = ("_timer", "_name", "_start")

    def __init__(self, timer: StageTimer, name: str) -> None:
        self._timer = timer
        self._name = name
        self._start: float | None = None

    def __enter__(self) -> StageMeasurement:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._finalize()

    async def __aenter__(self) -> StageMeasurement:
        self._start = time.perf_counter()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._finalize()

    def _finalize(self) -> None:
        if self._start is None:
            # Defensive — should not happen in practice.
            return
        duration_ms = (time.perf_counter() - self._start) * 1000.0
        self._timer._record(self._name, duration_ms)


class StageTimer:
    """Collect per-stage durations for one request/session.

    A single instance is created at the start of a request/session and
    passed through the handler. Each ``measure(stage_name)`` call
    returns a context manager that records its body's duration in
    milliseconds under ``stage_name``.

    Stages that never run remain absent from :meth:`summary` — callers
    that need ``null`` placeholders for unused stages should overlay
    them explicitly on the returned dict.
    """

    __slots__ = ("_stages", "_created_at")

    def __init__(self) -> None:
        self._stages: dict[str, float] = {}
        self._created_at = time.perf_counter()

    def measure(self, name: str) -> StageMeasurement:
        """Return a context manager that records the named stage's duration."""
        return StageMeasurement(self, name)

    def record(self, name: str, duration_ms: float) -> None:
        """Record a pre-computed duration (e.g. from an existing timer).

        If the stage already has a recorded value, the new value
        replaces it. This matches the semantics of ``measure`` — the
        most recent measurement for a named stage wins. Callers that
        need accumulation should aggregate externally before calling
        :meth:`record`.
        """
        self._stages[name] = float(duration_ms)

    def _record(self, name: str, duration_ms: float) -> None:
        """Internal: record a duration from a ``StageMeasurement``."""
        self._stages[name] = duration_ms

    def elapsed_ms(self) -> float:
        """Return total milliseconds since the timer was created."""
        return (time.perf_counter() - self._created_at) * 1000.0

    def summary(self) -> dict[str, float]:
        """Return a snapshot of the recorded stage durations (in ms)."""
        return dict(self._stages)

    def __contains__(self, name: str) -> bool:
        return name in self._stages


async def emit_stage_timings_log(
    *,
    path: str,
    request_id: str,
    session_id: str,
    stage_timer: StageTimer,
    expected_stages: Iterable[str],
    metrics: Any | None = None,
) -> None:
    """Emit one structured log line of stage timings + record Prometheus series.

    ``expected_stages`` guarantees that every stage the handler plans
    to instrument shows up in the log line, even when it never ran
    (``None`` placeholder). This makes the log stream trivial to parse
    without knowing the full stage vocabulary per path up-front.

    ``metrics`` (optional) is anything with an
    ``async record_stage_timings(path, timings)`` method — typically
    ``HeadroomProxy.metrics``. Errors from the metrics sink are logged
    at DEBUG and do not propagate.
    """
    summary = stage_timer.summary()
    padded: dict[str, float | None] = {s: summary.get(s) for s in expected_stages}
    # Include any extra stages that were recorded but not listed in
    # ``expected_stages`` — defensive, covers future additions.
    for extra_stage, extra_value in summary.items():
        if extra_stage not in padded:
            padded[extra_stage] = extra_value

    try:
        payload = json.dumps(
            {
                "event": "stage_timings",
                "path": path,
                "request_id": request_id,
                "session_id": session_id,
                "stages": padded,
            },
            default=str,
        )
        logger.info(f"[{request_id}] STAGE_TIMINGS {payload}")
    except (TypeError, ValueError):
        logger.info(
            f"[{request_id}] STAGE_TIMINGS path={path} session_id={session_id} stages={padded!r}"
        )

    if metrics is not None and hasattr(metrics, "record_stage_timings"):
        try:
            await metrics.record_stage_timings(path, summary)
        except Exception as metric_err:  # pragma: no cover - defensive
            logger.debug(f"[{request_id}] record_stage_timings failed for {path}: {metric_err}")
