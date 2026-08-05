"""The proxy savings-ledger append must not block the loop or hold the metrics lock.

``PrometheusMetrics.record_request`` appends one durable JSONL event per
compressed request. That append does synchronous ``open`` + ``fcntl.flock`` +
``write``, and rewrites the whole file once it passes 1 MB. Running it on the
event loop stalls every other request. Running it under ``self._lock`` also
queues every other metrics caller behind it, including the ``/metrics`` scrape
(``export`` holds that same lock for the full Prometheus serialization).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from headroom import savings_ledger
from headroom.proxy import prometheus_metrics

# Long enough to dwarf scheduler noise, short enough to keep the suite quick.
_WRITE_SECONDS = 0.5


class _FakeSavingsTracker:
    def snapshot(self) -> dict[str, dict[str, int | float]]:
        return {"lifetime": {"total_input_tokens": 0, "total_input_cost_usd": 0.0}}

    def record_request(self, **kwargs: Any) -> None:
        pass

    def record_lifetime_request(self, **kwargs: Any) -> None:
        pass


class _FakeOtelMetrics:
    def record_proxy_request(self, **kwargs: Any) -> None:
        pass


def _metrics(**kwargs: Any) -> prometheus_metrics.PrometheusMetrics:
    return prometheus_metrics.PrometheusMetrics(
        savings_tracker=_FakeSavingsTracker(),
        otel_metrics=_FakeOtelMetrics(),
        **kwargs,
    )


async def _record(
    metrics: prometheus_metrics.PrometheusMetrics, *, tokens_saved: int = 400
) -> None:
    await metrics.record_request(
        provider="anthropic",
        model="claude-opus-4-6",
        input_tokens=600,
        output_tokens=25,
        tokens_saved=tokens_saved,
        latency_ms=10.0,
        client="claude-code",
    )


@pytest.mark.asyncio
async def test_metrics_lock_is_free_while_the_ledger_write_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent ``self._lock`` holder proceeds mid-write, not after it.

    ``export()`` takes this same lock, so a write held under it blocks
    ``/metrics`` for the full duration of the disk write.
    """

    window: dict[str, float] = {}

    def slow_record(**kwargs: Any) -> None:
        window["start"] = time.perf_counter()
        time.sleep(_WRITE_SECONDS)
        window["end"] = time.perf_counter()

    monkeypatch.setattr(prometheus_metrics.savings_ledger, "record_savings_event", slow_record)
    metrics = _metrics()

    async def competitor() -> float:
        async with metrics._lock:
            return time.perf_counter()

    _, acquired = await asyncio.gather(_record(metrics), competitor())

    assert window, "the ledger write never ran"
    # Only an upper bound. Once the write is offloaded, the competitor takes the
    # free lock on the loop thread before the worker has even started, so
    # `acquired` legitimately precedes `window["start"]`. What must not happen is
    # the competitor queueing until the write is done.
    assert acquired < window["end"] - _WRITE_SECONDS / 2, (
        "the metrics lock was held across the ledger write: acquired "
        f"{acquired - window['start']:+.3f}s relative to write start "
        f"(write took {window['end'] - window['start']:.3f}s)"
    )


@pytest.mark.asyncio
async def test_event_is_on_disk_once_record_request_returns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Offloading must stay awaited: callers still see a durable write on return."""

    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(tmp_path / "savings_events.jsonl"))

    await _record(_metrics())

    report = savings_ledger.aggregate_savings()
    assert report.lifetime["calls"] == 1
    assert report.lifetime["tokens_saved"] == 400
    assert report.lifetime["tokens_before"] == 1000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tokens_saved", "stateless"),
    [(0, False), (400, True)],
)
async def test_no_ledger_write_when_gated_out(
    monkeypatch: pytest.MonkeyPatch, tokens_saved: int, stateless: bool
) -> None:
    """The ``tokens_saved > 0 and not stateless`` gate survives the move."""

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        prometheus_metrics.savings_ledger,
        "record_savings_event",
        lambda **kwargs: calls.append(kwargs),
    )

    await _record(_metrics(stateless=stateless), tokens_saved=tokens_saved)

    assert calls == []


@pytest.mark.asyncio
async def test_concurrent_requests_all_land_their_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Offloading means N in-flight requests append from N worker threads.

    Before the move every proxy ledger write ran on the one event-loop thread,
    so they were serialised for free. Now they are not, and the ledger's own
    ``flock`` plus its past-1 MB full-file rewrite are what has to hold the line.
    """

    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(tmp_path / "savings_events.jsonl"))
    metrics = _metrics()

    await asyncio.gather(*(_record(metrics) for _ in range(24)))

    report = savings_ledger.aggregate_savings()
    assert report.lifetime["calls"] == 24, "a concurrent append was lost"
    assert report.lifetime["tokens_saved"] == 24 * 400
