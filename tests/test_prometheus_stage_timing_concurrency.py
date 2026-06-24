"""Concurrency test for PrometheusMetrics stage-timing lock-free path.

Covers the P2 fix that moves ``record_stage_timings`` off the global
``asyncio.Lock`` and onto a tiny synchronous ``threading.Lock`` so
request finalizations don't queue behind ``export()``'s string-building.

The invariant we assert: sum / count / max agree after many concurrent
writes + concurrent scrapes.
"""

from __future__ import annotations

import asyncio

import pytest

from headroom.proxy.prometheus_metrics import PrometheusMetrics


@pytest.mark.asyncio
async def test_record_stage_timings_agrees_under_concurrent_scrapes():
    metrics = PrometheusMetrics()

    writes_per_task = 25
    writer_count = 20
    scrape_count = 5

    async def _writer(stage_id: int) -> None:
        for i in range(writes_per_task):
            await metrics.record_stage_timings(
                "some_path",
                {f"stage-{stage_id}": float(i + 1)},
            )

    async def _scraper() -> None:
        for _ in range(10):
            await metrics.export()
            await asyncio.sleep(0)  # yield

    writers = [asyncio.create_task(_writer(i)) for i in range(writer_count)]
    scrapers = [asyncio.create_task(_scraper()) for _ in range(scrape_count)]
    await asyncio.gather(*writers, *scrapers)

    # Invariant: for each (path, stage) key, count equals writes_per_task
    # and sum equals 1+2+...+writes_per_task.
    expected_count = writes_per_task
    expected_sum = sum(range(1, writes_per_task + 1))
    expected_max = float(writes_per_task)

    for stage_id in range(writer_count):
        key = ("some_path", f"stage-{stage_id}")
        assert metrics.stage_timing_count[key] == expected_count, (
            f"stage {stage_id}: count mismatch "
            f"got={metrics.stage_timing_count[key]}, want={expected_count}"
        )
        assert metrics.stage_timing_sum[key] == pytest.approx(expected_sum), (
            f"stage {stage_id}: sum mismatch "
            f"got={metrics.stage_timing_sum[key]}, want={expected_sum}"
        )
        assert metrics.stage_timing_max[key] == pytest.approx(expected_max), (
            f"stage {stage_id}: max mismatch "
            f"got={metrics.stage_timing_max[key]}, want={expected_max}"
        )


@pytest.mark.asyncio
async def test_record_stage_timings_does_not_hold_async_lock():
    """Sanity: ``record_stage_timings`` must not contend with the async
    lock. If it did, a long-running holder of the async lock would block
    stage-timing writes."""
    metrics = PrometheusMetrics()

    # Hold the async lock from a background task and ensure
    # record_stage_timings completes anyway.
    lock_acquired = asyncio.Event()
    release_lock = asyncio.Event()

    async def _hold_async_lock() -> None:
        async with metrics._lock:
            lock_acquired.set()
            await release_lock.wait()

    holder = asyncio.create_task(_hold_async_lock())
    await lock_acquired.wait()

    # This should complete even though _lock is held.
    await asyncio.wait_for(
        metrics.record_stage_timings("p", {"s": 1.0}),
        timeout=1.0,
    )
    assert metrics.stage_timing_count[("p", "s")] == 1

    release_lock.set()
    await holder
