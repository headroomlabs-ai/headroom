"""Tests for ``headroom.proxy.stage_timer.StageTimer``."""

from __future__ import annotations

import asyncio
import time

import pytest

from headroom.proxy.stage_timer import StageTimer


def test_sync_measure_records_duration():
    timer = StageTimer()

    with timer.measure("stage_a"):
        time.sleep(0.01)

    summary = timer.summary()
    assert "stage_a" in summary
    assert summary["stage_a"] >= 10.0  # at least 10 ms
    # Sanity: should not be wildly off (upper bound generous for slow CI).
    assert summary["stage_a"] < 5000.0


def test_sync_measure_records_even_when_body_raises():
    timer = StageTimer()

    with pytest.raises(RuntimeError, match="boom"):
        with timer.measure("stage_err"):
            time.sleep(0.005)
            raise RuntimeError("boom")

    summary = timer.summary()
    assert "stage_err" in summary
    assert summary["stage_err"] >= 5.0


@pytest.mark.asyncio
async def test_async_measure_records_duration():
    timer = StageTimer()

    async with timer.measure("async_stage"):
        await asyncio.sleep(0.01)

    summary = timer.summary()
    assert "async_stage" in summary
    assert summary["async_stage"] >= 10.0


@pytest.mark.asyncio
async def test_async_measure_records_even_when_body_raises():
    timer = StageTimer()

    with pytest.raises(ValueError, match="oops"):
        async with timer.measure("async_err"):
            await asyncio.sleep(0.002)
            raise ValueError("oops")

    summary = timer.summary()
    assert "async_err" in summary
    assert summary["async_err"] >= 2.0


@pytest.mark.asyncio
async def test_concurrent_measure_calls_are_independent():
    timer = StageTimer()

    async def _stage(name: str, delay: float) -> None:
        async with timer.measure(name):
            await asyncio.sleep(delay)

    await asyncio.gather(
        _stage("fast", 0.005),
        _stage("medium", 0.02),
        _stage("slow", 0.04),
    )

    summary = timer.summary()
    assert set(summary) == {"fast", "medium", "slow"}
    # Each stage's duration reflects its own body, not the whole gather.
    assert summary["fast"] < summary["medium"]
    assert summary["medium"] < summary["slow"]
    # All stages measured while scheduled concurrently should each be
    # close to their own sleep, not to the serialized sum (~65 ms).
    assert summary["slow"] < 200.0


def test_summary_returns_independent_snapshot():
    timer = StageTimer()

    with timer.measure("one"):
        pass

    snapshot = timer.summary()
    snapshot["mutated"] = 999.0

    # The timer itself does not retain the mutation.
    assert "mutated" not in timer.summary()


def test_record_allows_preexisting_duration():
    timer = StageTimer()
    timer.record("precomputed_ms", 42.0)

    summary = timer.summary()
    assert summary["precomputed_ms"] == pytest.approx(42.0)


def test_elapsed_ms_is_monotonic():
    timer = StageTimer()
    first = timer.elapsed_ms()
    time.sleep(0.005)
    second = timer.elapsed_ms()
    assert second > first
    assert second - first >= 5.0


def test_contains_operator_checks_recorded_stages():
    timer = StageTimer()

    assert "absent" not in timer

    with timer.measure("present"):
        pass

    assert "present" in timer


def test_unused_stages_absent_from_summary():
    timer = StageTimer()

    with timer.measure("only_one"):
        pass

    summary = timer.summary()
    assert summary == {"only_one": pytest.approx(summary["only_one"])}
    # No sentinel placeholders — callers overlay ``null`` themselves.
    assert "not_measured" not in summary
