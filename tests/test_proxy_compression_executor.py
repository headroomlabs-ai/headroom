"""Audit follow-up C3: bounded compression executor + cancel-aware metrics.

Replaces ``asyncio.to_thread`` for ``pipeline.apply()`` calls with a dedicated
``ThreadPoolExecutor`` that's bounded by ``ProxyConfig.compression_max_workers``.

Locks the following invariants:

1. The pool exists and respects ``compression_max_workers`` (auto and explicit).
2. ``compression_in_flight`` increments while a compression is running and
   decrements after it completes — under load, the high-water mark moves up
   as expected.
3. When a compression call exceeds its timeout, the awaiter unblocks with
   ``TimeoutError`` — but the worker thread keeps running (Python cannot
   preempt running CPython bytecode or in-flight Rust calls), and when the
   work eventually completes, ``compression_leaked_threads`` increments.
4. ``/stats runtime.compression_executor`` surfaces the gauge + counter so
   operators can see leaked-thread rate.

These tests also serve as documentation: anyone reading them sees that
"timeout fired" does not mean "compression was cancelled" — it means "we
stopped waiting; the worker is still going". A bounded pool plus the
leaked-thread counter is how we make that visible.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

pytest.importorskip("fastapi")

from headroom.proxy.helpers import COMPRESSION_TIMEOUT_SECONDS  # noqa: F401
from headroom.proxy.server import ProxyConfig, create_app


def _make_proxy(compression_max_workers: int | None = None):
    """Construct a HeadroomProxy with a no-op pipeline. Returns the proxy."""
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        compression_max_workers=compression_max_workers,
    )
    app = create_app(config)
    return app.state.proxy


def test_compression_executor_default_size_matches_asyncio_default() -> None:
    """When ``compression_max_workers`` is None, the resolved size should
    match asyncio's default executor sizing (``min(32, (cpu+1)*4)`` style).
    """
    import os

    proxy = _make_proxy(compression_max_workers=None)
    expected = min(32, (os.cpu_count() or 1) * 4)
    assert proxy.compression_max_workers == expected
    assert proxy._compression_executor._max_workers == expected


def test_compression_executor_explicit_override() -> None:
    """``ProxyConfig.compression_max_workers=N`` is honored verbatim."""
    proxy = _make_proxy(compression_max_workers=3)
    assert proxy.compression_max_workers == 3
    assert proxy._compression_executor._max_workers == 3


def test_compression_executor_minimum_one_worker() -> None:
    """A non-positive override clamps to 1 (zero workers would deadlock)."""
    proxy = _make_proxy(compression_max_workers=0)
    assert proxy.compression_max_workers == 1


def test_in_flight_gauge_tracks_running_compressions() -> None:
    """While a compression is running, ``_compression_in_flight`` reads ≥ 1.
    After it completes, it returns to 0. The high-water mark records the
    peak observed.
    """
    proxy = _make_proxy(compression_max_workers=4)

    enter_event = threading.Event()
    release_event = threading.Event()
    observed: dict[str, int] = {}

    def _slow_compression():
        enter_event.set()
        # Block until the test thread reads in_flight from the gauge.
        release_event.wait(timeout=5.0)
        return "done"

    async def _drive():
        task = asyncio.create_task(
            proxy._run_compression_in_executor(_slow_compression, timeout=10.0)
        )
        # Wait for the worker to actually start.
        for _ in range(50):
            if enter_event.is_set():
                break
            await asyncio.sleep(0.01)
        with proxy._compression_metrics_lock:
            observed["mid_flight"] = proxy._compression_in_flight
            observed["mid_flight_max"] = proxy._compression_in_flight_max
        release_event.set()
        result = await task
        return result

    result = asyncio.run(_drive())
    assert result == "done"
    assert observed["mid_flight"] == 1, (
        f"in_flight should be 1 mid-call, got {observed['mid_flight']}"
    )
    assert observed["mid_flight_max"] >= 1
    # Decremented after task completes.
    with proxy._compression_metrics_lock:
        assert proxy._compression_in_flight == 0


def test_high_water_mark_persists_after_completion() -> None:
    """``_compression_in_flight_max`` is monotonic — never decreases."""
    proxy = _make_proxy(compression_max_workers=8)

    enter_events = [threading.Event() for _ in range(3)]
    release_events = [threading.Event() for _ in range(3)]

    def _make_slow(idx: int):
        def _slow():
            enter_events[idx].set()
            release_events[idx].wait(timeout=5.0)
            return idx

        return _slow

    async def _drive():
        tasks = [
            asyncio.create_task(proxy._run_compression_in_executor(_make_slow(i), timeout=10.0))
            for i in range(3)
        ]
        # Wait for all 3 to enter.
        for ev in enter_events:
            for _ in range(50):
                if ev.is_set():
                    break
                await asyncio.sleep(0.01)
        peak = proxy._compression_in_flight
        for ev in release_events:
            ev.set()
        for t in tasks:
            await t
        return peak

    peak = asyncio.run(_drive())
    assert peak == 3, f"Should have observed 3 concurrent compressions, got {peak}"
    # After all complete, in_flight is back to 0 but max remains 3.
    with proxy._compression_metrics_lock:
        assert proxy._compression_in_flight == 0
        assert proxy._compression_in_flight_max >= 3


def test_timeout_fires_and_leaked_thread_is_counted() -> None:
    """When the compression exceeds ``timeout``, the awaiter sees
    ``TimeoutError`` immediately. The worker keeps running; when it finishes,
    ``_compression_leaked_threads`` increments by 1.
    """
    proxy = _make_proxy(compression_max_workers=2)
    finished_event = threading.Event()
    timeout_seconds = 0.10

    def _slow_compression():
        # Sleep well past the timeout so the asyncio side cancels first.
        time.sleep(timeout_seconds * 5)
        finished_event.set()
        return "completed-after-deadline"

    async def _drive():
        with pytest.raises(asyncio.TimeoutError):
            await proxy._run_compression_in_executor(_slow_compression, timeout=timeout_seconds)

    asyncio.run(_drive())

    # Wait for the worker to actually finish (it ran past the deadline).
    finished_event.wait(timeout=2.0)
    # Give the worker thread a moment to update the counter under the lock.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with proxy._compression_metrics_lock:
            if proxy._compression_leaked_threads >= 1:
                break
        time.sleep(0.01)

    with proxy._compression_metrics_lock:
        assert proxy._compression_leaked_threads >= 1, (
            f"leaked_threads should be ≥ 1; got {proxy._compression_leaked_threads}. "
            f"The worker either didn't finish past the deadline, or the wrapper "
            f"didn't increment the counter."
        )
        # In-flight gauge restored.
        assert proxy._compression_in_flight == 0


def test_compression_executor_metrics_appear_in_runtime_payload() -> None:
    """``/stats runtime.compression_executor`` surfaces the new gauges."""
    from fastapi.testclient import TestClient

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        compression_max_workers=5,
    )
    app = create_app(config)

    with TestClient(app) as client:
        # The compression_executor metrics are published from the runtime
        # payload (also surfaced in /health). Hit /health and look there.
        r = client.get("/health")
        assert r.status_code == 200
        runtime = r.json()["runtime"]
        assert "compression_executor" in runtime
        ce = runtime["compression_executor"]
        assert ce["max_workers"] == 5
        assert ce["in_flight"] == 0
        assert ce["leaked_threads_total"] == 0
        assert ce["source"] == "explicit"


def test_explicit_None_resolves_to_auto_source() -> None:
    """When max_workers is None (default), the runtime payload reports
    ``source: auto``."""
    from fastapi.testclient import TestClient

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.json()["runtime"]["compression_executor"]["source"] == "auto"
