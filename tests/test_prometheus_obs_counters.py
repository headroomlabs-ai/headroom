"""Unit tests for fail-open compression observability counters.

Covers the related counters added to ``PrometheusMetrics``:

* ``headroom_compression_failed_total{reason}`` — recorded at the proxy's
  optimization fail-open site, split into "timeout" vs "error".
* ``headroom_kompress_size_gate_total{outcome}`` — recorded by ContentRouter
  via the observer hook, split into "exceeded" vs "within".
* ``headroom_compression_quarantine_total{event}`` — records quarantine
  activation and immediate executor skips while a timed-out worker remains.

Imports only the metrics module so the test stays free of heavy ML deps.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from headroom.proxy.prometheus_metrics import PrometheusMetrics


def test_record_compression_failed_buckets_by_reason() -> None:
    metrics = PrometheusMetrics()

    metrics.record_compression_failed("timeout")
    metrics.record_compression_failed("error")
    metrics.record_compression_failed("error")

    assert metrics.compression_failed_by_reason["timeout"] == 1
    assert metrics.compression_failed_by_reason["error"] == 2


def test_record_compression_failed_empty_reason_defaults_to_error() -> None:
    metrics = PrometheusMetrics()

    metrics.record_compression_failed("")

    assert metrics.compression_failed_by_reason["error"] == 1


def test_record_kompress_size_gate_buckets_by_outcome() -> None:
    metrics = PrometheusMetrics()

    metrics.record_kompress_size_gate("exceeded")
    metrics.record_kompress_size_gate("within")
    metrics.record_kompress_size_gate("within")

    assert metrics.kompress_size_gate_by_outcome["exceeded"] == 1
    assert metrics.kompress_size_gate_by_outcome["within"] == 2


def test_record_compression_quarantine_buckets_by_event() -> None:
    metrics = PrometheusMetrics()

    metrics.record_compression_quarantine("activated")
    metrics.record_compression_quarantine("skipped")
    metrics.record_compression_quarantine("skipped")

    assert metrics.compression_quarantine_by_event["activated"] == 1
    assert metrics.compression_quarantine_by_event["skipped"] == 2


@pytest.mark.asyncio
async def test_counters_exported_in_prometheus_text() -> None:
    metrics = PrometheusMetrics()

    metrics.record_compression_failed("timeout")
    metrics.record_compression_failed("error")
    metrics.record_kompress_size_gate("exceeded")
    metrics.record_kompress_size_gate("within")
    metrics.record_compression_quarantine("activated")
    metrics.record_compression_quarantine("skipped")

    text = await metrics.export()

    assert "# TYPE headroom_compression_failed_total counter" in text
    assert 'headroom_compression_failed_total{reason="timeout"} 1' in text
    assert 'headroom_compression_failed_total{reason="error"} 1' in text

    assert "# TYPE headroom_kompress_size_gate_total counter" in text
    assert 'headroom_kompress_size_gate_total{outcome="exceeded"} 1' in text
    assert 'headroom_kompress_size_gate_total{outcome="within"} 1' in text

    assert "# TYPE headroom_compression_quarantine_total counter" in text
    assert 'headroom_compression_quarantine_total{event="activated"} 1' in text
    assert 'headroom_compression_quarantine_total{event="skipped"} 1' in text


@pytest.mark.asyncio
async def test_counters_absent_from_export_until_recorded() -> None:
    metrics = PrometheusMetrics()

    text = await metrics.export()

    # Conditional emission: the families only appear once a sample exists,
    # matching the other labelled-counter blocks in export().
    assert "headroom_compression_failed_total" not in text
    assert "headroom_kompress_size_gate_total" not in text
    assert "headroom_compression_quarantine_total" not in text


@pytest.mark.asyncio
async def test_reset_runtime_clears_observability_counters() -> None:
    metrics = PrometheusMetrics()

    metrics.record_compression_failed("timeout")
    metrics.record_kompress_size_gate("exceeded")
    metrics.record_compression_quarantine("activated")

    await metrics.reset_runtime()

    assert dict(metrics.compression_failed_by_reason) == {}
    assert dict(metrics.kompress_size_gate_by_outcome) == {}
    assert dict(metrics.compression_quarantine_by_event) == {}


@pytest.mark.asyncio
async def test_gate_counter_is_thread_safe_under_concurrent_export() -> None:
    # record_kompress_size_gate runs on the compression executor thread while
    # export() reads from the event loop. Concurrent unguarded access would
    # lose increments or raise "dictionary changed size during iteration".
    metrics = PrometheusMetrics()
    n_threads, per_thread = 8, 4000
    errors: list[str] = []

    def hammer() -> None:
        for i in range(per_thread):
            metrics.record_kompress_size_gate("within" if i % 2 else "exceeded")

    threads = [threading.Thread(target=hammer) for _ in range(n_threads)]
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        try:
            await metrics.export()
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(repr(exc))
        await asyncio.sleep(0)
    for t in threads:
        t.join()

    assert not errors, f"export() raced the writer: {errors[:3]}"
    totals = dict(metrics.kompress_size_gate_by_outcome)
    assert sum(totals.values()) == n_threads * per_thread
