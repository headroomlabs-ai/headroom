"""Unit tests for fail-open compression observability counters.

Covers the related counters added to ``PrometheusMetrics``:

* ``headroom_compression_failed_total{reason}`` — recorded at the proxy's
  optimization fail-open site, split into "timeout" vs "error".
* ``headroom_kompress_size_gate_total{outcome}`` — recorded by ContentRouter
  via the observer hook, split into "exceeded" vs "within".
* ``headroom_compression_quarantine_total{event}`` — records quarantine
  activation and immediate executor skips while a timed-out worker remains.
* ``headroom_upstream_connection_errors_total{provider}`` — recorded when the
  streaming path exhausts its connect retries and answers 502 itself.

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


def test_record_upstream_connection_error_buckets_by_provider() -> None:
    metrics = PrometheusMetrics()

    metrics.record_upstream_connection_error("anthropic")
    metrics.record_upstream_connection_error("openai")
    metrics.record_upstream_connection_error("openai")

    assert metrics.upstream_connection_errors_by_provider["anthropic"] == 1
    assert metrics.upstream_connection_errors_by_provider["openai"] == 2


def test_record_upstream_connection_error_empty_provider_defaults_to_unknown() -> None:
    metrics = PrometheusMetrics()

    metrics.record_upstream_connection_error("")

    assert metrics.upstream_connection_errors_by_provider["unknown"] == 1


async def test_upstream_connection_errors_exported_in_prometheus_text() -> None:
    metrics = PrometheusMetrics()
    metrics.record_upstream_connection_error("anthropic")

    text = await metrics.export()

    assert "# TYPE headroom_upstream_connection_errors_total counter" in text
    assert 'headroom_upstream_connection_errors_total{provider="anthropic"} 1' in text


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


# ---------------------------------------------------------------------------
# headroom_requests_rate_limited_total{source} / headroom_requests_failed_total{provider}
#
# Issue #3696: #3615 routed upstream 429s and 4xx into these two counters, which
# silently changed what they mean. The labels make the new meaning queryable —
# "my limiter throttled me" vs "the provider throttled me" are acted on
# differently, and a failure with no provider attached names no culprit.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limited_export_splits_headroom_from_upstream() -> None:
    metrics = PrometheusMetrics(stateless=True)

    # What the four handler sites do when Headroom's OWN limiter rejects.
    await metrics.record_rate_limited(provider="anthropic", source="headroom")
    # What the outcome funnel does with a provider 429.
    await metrics.record_rate_limited(provider="anthropic", source="upstream")
    await metrics.record_rate_limited(provider="openai", source="upstream")

    text = await metrics.export()

    assert "# TYPE headroom_requests_rate_limited_total counter" in text
    assert 'headroom_requests_rate_limited_total{source="headroom"} 1' in text
    assert 'headroom_requests_rate_limited_total{source="upstream"} 2' in text
    # `sum without (source)` must reproduce the pre-label value.
    assert metrics.requests_rate_limited == 3
    # No unlabelled sample alongside the labelled ones — that would double-count.
    assert "\nheadroom_requests_rate_limited_total 3" not in text


@pytest.mark.asyncio
async def test_rate_limited_source_series_exist_before_any_429() -> None:
    """Both series export from startup, so a rate() never appears mid-incident."""
    metrics = PrometheusMetrics(stateless=True)

    text = await metrics.export()

    assert 'headroom_requests_rate_limited_total{source="headroom"} 0' in text
    assert 'headroom_requests_rate_limited_total{source="upstream"} 0' in text


@pytest.mark.asyncio
async def test_rate_limited_source_defaults_to_headroom_and_clamps_unknown() -> None:
    metrics = PrometheusMetrics(stateless=True)

    await metrics.record_rate_limited(provider="anthropic")
    await metrics.record_rate_limited(provider="anthropic", source="bogus")

    text = await metrics.export()

    assert 'headroom_requests_rate_limited_total{source="headroom"} 2' in text
    assert 'headroom_requests_rate_limited_total{source="upstream"} 0' in text
    assert "bogus" not in text


@pytest.mark.asyncio
async def test_failed_export_is_attributed_to_the_provider_that_failed() -> None:
    metrics = PrometheusMetrics(stateless=True)

    await metrics.record_failed(provider="anthropic")
    await metrics.record_failed(provider="openai")
    await metrics.record_failed(provider="openai")

    text = await metrics.export()

    assert "# TYPE headroom_requests_failed_total counter" in text
    assert 'headroom_requests_failed_total{provider="anthropic"} 1' in text
    assert 'headroom_requests_failed_total{provider="openai"} 2' in text
    # `sum without (provider)` must reproduce the pre-label value.
    assert metrics.requests_failed == 3
    assert "\nheadroom_requests_failed_total 3" not in text


@pytest.mark.asyncio
async def test_failed_without_a_provider_lands_in_unknown() -> None:
    metrics = PrometheusMetrics(stateless=True)

    await metrics.record_failed()

    text = await metrics.export()

    assert 'headroom_requests_failed_total{provider="unknown"} 1' in text


@pytest.mark.asyncio
async def test_reset_runtime_clears_rate_limit_and_failure_splits() -> None:
    metrics = PrometheusMetrics(stateless=True)

    await metrics.record_rate_limited(provider="anthropic", source="upstream")
    await metrics.record_failed(provider="anthropic")

    await metrics.reset_runtime()

    assert metrics.requests_rate_limited == 0
    assert metrics.requests_failed == 0
    assert metrics.requests_rate_limited_by_source == {"headroom": 0, "upstream": 0}
    # Re-seeded, not emptied: the metric must keep exporting a sample after a
    # reset, same as the rate-limit source series above.
    assert dict(metrics.requests_failed_by_provider) == {"unknown": 0}


@pytest.mark.asyncio
async def test_failed_total_exports_a_sample_before_any_failure() -> None:
    """A healthy proxy must still export ``headroom_requests_failed_total``.

    Before the provider label this counter was always present as a bare
    ``headroom_requests_failed_total 0``. A labelled map that starts empty
    emits no sample at all, which makes the documented failure-rate query
    return an empty vector (an empty numerator empties the whole expression),
    so the panel reads "No data" instead of 0% while everything is fine --
    and `absent()` alerts fire on healthy proxies.
    """
    metrics = PrometheusMetrics(stateless=True)

    text = await metrics.export()

    assert 'headroom_requests_failed_total{provider="unknown"} 0' in text

    # The seed must not survive as a duplicate once a real provider fails.
    await metrics.record_failed(provider="anthropic")
    text = await metrics.export()
    assert 'headroom_requests_failed_total{provider="anthropic"} 1' in text
    assert 'headroom_requests_failed_total{provider="unknown"} 0' in text
    assert metrics.requests_failed == 1
