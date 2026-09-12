"""Proxy savings-ledger events record the original input as ``before``."""

from __future__ import annotations

from typing import Any

import pytest

from headroom.proxy import prometheus_metrics


class _FakeSavingsTracker:
    def snapshot(self) -> dict[str, dict[str, int | float]]:
        return {"lifetime": {"total_input_tokens": 0, "total_input_cost_usd": 0.0}}

    def record_request(self, **kwargs: Any) -> None:
        pass

    def record_lifetime_request(self, **kwargs: Any) -> None:
        pass


class _FakeOtelMetrics:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record_proxy_request(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


@pytest.mark.asyncio
async def test_record_savings_event_uses_original_input_as_before(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    def record_savings_event(**kwargs: Any) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(
        prometheus_metrics.savings_ledger,
        "record_savings_event",
        record_savings_event,
    )

    metrics = prometheus_metrics.PrometheusMetrics(
        savings_tracker=_FakeSavingsTracker(),
        otel_metrics=_FakeOtelMetrics(),
    )
    await metrics.record_request(
        provider="anthropic",
        model="claude-opus-4-6",
        input_tokens=600,
        output_tokens=25,
        tokens_saved=400,
        latency_ms=10.0,
        client="claude-code",
    )

    assert calls == [
        {
            "tokens_before": 1000,
            "tokens_after": 600,
            "model": "claude-opus-4-6",
            "client": "claude-code",
            "source": "proxy",
            # No cache breakdown on this request: the ledger gets no new-input
            # denominator rather than a zero it would divide by.
            "new_input_tokens": None,
            "deferred_tokens": 0,
        }
    ]

    # With a provider cache breakdown the ledger also gets the /stats
    # new-input denominator (uncached + cache write) and the deferral share of
    # the saving, so `headroom savings` can pair compression-only with new input.
    calls.clear()
    await metrics.record_request(
        provider="anthropic",
        model="claude-opus-4-6",
        input_tokens=600,
        output_tokens=25,
        tokens_saved=400,
        latency_ms=10.0,
        client="claude-code",
        cache_read_tokens=5000,
        cache_write_tokens=700,
        uncached_input_tokens=200,
        tool_search_saved=150,
    )
    assert calls[0]["tokens_before"] == 1000 + 150
    assert calls[0]["new_input_tokens"] == 900
    assert calls[0]["deferred_tokens"] == 150


@pytest.mark.asyncio
async def test_record_request_forwards_tool_savings_to_otel() -> None:
    otel = _FakeOtelMetrics()
    metrics = prometheus_metrics.PrometheusMetrics(
        savings_tracker=_FakeSavingsTracker(),
        otel_metrics=otel,
    )

    await metrics.record_request(
        provider="anthropic",
        model="claude-opus-4-6",
        input_tokens=600,
        output_tokens=25,
        tokens_saved=400,
        tool_search_saved=13182,
        latency_ms=10.0,
    )

    assert len(otel.calls) == 1
    assert otel.calls[0]["tokens_saved"] == 400
    assert otel.calls[0]["tool_search_saved"] == 13182


def _capture_ledger(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        prometheus_metrics.savings_ledger,
        "record_savings_event",
        lambda **kwargs: calls.append(kwargs),
    )
    return calls


@pytest.mark.asyncio
async def test_record_savings_event_includes_tool_search_deferral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """tool_search_deferral savings must ride into the ledger delta so
    `headroom savings` does not undercount tool-search sessions ~7-10x (#2795)."""
    calls = _capture_ledger(monkeypatch)
    metrics = prometheus_metrics.PrometheusMetrics(
        savings_tracker=_FakeSavingsTracker(),
        otel_metrics=_FakeOtelMetrics(),
    )
    await metrics.record_request(
        provider="anthropic",
        model="claude-opus-4-6",
        input_tokens=109844,  # forwarded (post-compression) message count
        output_tokens=25,
        tokens_saved=1896,
        tool_search_saved=13182,  # deferred tool schemas never sent
        latency_ms=10.0,
        client="claude-code",
    )

    assert len(calls) == 1
    # saved = tokens_saved + tool_search_saved; before = forwarded + saved.
    assert calls[0]["tokens_after"] == 109844
    assert calls[0]["tokens_before"] == 109844 + 1896 + 13182  # == 124922


@pytest.mark.asyncio
async def test_record_savings_event_written_for_deferral_only_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool-heavy turn can defer thousands of tool-schema tokens while
    tokens_saved is 0 (deferral does not move the message-level count). It must
    still be recorded, not dropped from the ledger (#2795)."""
    calls = _capture_ledger(monkeypatch)
    metrics = prometheus_metrics.PrometheusMetrics(
        savings_tracker=_FakeSavingsTracker(),
        otel_metrics=_FakeOtelMetrics(),
    )
    await metrics.record_request(
        provider="anthropic",
        model="claude-opus-4-6",
        input_tokens=50000,
        output_tokens=25,
        tokens_saved=0,
        tool_search_saved=13182,
        latency_ms=10.0,
        client="claude-code",
    )

    assert len(calls) == 1
    assert calls[0]["tokens_before"] == 50000 + 13182
    assert calls[0]["tokens_after"] == 50000


@pytest.mark.asyncio
async def test_zero_saving_request_with_cache_breakdown_reaches_the_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request that saved nothing but newly billed input is a denominator
    observation for the new-input basis, so it must reach the ledger. One
    without any breakdown is skipped as before."""
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        prometheus_metrics.savings_ledger,
        "record_savings_event",
        lambda **kwargs: calls.append(kwargs),
    )
    metrics = prometheus_metrics.PrometheusMetrics(
        savings_tracker=_FakeSavingsTracker(),
        otel_metrics=_FakeOtelMetrics(),
    )
    await metrics.record_request(
        provider="anthropic",
        model="claude-opus-4-6",
        input_tokens=10_000,
        output_tokens=25,
        tokens_saved=0,
        latency_ms=10.0,
        client="claude-code",
        cache_read_tokens=5_000,
        uncached_input_tokens=10_000,
    )
    assert len(calls) == 1
    assert calls[0]["tokens_before"] == calls[0]["tokens_after"]
    assert calls[0]["new_input_tokens"] == 10_000
    await metrics.record_request(
        provider="bedrock",
        model="claude-opus-4-6",
        input_tokens=10_000,
        output_tokens=25,
        tokens_saved=0,
        latency_ms=10.0,
        client="claude-code",
    )
    assert len(calls) == 1
