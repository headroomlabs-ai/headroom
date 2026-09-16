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
        }
    ]


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
