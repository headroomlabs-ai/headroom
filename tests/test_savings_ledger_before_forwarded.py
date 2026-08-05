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
    def record_proxy_request(self, **kwargs: Any) -> None:
        pass


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
