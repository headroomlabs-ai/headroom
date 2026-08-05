"""Tests forwarding existing proxy metric events to Lifetime storage."""

from __future__ import annotations

import asyncio

from headroom.proxy.prometheus_metrics import PrometheusMetrics
from headroom.proxy.savings_tracker import SavingsTracker


def test_runtime_metric_events_feed_lifetime_without_resetting_runtime_counters(tmp_path) -> None:
    tracker = SavingsTracker(path=str(tmp_path / "proxy_savings.json"), save_flush_every=25)
    metrics = PrometheusMetrics(savings_tracker=tracker)

    metrics.record_stack("codex")
    asyncio.run(
        metrics.record_request(
            provider="anthropic",
            model="claude-test",
            input_tokens=10,
            output_tokens=3,
            tokens_saved=2,
            latency_ms=1,
            cached=True,
            attempted_input_tokens=12,
            cache_read_tokens=5,
            cache_write_1h_tokens=2,
            waste_signals={"repetition": 4},
        )
    )
    asyncio.run(metrics.record_failed(provider="anthropic", model="claude-test"))
    asyncio.run(metrics.record_rate_limited(provider="anthropic", model="claude-test"))
    asyncio.run(metrics.record_cache_bust(tokens_lost=7))
    asyncio.run(metrics.record_cache_miss_attribution("anthropic", "prefix_change"))

    lifetime = tracker.lifetime_response()

    assert lifetime["requests"]["total"] == 1
    assert lifetime["requests"]["cached"] == 1
    assert lifetime["requests"]["failed"] == 1
    assert lifetime["requests"]["rate_limited"] == 1
    assert lifetime["requests"]["by_provider"] == {"anthropic": 1}
    assert lifetime["requests"]["by_stack"] == {"codex": 1}
    assert lifetime["tokens"]["output"] == 3
    assert lifetime["tokens"]["attempted_input"] == 12
    assert lifetime["prefix_cache"]["bust_tokens"] == 7
    assert lifetime["prefix_cache"]["misses_by_reason"] == {"prefix_change": 1}
    assert lifetime["waste_signals"] == {"repetition": 4}
    assert metrics.requests_total == 1
