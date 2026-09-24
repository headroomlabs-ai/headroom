"""Cumulative cache-bust losses overtaking savings must reach the operator.

``tokens_saved_total`` and ``cache_bust_tokens_lost`` were both recorded and
never compared, so a deployment where compression costs more cache than it
saves tokens looked, from the logs, exactly like one where it did not. The
dashboard has rendered the net for a while, which only helps an operator who
is looking at it.

The warning is edge-triggered: a losing deployment loses on every bust, so a
line per bust would be the same noise problem in a different place.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from headroom.proxy.cost import CostTracker, build_prefix_cache_stats
from headroom.proxy.prometheus_metrics import PrometheusMetrics


def _metrics_with_savings(tokens_saved: int) -> PrometheusMetrics:
    metrics = PrometheusMetrics()
    if tokens_saved:
        asyncio.run(
            metrics.record_request(
                provider="anthropic",
                model="claude-opus-4-6",
                input_tokens=1000,
                output_tokens=20,
                tokens_saved=tokens_saved,
                latency_ms=10.0,
            )
        )
    return metrics


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "event=net_tokens_negative" in r.getMessage()]


def test_warns_when_busts_overtake_savings(caplog: pytest.LogCaptureFixture) -> None:
    metrics = _metrics_with_savings(100)

    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        asyncio.run(metrics.record_cache_bust(500))

    warnings = _warnings(caplog)
    assert len(warnings) == 1
    assert "tokens_saved=100" in warnings[0]
    assert "tokens_lost_to_cache_bust=500" in warnings[0]
    assert "net_tokens=-400" in warnings[0]


def test_silent_while_still_net_positive(caplog: pytest.LogCaptureFixture) -> None:
    metrics = _metrics_with_savings(1000)

    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        asyncio.run(metrics.record_cache_bust(10))
        asyncio.run(metrics.record_cache_bust(10))

    assert _warnings(caplog) == []


def test_warns_once_per_crossing_not_once_per_bust(caplog: pytest.LogCaptureFixture) -> None:
    metrics = _metrics_with_savings(100)

    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        for _ in range(5):
            asyncio.run(metrics.record_cache_bust(200))

    assert len(_warnings(caplog)) == 1


def test_rearms_after_recovering(caplog: pytest.LogCaptureFixture) -> None:
    """Crossing back into profit and out again is a second event, not silence."""
    metrics = _metrics_with_savings(100)

    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        asyncio.run(metrics.record_cache_bust(200))  # net -100, warns
        asyncio.run(
            metrics.record_request(
                provider="anthropic",
                model="claude-opus-4-6",
                input_tokens=1000,
                output_tokens=20,
                tokens_saved=1000,
                latency_ms=10.0,
            )
        )
        asyncio.run(metrics.record_cache_bust(1))  # net +899, re-arms
        asyncio.run(metrics.record_cache_bust(5000))  # net negative again, warns

    assert len(_warnings(caplog)) == 2


def test_stats_payload_carries_the_flag() -> None:
    metrics = _metrics_with_savings(100)
    asyncio.run(metrics.record_cache_bust(500))

    stats = build_prefix_cache_stats(metrics, CostTracker())
    cvc = stats["compression_vs_cache"]

    assert cvc["net_tokens"] == -400
    assert cvc["net_is_negative"] is True


def test_stats_flag_is_false_when_winning() -> None:
    metrics = _metrics_with_savings(1000)
    asyncio.run(metrics.record_cache_bust(10))

    cvc = build_prefix_cache_stats(metrics, CostTracker())["compression_vs_cache"]

    assert cvc["net_tokens"] == 990
    assert cvc["net_is_negative"] is False
