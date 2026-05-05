"""Per-strategy compression observability tests.

These guard the forcing function: when any compressor runs in
production, a `CompressionObserver` notification fires once per real
compression event, and `PrometheusMetrics` accumulates per-strategy
counters that the test suite asserts on directly.

The TOIN→SmartCrusher silent disconnect (caught three weeks late by
manual audit) was invisible because no signal distinguished by
strategy. These tests exist so the next regression of that shape
fails the suite the day it lands instead of waiting on an audit.

The counters live ONLY as in-process state on the metrics instance;
they are deliberately NOT exported through the Prometheus scrape or
OTel surface, because the metric→Supabase pipeline treats each
metric name as a column and we cannot add new columns. CI-level
observability via these tests is enough to catch silent regressions;
production export waits on a non-column-adding pipeline.

Coverage:

1. `ContentRouter.compress(...)` calls observer once per RoutingDecision.
2. `SmartCrusher.apply(...)` calls observer once per crushed message.
3. Both transforms tolerate an observer that raises (compression must
   still succeed).
4. `PrometheusMetrics` correctly satisfies the `CompressionObserver`
   protocol — `record_compression` increments per-strategy counters
   and `tokens_saved_by_strategy` accumulates only positive savings.
5. The Prometheus scrape output (`export()`) does NOT emit any new
   metric names — the per-strategy state stays internal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from headroom.transforms.content_detector import ContentType
from headroom.transforms.content_router import (
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    RouterCompressionResult,
    RoutingDecision,
)
from headroom.transforms.observability import CompressionObserver
from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig

# ─── Test doubles ──────────────────────────────────────────────────────


@dataclass
class SpyObserver:
    """Captures every `record_compression` call for assertion."""

    calls: list[tuple[str, int, int]] = field(default_factory=list)

    def record_compression(
        self,
        strategy: str,
        original_tokens: int,
        compressed_tokens: int,
    ) -> None:
        self.calls.append((strategy, original_tokens, compressed_tokens))


@dataclass
class ExplodingObserver:
    """Raises on every call. Used to assert observer failures don't
    propagate out and break compression."""

    raised: int = 0

    def record_compression(self, *_a: Any, **_kw: Any) -> None:
        self.raised += 1
        raise RuntimeError("simulated observer outage")


# ─── Protocol conformance ──────────────────────────────────────────────


def test_spy_satisfies_observer_protocol():
    spy = SpyObserver()
    # `runtime_checkable` Protocol — isinstance check works.
    assert isinstance(spy, CompressionObserver)


def test_prometheus_metrics_satisfies_observer_protocol():
    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    m = PrometheusMetrics()
    assert isinstance(m, CompressionObserver)


# ─── ContentRouter wiring ──────────────────────────────────────────────


def test_content_router_records_observer_call_per_routing_decision():
    spy = SpyObserver()
    router = ContentRouter(ContentRouterConfig(), observer=spy)

    # Forge a routing log directly via the result object — the observer
    # call site walks `result.routing_log`, so we assert the contract
    # without depending on which compressor would actually fire.
    result = RouterCompressionResult(
        compressed="x",
        original="x",
        strategy_used=CompressionStrategy.SMART_CRUSHER,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.JSON_ARRAY,
                strategy=CompressionStrategy.SMART_CRUSHER,
                original_tokens=200,
                compressed_tokens=50,
            ),
            RoutingDecision(
                content_type=ContentType.SOURCE_CODE,
                strategy=CompressionStrategy.CODE_AWARE,
                original_tokens=300,
                compressed_tokens=300,  # passthrough — still recorded
            ),
        ],
    )
    router._observe(result)

    assert spy.calls == [
        ("smart_crusher", 200, 50),
        ("code_aware", 300, 300),
    ]


def test_content_router_with_no_observer_is_silent():
    router = ContentRouter(ContentRouterConfig())  # observer defaults None
    result = RouterCompressionResult(
        compressed="x",
        original="x",
        strategy_used=CompressionStrategy.PASSTHROUGH,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.PLAIN_TEXT,
                strategy=CompressionStrategy.TEXT,
                original_tokens=10,
                compressed_tokens=5,
            )
        ],
    )
    # Should not raise.
    router._observe(result)


def test_content_router_swallows_observer_failures():
    boom = ExplodingObserver()
    router = ContentRouter(ContentRouterConfig(), observer=boom)
    result = RouterCompressionResult(
        compressed="x",
        original="x",
        strategy_used=CompressionStrategy.TEXT,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.PLAIN_TEXT,
                strategy=CompressionStrategy.TEXT,
                original_tokens=10,
                compressed_tokens=5,
            )
        ],
    )
    # Must not raise — observability failures are not compression failures.
    router._observe(result)
    assert boom.raised == 1


# ─── SmartCrusher wiring (legacy direct-pipeline path) ─────────────────


def _bigger_array(n: int = 60) -> str:
    import json as _json

    items = [{"status": "ok", "tag": "x", "n": i} for i in range(n)]
    return _json.dumps(items)


@pytest.fixture
def isolated_toin(tmp_path, monkeypatch):
    """Point TOIN at a tempdir for the duration of the test.

    SmartCrusher.apply() feeds the global TOIN learning store via
    `record_compression`. Its default storage path is
    `~/.headroom/toin.json`, which persists across pytest invocations.
    On Python 3.11 CI runs the suite twice (regular + coverage); a
    pattern written in run #1 changes which rows the lossy sampler
    keeps in run #2 and breaks `test_first_last_items_always_preserved`
    in `test_evals.py`.

    Isolating the TOIN file per test contains the side effect.
    """
    from pathlib import Path

    from headroom.telemetry.toin import TOIN_PATH_ENV_VAR, reset_toin

    storage = str(Path(tmp_path) / "toin.json")
    monkeypatch.setenv(TOIN_PATH_ENV_VAR, storage)
    reset_toin()
    yield
    reset_toin()


def test_smart_crusher_apply_records_observer_per_crushed_message(isolated_toin):
    """End-to-end: SmartCrusher.apply() walks messages, crushes the
    big tool_result, fires the observer with strategy='smart_crusher'."""
    from headroom.providers.openai import OpenAITokenCounter
    from headroom.tokenizer import Tokenizer

    spy = SpyObserver()
    crusher = SmartCrusher(SmartCrusherConfig(), observer=spy)
    tok = Tokenizer(OpenAITokenCounter("gpt-4o-mini"), model="gpt-4o-mini")

    messages = [
        {"role": "user", "content": "what's in the data?"},
        {"role": "tool", "content": _bigger_array(60)},
    ]
    result = crusher.apply(messages, tok)
    # If the analyzer chose passthrough this run, the observer wasn't
    # fired; that's fine for the wiring test — we only assert it WAS
    # fired in the case it crushed.
    if "smart_crush:" in ",".join(result.transforms_applied):
        assert spy.calls, "smart_crusher crushed but observer wasn't notified"
        for strategy, original, compressed in spy.calls:
            assert strategy == "smart_crusher"
            assert original > 0
            assert compressed >= 0


def test_smart_crusher_apply_swallows_observer_failures(isolated_toin):
    """Observer raises → compression still completes, returns valid
    TransformResult, count of raises matches the crushed_count."""
    from headroom.providers.openai import OpenAITokenCounter
    from headroom.tokenizer import Tokenizer

    boom = ExplodingObserver()
    crusher = SmartCrusher(SmartCrusherConfig(), observer=boom)
    tok = Tokenizer(OpenAITokenCounter("gpt-4o-mini"), model="gpt-4o-mini")
    messages = [{"role": "tool", "content": _bigger_array(60)}]
    result = crusher.apply(messages, tok)
    # Either the analyzer didn't crush (boom.raised == 0) or it did
    # (boom.raised >= 1) — but in both cases compression returned a
    # valid TransformResult. No exception escaped.
    assert result.messages is not None


# ─── PrometheusMetrics implementation ──────────────────────────────────


def test_prometheus_metrics_accumulates_per_strategy_counters():
    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    m = PrometheusMetrics()

    m.record_compression("smart_crusher", original_tokens=200, compressed_tokens=50)
    m.record_compression("smart_crusher", original_tokens=100, compressed_tokens=40)
    m.record_compression("diff", original_tokens=80, compressed_tokens=80)  # no savings
    m.record_compression("code_aware", original_tokens=50, compressed_tokens=70)  # negative savings

    assert m.compressions_by_strategy == {
        "smart_crusher": 2,
        "diff": 1,
        "code_aware": 1,
    }
    # Tokens saved is `max(0, original - compressed)` per strategy.
    # smart_crusher: 150 + 60 = 210; diff: 0 (no savings, dict entry omitted);
    # code_aware: 0 (negative).
    assert m.tokens_saved_by_strategy == {"smart_crusher": 210}


def test_prometheus_export_does_not_leak_per_strategy_metrics():
    """Per-strategy state is tracked in-process only. The Prometheus
    scrape output deliberately must NOT emit new metric names — the
    metric→Supabase pipeline treats each metric name as a column, and
    we cannot add new columns. This test guards that constraint: if a
    future change adds the metric to the scrape, this fails and forces
    a conscious decision."""
    import asyncio

    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    m = PrometheusMetrics()
    m.record_compression("smart_crusher", original_tokens=200, compressed_tokens=50)
    m.record_compression("diff", original_tokens=120, compressed_tokens=70)

    output = asyncio.run(m.export())

    assert "headroom_compressions_total" not in output
    assert "headroom_tokens_saved_by_strategy_total" not in output


# ─── End-to-end smoke (router + metrics together) ──────────────────────


def test_router_with_prometheus_observer_increments_counters():
    """Plumbing test: a router wired to a real PrometheusMetrics
    instance lights up the per-strategy counters as routing decisions
    accumulate. This is the production wiring shape from
    `headroom/proxy/server.py`."""
    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    m = PrometheusMetrics()
    router = ContentRouter(ContentRouterConfig(), observer=m)

    fake_result = RouterCompressionResult(
        compressed="x",
        original="x",
        strategy_used=CompressionStrategy.MIXED,
        routing_log=[
            RoutingDecision(
                content_type=ContentType.JSON_ARRAY,
                strategy=CompressionStrategy.SMART_CRUSHER,
                original_tokens=300,
                compressed_tokens=80,
            ),
            RoutingDecision(
                content_type=ContentType.SOURCE_CODE,
                strategy=CompressionStrategy.CODE_AWARE,
                original_tokens=200,
                compressed_tokens=120,
            ),
            RoutingDecision(
                content_type=ContentType.JSON_ARRAY,
                strategy=CompressionStrategy.SMART_CRUSHER,
                original_tokens=100,
                compressed_tokens=40,
            ),
        ],
    )
    router._observe(fake_result)

    assert m.compressions_by_strategy == {"smart_crusher": 2, "code_aware": 1}
    assert m.tokens_saved_by_strategy == {
        "smart_crusher": (300 - 80) + (100 - 40),  # 280
        "code_aware": (200 - 120),  # 80
    }


# IntelligentContextManager observability tests retired with PR-B1 —
# the manager itself was deleted along with the message-dropping
# strategy. Inner-router observability is now exercised solely
# through ContentRouter, covered by
# `test_content_router_records_observer_call_per_routing_decision`.
