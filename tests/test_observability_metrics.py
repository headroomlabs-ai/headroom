"""Tests for OTEL-backed operational observability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from headroom.observability import HeadroomOtelMetrics, reset_otel_metrics, set_otel_metrics
from headroom.proxy.prometheus_metrics import PrometheusMetrics
from headroom.transforms.pipeline import TransformPipeline


def _collect_metrics(reader: InMemoryMetricReader) -> dict[str, Any]:
    data = reader.get_metrics_data()
    collected: dict[str, Any] = {}

    for resource_metric in data.resource_metrics:
        for scope_metric in resource_metric.scope_metrics:
            for metric in scope_metric.metrics:
                collected[metric.name] = metric

    return collected


def _find_point(metric: Any, **expected_attributes: Any) -> Any:
    for point in metric.data.data_points:
        if all(point.attributes.get(key) == value for key, value in expected_attributes.items()):
            return point
    raise AssertionError(f"No datapoint matched attributes: {expected_attributes}")


def test_headroom_otel_metrics_records_proxy_and_pipeline_metrics() -> None:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    otel_metrics = HeadroomOtelMetrics(meter_provider=provider)

    otel_metrics.record_proxy_request(
        provider="anthropic",
        model="claude-opus-4-6",
        input_tokens=120,
        output_tokens=30,
        tokens_saved=45,
        latency_ms=18.5,
        cached=True,
        overhead_ms=4.0,
        ttfb_ms=12.0,
        cache_read_tokens=25,
        cache_write_tokens=35,
        cache_write_5m_tokens=10,
        cache_write_1h_tokens=25,
        uncached_input_tokens=60,
    )
    otel_metrics.record_proxy_cache_bust(tokens_lost=7)
    otel_metrics.record_pipeline_run(
        model="claude-opus-4-6",
        provider="anthropic",
        tokens_before=120,
        tokens_after=75,
        duration_ms=6.5,
        timing={"_deep_copy": 0.2, "router": 3.5, "pipeline_total": 6.5},
        transforms_applied=["router:smart_crusher:0.35"],
        waste_signals={"json_bloat": 12},
    )

    metrics = _collect_metrics(reader)

    requests = metrics["headroom.proxy.requests"]
    request_point = _find_point(
        requests,
        provider="anthropic",
        model="claude-opus-4-6",
        cached=True,
    )
    assert request_point.value == 1

    latency = metrics["headroom.proxy.request.duration"]
    latency_point = _find_point(
        latency,
        provider="anthropic",
        model="claude-opus-4-6",
        cached=True,
    )
    assert latency_point.count == 1
    assert latency_point.sum == pytest.approx(0.0185)

    ttl_tokens = metrics["headroom.proxy.cache.write_ttl_tokens"]
    five_minute_ttl = _find_point(
        ttl_tokens,
        provider="anthropic",
        model="claude-opus-4-6",
        ttl="5m",
    )
    assert five_minute_ttl.value == 10

    compression_runs = metrics["headroom.compression.runs"]
    compression_point = _find_point(
        compression_runs,
        provider="anthropic",
        model="claude-opus-4-6",
    )
    assert compression_point.value == 1

    stage_duration = metrics["headroom.compression.stage.duration"]
    router_stage = _find_point(
        stage_duration,
        provider="anthropic",
        model="claude-opus-4-6",
        stage="router",
    )
    assert router_stage.count == 1
    assert router_stage.sum == pytest.approx(0.0035)

    assert len(stage_duration.data.data_points) == 1

    waste_tokens = metrics["headroom.compression.waste.tokens"]
    waste_point = _find_point(
        waste_tokens,
        provider="anthropic",
        model="claude-opus-4-6",
        signal="json_bloat",
    )
    assert waste_point.value == 12


@dataclass
class _SpyMetrics:
    pipeline_calls: list[dict[str, Any]] = field(default_factory=list)

    def record_pipeline_run(self, **kwargs: Any) -> None:
        self.pipeline_calls.append(kwargs)


@dataclass
class _SpyProxyMetrics:
    failed_calls: list[dict[str, Any]] = field(default_factory=list)
    rate_limited_calls: list[dict[str, Any]] = field(default_factory=list)

    def record_proxy_failed(self, **kwargs: Any) -> None:
        self.failed_calls.append(kwargs)

    def record_proxy_rate_limited(self, **kwargs: Any) -> None:
        self.rate_limited_calls.append(kwargs)


def test_transform_pipeline_simulate_skips_metric_recording() -> None:
    spy = _SpyMetrics()
    set_otel_metrics(spy)  # type: ignore[arg-type]

    try:
        pipeline = TransformPipeline(transforms=[])
        messages = [{"role": "user", "content": "hello world"}]

        pipeline.apply(messages, model="gpt-4o", model_limit=1024)
        assert len(spy.pipeline_calls) == 1

        pipeline.simulate(messages, model="gpt-4o", model_limit=1024)
        assert len(spy.pipeline_calls) == 1
    finally:
        reset_otel_metrics()


def test_proxy_failure_and_rate_limit_metrics_include_provider_labels() -> None:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    otel_metrics = HeadroomOtelMetrics(meter_provider=provider)

    otel_metrics.record_proxy_failed(provider="openai")
    otel_metrics.record_proxy_rate_limited(provider="anthropic", model="claude-sonnet")

    metrics = _collect_metrics(reader)

    failed_point = _find_point(metrics["headroom.proxy.requests.failed"], provider="openai")
    assert failed_point.value == 1

    rate_limited_point = _find_point(
        metrics["headroom.proxy.requests.rate_limited"],
        provider="anthropic",
        model="claude-sonnet",
    )
    assert rate_limited_point.value == 1


@pytest.mark.asyncio
async def test_prometheus_metrics_reads_late_configured_otel_metrics() -> None:
    spy = _SpyProxyMetrics()
    metrics = PrometheusMetrics()
    set_otel_metrics(spy)  # type: ignore[arg-type]

    try:
        await metrics.record_failed(provider="openai")
        await metrics.record_rate_limited(provider="anthropic", model="claude-sonnet")

        assert spy.failed_calls == [{"provider": "openai", "model": None}]
        assert spy.rate_limited_calls == [{"provider": "anthropic", "model": "claude-sonnet"}]
    finally:
        reset_otel_metrics()
