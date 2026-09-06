"""Tests for OTEL-backed operational observability."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from headroom.observability import (
    HeadroomOtelMetrics,
    get_otel_meter,
    register_otel_metric_attribute_provider,
    reset_otel_metrics,
    set_otel_metrics,
    unregister_otel_metric_attribute_provider,
)
from headroom.proxy.prometheus_metrics import MAX_DISTINCT_PATHS, PrometheusMetrics
from headroom.telemetry.context import MAX_DISTINCT_MODELS
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
        tool_search_saved=15,
        latency_ms=18.5,
        cached=True,
        overhead_ms=4.0,
        ttfb_ms=12.0,
        cache_read_tokens=25,
        cache_write_tokens=35,
        cache_write_5m_tokens=10,
        cache_write_1h_tokens=25,
        uncached_input_tokens=60,
        attempted_input_tokens=165,
        output_tokens_saved=8,
        savings_usd={
            "compression": 0.001,
            "tool_schema": 0.0003,
            "output_shaping": 0.0008,
            "provider_cache": 0.0002,
        },
        project="checkout",
        client="claude-code",
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

    saved_tokens = metrics["headroom.proxy.tokens.saved"]
    saved_point = _find_point(
        saved_tokens,
        provider="anthropic",
        model="claude-opus-4-6",
        cached=True,
    )
    assert saved_point.value == 60

    tool_schema_saved = metrics["headroom.proxy.tokens.tool_schema_saved"]
    tool_schema_point = _find_point(
        tool_schema_saved,
        provider="anthropic",
        model="claude-opus-4-6",
        cached=True,
    )
    assert tool_schema_point.value == 15

    attempted_input = metrics["headroom.proxy.tokens.attempted_input"]
    attempted_point = _find_point(
        attempted_input,
        **{
            "headroom.project": "checkout",
            "headroom.client": "claude-code",
        },
    )
    assert attempted_point.value == 165

    output_saved = metrics["headroom.proxy.tokens.output_saved"]
    output_saved_point = _find_point(
        output_saved,
        **{
            "headroom.project": "checkout",
            "headroom.client": "claude-code",
        },
    )
    assert output_saved_point.value == 8

    savings_usd = metrics["headroom.proxy.savings.usd"]
    compression_usd = _find_point(savings_usd, source="compression", estimated=True)
    assert compression_usd.value == pytest.approx(0.001)

    compression_saved = metrics["headroom.compression.tokens.saved"]
    compression_saved_point = _find_point(
        compression_saved,
        provider="anthropic",
        model="claude-opus-4-6",
    )
    assert compression_saved_point.value == 45

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


def test_get_otel_meter_uses_headrooms_configured_provider() -> None:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    set_otel_metrics(HeadroomOtelMetrics(meter_provider=provider))

    try:
        meter = get_otel_meter("example.integration", "1.0.0")
        meter.create_counter("example.integration.events").add(1, {"source": "test"})

        metric = _collect_metrics(reader)["example.integration.events"]
        point = _find_point(metric, source="test")
        assert point.value == 1
    finally:
        reset_otel_metrics()


def test_request_attribute_provider_enriches_core_and_savings_metrics() -> None:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    otel_metrics = HeadroomOtelMetrics(meter_provider=provider)

    def identity_attributes() -> dict[str, str]:
        return {
            "headroom.org": "acme",
            "headroom.team": "payments",
            "headroom.user": "alice",
            # Canonical call-site dimensions must win over an extension.
            "model": "must-not-override",
            "source": "must-not-override",
        }

    register_otel_metric_attribute_provider(identity_attributes)
    try:
        otel_metrics.record_proxy_request(
            provider="anthropic",
            model="claude-sonnet-4-5",
            input_tokens=100,
            output_tokens=10,
            tokens_saved=25,
            latency_ms=20,
        )
        otel_metrics.record_savings_attribution(
            [{"source": "tool_search", "tokens": 20, "usd": 0.001}]
        )

        metrics = _collect_metrics(reader)
        request = _find_point(
            metrics["headroom.proxy.requests"],
            model="claude-sonnet-4-5",
            **{
                "headroom.org": "acme",
                "headroom.team": "payments",
                "headroom.user": "alice",
            },
        )
        assert request.value == 1
        attributed = _find_point(
            metrics["headroom.savings.attributed.tokens"],
            source="tool_search",
            **{"headroom.user": "alice"},
        )
        assert attributed.value == 20
    finally:
        unregister_otel_metric_attribute_provider(identity_attributes)


def test_failing_request_attribute_provider_is_fail_open() -> None:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    otel_metrics = HeadroomOtelMetrics(meter_provider=provider)

    def broken_provider() -> dict[str, str]:
        raise RuntimeError("identity unavailable")

    register_otel_metric_attribute_provider(broken_provider)
    try:
        otel_metrics.record_proxy_failed(provider="openai", model="gpt-5")
        point = _find_point(
            _collect_metrics(reader)["headroom.proxy.requests.failed"],
            provider="openai",
            model="gpt-5",
        )
        assert point.value == 1
    finally:
        unregister_otel_metric_attribute_provider(broken_provider)


@dataclass
class _SpyMetrics:
    pipeline_calls: list[dict[str, Any]] = field(default_factory=list)

    def record_pipeline_run(self, **kwargs: Any) -> None:
        self.pipeline_calls.append(kwargs)


@dataclass
class _SpyProxyMetrics:
    request_calls: list[dict[str, Any]] = field(default_factory=list)
    failed_calls: list[dict[str, Any]] = field(default_factory=list)
    rate_limited_calls: list[dict[str, Any]] = field(default_factory=list)

    def record_proxy_request(self, **kwargs: Any) -> None:
        self.request_calls.append(kwargs)

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


@pytest.mark.asyncio
async def test_prometheus_metrics_forwards_savings_drilldown_fields_to_otel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_usd = {
        "compression": 0.003,
        "tool_schema": 0.0,
        "output_shaping": 0.004,
        "provider_cache": 0.0,
    }
    monkeypatch.setattr(
        "headroom.proxy.prometheus_metrics.estimate_request_savings_usd",
        lambda *_args, **_kwargs: expected_usd,
    )
    spy = _SpyProxyMetrics()
    metrics = PrometheusMetrics(stateless=True)
    set_otel_metrics(spy)  # type: ignore[arg-type]

    try:
        await metrics.record_request(
            provider="anthropic",
            model="claude-sonnet-4-5",
            input_tokens=90,
            output_tokens=12,
            tokens_saved=30,
            latency_ms=5.0,
            attempted_input_tokens=120,
            output_tokens_saved=4,
            project="checkout",
            client="claude-code",
        )

        assert len(spy.request_calls) == 1
        call = spy.request_calls[0]
        assert call["attempted_input_tokens"] == 120
        assert call["output_tokens_saved"] == 4
        assert call["savings_usd"] == expected_usd
        assert call["project"] == "checkout"
        assert call["client"] == "claude-code"
    finally:
        reset_otel_metrics()


@pytest.mark.asyncio
async def test_prometheus_metrics_clamps_negative_token_savings() -> None:
    metrics = PrometheusMetrics()

    await metrics.record_request(
        provider="openai",
        model="openai-compatible",
        input_tokens=100,
        output_tokens=5,
        tokens_saved=-25,
        latency_ms=1.0,
    )

    assert metrics.tokens_saved_total == 0
    assert metrics.savings_history[-1][1] == 0


@pytest.mark.asyncio
async def test_prometheus_metrics_caps_model_cardinality() -> None:
    """A client sending unbounded distinct models cannot grow the per-model dicts
    past MAX_DISTINCT_MODELS + the "other" sentinel, while accounting stays exact."""
    metrics = PrometheusMetrics(stateless=True)

    async def record(model: str) -> None:
        await metrics.record_request(
            provider="anthropic",
            model=model,
            input_tokens=10,
            output_tokens=1,
            tokens_saved=1,
            latency_ms=1.0,
            cache_read_tokens=1,  # enter the prefix-cache block -> _cache_requests_by_model
        )

    # Fill exactly to the cap with distinct models: no bucketing yet.
    for i in range(MAX_DISTINCT_MODELS):
        await record(f"model_{i}")
    assert len(metrics.requests_by_model) == MAX_DISTINCT_MODELS
    assert len(metrics._cache_requests_by_model) == MAX_DISTINCT_MODELS
    assert "other" not in metrics.requests_by_model

    # New distinct models past the cap bucket into "other", never their own key.
    for i in range(5):
        await record(f"overflow_{i}")
    assert "overflow_0" not in metrics.requests_by_model
    assert metrics.requests_by_model["other"] == 5
    assert metrics._cache_requests_by_model["other"] == 5
    assert len(metrics.requests_by_model) == MAX_DISTINCT_MODELS + 1
    assert len(metrics._cache_requests_by_model) == MAX_DISTINCT_MODELS + 1

    # An already-tracked model keeps incrementing after the cap is reached.
    await record("model_0")
    assert metrics.requests_by_model["model_0"] == 2

    # Accounting is preserved: every request is counted somewhere.
    total_calls = MAX_DISTINCT_MODELS + 5 + 1
    assert metrics.requests_total == total_calls
    assert sum(metrics.requests_by_model.values()) == total_calls


@pytest.mark.asyncio
async def test_prometheus_metrics_model_cardinality_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Bucketing into "other" logs exactly one warning, not one per request."""
    metrics = PrometheusMetrics(stateless=True)
    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        for i in range(MAX_DISTINCT_MODELS + 10):
            await metrics.record_request(
                provider="openai",
                model=f"model_{i}",
                input_tokens=10,
                output_tokens=1,
                tokens_saved=1,
                latency_ms=1.0,
            )
    cap_warnings = [r for r in caplog.records if "cardinality cap" in r.getMessage()]
    assert len(cap_warnings) == 1


@pytest.mark.asyncio
async def test_prometheus_metrics_reset_rearms_cardinality_warning() -> None:
    """reset_runtime clears the model dicts and re-arms the one-shot cap warning."""
    metrics = PrometheusMetrics(stateless=True)
    for i in range(MAX_DISTINCT_MODELS + 5):
        await metrics.record_request(
            provider="openai",
            model=f"model_{i}",
            input_tokens=1,
            output_tokens=1,
            tokens_saved=1,
            latency_ms=1.0,
            cache_read_tokens=1,
        )
    assert metrics._model_cardinality_warned is True

    await metrics.reset_runtime()

    assert metrics._model_cardinality_warned is False
    assert len(metrics.requests_by_model) == 0
    assert len(metrics._cache_requests_by_model) == 0


def test_prometheus_metrics_caps_inbound_path_cardinality() -> None:
    """A client hitting unbounded distinct paths (ID-bearing passthrough routes)
    cannot grow inbound_requests_by_path past MAX_DISTINCT_PATHS + the "other"
    sentinel, while the total request count stays exact."""
    metrics = PrometheusMetrics(stateless=True)

    # Fill exactly to the cap with distinct paths: no bucketing yet.
    for i in range(MAX_DISTINCT_PATHS):
        metrics.record_inbound_request(method="GET", path=f"/v1/files/file_{i}")
    assert len(metrics.inbound_requests_by_path) == MAX_DISTINCT_PATHS
    assert "other" not in metrics.inbound_requests_by_path

    # New distinct paths past the cap bucket into "other", never their own key.
    for i in range(5):
        metrics.record_inbound_request(method="GET", path=f"/v1/responses/resp_{i}")
    assert "/v1/responses/resp_0" not in metrics.inbound_requests_by_path
    assert metrics.inbound_requests_by_path["other"] == 5
    assert len(metrics.inbound_requests_by_path) == MAX_DISTINCT_PATHS + 1

    # An already-tracked path keeps incrementing after the cap is reached.
    metrics.record_inbound_request(method="GET", path="/v1/files/file_0")
    assert metrics.inbound_requests_by_path["/v1/files/file_0"] == 2

    # Accounting is preserved: every inbound request is counted somewhere.
    total_calls = MAX_DISTINCT_PATHS + 5 + 1
    assert metrics.inbound_requests_total == total_calls
    assert sum(metrics.inbound_requests_by_path.values()) == total_calls


def test_prometheus_metrics_path_cardinality_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Bucketing inbound paths into "other" logs exactly one warning."""
    metrics = PrometheusMetrics(stateless=True)
    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        for i in range(MAX_DISTINCT_PATHS + 10):
            metrics.record_inbound_request(method="GET", path=f"/v1/files/file_{i}")
    cap_warnings = [r for r in caplog.records if "path cardinality cap" in r.getMessage()]
    assert len(cap_warnings) == 1


@pytest.mark.asyncio
async def test_prometheus_metrics_reset_rearms_path_cardinality_warning() -> None:
    """reset_runtime clears the path dict and re-arms the one-shot cap warning."""
    metrics = PrometheusMetrics(stateless=True)
    for i in range(MAX_DISTINCT_PATHS + 5):
        metrics.record_inbound_request(method="GET", path=f"/v1/files/file_{i}")
    assert metrics._path_cardinality_warned is True

    await metrics.reset_runtime()

    assert metrics._path_cardinality_warned is False
    assert len(metrics.inbound_requests_by_path) == 0


@pytest.mark.asyncio
async def test_prometheus_metrics_export_bounds_model_series() -> None:
    """export() emits at most MAX_DISTINCT_MODELS model series plus the 'other' bucket."""
    metrics = PrometheusMetrics(stateless=True)
    for i in range(MAX_DISTINCT_MODELS + 20):
        await metrics.record_request(
            provider="openai",
            model=f"model_{i}",
            input_tokens=1,
            output_tokens=1,
            tokens_saved=1,
            latency_ms=1.0,
        )
    text = await metrics.export()
    series = text.count("headroom_requests_by_model{")
    assert series <= MAX_DISTINCT_MODELS + 1
    assert 'headroom_requests_by_model{model="other"}' in text
