"""Tests for Langfuse/OTEL tracing helpers."""

from __future__ import annotations

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from headroom.observability import (
    HeadroomTracer,
    LangfuseTracingConfig,
    get_langfuse_tracing_status,
    reset_headroom_tracing,
    set_headroom_tracer,
)
from headroom.transforms.pipeline import TransformPipeline


def test_langfuse_tracing_config_builds_trace_endpoint() -> None:
    config = LangfuseTracingConfig(
        enabled=True,
        public_key="pk-lf-test",
        secret_key="sk-lf-test",
        base_url="https://cloud.langfuse.com",
        service_name="headroom-proxy",
    )

    assert config.endpoint == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    assert config.headers["x-langfuse-ingestion-version"] == "4"
    assert config.headers["Authorization"].startswith("Basic ")
    assert "sk-lf-test" not in repr(config)


def test_transform_pipeline_emits_trace_spans() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource.create({"service.name": "headroom-test"}))
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_headroom_tracer(HeadroomTracer(tracer_provider=provider))

    try:
        pipeline = TransformPipeline(transforms=[])
        messages = [{"role": "user", "content": "hello world"}]
        pipeline.apply(messages, model="gpt-4o", model_limit=1024)

        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        span = spans[0]
        assert span.name == "headroom.compression.pipeline"
        assert span.attributes["headroom.model"] == "gpt-4o"
        assert span.attributes["headroom.tokens.before"] >= 1
        assert span.attributes["headroom.tokens.after"] >= 1
    finally:
        reset_headroom_tracing()


def test_langfuse_tracing_status_defaults_to_unconfigured() -> None:
    reset_headroom_tracing()
    status = get_langfuse_tracing_status()
    assert status["configured"] is False
    assert status["enabled"] is False


def test_langfuse_tracing_requires_explicit_enable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")

    config = LangfuseTracingConfig.from_env(default_service_name="headroom-proxy")

    assert config.enabled is False
