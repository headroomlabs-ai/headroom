"""Regression tests for observability status without optional OTEL extras."""

from __future__ import annotations

import builtins
from typing import Any

import pytest

from headroom.observability import (
    LangfuseTracingConfig,
    OTelMetricsConfig,
    configure_langfuse_tracing,
    configure_otel_metrics,
    get_langfuse_tracing_status,
    get_otel_metrics_status,
    reset_headroom_tracing,
    reset_otel_metrics,
)


def _block_otel_sdk_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def blocked_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name.startswith(("opentelemetry.exporter", "opentelemetry.sdk")):
            raise ImportError(f"blocked optional OTEL dependency: {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)


def test_otel_metrics_status_keeps_enabled_config_without_exporter_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_otel_metrics()
    _block_otel_sdk_imports(monkeypatch)

    try:
        configure_otel_metrics(
            OTelMetricsConfig(
                enabled=True,
                service_name="headroom-proxy",
                exporter="console",
            )
        )

        status = get_otel_metrics_status()
        assert status["configured"] is True
        assert status["enabled"] is True
        assert status["service_name"] == "headroom-proxy"
        assert status["exporter"] == "console"
    finally:
        reset_otel_metrics()


def test_langfuse_status_keeps_enabled_config_without_exporter_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_headroom_tracing()
    _block_otel_sdk_imports(monkeypatch)

    try:
        configure_langfuse_tracing(
            LangfuseTracingConfig(
                enabled=True,
                public_key="pk-lf-test",
                secret_key="sk-lf-test",
                base_url="https://cloud.langfuse.com",
                service_name="headroom-proxy",
            )
        )

        status = get_langfuse_tracing_status()
        assert status["configured"] is True
        assert status["enabled"] is True
        assert status["service_name"] == "headroom-proxy"
        assert status["endpoint"] == "https://cloud.langfuse.com/api/public/otel/v1/traces"
    finally:
        reset_headroom_tracing()
