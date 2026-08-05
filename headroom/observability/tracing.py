"""OTEL tracing helpers for Headroom and Langfuse."""

from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from opentelemetry import trace

from .metrics import _headroom_version, _parse_bool, _parse_key_value_pairs

logger = logging.getLogger(__name__)

_SCOPE_NAME = "headroom"

_tracing_lock = Lock()
_global_tracer: HeadroomTracer | None = None
_owned_tracer_provider: Any | None = None
_owned_langfuse_config: LangfuseTracingConfig | None = None


@dataclass(slots=True)
class LangfuseTracingConfig:
    """Configuration for Headroom-managed Langfuse OTLP trace export."""

    enabled: bool = False
    public_key: str = field(default="", repr=False)
    secret_key: str = field(default="", repr=False)
    base_url: str = "https://cloud.langfuse.com"
    service_name: str = "headroom"
    resource_attributes: dict[str, str] = field(default_factory=dict)

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/api/public/otel/v1/traces"

    @property
    def auth_header(self) -> str:
        encoded = base64.b64encode(f"{self.public_key}:{self.secret_key}".encode()).decode()
        return f"Basic {encoded}"

    @property
    def headers(self) -> dict[str, str]:
        return {
            "Authorization": self.auth_header,
            "x-langfuse-ingestion-version": "4",
        }

    @classmethod
    def from_env(cls, *, default_service_name: str = "headroom") -> LangfuseTracingConfig:
        public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()

        return cls(
            enabled=_parse_bool(
                os.environ.get("HEADROOM_LANGFUSE_ENABLED"),
                default=False,
            ),
            public_key=public_key,
            secret_key=secret_key,
            base_url=(
                os.environ.get("LANGFUSE_BASE_URL")
                or os.environ.get("LANGFUSE_OTEL_HOST")
                or "https://cloud.langfuse.com"
            ).strip(),
            service_name=os.environ.get(
                "HEADROOM_LANGFUSE_SERVICE_NAME", default_service_name
            ).strip()
            or default_service_name,
            resource_attributes=_parse_key_value_pairs(
                os.environ.get("HEADROOM_LANGFUSE_RESOURCE_ATTRIBUTES")
            ),
        )

    def is_complete(self) -> bool:
        return bool(self.public_key and self.secret_key)

    def status(self) -> dict[str, Any]:
        return {
            "configured": True,
            "enabled": self.enabled,
            "service_name": self.service_name,
            "base_url": self.base_url,
            "endpoint": self.endpoint,
        }


class HeadroomTracer:
    """Tracer facade used by shared Headroom compression paths."""

    def __init__(self, tracer_provider: Any | None = None):
        if tracer_provider is None:
            self._tracer = trace.get_tracer(_SCOPE_NAME, _headroom_version())
        else:
            self._tracer = tracer_provider.get_tracer(_SCOPE_NAME, _headroom_version())

    def start_as_current_span(
        self,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
    ) -> Any:
        return self._tracer.start_as_current_span(
            name,
            attributes=attributes,
            record_exception=True,
            set_status_on_exception=True,
        )


def get_headroom_tracer() -> HeadroomTracer:
    global _global_tracer

    if _global_tracer is None:
        with _tracing_lock:
            if _global_tracer is None:
                _global_tracer = HeadroomTracer()

    return _global_tracer


def set_headroom_tracer(headroom_tracer: HeadroomTracer) -> HeadroomTracer:
    global _global_tracer
    with _tracing_lock:
        _global_tracer = headroom_tracer
    return headroom_tracer


def configure_langfuse_tracing(
    config: LangfuseTracingConfig | None = None,
) -> HeadroomTracer:
    global _global_tracer
    global _owned_tracer_provider
    global _owned_langfuse_config

    resolved = config or LangfuseTracingConfig()
    if not resolved.enabled:
        return get_headroom_tracer()
    if not resolved.is_complete():
        logger.warning(
            "Langfuse tracing is enabled but LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY are missing."
        )
        return get_headroom_tracer()

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "OpenTelemetry SDK/exporter packages are not installed. "
            "Install headroom-ai[otel] to enable Langfuse OTLP tracing."
        )
        return get_headroom_tracer()

    resource = Resource.create(
        {
            SERVICE_NAME: resolved.service_name,
            SERVICE_VERSION: _headroom_version(),
            **resolved.resource_attributes,
        }
    )
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(
                endpoint=resolved.endpoint,
                headers=resolved.headers,
            )
        )
    )
    headroom_tracer = HeadroomTracer(tracer_provider=tracer_provider)

    previous_provider = None
    with _tracing_lock:
        previous_provider = _owned_tracer_provider
        _owned_tracer_provider = tracer_provider
        _owned_langfuse_config = resolved
        _global_tracer = headroom_tracer

    if previous_provider is not None:
        try:
            previous_provider.shutdown()
        except Exception:
            logger.debug("Failed to shut down previous Langfuse tracer provider", exc_info=True)

    return headroom_tracer


def get_langfuse_tracing_status() -> dict[str, Any]:
    with _tracing_lock:
        if _owned_langfuse_config is not None:
            return _owned_langfuse_config.status()
    if not any(
        os.environ.get(name)
        for name in (
            "HEADROOM_LANGFUSE_ENABLED",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "LANGFUSE_BASE_URL",
        )
    ):
        return {
            "configured": False,
            "enabled": False,
            "service_name": None,
            "base_url": None,
            "endpoint": None,
        }
    return LangfuseTracingConfig.from_env(default_service_name="headroom-proxy").status()


def shutdown_headroom_tracing() -> None:
    global _global_tracer
    global _owned_tracer_provider
    global _owned_langfuse_config

    provider = None
    with _tracing_lock:
        provider = _owned_tracer_provider
        _owned_tracer_provider = None
        _owned_langfuse_config = None
        _global_tracer = None

    if provider is not None:
        try:
            provider.shutdown()
        except Exception:
            logger.debug("Failed to shut down Langfuse tracer provider", exc_info=True)


def reset_headroom_tracing() -> None:
    shutdown_headroom_tracing()
