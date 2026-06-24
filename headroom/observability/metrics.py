"""OpenTelemetry-backed operational metrics for Headroom."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from threading import Lock
from typing import Any, Literal

from opentelemetry import metrics
from opentelemetry.metrics import CallbackOptions, Observation

logger = logging.getLogger(__name__)

MetricExporter = Literal["console", "otlp_http"]

_SCOPE_NAME = "headroom"
_DEFAULT_EXPORT_INTERVAL_MS = 10000
_MILLISECONDS_TO_SECONDS = 1000.0

_metrics_lock = Lock()
_global_metrics: HeadroomOtelMetrics | None = None
_owned_meter_provider: Any | None = None
_owned_metrics_config: OTelMetricsConfig | None = None


def _headroom_version() -> str:
    try:
        return package_version("headroom-ai")
    except PackageNotFoundError:
        return "unknown"


def _parse_bool(raw: str | None, default: bool = False) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_int(raw: str | None, default: int) -> int:
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return value if value > 0 else default


def _parse_key_value_pairs(raw: str | None) -> dict[str, str]:
    if raw is None:
        return {}

    pairs: dict[str, str] = {}
    for item in raw.split(","):
        part = item.strip()
        if not part or "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            pairs[key] = value
    return pairs


@dataclass(slots=True)
class OTelMetricsConfig:
    """Configuration for Headroom-managed OTEL metric export."""

    enabled: bool = False
    service_name: str = "headroom"
    exporter: MetricExporter = "otlp_http"
    endpoint: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    export_interval_millis: int = _DEFAULT_EXPORT_INTERVAL_MS
    resource_attributes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls, *, default_service_name: str = "headroom") -> OTelMetricsConfig:
        exporter_raw = (
            os.environ.get("HEADROOM_OTEL_METRICS_EXPORTER", "otlp_http")
            .strip()
            .lower()
            .replace("-", "_")
        )
        if exporter_raw not in {"console", "otlp_http"}:
            logger.warning(
                "Unknown HEADROOM_OTEL_METRICS_EXPORTER=%s; falling back to otlp_http",
                exporter_raw,
            )
            exporter_raw = "otlp_http"

        return cls(
            enabled=_parse_bool(os.environ.get("HEADROOM_OTEL_METRICS_ENABLED"), default=False),
            service_name=os.environ.get("HEADROOM_OTEL_SERVICE_NAME", default_service_name).strip()
            or default_service_name,
            exporter=exporter_raw,  # type: ignore[arg-type]
            endpoint=os.environ.get("HEADROOM_OTEL_METRICS_ENDPOINT") or None,
            headers=_parse_key_value_pairs(os.environ.get("HEADROOM_OTEL_METRICS_HEADERS")),
            export_interval_millis=_parse_int(
                os.environ.get("HEADROOM_OTEL_METRICS_EXPORT_INTERVAL_MS"),
                _DEFAULT_EXPORT_INTERVAL_MS,
            ),
            resource_attributes=_parse_key_value_pairs(
                os.environ.get("HEADROOM_OTEL_RESOURCE_ATTRIBUTES")
            ),
        )

    def status(self) -> dict[str, Any]:
        return {
            "configured": True,
            "enabled": self.enabled,
            "service_name": self.service_name,
            "exporter": self.exporter,
            "endpoint": self.endpoint,
            "resource_attributes": dict(self.resource_attributes),
        }


class HeadroomOtelMetrics:
    """Shared OTEL metrics facade for Headroom operations."""

    def __init__(self, meter_provider: Any | None = None):
        if meter_provider is None:
            self._meter = metrics.get_meter(_SCOPE_NAME, _headroom_version())
        else:
            self._meter = meter_provider.get_meter(_SCOPE_NAME, _headroom_version())

        self._proxy_requests = self._meter.create_counter(
            "headroom.proxy.requests",
            description="Proxy requests handled by Headroom.",
            unit="1",
        )
        self._proxy_cached_requests = self._meter.create_counter(
            "headroom.proxy.requests.cached",
            description="Proxy requests served with provider cache participation.",
            unit="1",
        )
        self._proxy_failed_requests = self._meter.create_counter(
            "headroom.proxy.requests.failed",
            description="Proxy requests that failed.",
            unit="1",
        )
        self._proxy_rate_limited_requests = self._meter.create_counter(
            "headroom.proxy.requests.rate_limited",
            description="Proxy requests rejected by rate limiting.",
            unit="1",
        )
        self._proxy_input_tokens = self._meter.create_counter(
            "headroom.proxy.tokens.input",
            description="Input tokens received by the proxy.",
            unit="1",
        )
        self._proxy_output_tokens = self._meter.create_counter(
            "headroom.proxy.tokens.output",
            description="Output tokens returned by upstream providers.",
            unit="1",
        )
        self._proxy_saved_tokens = self._meter.create_counter(
            "headroom.proxy.tokens.saved",
            description="Input tokens saved by Headroom compression.",
            unit="1",
        )
        self._proxy_cache_read_tokens = self._meter.create_counter(
            "headroom.proxy.cache.read_tokens",
            description="Provider cache read tokens observed by the proxy.",
            unit="1",
        )
        self._proxy_cache_write_tokens = self._meter.create_counter(
            "headroom.proxy.cache.write_tokens",
            description="Provider cache write tokens observed by the proxy.",
            unit="1",
        )
        self._proxy_cache_write_ttl_tokens = self._meter.create_counter(
            "headroom.proxy.cache.write_ttl_tokens",
            description="Provider cache write tokens by observed TTL bucket.",
            unit="1",
        )
        self._proxy_uncached_input_tokens = self._meter.create_counter(
            "headroom.proxy.cache.uncached_input_tokens",
            description="Proxy input tokens not served from provider cache.",
            unit="1",
        )
        self._proxy_cache_busts = self._meter.create_counter(
            "headroom.proxy.cache.busts",
            description="Requests that lost provider cache efficiency.",
            unit="1",
        )
        self._proxy_cache_bust_tokens_lost = self._meter.create_counter(
            "headroom.proxy.cache.bust_tokens_lost",
            description="Tokens that lost provider cache discount because of compression.",
            unit="1",
        )
        self._proxy_latency = self._meter.create_histogram(
            "headroom.proxy.request.duration",
            description="End-to-end proxy request duration.",
            unit="s",
        )
        self._proxy_overhead = self._meter.create_histogram(
            "headroom.proxy.overhead.duration",
            description="Time spent inside Headroom optimization logic.",
            unit="s",
        )
        self._proxy_ttfb = self._meter.create_histogram(
            "headroom.proxy.ttfb.duration",
            description="Upstream time to first byte observed by Headroom.",
            unit="s",
        )
        self._compression_runs = self._meter.create_counter(
            "headroom.compression.runs",
            description="Compression pipeline runs executed by Headroom.",
            unit="1",
        )
        self._compression_failures = self._meter.create_counter(
            "headroom.compression.failures",
            description="Compression operations that failed before producing a result.",
            unit="1",
        )
        self._compression_input_tokens = self._meter.create_counter(
            "headroom.compression.tokens.input",
            description="Input tokens analyzed by Headroom compression.",
            unit="1",
        )
        self._compression_output_tokens = self._meter.create_counter(
            "headroom.compression.tokens.output",
            description="Output tokens produced by Headroom compression.",
            unit="1",
        )
        self._compression_saved_tokens = self._meter.create_counter(
            "headroom.compression.tokens.saved",
            description="Tokens removed by Headroom compression.",
            unit="1",
        )
        self._compression_duration = self._meter.create_histogram(
            "headroom.compression.pipeline.duration",
            description="Compression pipeline execution duration.",
            unit="s",
        )
        self._compression_stage_duration = self._meter.create_histogram(
            "headroom.compression.stage.duration",
            description="Per-stage compression timing emitted by the pipeline.",
            unit="s",
        )
        self._compression_transforms = self._meter.create_counter(
            "headroom.compression.transforms",
            description="Transforms applied during compression.",
            unit="1",
        )
        self._waste_signal_tokens = self._meter.create_counter(
            "headroom.compression.waste.tokens",
            description="Waste tokens detected in compressed inputs.",
            unit="1",
        )

        # Backing values updated by record_subscription_window()
        self._sub_5h_util_val: float = 0.0
        self._sub_7d_util_val: float = 0.0
        self._sub_5h_reset_val: float = 0.0
        self._sub_7d_reset_val: float = 0.0
        self._sub_overage_val: float = 0.0

        # Subscription window gauges (Anthropic OAuth accounts)
        def _cb_5h_util(opts: CallbackOptions) -> list[Observation]:
            return [Observation(self._sub_5h_util_val)]

        def _cb_7d_util(opts: CallbackOptions) -> list[Observation]:
            return [Observation(self._sub_7d_util_val)]

        def _cb_5h_reset(opts: CallbackOptions) -> list[Observation]:
            return [Observation(self._sub_5h_reset_val)]

        def _cb_7d_reset(opts: CallbackOptions) -> list[Observation]:
            return [Observation(self._sub_7d_reset_val)]

        def _cb_overage(opts: CallbackOptions) -> list[Observation]:
            return [Observation(self._sub_overage_val)]

        self._meter.create_observable_gauge(
            "headroom.subscription.5h_utilization_pct",
            description="Anthropic 5-hour rate-limit window utilisation (0–100%).",
            unit="1",
            callbacks=[_cb_5h_util],
        )
        self._meter.create_observable_gauge(
            "headroom.subscription.7d_utilization_pct",
            description="Anthropic 7-day rate-limit window utilisation (0–100%).",
            unit="1",
            callbacks=[_cb_7d_util],
        )
        self._meter.create_observable_gauge(
            "headroom.subscription.5h_seconds_to_reset",
            description="Seconds until the Anthropic 5-hour window resets.",
            unit="s",
            callbacks=[_cb_5h_reset],
        )
        self._meter.create_observable_gauge(
            "headroom.subscription.7d_seconds_to_reset",
            description="Seconds until the Anthropic 7-day window resets.",
            unit="s",
            callbacks=[_cb_7d_reset],
        )
        self._meter.create_observable_gauge(
            "headroom.subscription.overage_usd",
            description="Anthropic extra-usage (overage) credits consumed in USD.",
            unit="USD",
            callbacks=[_cb_overage],
        )

    @staticmethod
    def _attrs(**attrs: Any) -> dict[str, Any]:
        filtered: dict[str, Any] = {}
        for key, value in attrs.items():
            if value is None or value == "":
                continue
            filtered[key] = value
        return filtered

    def record_proxy_request(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        tokens_saved: int,
        latency_ms: float,
        cached: bool = False,
        overhead_ms: float = 0.0,
        ttfb_ms: float = 0.0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        cache_write_5m_tokens: int = 0,
        cache_write_1h_tokens: int = 0,
        uncached_input_tokens: int = 0,
    ) -> None:
        attrs = self._attrs(provider=provider, model=model, cached=cached)

        self._proxy_requests.add(1, attrs)
        if cached:
            self._proxy_cached_requests.add(1, attrs)

        self._proxy_input_tokens.add(max(input_tokens, 0), attrs)
        self._proxy_output_tokens.add(max(output_tokens, 0), attrs)
        self._proxy_saved_tokens.add(max(tokens_saved, 0), attrs)
        self._proxy_latency.record(max(latency_ms, 0.0) / _MILLISECONDS_TO_SECONDS, attrs)

        if overhead_ms > 0:
            self._proxy_overhead.record(overhead_ms / _MILLISECONDS_TO_SECONDS, attrs)
        if ttfb_ms > 0:
            self._proxy_ttfb.record(ttfb_ms / _MILLISECONDS_TO_SECONDS, attrs)
        if cache_read_tokens > 0:
            self._proxy_cache_read_tokens.add(cache_read_tokens, attrs)
        if cache_write_tokens > 0:
            self._proxy_cache_write_tokens.add(cache_write_tokens, attrs)
        if uncached_input_tokens > 0:
            self._proxy_uncached_input_tokens.add(uncached_input_tokens, attrs)
        if cache_write_5m_tokens > 0:
            self._proxy_cache_write_ttl_tokens.add(
                cache_write_5m_tokens,
                self._attrs(provider=provider, model=model, ttl="5m"),
            )
        if cache_write_1h_tokens > 0:
            self._proxy_cache_write_ttl_tokens.add(
                cache_write_1h_tokens,
                self._attrs(provider=provider, model=model, ttl="1h"),
            )

    def record_proxy_failed(self, *, provider: str | None = None, model: str | None = None) -> None:
        self._proxy_failed_requests.add(1, self._attrs(provider=provider, model=model))

    def record_proxy_rate_limited(
        self,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        self._proxy_rate_limited_requests.add(1, self._attrs(provider=provider, model=model))

    def record_proxy_cache_bust(self, *, tokens_lost: int) -> None:
        self._proxy_cache_busts.add(1)
        self._proxy_cache_bust_tokens_lost.add(max(tokens_lost, 0))

    def record_pipeline_run(
        self,
        *,
        model: str,
        provider: str | None,
        tokens_before: int,
        tokens_after: int,
        duration_ms: float,
        timing: dict[str, float] | None = None,
        transforms_applied: list[str] | None = None,
        waste_signals: dict[str, int] | None = None,
    ) -> None:
        attrs = self._attrs(model=model, provider=provider)
        tokens_saved = max(tokens_before - tokens_after, 0)

        self._compression_runs.add(1, attrs)
        self._compression_input_tokens.add(max(tokens_before, 0), attrs)
        self._compression_output_tokens.add(max(tokens_after, 0), attrs)
        self._compression_saved_tokens.add(tokens_saved, attrs)
        self._compression_duration.record(max(duration_ms, 0.0) / _MILLISECONDS_TO_SECONDS, attrs)

        if transforms_applied:
            for transform in transforms_applied:
                self._compression_transforms.add(
                    1, self._attrs(model=model, provider=provider, transform=transform)
                )

        if timing:
            for stage, stage_ms in timing.items():
                if stage == "pipeline_total" or stage.startswith("_"):
                    continue
                self._compression_stage_duration.record(
                    max(stage_ms, 0.0) / _MILLISECONDS_TO_SECONDS,
                    self._attrs(model=model, provider=provider, stage=stage),
                )

        if waste_signals:
            for signal_name, token_count in waste_signals.items():
                if token_count > 0:
                    self._waste_signal_tokens.add(
                        token_count,
                        self._attrs(model=model, provider=provider, signal=signal_name),
                    )

    def record_compression_failure(
        self,
        *,
        model: str,
        operation: str,
        error_type: str,
    ) -> None:
        self._compression_failures.add(
            1,
            self._attrs(model=model, operation=operation, error_type=error_type),
        )

    def record_subscription_window(self, state: dict[str, Any]) -> None:
        """Update OTEL subscription gauge backing values from the tracker state dict."""
        latest = state.get("latest") or {}

        five_hour = latest.get("five_hour") or {}
        if five_hour:
            self._sub_5h_util_val = float(five_hour.get("utilization_pct", 0.0))
            self._sub_5h_reset_val = float(five_hour.get("seconds_to_reset") or 0.0)

        seven_day = latest.get("seven_day") or {}
        if seven_day:
            self._sub_7d_util_val = float(seven_day.get("utilization_pct", 0.0))
            self._sub_7d_reset_val = float(seven_day.get("seconds_to_reset") or 0.0)

        extra = latest.get("extra_usage") or {}
        if extra.get("is_enabled"):
            self._sub_overage_val = float(extra.get("used_credits_usd") or 0.0)


def get_otel_metrics() -> HeadroomOtelMetrics:
    global _global_metrics

    if _global_metrics is None:
        with _metrics_lock:
            if _global_metrics is None:
                _global_metrics = HeadroomOtelMetrics()

    return _global_metrics


def set_otel_metrics(otel_metrics: HeadroomOtelMetrics) -> HeadroomOtelMetrics:
    global _global_metrics
    with _metrics_lock:
        _global_metrics = otel_metrics
    return otel_metrics


def configure_otel_metrics(config: OTelMetricsConfig | None = None) -> HeadroomOtelMetrics:
    global _global_metrics
    global _owned_meter_provider
    global _owned_metrics_config

    resolved = config or OTelMetricsConfig()
    if not resolved.enabled:
        return get_otel_metrics()

    try:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import (
            ConsoleMetricExporter,
            PeriodicExportingMetricReader,
        )
        from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
    except ImportError:
        logger.warning(
            "OpenTelemetry SDK/exporter packages are not installed. "
            "Install headroom-ai[otel] to enable managed OTEL metric export."
        )
        return get_otel_metrics()

    exporter: Any
    if resolved.exporter == "console":
        exporter = ConsoleMetricExporter()
    else:
        exporter_kwargs: dict[str, Any] = {}
        if resolved.endpoint is not None:
            exporter_kwargs["endpoint"] = resolved.endpoint
        if resolved.headers:
            exporter_kwargs["headers"] = resolved.headers
        exporter = OTLPMetricExporter(**exporter_kwargs)

    reader = PeriodicExportingMetricReader(
        exporter,
        export_interval_millis=resolved.export_interval_millis,
    )
    resource = Resource.create(
        {
            SERVICE_NAME: resolved.service_name,
            SERVICE_VERSION: _headroom_version(),
            **resolved.resource_attributes,
        }
    )
    meter_provider = MeterProvider(resource=resource, metric_readers=[reader])
    otel_metrics = HeadroomOtelMetrics(meter_provider=meter_provider)

    previous_provider = None
    with _metrics_lock:
        previous_provider = _owned_meter_provider
        _owned_meter_provider = meter_provider
        _owned_metrics_config = resolved
        _global_metrics = otel_metrics

    if previous_provider is not None:
        try:
            previous_provider.shutdown()
        except Exception:
            logger.debug("Failed to shut down previous OTEL metrics provider", exc_info=True)

    return otel_metrics


def get_otel_metrics_status() -> dict[str, Any]:
    with _metrics_lock:
        if _owned_metrics_config is not None:
            return _owned_metrics_config.status()
    return OTelMetricsConfig.from_env(default_service_name="headroom-proxy").status()


def shutdown_otel_metrics() -> None:
    global _global_metrics
    global _owned_meter_provider
    global _owned_metrics_config

    provider = None
    with _metrics_lock:
        provider = _owned_meter_provider
        _owned_meter_provider = None
        _owned_metrics_config = None
        _global_metrics = None

    if provider is not None:
        try:
            provider.shutdown()
        except Exception:
            logger.debug("Failed to shut down OTEL metrics provider", exc_info=True)


def reset_otel_metrics() -> None:
    shutdown_otel_metrics()
