"""Runtime enumeration of Headroom telemetry and observability surfaces."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

from headroom import paths
from headroom.observability import LangfuseTracingConfig, OTelMetricsConfig
from headroom.proxy.savings_tracker import get_default_savings_storage_path
from headroom.telemetry.beacon import is_telemetry_enabled
from headroom.telemetry.toin import get_default_toin_storage_path


@dataclass(frozen=True)
class TelemetrySurface:
    surface: str
    status: str
    emits: str
    includes_prompt_content: bool
    leaves_host_by_default: bool
    observe: str
    export: str
    retention: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _status(enabled: bool) -> str:
    return "on" if enabled else "off"


def list_telemetry_surfaces() -> list[TelemetrySurface]:
    """Return locally configured telemetry surfaces without network access."""

    otel = OTelMetricsConfig.from_env(default_service_name="headroom-proxy")
    langfuse = LangfuseTracingConfig.from_env(default_service_name="headroom-proxy")
    toin_backend = (os.environ.get("HEADROOM_TOIN_BACKEND") or "filesystem").strip() or "filesystem"
    toin_status = "off" if toin_backend.lower() == "none" else "on"

    return [
        TelemetrySurface(
            surface="anonymous_beacon",
            status=_status(is_telemetry_enabled()),
            emits="Aggregate usage counters, version, platform, install mode",
            includes_prompt_content=False,
            leaves_host_by_default=True,
            observe="Proxy log startup line",
            export="Supabase REST beacon; disable with HEADROOM_TELEMETRY=off or --no-telemetry",
            retention="Headroom service controlled",
            notes="No prompts, responses, or tool payloads.",
        ),
        TelemetrySurface(
            surface="prometheus",
            status="on",
            emits="Request counts, latency, token savings, cache metrics",
            includes_prompt_content=False,
            leaves_host_by_default=False,
            observe="GET /metrics",
            export="Prometheus scrape",
            retention="Scraper controlled; proxy holds current counters in memory",
        ),
        TelemetrySurface(
            surface="otel_metrics",
            status=_status(otel.enabled),
            emits="Operational metrics equivalent to proxy counters",
            includes_prompt_content=False,
            leaves_host_by_default=otel.enabled and otel.exporter == "otlp_http",
            observe="OTEL collector or console exporter",
            export=otel.endpoint or otel.exporter,
            retention="Exporter controlled",
            notes="Enable with HEADROOM_OTEL_METRICS_ENABLED=true.",
        ),
        TelemetrySurface(
            surface="langfuse_tracing",
            status=_status(langfuse.enabled),
            emits="Trace spans for Headroom operations",
            includes_prompt_content=False,
            leaves_host_by_default=langfuse.enabled,
            observe="Langfuse project dashboard",
            export=langfuse.endpoint,
            retention="Langfuse controlled",
            notes="Requires HEADROOM_LANGFUSE_ENABLED=true and Langfuse keys.",
        ),
        TelemetrySurface(
            surface="savings_tracker",
            status="on",
            emits="Per-request token savings and estimated input cost rollups",
            includes_prompt_content=False,
            leaves_host_by_default=False,
            observe=get_default_savings_storage_path(),
            export="Local JSON file or /stats-history",
            retention="Defaults to 365 days / 5000 history points",
            notes="Override path with HEADROOM_SAVINGS_PATH.",
        ),
        TelemetrySurface(
            surface="dashboard",
            status="on",
            emits="Browser-visible aggregate stats and recent request summaries",
            includes_prompt_content=False,
            leaves_host_by_default=False,
            observe="GET /dashboard and /stats",
            export="None built in; scrape /stats for JSON",
            retention="In-memory plus savings tracker history",
        ),
        TelemetrySurface(
            surface="toin",
            status=toin_status,
            emits="Compression outcome patterns and tool signatures",
            includes_prompt_content=False,
            leaves_host_by_default=toin_backend.lower() not in {"", "filesystem", "none"},
            observe=get_default_toin_storage_path(),
            export=f"HEADROOM_TOIN_BACKEND={toin_backend}",
            retention="Backend controlled",
            notes="Custom backends use HEADROOM_TOIN_URL and HEADROOM_TOIN_TENANT_PREFIX.",
        ),
        TelemetrySurface(
            surface="proxy_log",
            status="on",
            emits="Startup, request metadata, errors, and operational events",
            includes_prompt_content=False,
            leaves_host_by_default=False,
            observe=str(paths.proxy_log_path()),
            export="Local log file",
            retention="Until log rotation or user cleanup",
            notes="Prompt logging requires explicit log_full_messages configuration.",
        ),
        TelemetrySurface(
            surface="active_sessions",
            status="on",
            emits="Per-process session manifest, heartbeat, and aggregate counters",
            includes_prompt_content=False,
            leaves_host_by_default=paths.cluster_enabled(),
            observe=str(paths.sessions_dir()),
            export=(
                str(paths.cluster_sessions_dir())
                if paths.cluster_enabled()
                else "Set HEADROOM_CLUSTER_ENABLED=true to mirror to cluster dir"
            ),
            retention="Stale manifests ignored after runtime TTL",
            notes="Cluster mirroring uses HEADROOM_CLUSTER_ID and HEADROOM_CLUSTER_DIR.",
        ),
    ]


def list_telemetry_surface_dicts() -> list[dict[str, Any]]:
    return [surface.to_dict() for surface in list_telemetry_surfaces()]
