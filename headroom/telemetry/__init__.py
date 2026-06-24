"""Telemetry module for building the data flywheel.

This module collects PRIVACY-PRESERVING statistics about compression patterns
to enable cross-user learning and improve compression over time.

What we collect (anonymized):
- Tool output structure patterns (field types, not values)
- Compression decisions and ratios
- Retrieval patterns (rate, type, not content)
- Strategy effectiveness

What we DON'T collect:
- Actual data values
- User identifiers
- Queries or search terms
- File paths or tool names (unless opted in)

Usage:
    from headroom.telemetry import get_telemetry_collector

    collector = get_telemetry_collector()

    # Record a compression event
    collector.record_compression(
        tool_signature="search_api:v1",
        original_items=1000,
        compressed_items=20,
        strategy="top_n",
        field_stats={...},
    )

    # Export for aggregation
    stats = collector.export_stats()

TOIN (Tool Output Intelligence Network) — observation-only since PR-B5:
    from headroom.telemetry import get_toin

    toin = get_toin()

    # Record compression outcome (the only request-time TOIN call).
    toin.record_compression(tool_signature, ...)

    # Record retrieval (automatic via compression_store).
    toin.record_retrieval(sig_hash, retrieval_type, query, query_fields)

    # Aggregated recommendations are emitted offline:
    #   python -m headroom.cli.toin_publish --output recommendations.toml
    # The Rust proxy loads that TOML at startup; there is no
    # request-time hint API.
"""

from .beacon import (
    format_telemetry_notice,
    is_telemetry_enabled,
    is_telemetry_warn_enabled,
)
from .collector import (
    TelemetryCollector,
    TelemetryConfig,
    get_telemetry_collector,
    reset_telemetry_collector,
)
from .models import (
    AnonymizedToolStats,
    CompressionEvent,
    FieldDistribution,
    RetrievalStats,
    ToolSignature,
)
from .toin import (
    DEFAULT_AUTH_MODE,
    DEFAULT_MIN_OBSERVATIONS_TO_PUBLISH,
    DEFAULT_MODEL_FAMILY,
    TOINConfig,
    ToolIntelligenceNetwork,
    ToolPattern,
    get_toin,
    reset_toin,
)

__all__ = [
    # Beacon helpers
    "format_telemetry_notice",
    "is_telemetry_enabled",
    "is_telemetry_warn_enabled",
    # Collector
    "TelemetryCollector",
    "TelemetryConfig",
    "get_telemetry_collector",
    "reset_telemetry_collector",
    # Models
    "AnonymizedToolStats",
    "CompressionEvent",
    "FieldDistribution",
    "RetrievalStats",
    "ToolSignature",
    # TOIN (observation-only since PR-B5)
    "DEFAULT_AUTH_MODE",
    "DEFAULT_MIN_OBSERVATIONS_TO_PUBLISH",
    "DEFAULT_MODEL_FAMILY",
    "TOINConfig",
    "ToolIntelligenceNetwork",
    "ToolPattern",
    "get_toin",
    "reset_toin",
]
