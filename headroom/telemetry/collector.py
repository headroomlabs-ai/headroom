"""TelemetryCollector for privacy-preserving statistics collection.

This module collects anonymized statistics about compression patterns
to enable cross-user learning and improve compression over time.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import (
    AnonymizedToolStats,
    CompressionEvent,
    FieldDistribution,
    RetrievalStats,
    ToolSignature,
)


@dataclass
class TelemetryConfig:
    """Configuration for telemetry collection."""

    # Enable/disable telemetry
    enabled: bool = True

    # Storage
    storage_path: str | None = None  # Path to store telemetry data (None = in-memory only)
    auto_save_interval: int = 300  # Auto-save every N seconds (0 = disabled)

    # Privacy settings
    anonymize_tool_names: bool = True  # Hash tool names
    collect_field_names: bool = False  # If False, only collect field hashes
    collect_timing: bool = True  # Collect processing time

    # Aggregation settings
    max_events_in_memory: int = 10000  # Max events to keep in memory
    min_samples_for_recommendation: int = 10  # Min samples before making recommendations

    # Export settings
    include_field_distributions: bool = True  # Include detailed field stats in export
    include_recommendations: bool = True  # Include learned recommendations


class TelemetryCollector:
    """Collects and aggregates compression telemetry.

    Thread-safe collector that maintains anonymized statistics about
    compression patterns. Can be used to:
    - Understand what tool outputs look like (structurally)
    - Track which compression strategies work best
    - Learn optimal settings per tool type
    - Export data for cross-user aggregation

    Privacy guarantees:
    - No actual data values are stored
    - Tool names are hashed by default
    - Field names can be hashed
    - No user identifiers
    - No query content
    """

    def __init__(self, config: TelemetryConfig | None = None):
        """Initialize the telemetry collector.

        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self._config = config or TelemetryConfig()
        self._lock = threading.Lock()

        # Event storage
        self._events: list[CompressionEvent] = []

        # Aggregated stats per tool signature
        self._tool_stats: dict[str, AnonymizedToolStats] = {}

        # Retrieval tracking
        self._retrieval_stats: dict[str, RetrievalStats] = {}

        # Global counters
        self._total_compressions: int = 0
        self._total_retrievals: int = 0
        self._total_tokens_saved: int = 0

        # Auto-save tracking
        self._last_save_time: float = time.time()
        self._dirty: bool = False

        # Load existing data if storage path exists
        if self._config.storage_path:
            self._load_from_disk()

    def record_compression(
        self,
        items: list[dict[str, Any]],
        original_count: int,
        compressed_count: int,
        original_tokens: int,
        compressed_tokens: int,
        strategy: str,
        *,
        tool_name: str | None = None,
        strategy_reason: str | None = None,
        crushability_score: float | None = None,
        crushability_reason: str | None = None,
        kept_first_n: int = 0,
        kept_last_n: int = 0,
        kept_errors: int = 0,
        kept_anomalies: int = 0,
        kept_by_relevance: int = 0,
        kept_by_score: int = 0,
        processing_time_ms: float = 0.0,
    ) -> None:
        """Record a compression event.

        Args:
            items: Sample items from the original array (for structure analysis).
            original_count: Original number of items.
            compressed_count: Number of items after compression.
            original_tokens: Original token count.
            compressed_tokens: Compressed token count.
            strategy: Compression strategy used.
            tool_name: Optional tool name (will be hashed if configured).
            strategy_reason: Why this strategy was chosen.
            crushability_score: Crushability analysis score.
            crushability_reason: Crushability analysis reason.
            kept_first_n: Items kept from start.
            kept_last_n: Items kept from end.
            kept_errors: Error items kept.
            kept_anomalies: Anomalous items kept.
            kept_by_relevance: Items kept by relevance score.
            kept_by_score: Items kept by score field.
            processing_time_ms: Processing time in milliseconds.
        """
        if not self._config.enabled:
            return

        # Create tool signature from items
        signature = ToolSignature.from_items(items[:10])  # Sample first 10

        # Analyze field distributions
        field_distributions: list[FieldDistribution] = []
        if self._config.include_field_distributions and items:
            field_distributions = self._analyze_fields(items[:100])  # Sample 100

        # Calculate ratios
        compression_ratio = compressed_count / original_count if original_count > 0 else 0.0
        token_reduction = 1 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0.0

        # Create event
        event = CompressionEvent(
            tool_signature=signature,
            original_item_count=original_count,
            compressed_item_count=compressed_count,
            compression_ratio=compression_ratio,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            token_reduction_ratio=token_reduction,
            strategy=strategy,
            strategy_reason=strategy_reason,
            crushability_score=crushability_score,
            crushability_reason=crushability_reason,
            field_distributions=field_distributions,
            kept_first_n=kept_first_n,
            kept_last_n=kept_last_n,
            kept_errors=kept_errors,
            kept_anomalies=kept_anomalies,
            kept_by_relevance=kept_by_relevance,
            kept_by_score=kept_by_score,
            timestamp=time.time(),
            processing_time_ms=processing_time_ms,
        )

        should_save = False
        with self._lock:
            # Store event
            self._events.append(event)
            if len(self._events) > self._config.max_events_in_memory:
                self._events = self._events[-self._config.max_events_in_memory :]

            # Update aggregated stats
            self._update_tool_stats(signature, event)

            # Update global counters
            self._total_compressions += 1
            self._total_tokens_saved += original_tokens - compressed_tokens
            self._dirty = True

            # Check if auto-save needed (don't actually save while holding lock)
            should_save = self._should_auto_save()

        # Auto-save outside lock to avoid blocking other operations
        if should_save:
            self.save()

    def record_retrieval(
        self,
        tool_signature_hash: str,
        retrieval_type: str,  # "full" or "search"
        query_fields: list[str] | None = None,
    ) -> None:
        """Record a retrieval event.

        This is called when an LLM retrieves compressed content, indicating
        the compression may have been too aggressive.

        Args:
            tool_signature_hash: Hash of the tool signature.
            retrieval_type: "full" (retrieved everything) or "search" (filtered).
            query_fields: Field names mentioned in search query (will be hashed).
        """
        if not self._config.enabled:
            return

        with self._lock:
            # Get or create retrieval stats
            if tool_signature_hash not in self._retrieval_stats:
                self._retrieval_stats[tool_signature_hash] = RetrievalStats(
                    tool_signature_hash=tool_signature_hash
                )

            stats = self._retrieval_stats[tool_signature_hash]
            stats.total_retrievals += 1

            if retrieval_type == "full":
                stats.full_retrievals += 1
            else:
                stats.search_retrievals += 1

            # Track queried fields (anonymized)
            if query_fields:
                for field_name in query_fields:
                    field_hash = self._hash_field_name(field_name)
                    stats.query_field_frequency[field_hash] = (
                        stats.query_field_frequency.get(field_hash, 0) + 1
                    )

            # Update global counter
            self._total_retrievals += 1
            self._dirty = True

            # Update tool stats with retrieval info
            if tool_signature_hash in self._tool_stats:
                self._tool_stats[tool_signature_hash].retrieval_stats = stats
                self._update_recommendations(tool_signature_hash)

    def get_stats(self) -> dict[str, Any]:
        """Get overall telemetry statistics.

        Returns:
            Dictionary with aggregated statistics.
        """
        with self._lock:
            return {
                "enabled": self._config.enabled,
                "total_compressions": self._total_compressions,
                "total_retrievals": self._total_retrievals,
                "total_tokens_saved": self._total_tokens_saved,
                "global_retrieval_rate": (
                    self._total_retrievals / self._total_compressions
                    if self._total_compressions > 0
                    else 0.0
                ),
                "tool_signatures_tracked": len(self._tool_stats),
                "events_in_memory": len(self._events),
                "avg_compression_ratio": self._calculate_avg_compression_ratio(),
                "avg_token_reduction": self._calculate_avg_token_reduction(),
            }

    def get_tool_stats(self, signature_hash: str) -> AnonymizedToolStats | None:
        """Get statistics for a specific tool signature.

        Args:
            signature_hash: The tool signature hash.

        Returns:
            AnonymizedToolStats if found, None otherwise.
        """
        with self._lock:
            return self._tool_stats.get(signature_hash)

    def get_all_tool_stats(self) -> dict[str, AnonymizedToolStats]:
        """Get statistics for all tracked tool signatures.

        Returns:
            Dictionary mapping signature hash to stats.
        """
        with self._lock:
            return dict(self._tool_stats)

    def get_recommendations(self, signature_hash: str) -> dict[str, Any] | None:
        """Get learned recommendations for a tool signature.

        Args:
            signature_hash: The tool signature hash.

        Returns:
            Recommendations dictionary if available, None otherwise.
        """
        with self._lock:
            stats = self._tool_stats.get(signature_hash)
            if not stats or stats.sample_size < self._config.min_samples_for_recommendation:
                return None

            return {
                "signature_hash": signature_hash,
                "recommended_min_items": stats.recommended_min_items,
                "recommended_preserve_fields": stats.recommended_preserve_fields,
                "skip_compression_recommended": stats.skip_compression_recommended,
                "confidence": stats.confidence,
                "based_on_samples": stats.sample_size,
                "retrieval_rate": (
                    stats.retrieval_stats.retrieval_rate if stats.retrieval_stats else None
                ),
            }

    def export_stats(self) -> dict[str, Any]:
        """Export all telemetry data for aggregation.

        This is the data that can be sent to a central server for
        cross-user learning (with user consent).

        Returns:
            Complete telemetry export.
        """
        with self._lock:
            export = {
                "version": "1.0",
                "export_timestamp": time.time(),
                "summary": {
                    "total_compressions": self._total_compressions,
                    "total_retrievals": self._total_retrievals,
                    "total_tokens_saved": self._total_tokens_saved,
                    "tool_signatures_tracked": len(self._tool_stats),
                },
                "tool_stats": {
                    sig_hash: stats.to_dict() for sig_hash, stats in self._tool_stats.items()
                },
            }

            if self._config.include_recommendations:
                export["recommendations"] = {
                    sig_hash: {
                        "recommended_min_items": stats.recommended_min_items,
                        "skip_compression_recommended": stats.skip_compression_recommended,
                        "confidence": stats.confidence,
                    }
                    for sig_hash, stats in self._tool_stats.items()
                    if stats.sample_size >= self._config.min_samples_for_recommendation
                }

            return export

    def import_stats(self, data: dict[str, Any]) -> None:
        """Import telemetry data from another source.

        This allows merging stats from multiple users for cross-user learning.

        Args:
            data: Exported telemetry data.
        """
        if not self._config.enabled:
            return

        with self._lock:
            # Import summary counters
            summary = data.get("summary", {})
            self._total_compressions += summary.get("total_compressions", 0)
            self._total_retrievals += summary.get("total_retrievals", 0)
            self._total_tokens_saved += summary.get("total_tokens_saved", 0)

            # Import tool stats
            tool_stats_data = data.get("tool_stats", {})
            for sig_hash, stats_dict in tool_stats_data.items():
                if sig_hash in self._tool_stats:
                    # Merge with existing
                    existing = self._tool_stats[sig_hash]
                    imported = AnonymizedToolStats.from_dict(stats_dict)
                    self._merge_tool_stats(existing, imported)
                else:
                    # Add new
                    self._tool_stats[sig_hash] = AnonymizedToolStats.from_dict(stats_dict)

            self._dirty = True

    def clear(self) -> None:
        """Clear all telemetry data. Mainly for testing."""
        with self._lock:
            self._events.clear()
            self._tool_stats.clear()
            self._retrieval_stats.clear()
            self._total_compressions = 0
            self._total_retrievals = 0
            self._total_tokens_saved = 0
            self._dirty = False

    def save(self) -> None:
        """Save telemetry data to disk."""
        if not self._config.storage_path:
            return

        with self._lock:
            # Build export data inline to avoid deadlock (export_stats also acquires lock)
            data = {
                "version": "1.0",
                "export_timestamp": time.time(),
                "summary": {
                    "total_compressions": self._total_compressions,
                    "total_retrievals": self._total_retrievals,
                    "total_tokens_saved": self._total_tokens_saved,
                    "tool_signatures_tracked": len(self._tool_stats),
                },
                "tool_stats": {
                    sig_hash: stats.to_dict() for sig_hash, stats in self._tool_stats.items()
                },
            }

            if self._config.include_recommendations:
                data["recommendations"] = {
                    sig_hash: {
                        "recommended_min_items": stats.recommended_min_items,
                        "skip_compression_recommended": stats.skip_compression_recommended,
                        "confidence": stats.confidence,
                    }
                    for sig_hash, stats in self._tool_stats.items()
                    if stats.sample_size >= self._config.min_samples_for_recommendation
                }

            path = Path(self._config.storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, "w") as f:
                json.dump(data, f, indent=2)

            self._dirty = False
            self._last_save_time = time.time()

    def _load_from_disk(self) -> None:
        """Load telemetry data from disk."""
        if not self._config.storage_path:
            return

        path = Path(self._config.storage_path)
        if not path.exists():
            return

        try:
            with open(path) as f:
                data = json.load(f)
            self.import_stats(data)
            self._dirty = False
        except (json.JSONDecodeError, OSError):
            pass  # Start fresh if file is corrupted

    def _analyze_fields(self, items: list[dict[str, Any]]) -> list[FieldDistribution]:
        """Analyze field distributions in items."""
        if not items:
            return []

        distributions: list[FieldDistribution] = []

        # Get all field names from first item
        sample = items[0] if isinstance(items[0], dict) else {}
        for field_name, _sample_value in sample.items():
            # Collect all values for this field
            values = [
                item.get(field_name)
                for item in items
                if isinstance(item, dict) and field_name in item
            ]

            if not values:
                continue

            dist = self._create_field_distribution(field_name, values)
            distributions.append(dist)

        return distributions

    def _create_field_distribution(
        self,
        field_name: str,
        values: list[Any],
    ) -> FieldDistribution:
        """Create a FieldDistribution from values."""
        field_hash = self._hash_field_name(field_name)

        # Determine type
        type_counts: dict[str, int] = {}
        for v in values:
            if isinstance(v, str):
                type_counts["string"] = type_counts.get("string", 0) + 1
            elif isinstance(v, bool):
                type_counts["boolean"] = type_counts.get("boolean", 0) + 1
            elif isinstance(v, int | float):
                type_counts["numeric"] = type_counts.get("numeric", 0) + 1
            elif isinstance(v, list):
                type_counts["array"] = type_counts.get("array", 0) + 1
            elif isinstance(v, dict):
                type_counts["object"] = type_counts.get("object", 0) + 1
            elif v is None:
                type_counts["null"] = type_counts.get("null", 0) + 1

        # Get dominant type
        if not type_counts:
            field_type = "null"
        elif len(type_counts) > 1:
            field_type = "mixed"
        else:
            field_type = list(type_counts.keys())[0]

        dist = FieldDistribution(
            field_name_hash=field_hash,
            field_type=field_type,  # type: ignore[arg-type]
        )

        # Type-specific analysis
        if field_type == "string":
            str_values = [v for v in values if isinstance(v, str)]
            if str_values:
                dist.avg_length = sum(len(s) for s in str_values) / len(str_values)
                unique_count = len(set(str_values))
                dist.unique_ratio = unique_count / len(str_values)
                dist.looks_like_id = dist.unique_ratio > 0.9 and dist.avg_length > 5

        elif field_type == "numeric":
            num_values = [v for v in values if isinstance(v, int | float)]
            # Filter out infinity and NaN which can cause issues
            num_values = [
                v
                for v in num_values
                if not (
                    isinstance(v, float) and (v != v or v == float("inf") or v == float("-inf"))
                )
            ]
            if num_values:
                dist.has_negative = any(v < 0 for v in num_values)
                # Safe integer check (avoid OverflowError from int(inf))
                dist.is_integer = all(
                    isinstance(v, int) or (isinstance(v, float) and v.is_integer())
                    for v in num_values
                )

                if len(num_values) > 1:
                    mean = sum(num_values) / len(num_values)
                    variance = sum((v - mean) ** 2 for v in num_values) / len(num_values)
                    dist.has_variance = variance > 0

                    if variance == 0:
                        dist.variance_bucket = "zero"
                    elif variance < 10:
                        dist.variance_bucket = "low"
                    elif variance < 1000:
                        dist.variance_bucket = "medium"
                    else:
                        dist.variance_bucket = "high"

                    # Check for outliers
                    std = variance**0.5
                    if std > 0:
                        outliers = sum(1 for v in num_values if abs(v - mean) > 2 * std)
                        dist.has_outliers = outliers > 0

                    # Pattern detection
                    sorted_vals = sorted(num_values)
                    is_monotonic = (
                        sorted_vals == num_values or list(reversed(sorted_vals)) == num_values
                    )
                    if is_monotonic and dist.variance_bucket in ("medium", "high"):
                        dist.is_likely_score = True

        elif field_type == "array":
            arr_values = [v for v in values if isinstance(v, list)]
            if arr_values:
                dist.avg_array_length = sum(len(a) for a in arr_values) / len(arr_values)

        return dist

    def _update_tool_stats(self, signature: ToolSignature, event: CompressionEvent) -> None:
        """Update aggregated stats for a tool signature."""
        sig_hash = signature.structure_hash

        if sig_hash not in self._tool_stats:
            self._tool_stats[sig_hash] = AnonymizedToolStats(signature=signature)

        stats = self._tool_stats[sig_hash]

        # Update counts
        stats.total_compressions += 1
        stats.total_items_seen += event.original_item_count
        stats.total_items_kept += event.compressed_item_count
        stats.sample_size += 1

        # Update averages (rolling)
        n = stats.total_compressions
        stats.avg_compression_ratio = (
            stats.avg_compression_ratio * (n - 1) + event.compression_ratio
        ) / n
        stats.avg_token_reduction = (
            stats.avg_token_reduction * (n - 1) + event.token_reduction_ratio
        ) / n

        # Update strategy counts
        strategy = event.strategy
        stats.strategy_counts[strategy] = stats.strategy_counts.get(strategy, 0) + 1

        # Update confidence based on sample size
        stats.confidence = min(0.95, stats.sample_size / 100)

        # Update recommendations
        self._update_recommendations(sig_hash)

    def _update_recommendations(self, sig_hash: str) -> None:
        """Update recommendations based on current data."""
        if sig_hash not in self._tool_stats:
            return

        stats = self._tool_stats[sig_hash]

        # Not enough data yet
        if stats.sample_size < self._config.min_samples_for_recommendation:
            return

        # Check retrieval rate to determine if compression is too aggressive
        if stats.retrieval_stats:
            retrieval_rate = stats.retrieval_stats.retrieval_rate
            full_rate = stats.retrieval_stats.full_retrieval_rate

            # High retrieval rate = compression too aggressive
            if retrieval_rate > 0.5:
                if full_rate > 0.8:
                    # Almost all retrievals are full = skip compression
                    stats.skip_compression_recommended = True
                else:
                    # Increase min items
                    stats.recommended_min_items = 50
            elif retrieval_rate > 0.2:
                # Medium retrieval rate = slightly less aggressive
                stats.recommended_min_items = 30
            else:
                # Low retrieval rate = current settings work
                stats.recommended_min_items = 15

            # Track frequently queried fields
            if stats.retrieval_stats.query_field_frequency:
                top_fields = sorted(
                    stats.retrieval_stats.query_field_frequency.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
                stats.recommended_preserve_fields = [f for f, _ in top_fields]

    def _merge_tool_stats(
        self,
        existing: AnonymizedToolStats,
        imported: AnonymizedToolStats,
    ) -> None:
        """Merge imported stats into existing."""
        # Weighted average based on sample sizes
        total_samples = existing.sample_size + imported.sample_size
        if total_samples == 0:
            return

        w_existing = existing.sample_size / total_samples
        w_imported = imported.sample_size / total_samples

        existing.total_compressions += imported.total_compressions
        existing.total_items_seen += imported.total_items_seen
        existing.total_items_kept += imported.total_items_kept
        existing.avg_compression_ratio = (
            existing.avg_compression_ratio * w_existing
            + imported.avg_compression_ratio * w_imported
        )
        existing.avg_token_reduction = (
            existing.avg_token_reduction * w_existing + imported.avg_token_reduction * w_imported
        )
        existing.sample_size = total_samples

        # Merge strategy counts
        for strategy, count in imported.strategy_counts.items():
            existing.strategy_counts[strategy] = existing.strategy_counts.get(strategy, 0) + count

        # Update confidence
        existing.confidence = min(0.95, total_samples / 100)

    def _hash_field_name(self, field_name: str) -> str:
        """Hash a field name for anonymization."""
        if self._config.collect_field_names:
            return field_name
        return hashlib.sha256(field_name.encode()).hexdigest()[:8]

    def _calculate_avg_compression_ratio(self) -> float:
        """Calculate average compression ratio across all tools."""
        if not self._tool_stats:
            return 0.0
        ratios = [s.avg_compression_ratio for s in self._tool_stats.values()]
        return sum(ratios) / len(ratios)

    def _calculate_avg_token_reduction(self) -> float:
        """Calculate average token reduction across all tools."""
        if not self._tool_stats:
            return 0.0
        reductions = [s.avg_token_reduction for s in self._tool_stats.values()]
        return sum(reductions) / len(reductions)

    def _should_auto_save(self) -> bool:
        """Check if auto-save should run. Must be called with lock held."""
        if not self._config.auto_save_interval or not self._config.storage_path:
            return False

        if not self._dirty:
            return False

        elapsed = time.time() - self._last_save_time
        return elapsed >= self._config.auto_save_interval


# Global collector instance (lazy initialization)
_telemetry_collector: TelemetryCollector | None = None
_collector_lock = threading.Lock()


def get_telemetry_collector(
    config: TelemetryConfig | None = None,
) -> TelemetryCollector:
    """Get the global telemetry collector instance.

    Args:
        config: Configuration (only used on first call).

    Returns:
        Global TelemetryCollector instance.
    """
    global _telemetry_collector

    if _telemetry_collector is None:
        with _collector_lock:
            if _telemetry_collector is None:
                # Honour HEADROOM_TELEMETRY (the documented opt-out var,
                # also used by the Supabase beacon at telemetry/beacon.py).
                # Pre-#390 this only checked HEADROOM_TELEMETRY_DISABLED,
                # so users who set HEADROOM_TELEMETRY=off (the value in
                # the docs) still saw /v1/telemetry report enabled=true.
                # HEADROOM_TELEMETRY_DISABLED stays accepted for back-compat.
                from headroom.telemetry.beacon import is_telemetry_enabled

                disabled_legacy = os.environ.get("HEADROOM_TELEMETRY_DISABLED", "").lower() in (
                    "1",
                    "true",
                )
                if disabled_legacy or not is_telemetry_enabled():
                    config = config or TelemetryConfig()
                    config.enabled = False

                _telemetry_collector = TelemetryCollector(config)

    return _telemetry_collector


def reset_telemetry_collector() -> None:
    """Reset the global telemetry collector. Mainly for testing."""
    global _telemetry_collector

    with _collector_lock:
        if _telemetry_collector is not None:
            _telemetry_collector.clear()
        _telemetry_collector = None
