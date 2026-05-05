"""Tool Output Intelligence Network (TOIN) — observation-only contract.

# Observation-only contract (PR-B5)

TOIN observes; it never mutates request-time compression decisions. The
request path is deterministic: SmartCrusher and the live-zone dispatcher
read their static configuration only. TOIN's role is to record what
happened so an offline aggregator (`headroom.cli.toin_publish`) can emit
a `recommendations.toml` file the deploy pipeline ships to the proxy at
the next restart.

Why this shape:
- Per-request mutation tied compression bytes to TOIN's mutable state,
  which made the same input produce different outputs across runs (P2-27,
  P5-56). That broke prompt caching and made bugs irreproducible.
- The request-time hint API (`get_recommendation()`) is retired. It now
  emits a `DeprecationWarning` and returns `None`. New code must not call
  it.
- Recording (`record_compression`, `record_retrieval`) and storage
  (save/load/export/import) are unchanged; the learning value is intact.

# Aggregation key

Patterns are keyed by `(auth_mode, model_family, structure_hash)` —
each tenant slice (PAYG vs OAuth vs subscription) and each model family
(claude-3-5, gpt-4o, …) learns independently. Defaults `"unknown"` when
either is not yet plumbed through (PR-F3 lights up real auth-mode
detection).

# Privacy
- No actual data values are stored.
- Tool names are structure hashes.
- Field names are SHA256[:8] hashes.
- No user identifiers.

# Network effect (preserved)
- More users → more compression events → better aggregated `optimal_*`
  fields on each `ToolPattern`. The `toin publish` CLI promotes those
  into `recommendations.toml`.
- Cross-instance pattern import (`import_patterns`) supports federated
  learning without sharing actual data.

# Usage
    from headroom.telemetry.toin import get_toin

    # Record a compression event (the only request-time TOIN call).
    get_toin().record_compression(
        tool_signature=signature,
        original_count=len(items),
        compressed_count=kept,
        original_tokens=before,
        compressed_tokens=after,
        strategy="smart_crusher",
    )

    # Aggregated recommendations are produced offline:
    #   python -m headroom.cli.toin_publish --output recommendations.toml
    # The Rust proxy loads that file at startup; no per-request hint API.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from .models import FieldSemantics, ToolSignature

logger = logging.getLogger(__name__)

# Environment variable for custom TOIN storage path
TOIN_PATH_ENV_VAR = "HEADROOM_TOIN_PATH"

# Default TOIN storage directory and file
DEFAULT_TOIN_DIR = ".headroom"
DEFAULT_TOIN_FILE = "toin.json"

# ── Aggregation-key defaults ────────────────────────────────────────────
# Used when callers haven't plumbed auth-mode / model-family detection
# (PR-F3 wires the real detectors). Shipping a real `"unknown"` slice is
# explicit — better than a magic empty string and lets the publish CLI
# filter on it deliberately.
DEFAULT_AUTH_MODE: Final[str] = "unknown"
DEFAULT_MODEL_FAMILY: Final[str] = "unknown"

# ── Aggregation thresholds (Final, not magic numbers) ───────────────────
# Minimum observations a pattern must have before the `toin publish` CLI
# emits a recommendation row for it. Below this, the recommendation
# would be noise. The CLI exposes `--min-observations` to override per
# environment; this is the production default the Rust proxy expects.
DEFAULT_MIN_OBSERVATIONS_TO_PUBLISH: Final[int] = 50

# Aggregation-key serialization separator. Used to encode the
# `(auth_mode, model_family, sig_hash)` tuple as a string for JSON
# storage (JSON object keys must be strings) and for cross-instance
# pattern imports. Pipe is illegal in all three components by
# construction (auth_mode ∈ {"unknown","payg","oauth","subscription"};
# model_family is a registry name with no `|`; sig_hash is hex).
_AGG_KEY_SEPARATOR: Final[str] = "|"


# ── Aggregation key helpers ─────────────────────────────────────────────
PatternKey = tuple[str, str, str]


def _make_pattern_key(
    auth_mode: str | None,
    model_family: str | None,
    sig_hash: str,
) -> PatternKey:
    """Build the canonical `(auth_mode, model_family, sig_hash)` key.

    Defaults populate to `DEFAULT_AUTH_MODE` / `DEFAULT_MODEL_FAMILY`
    when callers haven't supplied a value — keeps callers terse during
    the Phase B realignment while PR-F3 wires real detectors.
    """
    return (
        auth_mode or DEFAULT_AUTH_MODE,
        model_family or DEFAULT_MODEL_FAMILY,
        sig_hash,
    )


def _serialize_pattern_key(key: PatternKey) -> str:
    """Serialize an aggregation key to a string for JSON / TOML storage."""
    return _AGG_KEY_SEPARATOR.join(key)


def _deserialize_pattern_key(serialized: str) -> PatternKey:
    """Parse a serialized aggregation key back to a tuple.

    Backward-compatible with pre-B5 dumps that stored keys as bare
    structure hashes (no separator): those parse as
    `(DEFAULT_AUTH_MODE, DEFAULT_MODEL_FAMILY, sig_hash)`. The realignment
    plan permits wiping the on-disk store, but this fallback keeps reads
    safe if a stale file appears in the wild.
    """
    parts = serialized.split(_AGG_KEY_SEPARATOR)
    if len(parts) == 3:
        return (parts[0], parts[1], parts[2])
    # Legacy format: bare sig_hash. Promote to default tenant slice.
    return (DEFAULT_AUTH_MODE, DEFAULT_MODEL_FAMILY, serialized)


def get_default_toin_storage_path() -> str:
    """Get the default TOIN storage path.

    Checks for the HEADROOM_TOIN_PATH environment variable first.
    Falls back to ``${HEADROOM_WORKSPACE_DIR}/toin.json`` (which defaults
    to ``~/.headroom/toin.json``) when unset.

    Returns:
        The path string for TOIN storage.
    """
    # Preserve legacy behavior: when HEADROOM_TOIN_PATH is set we return the
    # raw string exactly as the user supplied it (no tilde expansion, no
    # path-separator normalization). This matches what existing tests and
    # users have relied on since the env var was introduced.
    env_path = os.environ.get(TOIN_PATH_ENV_VAR, "").strip()
    if env_path:
        return env_path

    from headroom import paths as _paths

    return str(_paths.toin_path())


# LOW FIX #22: Define callback types for metrics/monitoring hooks
# These allow users to plug in their own metrics collection (Prometheus, StatsD, etc.)
MetricsCallback = Callable[[str, dict[str, Any]], None]  # (event_name, event_data) -> None


@dataclass
class ToolPattern:
    """Aggregated intelligence about a tool type across all users.

    This is the core TOIN data structure. It represents everything we've
    learned about how to compress outputs from tools with a specific structure.
    """

    tool_signature_hash: str

    # === Aggregation Key (PR-B5) ===
    # Per-tenant aggregation key extension. The Pattern is keyed inside
    # the TOIN store by `(auth_mode, model_family, tool_signature_hash)` —
    # these two fields carry the same values onto the dataclass so dumps,
    # imports, and publish-CLI rows are self-describing without
    # cross-referencing the dict key.
    auth_mode: str = DEFAULT_AUTH_MODE
    model_family: str = DEFAULT_MODEL_FAMILY

    # === Compression Statistics ===
    total_compressions: int = 0
    total_items_seen: int = 0
    total_items_kept: int = 0
    avg_compression_ratio: float = 0.0
    avg_token_reduction: float = 0.0

    # === Retrieval Statistics ===
    total_retrievals: int = 0
    full_retrievals: int = 0  # Retrieved everything
    search_retrievals: int = 0  # Used search filter

    @property
    def retrieval_rate(self) -> float:
        """Fraction of compressions that triggered retrieval."""
        if self.total_compressions == 0:
            return 0.0
        return self.total_retrievals / self.total_compressions

    @property
    def full_retrieval_rate(self) -> float:
        """Fraction of retrievals that were full (not search)."""
        if self.total_retrievals == 0:
            return 0.0
        return self.full_retrievals / self.total_retrievals

    # === Learned Patterns ===
    # Fields that are frequently retrieved (should preserve)
    commonly_retrieved_fields: list[str] = field(default_factory=list)
    field_retrieval_frequency: dict[str, int] = field(default_factory=dict)

    # Query patterns that trigger retrieval
    common_query_patterns: list[str] = field(default_factory=list)
    # MEDIUM FIX #10: Track query pattern frequency to keep most common, not just recent
    query_pattern_frequency: dict[str, int] = field(default_factory=dict)

    # Best compression strategy for this tool type
    optimal_strategy: str = "default"
    strategy_success_rates: dict[str, float] = field(default_factory=dict)

    # === Learned Recommendations ===
    optimal_max_items: int = 20
    skip_compression_recommended: bool = False
    preserve_fields: list[str] = field(default_factory=list)

    # === Field-Level Semantics (TOIN Evolution) ===
    # Learned semantic types for each field based on retrieval patterns
    # This enables zero-latency signal detection without hardcoded patterns
    field_semantics: dict[str, FieldSemantics] = field(default_factory=dict)

    # === Observation Counter ===
    # PR-B5: legacy counter from the retired `get_recommendation()` API.
    # Held for serialization compatibility with v1.0 dumps; new
    # increments only happen via record_compression / record_retrieval.
    observations: int = 0

    # === Confidence ===
    sample_size: int = 0
    user_count: int = 0  # Number of unique users (anonymized)
    confidence: float = 0.0  # 0.0 = no data, 1.0 = high confidence
    last_updated: float = 0.0

    # === Instance Tracking (for user_count) ===
    # Hashed instance IDs of users who have contributed to this pattern
    # Limited to avoid unbounded growth (for serialization)
    _seen_instance_hashes: list[str] = field(default_factory=list)
    # FIX: Separate set for ALL seen instances to prevent double-counting
    # CRITICAL FIX #1: Capped at MAX_SEEN_INSTANCES to prevent OOM with millions of users.
    # When cap is reached, we rely on user_count for accurate counting and
    # accept some potential double-counting for new users (negligible at scale).
    _all_seen_instances: set[str] = field(default_factory=set)

    # CRITICAL FIX: Track whether instance tracking was truncated during serialization
    # If True, we know some users were lost and should be conservative about user_count
    _tracking_truncated: bool = False

    # CRITICAL FIX #1: Maximum entries in _all_seen_instances to prevent OOM
    # This is a class constant, not a field (not serialized)
    MAX_SEEN_INSTANCES: int = 10000

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_signature_hash": self.tool_signature_hash,
            "auth_mode": self.auth_mode,
            "model_family": self.model_family,
            "total_compressions": self.total_compressions,
            "total_items_seen": self.total_items_seen,
            "total_items_kept": self.total_items_kept,
            "avg_compression_ratio": self.avg_compression_ratio,
            "avg_token_reduction": self.avg_token_reduction,
            "total_retrievals": self.total_retrievals,
            "full_retrievals": self.full_retrievals,
            "search_retrievals": self.search_retrievals,
            "retrieval_rate": self.retrieval_rate,
            "full_retrieval_rate": self.full_retrieval_rate,
            "commonly_retrieved_fields": self.commonly_retrieved_fields,
            "field_retrieval_frequency": self.field_retrieval_frequency,
            "common_query_patterns": self.common_query_patterns,
            "query_pattern_frequency": self.query_pattern_frequency,
            "optimal_strategy": self.optimal_strategy,
            "strategy_success_rates": self.strategy_success_rates,
            "optimal_max_items": self.optimal_max_items,
            "skip_compression_recommended": self.skip_compression_recommended,
            "preserve_fields": self.preserve_fields,
            # Field-level semantics (TOIN Evolution)
            "field_semantics": {k: v.to_dict() for k, v in self.field_semantics.items()},
            "observations": self.observations,
            "sample_size": self.sample_size,
            "user_count": self.user_count,
            "confidence": self.confidence,
            "last_updated": self.last_updated,
            # Serialize instance hashes (limited to 100 for bounded storage)
            "seen_instance_hashes": self._seen_instance_hashes[:100],
            # CRITICAL FIX: Track if truncation occurred during serialization
            # This tells from_dict() that some users were lost and prevents double-counting
            "tracking_truncated": (
                self._tracking_truncated
                or self.user_count > len(self._seen_instance_hashes)
                or len(self._all_seen_instances) > 100
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolPattern:
        """Create from dictionary."""
        # Filter to only valid fields
        valid_fields = {
            "tool_signature_hash",
            "auth_mode",
            "model_family",
            "total_compressions",
            "total_items_seen",
            "total_items_kept",
            "avg_compression_ratio",
            "avg_token_reduction",
            "total_retrievals",
            "full_retrievals",
            "search_retrievals",
            "commonly_retrieved_fields",
            "field_retrieval_frequency",
            "common_query_patterns",
            "query_pattern_frequency",
            "optimal_strategy",
            "strategy_success_rates",
            "optimal_max_items",
            "skip_compression_recommended",
            "preserve_fields",
            "observations",
            "sample_size",
            "user_count",
            "confidence",
            "last_updated",
        }
        filtered = {k: v for k, v in data.items() if k in valid_fields}

        # Handle seen_instance_hashes (serialized without underscore prefix)
        seen_hashes = data.get("seen_instance_hashes", [])

        pattern = cls(**filtered)
        pattern._seen_instance_hashes = seen_hashes[:100]  # Limit on load

        # CRITICAL FIX: Populate _all_seen_instances from loaded hashes
        # This prevents double-counting after restart - without this, the same
        # instances would be counted again because the lookup set was empty
        pattern._all_seen_instances = set(pattern._seen_instance_hashes)

        # CRITICAL FIX: Restore truncation flag to prevent double-counting
        # If truncated, we know some users were lost in serialization
        pattern._tracking_truncated = data.get("tracking_truncated", False)
        # Also detect truncation if user_count > loaded hashes (backward compat)
        if pattern.user_count > len(pattern._seen_instance_hashes):
            pattern._tracking_truncated = True

        # Load field semantics (TOIN Evolution)
        field_semantics_data = data.get("field_semantics", {})
        if field_semantics_data:
            pattern.field_semantics = {
                k: FieldSemantics.from_dict(v) for k, v in field_semantics_data.items()
            }

        return pattern


@dataclass
class _CompressionHint:
    """Internal recommendation envelope (PR-B5: private, observation-only).

    Pre-B5 this was the public return type of `get_recommendation()` —
    the request-time hint API now retired. The dataclass is retained as
    `_CompressionHint` purely for the deprecated stub's signature and
    for the publish CLI's internal aggregation; no new code should
    construct or consume it. Read recommendations from
    `recommendations.toml` produced by `headroom.cli.toin_publish`.
    """

    # Should we compress at all?
    skip_compression: bool = False

    # How aggressively to compress
    max_items: int = 20
    compression_level: Literal["none", "conservative", "moderate", "aggressive"] = "moderate"

    # Which fields to preserve (never remove)
    preserve_fields: list[str] = field(default_factory=list)

    # Which strategy to use
    recommended_strategy: str = "default"

    # Why this recommendation
    reason: str = ""
    confidence: float = 0.0

    # Source of recommendation
    source: Literal["network", "local", "default"] = "default"
    based_on_samples: int = 0

    # === TOIN Evolution: Learned Field Semantics ===
    # These enable zero-latency signal detection in SmartCrusher.
    # field_hash -> FieldSemantics (learned semantic type, important values, etc.)
    field_semantics: dict[str, FieldSemantics] = field(default_factory=dict)


@dataclass
class TOINConfig:
    """Configuration for the Tool Output Intelligence Network."""

    # Enable/disable TOIN
    enabled: bool = True

    # Storage
    # Default path is ~/.headroom/toin.json (or HEADROOM_TOIN_PATH env var)
    storage_path: str = field(default_factory=get_default_toin_storage_path)
    auto_save_interval: int = 600  # Auto-save every 10 minutes

    # Network learning thresholds
    min_samples_for_recommendation: int = 10
    min_users_for_network_effect: int = 3

    # Recommendation thresholds
    high_retrieval_threshold: float = 0.5  # Above this = compress less
    medium_retrieval_threshold: float = 0.2  # Between medium and high = moderate

    # Privacy
    anonymize_queries: bool = True
    max_query_patterns: int = 10

    # LOW FIX #22: Metrics/monitoring hooks
    # Callback for emitting metrics events. Signature: (event_name, event_data) -> None
    # Event names: "toin.compression", "toin.retrieval", "toin.recommendation", "toin.save"
    # This allows integration with Prometheus, StatsD, OpenTelemetry, etc.
    metrics_callback: MetricsCallback | None = None


class ToolIntelligenceNetwork:
    """Aggregates tool patterns across all Headroom users (observation-only).

    This is the offline brain of TOIN. It maintains a database of learned
    patterns for different `(auth_mode, model_family, tool_signature)`
    slices. The `record_compression` / `record_retrieval` calls are the
    only request-time API; aggregated recommendations are emitted by
    `headroom.cli.toin_publish` and consumed by the Rust proxy at startup.

    Thread-safe for concurrent access.
    """

    # ── Deprecation warning de-dupe (PR-B5) ───────────────────────────────
    # `get_recommendation` is retired as a per-request mutator. We emit
    # `DeprecationWarning` once per process; if every call warned, busy
    # call sites would flood logs and obscure other warnings. Class-level
    # so all instances share the flag.
    _DEPRECATION_WARNED: bool = False

    def __init__(
        self,
        config: TOINConfig | None = None,
        backend: Any | None = None,
    ):
        """Initialize TOIN.

        Args:
            config: Configuration options.
            backend: Storage backend implementing TOINBackend protocol.
                     If None, creates a FileSystemTOINBackend from config.storage_path.
                     Pass a custom backend for Redis, PostgreSQL, etc.
        """
        from .backends import FileSystemTOINBackend

        self._config = config or TOINConfig()
        self._lock = threading.RLock()  # RLock for reentrant locking (save calls export_patterns)

        # Storage backend
        if backend is not None:
            self._backend = backend
        elif self._config.storage_path:
            self._backend = FileSystemTOINBackend(self._config.storage_path)
        else:
            self._backend = None

        # Pattern database: (auth_mode, model_family, structure_hash) -> ToolPattern
        # PR-B5 extended the key from a bare structure_hash to the per-tenant
        # tuple. The serialized form on disk encodes the tuple as
        # "auth|model|hash"; see `_serialize_pattern_key`.
        self._patterns: dict[PatternKey, ToolPattern] = {}

        # Instance ID for user counting (anonymized)
        # IMPORTANT: Must be STABLE across restarts to avoid false user count inflation
        # Derive from storage path if available, otherwise use machine-specific ID
        self._instance_id = self._generate_stable_instance_id()

        # Tracking
        self._last_save_time = time.time()
        self._dirty = False

        # Load existing data from backend
        if self._backend is not None:
            self._load_from_backend()

    def _generate_stable_instance_id(self) -> str:
        """Generate a stable instance ID that doesn't change across restarts.

        Uses storage path if available, otherwise uses machine-specific info.
        This prevents false user count inflation when reloading from disk.

        HIGH FIX: Instance ID collision risk
        Previously used SHA256[:8] (32 bits) which has 50% collision probability
        at sqrt(2^32) ≈ 65,536 users (birthday paradox). Increased to SHA256[:16]
        (64 bits) for 50% collision at ~4 billion users, which is acceptable.
        """
        if self._config.storage_path:
            # Derive from storage path - same path = same instance
            return hashlib.sha256(self._config.storage_path.encode()).hexdigest()[
                :16
            ]  # HIGH FIX: 64 bits instead of 32
        else:
            # No storage - use a combination of hostname and process info
            # This is less stable but better than pure random
            import os
            import socket

            machine_info = (
                f"{socket.gethostname()}:{os.getuid() if hasattr(os, 'getuid') else 'unknown'}"
            )
            return hashlib.sha256(machine_info.encode()).hexdigest()[:16]  # HIGH FIX: 64 bits

    def _emit_metric(self, event_name: str, event_data: dict[str, Any]) -> None:
        """Emit a metrics event via the configured callback.

        LOW FIX #22: Provides monitoring integration for external metrics systems.

        Args:
            event_name: Name of the event (e.g., "toin.compression").
            event_data: Dictionary of event data to emit.
        """
        if self._config.metrics_callback is not None:
            try:
                self._config.metrics_callback(event_name, event_data)
            except Exception as e:
                # Never let metrics callback failures break TOIN
                logger.debug(f"Metrics callback failed for {event_name}: {e}")

    def record_compression(
        self,
        tool_signature: ToolSignature,
        original_count: int,
        compressed_count: int,
        original_tokens: int,
        compressed_tokens: int,
        strategy: str,
        query_context: str | None = None,
        items: list[dict[str, Any]] | None = None,
        auth_mode: str | None = None,
        model_family: str | None = None,
    ) -> None:
        """Record a compression event.

        Called after SmartCrusher compresses data. Updates the pattern
        for this `(auth_mode, model_family, tool_signature)` slice.

        TOIN Evolution: When items are provided, we capture field statistics
        for learning semantic types (uniqueness, default values, etc.).

        Args:
            tool_signature: Signature of the tool output structure.
            original_count: Original number of items.
            compressed_count: Number of items after compression.
            original_tokens: Original token count.
            compressed_tokens: Compressed token count.
            strategy: Compression strategy used.
            query_context: Optional user query that triggered this tool call.
            items: Optional list of items being compressed for field-level learning.
            auth_mode: Tenant auth slice (`payg` / `oauth` / `subscription`).
                Defaults to `DEFAULT_AUTH_MODE` when not provided.
            model_family: Target model family (`claude-3-5`, `gpt-4o`, …).
                Defaults to `DEFAULT_MODEL_FAMILY` when not provided.
        """
        # HIGH FIX: Check enabled FIRST to avoid computing structure_hash if disabled
        # This saves CPU when TOIN is turned off
        if not self._config.enabled:
            return

        # Computing structure_hash can be expensive for large structures
        sig_hash = tool_signature.structure_hash
        key = _make_pattern_key(auth_mode, model_family, sig_hash)

        # LOW FIX #22: Emit compression metric
        self._emit_metric(
            "toin.compression",
            {
                "signature_hash": sig_hash,
                "auth_mode": key[0],
                "model_family": key[1],
                "original_count": original_count,
                "compressed_count": compressed_count,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "strategy": strategy,
                "compression_ratio": compressed_count / original_count if original_count > 0 else 0,
            },
        )

        with self._lock:
            # Get or create pattern
            if key not in self._patterns:
                self._patterns[key] = ToolPattern(
                    tool_signature_hash=sig_hash,
                    auth_mode=key[0],
                    model_family=key[1],
                )

            pattern = self._patterns[key]

            # Update compression stats
            pattern.total_compressions += 1
            pattern.total_items_seen += original_count
            pattern.total_items_kept += compressed_count
            pattern.sample_size += 1

            # Update rolling averages
            n = pattern.total_compressions
            compression_ratio = compressed_count / original_count if original_count > 0 else 0.0
            token_reduction = (
                1 - (compressed_tokens / original_tokens) if original_tokens > 0 else 0.0
            )

            pattern.avg_compression_ratio = (
                pattern.avg_compression_ratio * (n - 1) + compression_ratio
            ) / n
            pattern.avg_token_reduction = (
                pattern.avg_token_reduction * (n - 1) + token_reduction
            ) / n

            # Update strategy stats
            if strategy not in pattern.strategy_success_rates:
                pattern.strategy_success_rates[strategy] = 1.0  # Start optimistic
            else:
                # Give a small boost for each compression without retrieval
                # This counteracts the penalty from record_retrieval() and prevents
                # all strategies from trending to 0.0 over time (one-way ratchet fix)
                # The boost is small (0.02) because retrieval penalties are larger (0.05-0.15)
                # This means strategies that cause retrievals will still trend down
                current_rate = pattern.strategy_success_rates[strategy]
                pattern.strategy_success_rates[strategy] = min(1.0, current_rate + 0.02)

            # HIGH FIX: Bound strategy_success_rates to prevent unbounded growth
            # Keep top 20 strategies by success rate
            if len(pattern.strategy_success_rates) > 20:
                sorted_strategies = sorted(
                    pattern.strategy_success_rates.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:20]
                pattern.strategy_success_rates = dict(sorted_strategies)

            # Track unique users via instance_id
            # FIX: Use _all_seen_instances set for lookup to prevent double-counting
            # after the storage list hits its cap
            # CRITICAL FIX #1: Check cap before adding to prevent OOM
            if self._instance_id not in pattern._all_seen_instances:
                # CRITICAL FIX: Check if we can verify this is a new user
                # If tracking was truncated (users lost after restart), we can only
                # count new users if we can add them to _all_seen_instances for dedup
                can_track = len(pattern._all_seen_instances) < ToolPattern.MAX_SEEN_INSTANCES

                if can_track:
                    # Add to the lookup set - we can verify this is new
                    pattern._all_seen_instances.add(self._instance_id)
                    # Also add to storage list (capped at 100 for serialization)
                    if len(pattern._seen_instance_hashes) < 100:
                        pattern._seen_instance_hashes.append(self._instance_id)
                    # Safe to increment user_count - we verified it's new
                    pattern.user_count += 1
                elif not pattern._tracking_truncated:
                    # Tracking set is full but we weren't truncated before
                    # This is a truly new user beyond our tracking capacity
                    pattern.user_count += 1
                # else: Can't verify if new, skip incrementing to prevent double-count

            # Track query context patterns for learning (privacy-preserving)
            if query_context and len(query_context) >= 3:
                # Normalize and anonymize: extract keywords, remove values
                query_pattern = self._anonymize_query_pattern(query_context)
                if query_pattern:
                    # MEDIUM FIX #10: Track frequency to keep most common patterns
                    pattern.query_pattern_frequency[query_pattern] = (
                        pattern.query_pattern_frequency.get(query_pattern, 0) + 1
                    )
                    # Update the list to contain top patterns by frequency
                    if query_pattern not in pattern.common_query_patterns:
                        pattern.common_query_patterns.append(query_pattern)
                    # Keep only the most common patterns (by frequency)
                    if len(pattern.common_query_patterns) > self._config.max_query_patterns:
                        pattern.common_query_patterns = sorted(
                            pattern.common_query_patterns,
                            key=lambda p: pattern.query_pattern_frequency.get(p, 0),
                            reverse=True,
                        )[: self._config.max_query_patterns]
                    # Also limit the frequency dict
                    if len(pattern.query_pattern_frequency) > self._config.max_query_patterns * 2:
                        top_patterns = sorted(
                            pattern.query_pattern_frequency.items(),
                            key=lambda x: x[1],
                            reverse=True,
                        )[: self._config.max_query_patterns * 2]
                        pattern.query_pattern_frequency = dict(top_patterns)

            # Periodically update recommendations even without retrievals
            # This ensures optimal_strategy is updated based on success rates
            if pattern.total_compressions % 10 == 0:
                self._update_recommendations(pattern)

            # === TOIN Evolution: Field Statistics for Semantic Learning ===
            # Capture field-level statistics to learn default values and uniqueness
            if items:
                self._update_field_statistics(pattern, items)

            pattern.last_updated = time.time()
            pattern.confidence = self._calculate_confidence(pattern)
            self._dirty = True

        # Auto-save if needed (outside lock)
        self._maybe_auto_save()

    def _update_field_statistics(
        self,
        pattern: ToolPattern,
        items: list[dict[str, Any]],
    ) -> None:
        """Update field statistics from compression items.

        Captures uniqueness, default values, and value distribution for
        learning field semantic types.

        Args:
            pattern: ToolPattern to update.
            items: Items being compressed.
        """
        if not items:
            return

        # Analyze field statistics (sample up to 100 items to limit CPU)
        sample_items = items[:100] if len(items) > 100 else items

        # Collect values for each field
        field_values: dict[str, list[str]] = {}  # field_hash -> list of value_hashes

        for item in sample_items:
            if not isinstance(item, dict):
                continue

            for field_name, value in item.items():
                field_hash = self._hash_field_name(field_name)
                value_hash = self._hash_value(value)

                if field_hash not in field_values:
                    field_values[field_hash] = []
                field_values[field_hash].append(value_hash)

        # Update FieldSemantics with statistics
        for field_hash, values in field_values.items():
            if not values:
                continue

            # Get or create FieldSemantics
            if field_hash not in pattern.field_semantics:
                pattern.field_semantics[field_hash] = FieldSemantics(field_hash=field_hash)

            field_sem = pattern.field_semantics[field_hash]

            # Calculate statistics
            unique_values = len(set(values))
            total_values = len(values)

            # Find most common value
            from collections import Counter

            value_counts = Counter(values)
            most_common_value, most_common_count = value_counts.most_common(1)[0]
            most_common_frequency = most_common_count / total_values if total_values > 0 else 0.0

            # Record compression stats
            field_sem.record_compression_stats(
                unique_values=unique_values,
                total_values=total_values,
                most_common_value_hash=most_common_value,
                most_common_frequency=most_common_frequency,
            )

        # Bound field_semantics to prevent unbounded growth (max 100 fields)
        if len(pattern.field_semantics) > 100:
            # Keep fields with highest activity (retrieval + compression count)
            sorted_fields = sorted(
                pattern.field_semantics.items(),
                key=lambda x: x[1].retrieval_count + x[1].compression_count,
                reverse=True,
            )[:100]
            pattern.field_semantics = dict(sorted_fields)

    def record_retrieval(
        self,
        tool_signature_hash: str,
        retrieval_type: str,
        query: str | None = None,
        query_fields: list[str] | None = None,
        strategy: str | None = None,
        retrieved_items: list[dict[str, Any]] | None = None,
        auth_mode: str | None = None,
        model_family: str | None = None,
    ) -> None:
        """Record a retrieval event.

        Called when LLM retrieves compressed content. This is the key
        feedback signal - it means compression was too aggressive.

        TOIN Evolution: When retrieved_items are provided, we learn field
        semantics from the values. This enables zero-latency signal detection.

        Args:
            tool_signature_hash: Hash of the tool signature.
            retrieval_type: "full" or "search".
            query: Optional search query (will be anonymized).
            query_fields: Fields mentioned in query (will be hashed).
            strategy: Compression strategy that was used (for success rate tracking).
            retrieved_items: Optional list of retrieved items for field-level learning.
            auth_mode: Tenant auth slice. Defaults to `DEFAULT_AUTH_MODE`.
            model_family: Target model family. Defaults to `DEFAULT_MODEL_FAMILY`.
        """
        if not self._config.enabled:
            return

        key = _make_pattern_key(auth_mode, model_family, tool_signature_hash)

        # LOW FIX #22: Emit retrieval metric
        self._emit_metric(
            "toin.retrieval",
            {
                "signature_hash": tool_signature_hash,
                "auth_mode": key[0],
                "model_family": key[1],
                "retrieval_type": retrieval_type,
                "has_query": query is not None,
                "query_fields_count": len(query_fields) if query_fields else 0,
                "strategy": strategy,
            },
        )

        with self._lock:
            if key not in self._patterns:
                # First time seeing this tool via retrieval
                self._patterns[key] = ToolPattern(
                    tool_signature_hash=tool_signature_hash,
                    auth_mode=key[0],
                    model_family=key[1],
                )

            pattern = self._patterns[key]

            # Update retrieval stats
            pattern.total_retrievals += 1
            if retrieval_type == "full":
                pattern.full_retrievals += 1
            else:
                pattern.search_retrievals += 1

            # Update strategy success rates - retrieval means the strategy was TOO aggressive
            # Decrease success rate for this strategy
            if strategy and strategy in pattern.strategy_success_rates:
                # Exponential moving average: penalize strategies that trigger retrieval
                # Full retrievals are worse than search retrievals
                penalty = 0.15 if retrieval_type == "full" else 0.05
                current_rate = pattern.strategy_success_rates[strategy]
                pattern.strategy_success_rates[strategy] = max(0.0, current_rate - penalty)

            # Track queried fields (anonymized)
            if query_fields:
                for field_name in query_fields:
                    field_hash = self._hash_field_name(field_name)
                    pattern.field_retrieval_frequency[field_hash] = (
                        pattern.field_retrieval_frequency.get(field_hash, 0) + 1
                    )

                    # Update commonly retrieved fields
                    if field_hash not in pattern.commonly_retrieved_fields:
                        # Add if frequently retrieved (check count from dict)
                        freq = pattern.field_retrieval_frequency.get(field_hash, 0)
                        if freq >= 3:
                            pattern.commonly_retrieved_fields.append(field_hash)
                            # HIGH: Limit commonly_retrieved_fields to prevent unbounded growth
                            if len(pattern.commonly_retrieved_fields) > 20:
                                # Keep only the most frequently retrieved fields
                                sorted_fields = sorted(
                                    pattern.commonly_retrieved_fields,
                                    key=lambda f: pattern.field_retrieval_frequency.get(f, 0),
                                    reverse=True,
                                )
                                pattern.commonly_retrieved_fields = sorted_fields[:20]

                # HIGH: Limit field_retrieval_frequency dict to prevent unbounded growth
                if len(pattern.field_retrieval_frequency) > 100:
                    sorted_freq_items = sorted(
                        pattern.field_retrieval_frequency.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:100]
                    pattern.field_retrieval_frequency = dict(sorted_freq_items)

            # Track query patterns (anonymized)
            if query and self._config.anonymize_queries:
                query_pattern = self._anonymize_query_pattern(query)
                if query_pattern:
                    # MEDIUM FIX #10: Track frequency to keep most common patterns
                    pattern.query_pattern_frequency[query_pattern] = (
                        pattern.query_pattern_frequency.get(query_pattern, 0) + 1
                    )
                    if query_pattern not in pattern.common_query_patterns:
                        pattern.common_query_patterns.append(query_pattern)
                    # Keep only the most common patterns (by frequency)
                    if len(pattern.common_query_patterns) > self._config.max_query_patterns:
                        pattern.common_query_patterns = sorted(
                            pattern.common_query_patterns,
                            key=lambda p: pattern.query_pattern_frequency.get(p, 0),
                            reverse=True,
                        )[: self._config.max_query_patterns]

            # === TOIN Evolution: Field-Level Semantic Learning ===
            # Learn from retrieved items to build zero-latency signal detection
            if retrieved_items:
                # Extract query operator from query string (for learning)
                query_operator = self._extract_query_operator(query) if query else "="

                for item in retrieved_items:
                    if not isinstance(item, dict):
                        continue

                    for field_name, value in item.items():
                        field_hash = self._hash_field_name(field_name)

                        # Get or create FieldSemantics for this field
                        if field_hash not in pattern.field_semantics:
                            pattern.field_semantics[field_hash] = FieldSemantics(
                                field_hash=field_hash
                            )

                        field_sem = pattern.field_semantics[field_hash]

                        # Hash the value for privacy
                        value_hash = self._hash_value(value)

                        # Record this retrieval
                        field_sem.record_retrieval_value(value_hash, query_operator)

                # Periodically infer types (every 5 retrievals to save CPU)
                if pattern.total_retrievals % 5 == 0:
                    for field_sem in pattern.field_semantics.values():
                        if field_sem.retrieval_count >= 3:  # Need minimum data
                            field_sem.infer_type()

                # Bound field_semantics to prevent unbounded growth (max 100 fields)
                if len(pattern.field_semantics) > 100:
                    # Keep fields with highest retrieval counts
                    sorted_semantics = sorted(
                        pattern.field_semantics.items(),
                        key=lambda x: x[1].retrieval_count,
                        reverse=True,
                    )[:100]
                    pattern.field_semantics = dict(sorted_semantics)

            # Update recommendations based on new retrieval data
            self._update_recommendations(pattern)

            pattern.last_updated = time.time()
            self._dirty = True

        self._maybe_auto_save()

    def get_recommendation(
        self,
        tool_signature: ToolSignature,  # noqa: ARG002 — kept for source compat
        query_context: str | None = None,  # noqa: ARG002
    ) -> None:
        """**Deprecated.** Returns `None`. PR-B5 retired the request-time hint API.

        TOIN is observation-only; recommendations are emitted by the
        offline `headroom.cli.toin_publish` CLI into `recommendations.toml`
        and loaded by the Rust proxy at startup. New code must not call
        this method. Existing call sites should migrate to reading the
        TOML file directly.

        Emits `DeprecationWarning` once per process to keep busy call
        sites from flooding logs.

        Returns:
            Always `None`. The legacy `_CompressionHint` envelope is no
            longer constructed at request time.
        """
        cls = type(self)
        if not cls._DEPRECATION_WARNED:
            cls._DEPRECATION_WARNED = True
            warnings.warn(
                "ToolIntelligenceNetwork.get_recommendation() is deprecated "
                "and now returns None. PR-B5 retired the request-time hint "
                "API; recommendations come from recommendations.toml at "
                "startup. See headroom/telemetry/toin.py module docstring.",
                DeprecationWarning,
                stacklevel=2,
            )
        return None

    def _update_recommendations(self, pattern: ToolPattern) -> None:
        """Update learned recommendations for a pattern."""
        # Calculate optimal max_items based on retrieval rate
        retrieval_rate = pattern.retrieval_rate

        if retrieval_rate > self._config.high_retrieval_threshold:
            if pattern.full_retrieval_rate > 0.8:
                pattern.skip_compression_recommended = True
                pattern.optimal_max_items = pattern.total_items_seen // max(
                    1, pattern.total_compressions
                )
            else:
                pattern.optimal_max_items = 50
        elif retrieval_rate > self._config.medium_retrieval_threshold:
            pattern.optimal_max_items = 30
        else:
            pattern.optimal_max_items = 20

        # Update preserve_fields from frequently retrieved fields
        if pattern.field_retrieval_frequency:
            # Get top 5 most retrieved fields
            sorted_fields = sorted(
                pattern.field_retrieval_frequency.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            pattern.preserve_fields = [f for f, _ in sorted_fields]

        # Update optimal strategy (pick most successful)
        if pattern.strategy_success_rates:
            best_strategy = max(
                pattern.strategy_success_rates.items(),
                key=lambda x: x[1],
            )[0]
            pattern.optimal_strategy = best_strategy

    def _calculate_confidence(self, pattern: ToolPattern) -> float:
        """Calculate confidence level for a pattern."""
        # Base confidence on sample size
        sample_confidence = min(0.7, pattern.sample_size / 100)

        # Boost if from multiple users
        # FIX: Changed from `user_count / 10 * 0.1` (= user_count * 0.01, too small)
        # to `user_count * 0.03` for meaningful boost at low user counts
        # - 3 users: 0.09 boost
        # - 10 users: 0.30 boost (capped)
        user_boost = 0.0
        if pattern.user_count >= self._config.min_users_for_network_effect:
            user_boost = min(0.3, pattern.user_count * 0.03)

        return min(0.95, sample_confidence + user_boost)

    def _hash_field_name(self, field_name: str) -> str:
        """Hash a field name for anonymization."""
        return hashlib.sha256(field_name.encode()).hexdigest()[:8]

    def _anonymize_query_pattern(self, query: str) -> str | None:
        """Extract anonymized pattern from a query.

        Keeps structural patterns, removes specific values.
        E.g., "status:error AND user:john" -> "status:* AND user:*"
        """
        if not query:
            return None

        # Simple pattern extraction: replace values after : or =
        import re

        # Match field:value or field="value" patterns, but don't include spaces in unquoted values
        pattern = re.sub(r'(\w+)[=:](?:"[^"]*"|\'[^\']*\'|\w+)', r"\1:*", query)

        # Remove if it's just generic
        if pattern in ("*", ""):
            return None

        return pattern

    def _hash_value(self, value: Any) -> str:
        """Hash a value for privacy-preserving storage.

        Handles all types by converting to a canonical string representation.
        """
        if value is None:
            canonical = "null"
        elif isinstance(value, bool):
            canonical = "true" if value else "false"
        elif isinstance(value, int | float):
            canonical = str(value)
        elif isinstance(value, str):
            canonical = value
        elif isinstance(value, list | dict):
            # For complex types, use JSON serialization
            try:
                canonical = json.dumps(value, sort_keys=True, default=str)
            except (TypeError, ValueError):
                canonical = str(value)
        else:
            canonical = str(value)

        return hashlib.sha256(canonical.encode()).hexdigest()[:8]

    def _extract_query_operator(self, query: str) -> str:
        """Extract the dominant query operator from a search query.

        Used for learning field semantic types from query patterns.

        Returns:
            Query operator: "=", "!=", ">", "<", ">=", "<=", "contains", or "="
        """
        if not query:
            return "="

        query_lower = query.lower()

        # Check for inequality operators
        if "!=" in query or " not " in query_lower or " ne " in query_lower:
            return "!="
        if ">=" in query or " gte " in query_lower:
            return ">="
        if "<=" in query or " lte " in query_lower:
            return "<="
        if ">" in query or " gt " in query_lower:
            return ">"
        if "<" in query or " lt " in query_lower:
            return "<"

        # Check for text search operators
        if " like " in query_lower or " contains " in query_lower or "*" in query:
            return "contains"

        # Default to equality
        return "="

    def get_stats(self) -> dict[str, Any]:
        """Get overall TOIN statistics."""
        with self._lock:
            total_compressions = sum(p.total_compressions for p in self._patterns.values())
            total_retrievals = sum(p.total_retrievals for p in self._patterns.values())

            return {
                "enabled": self._config.enabled,
                "patterns_tracked": len(self._patterns),
                "total_compressions": total_compressions,
                "total_retrievals": total_retrievals,
                "global_retrieval_rate": (
                    total_retrievals / total_compressions if total_compressions > 0 else 0.0
                ),
                "patterns_with_recommendations": sum(
                    1
                    for p in self._patterns.values()
                    if p.sample_size >= self._config.min_samples_for_recommendation
                ),
            }

    def get_pattern(
        self,
        signature_hash: str,
        auth_mode: str | None = None,
        model_family: str | None = None,
    ) -> ToolPattern | None:
        """Get pattern data for a specific `(auth_mode, model_family, sig_hash)` slice.

        Defaults to `(DEFAULT_AUTH_MODE, DEFAULT_MODEL_FAMILY, signature_hash)`
        when callers haven't supplied tenant info — preserves source-compat
        with pre-B5 callers that look up by bare hash.

        HIGH FIX: Returns a deep copy to prevent external mutation of internal state.
        """
        import copy

        key = _make_pattern_key(auth_mode, model_family, signature_hash)

        with self._lock:
            pattern = self._patterns.get(key)
            if pattern is not None:
                return copy.deepcopy(pattern)
            return None

    def iter_patterns(self) -> list[tuple[PatternKey, ToolPattern]]:
        """Snapshot of `(key, pattern)` pairs for offline aggregation.

        Used by `headroom.cli.toin_publish` to walk every aggregated
        slice without exposing the live `_patterns` dict to external
        callers (deep-copies each pattern to prevent mutation).
        """
        import copy

        with self._lock:
            return [(k, copy.deepcopy(p)) for k, p in self._patterns.items()]

    def export_patterns(self) -> dict[str, Any]:
        """Export all patterns for sharing/aggregation.

        The aggregation key tuple is encoded as a `"auth|model|hash"`
        string for JSON storage (JSON object keys must be strings).
        See `_serialize_pattern_key` for the canonical encoding.
        """
        with self._lock:
            return {
                "version": "2.0",  # PR-B5: tuple aggregation key
                "export_timestamp": time.time(),
                "instance_id": self._instance_id,
                "patterns": {
                    _serialize_pattern_key(key): pattern.to_dict()
                    for key, pattern in self._patterns.items()
                },
            }

    def import_patterns(self, data: dict[str, Any]) -> None:
        """Import patterns from another source.

        Used for federated learning: aggregate patterns from multiple
        Headroom instances without sharing actual data.

        Backward-compatible with v1.0 dumps that keyed patterns by bare
        structure_hash: those are promoted to the
        `(DEFAULT_AUTH_MODE, DEFAULT_MODEL_FAMILY, sig_hash)` slice via
        `_deserialize_pattern_key`.

        Args:
            data: Exported pattern data.
        """
        if not self._config.enabled:
            return

        patterns_data = data.get("patterns", {})
        source_instance = data.get("instance_id", "unknown")

        with self._lock:
            for serialized_key, pattern_dict in patterns_data.items():
                key = _deserialize_pattern_key(serialized_key)
                imported = ToolPattern.from_dict(pattern_dict)
                # Make sure dataclass fields agree with the dict key — pre-B5
                # dumps don't carry auth_mode/model_family on the pattern;
                # promote from the (possibly default) key.
                imported.auth_mode = key[0]
                imported.model_family = key[1]

                if key in self._patterns:
                    # Merge with existing
                    self._merge_patterns(self._patterns[key], imported)
                else:
                    # Add new pattern - need to track source instance
                    self._patterns[key] = imported

                    # For NEW patterns from another instance, track the source in
                    # _seen_instance_hashes so user_count reflects cross-user data
                    if source_instance != self._instance_id:
                        pattern = self._patterns[key]
                        if source_instance not in pattern._seen_instance_hashes:
                            # Limit storage to 100 unique instances to bound memory
                            if len(pattern._seen_instance_hashes) < 100:
                                pattern._seen_instance_hashes.append(source_instance)
                            # CRITICAL: Always increment user_count (even after cap)
                            pattern.user_count += 1

            self._dirty = True

    def _merge_patterns(self, existing: ToolPattern, imported: ToolPattern) -> None:
        """Merge imported pattern into existing."""
        total = existing.sample_size + imported.sample_size
        if total == 0:
            return

        w_existing = existing.sample_size / total
        w_imported = imported.sample_size / total

        # Merge counts
        existing.total_compressions += imported.total_compressions
        existing.total_retrievals += imported.total_retrievals
        existing.full_retrievals += imported.full_retrievals
        existing.search_retrievals += imported.search_retrievals
        existing.total_items_seen += imported.total_items_seen
        existing.total_items_kept += imported.total_items_kept

        # Weighted averages
        existing.avg_compression_ratio = (
            existing.avg_compression_ratio * w_existing
            + imported.avg_compression_ratio * w_imported
        )
        existing.avg_token_reduction = (
            existing.avg_token_reduction * w_existing + imported.avg_token_reduction * w_imported
        )

        # Merge field frequencies
        for field_hash, count in imported.field_retrieval_frequency.items():
            existing.field_retrieval_frequency[field_hash] = (
                existing.field_retrieval_frequency.get(field_hash, 0) + count
            )
        # HIGH: Limit field_retrieval_frequency dict to prevent unbounded growth
        if len(existing.field_retrieval_frequency) > 100:
            # Keep only the most frequently retrieved fields
            sorted_fields = sorted(
                existing.field_retrieval_frequency.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:100]
            existing.field_retrieval_frequency = dict(sorted_fields)

        # Merge commonly retrieved fields
        for field_hash in imported.commonly_retrieved_fields:
            if field_hash not in existing.commonly_retrieved_fields:
                existing.commonly_retrieved_fields.append(field_hash)
        # HIGH: Limit commonly_retrieved_fields to prevent unbounded growth
        if len(existing.commonly_retrieved_fields) > 20:
            # Prioritize by retrieval frequency if available
            if existing.field_retrieval_frequency:
                existing.commonly_retrieved_fields = sorted(
                    existing.commonly_retrieved_fields,
                    key=lambda f: existing.field_retrieval_frequency.get(f, 0),
                    reverse=True,
                )[:20]
            else:
                existing.commonly_retrieved_fields = existing.commonly_retrieved_fields[:20]

        # Merge query patterns (for federated learning)
        # MEDIUM FIX #10: Also merge query_pattern_frequency for proper ranking
        for query_pattern, freq in imported.query_pattern_frequency.items():
            existing.query_pattern_frequency[query_pattern] = (
                existing.query_pattern_frequency.get(query_pattern, 0) + freq
            )
        for query_pattern in imported.common_query_patterns:
            if query_pattern not in existing.common_query_patterns:
                existing.common_query_patterns.append(query_pattern)
        # Keep only the most common patterns (by frequency)
        if len(existing.common_query_patterns) > self._config.max_query_patterns:
            existing.common_query_patterns = sorted(
                existing.common_query_patterns,
                key=lambda p: existing.query_pattern_frequency.get(p, 0),
                reverse=True,
            )[: self._config.max_query_patterns]
        # Limit frequency dict
        if len(existing.query_pattern_frequency) > self._config.max_query_patterns * 2:
            top_patterns = sorted(
                existing.query_pattern_frequency.items(),
                key=lambda x: x[1],
                reverse=True,
            )[: self._config.max_query_patterns * 2]
            existing.query_pattern_frequency = dict(top_patterns)

        # Merge strategy success rates (weighted average)
        for strategy, rate in imported.strategy_success_rates.items():
            if strategy in existing.strategy_success_rates:
                existing.strategy_success_rates[strategy] = (
                    existing.strategy_success_rates[strategy] * w_existing + rate * w_imported
                )
            else:
                existing.strategy_success_rates[strategy] = rate

        # HIGH FIX: Bound strategy_success_rates after merge
        if len(existing.strategy_success_rates) > 20:
            sorted_strategies = sorted(
                existing.strategy_success_rates.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:20]
            existing.strategy_success_rates = dict(sorted_strategies)

        # Merge preserve_fields (union of both, deduplicated)
        for preserve_field in imported.preserve_fields:
            if preserve_field not in existing.preserve_fields:
                existing.preserve_fields.append(preserve_field)
        # Keep only top 10 most important fields
        if len(existing.preserve_fields) > 10:
            # Prioritize by retrieval frequency if available
            if existing.field_retrieval_frequency:
                existing.preserve_fields = sorted(
                    existing.preserve_fields,
                    key=lambda f: existing.field_retrieval_frequency.get(f, 0),
                    reverse=True,
                )[:10]
            else:
                existing.preserve_fields = existing.preserve_fields[:10]

        # Merge skip_compression_recommended (true if either recommends skip)
        if imported.skip_compression_recommended:
            # Imported has more data suggesting skip - consider it
            if imported.sample_size > existing.sample_size // 2:
                existing.skip_compression_recommended = True

        # Merge optimal_strategy (prefer the one with better success rate)
        if imported.optimal_strategy != "default":
            imported_rate = imported.strategy_success_rates.get(imported.optimal_strategy, 0.5)
            existing_rate = (
                existing.strategy_success_rates.get(existing.optimal_strategy, 0.5)
                if existing.optimal_strategy != "default"
                else 0.0
            )

            if imported_rate > existing_rate:
                existing.optimal_strategy = imported.optimal_strategy

        # Merge optimal_max_items (weighted average with bounds)
        if imported.optimal_max_items > 0:
            merged_max_items = int(
                existing.optimal_max_items * w_existing + imported.optimal_max_items * w_imported
            )
            # Ensure valid bounds: min 3 items, max 1000 items
            existing.optimal_max_items = max(3, min(1000, merged_max_items))

        existing.sample_size = total

        # Merge seen instance hashes (union of both, limited to 100 for storage)
        # CRITICAL FIX #1 & #3: Simplified user count merge logic with cap enforcement.
        # user_count is the authoritative count even when sets hit their caps.
        new_users_found = 0
        for instance_hash in imported._seen_instance_hashes:
            # Use _all_seen_instances for deduplication (the authoritative set)
            if instance_hash not in existing._all_seen_instances:
                # Add to lookup set (with cap to prevent OOM)
                if len(existing._all_seen_instances) < ToolPattern.MAX_SEEN_INSTANCES:
                    existing._all_seen_instances.add(instance_hash)
                # Limit storage list to 100 unique instances to bound serialization
                if len(existing._seen_instance_hashes) < 100:
                    existing._seen_instance_hashes.append(instance_hash)
                new_users_found += 1

        # Also merge instances from imported._all_seen_instances that weren't in list
        # (in case imported had more than 100 instances)
        for instance_hash in imported._all_seen_instances:
            if instance_hash not in existing._all_seen_instances:
                # Add with cap check
                if len(existing._all_seen_instances) < ToolPattern.MAX_SEEN_INSTANCES:
                    existing._all_seen_instances.add(instance_hash)
                # Storage list already at limit, just track for dedup
                new_users_found += 1

        # CRITICAL FIX #3: Simplified user count calculation.
        # We count new users from both the list and set, then add any users
        # that imported had beyond what we could deduplicate (when both hit caps).
        # imported.user_count may be > len(imported._all_seen_instances) if they hit cap
        users_beyond_imported_tracking = max(
            0, imported.user_count - len(imported._all_seen_instances)
        )
        existing.user_count += new_users_found + users_beyond_imported_tracking

        existing.last_updated = time.time()

        # Recalculate recommendations based on merged data
        self._update_recommendations(existing)

    def save(self) -> None:
        """Save TOIN data via the storage backend.

        HIGH FIX: Serialize under lock but write outside lock to prevent
        blocking other threads during slow file I/O.
        """
        if self._backend is None:
            return

        # Step 1: Serialize under lock (fast in-memory operation)
        with self._lock:
            data = self.export_patterns()

        # Step 2: Write outside lock (slow I/O operation)
        try:
            self._backend.save(data)

            # Step 3: Update state under lock (fast)
            with self._lock:
                self._dirty = False
                self._last_save_time = time.time()

        except Exception as e:
            # Surface storage failures structured so log aggregators can
            # alert on `event=toin_save_failed` without false positives
            # from generic exception lines. Per project memory
            # `feedback_no_silent_fallbacks.md`: never swallow.
            logger.warning(
                "TOIN storage save failed",
                extra={
                    "event": "toin_save_failed",
                    "backend": type(self._backend).__name__,
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )

    def _load_from_backend(self) -> None:
        """Load TOIN data from the storage backend."""
        if self._backend is None:
            return

        try:
            data = self._backend.load()
            if data:
                self.import_patterns(data)
                self._dirty = False
        except Exception as e:
            logger.warning(
                "TOIN storage load failed",
                extra={
                    "event": "toin_load_failed",
                    "backend": type(self._backend).__name__,
                    "error_type": type(e).__name__,
                    "error": str(e),
                },
            )

    def _maybe_auto_save(self) -> None:
        """Auto-save if enough time has passed.

        HIGH FIX: Check conditions under lock to prevent race where another
        thread modifies _dirty or _last_save_time between check and save.
        The save() method already acquires the lock, and we use RLock so
        it's safe to hold the lock when calling save().
        """
        if self._backend is None or not self._config.auto_save_interval:
            return

        # Check under lock to prevent race conditions
        with self._lock:
            if not self._dirty:
                return

            elapsed = time.time() - self._last_save_time
            if elapsed >= self._config.auto_save_interval:
                # save() uses the same RLock, so this is safe
                self.save()

    def clear(self) -> None:
        """Clear all TOIN data. Mainly for testing."""
        with self._lock:
            self._patterns.clear()
            self._dirty = False


# Global TOIN instance (lazy initialization)
_toin_instance: ToolIntelligenceNetwork | None = None
_toin_lock = threading.Lock()

# Environment variable for custom TOIN backend
TOIN_BACKEND_ENV_VAR = "HEADROOM_TOIN_BACKEND"


def _create_default_toin_backend() -> Any:
    """Create a TOIN backend from env (e.g. HEADROOM_TOIN_BACKEND=redis).

    Loads adapters via setuptools entry point 'headroom.toin_backend'.
    Returns None to use default FileSystemTOINBackend.
    """
    backend_type = (os.environ.get(TOIN_BACKEND_ENV_VAR) or "").strip().lower()
    if not backend_type or backend_type == "filesystem":
        return None
    if backend_type == "none":
        return None  # Explicit in-memory-only (e.g. --stateless mode)
    try:
        from importlib.metadata import entry_points

        all_eps = entry_points(group="headroom.toin_backend")
        ep = next((e for e in all_eps if e.name == backend_type), None)
        if ep is None:
            logger.warning(
                "HEADROOM_TOIN_BACKEND=%s but no entry point headroom.toin_backend[%s]",
                backend_type,
                backend_type,
            )
            return None
        fn = ep.load()
        # `tenant_prefix` is retained for storage-backend namespacing
        # (Redis key prefix, Postgres schema name, etc.) so multi-tenant
        # SaaS deployments can carve up shared infrastructure. PR-B5 made
        # the in-memory aggregation key per-tenant via `auth_mode` /
        # `model_family`, so `tenant_prefix` is now functionally redundant
        # for *learning* — it only matters for storage layout. Keep it.
        kwargs = {
            "url": os.environ.get("HEADROOM_TOIN_URL", ""),
            "tenant_prefix": os.environ.get("HEADROOM_TOIN_TENANT_PREFIX", ""),
        }
        return fn(**kwargs)
    except Exception as e:
        logger.warning("Failed to load TOIN backend %s: %s", backend_type, e)
        return None


def get_toin(config: TOINConfig | None = None) -> ToolIntelligenceNetwork:
    """Get the global TOIN instance.

    Thread-safe singleton pattern. Always acquires lock to avoid subtle
    race conditions in double-checked locking on non-CPython implementations.

    On first call, checks HEADROOM_TOIN_BACKEND env var. If set, loads the
    backend via setuptools entry point 'headroom.toin_backend'. Otherwise
    uses the default FileSystemTOINBackend.

    Args:
        config: Configuration (only used on first call). If the instance
            already exists, config is ignored and a warning is logged.

    Returns:
        Global ToolIntelligenceNetwork instance.
    """
    global _toin_instance

    # CRITICAL FIX: Always acquire lock for thread safety across all Python
    # implementations. The overhead is negligible since we only construct once.
    with _toin_lock:
        if _toin_instance is None:
            backend = _create_default_toin_backend()
            _toin_instance = ToolIntelligenceNetwork(config, backend=backend)
        elif config is not None:
            # Warn when config is silently ignored
            logger.warning(
                "TOIN config ignored: instance already exists. "
                "Call reset_toin() first if you need to change config."
            )

    return _toin_instance


def reset_toin() -> None:
    """Reset the global TOIN instance. Mainly for testing."""
    global _toin_instance

    with _toin_lock:
        if _toin_instance is not None:
            _toin_instance.clear()
        _toin_instance = None
