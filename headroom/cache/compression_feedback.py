"""Compression Feedback Loop for learning optimal compression strategies.

This module analyzes retrieval patterns from the CompressionStore to learn
what kinds of compression work well and what doesn't. It provides hints to
SmartCrusher to improve compression over time.

Key insight from ACON research: Learn compression guidelines by analyzing failures.
When compression causes the LLM to retrieve more data, that's a signal that
we compressed too aggressively.

Features:
- Track retrieval rates per tool type
- Learn common search queries for each tool
- Adjust compression aggressiveness based on patterns
- Provide hints: max_items, fields to preserve, etc.

Usage:
    feedback = CompressionFeedback(compression_store)

    # Get hints before compressing
    hints = feedback.get_compression_hints("github_search_repos")
    # hints = {"max_items": 50, "preserve_fields": ["id", "name"], ...}

    # Apply hints in SmartCrusher config
    config = SmartCrusherConfig(max_items=hints.get("max_items", 15))
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .compression_store import CompressionStore, RetrievalEvent


@dataclass
class LocalToolPattern:
    """Learned patterns for a specific tool type (local feedback).

    MEDIUM FIX #18: Renamed from ToolPattern to avoid confusion with
    headroom.telemetry.toin.ToolPattern which serves a different purpose:
    - LocalToolPattern: Local feedback patterns keyed by tool_name
    - toin.ToolPattern: Cross-user TOIN patterns keyed by tool_signature_hash
    """

    tool_name: str

    # Retrieval statistics
    total_compressions: int = 0
    total_retrievals: int = 0
    full_retrievals: int = 0  # Retrieved entire original content
    search_retrievals: int = 0  # Used search within content

    # Query analysis
    common_queries: dict[str, int] = field(default_factory=dict)
    queried_fields: dict[str, int] = field(default_factory=dict)

    # Strategy analysis - track which strategies work for this tool
    strategy_compressions: dict[str, int] = field(default_factory=dict)
    strategy_retrievals: dict[str, int] = field(default_factory=dict)

    # Signature hash tracking - correlate with TOIN patterns
    signature_hashes: set[str] = field(default_factory=set)

    # Timing
    last_compression: float = 0.0
    last_retrieval: float = 0.0

    # Calculated metrics
    @property
    def retrieval_rate(self) -> float:
        """Fraction of compressions that resulted in retrieval."""
        if self.total_compressions == 0:
            return 0.0
        return self.total_retrievals / self.total_compressions

    @property
    def full_retrieval_rate(self) -> float:
        """Fraction of retrievals that were full (not search)."""
        if self.total_retrievals == 0:
            return 0.0
        return self.full_retrievals / self.total_retrievals

    @property
    def search_rate(self) -> float:
        """Fraction of retrievals that used search."""
        if self.total_retrievals == 0:
            return 0.0
        return self.search_retrievals / self.total_retrievals

    def strategy_retrieval_rate(self, strategy: str) -> float:
        """Get retrieval rate for a specific compression strategy."""
        compressions = self.strategy_compressions.get(strategy, 0)
        if compressions == 0:
            return 0.0
        retrievals = self.strategy_retrievals.get(strategy, 0)
        return retrievals / compressions

    def best_strategy(self) -> str | None:
        """Find the strategy with lowest retrieval rate (most successful)."""
        if not self.strategy_compressions:
            return None

        best = None
        best_rate = 1.0

        for strategy in self.strategy_compressions:
            rate = self.strategy_retrieval_rate(strategy)
            # Only consider strategies with enough samples
            if self.strategy_compressions[strategy] >= 3 and rate < best_rate:
                best_rate = rate
                best = strategy

        return best


@dataclass
class CompressionHints:
    """Hints for optimizing compression of a specific tool's output."""

    # Item count hints
    max_items: int = 15  # Default from SmartCrusher
    min_items: int = 3
    suggested_items: int | None = None  # Calculated optimal

    # Field preservation
    preserve_fields: list[str] = field(default_factory=list)

    # Compression aggressiveness (0.0 = aggressive, 1.0 = conservative)
    aggressiveness: float = 0.7

    # Reasoning
    reason: str = ""

    # Whether to skip compression entirely
    skip_compression: bool = False

    # Recommended compression strategy based on local learning
    recommended_strategy: str | None = None


class CompressionFeedback:
    """Learn from retrieval patterns to improve compression.

    This class analyzes retrieval events from CompressionStore and builds
    tool-specific patterns. These patterns inform compression decisions.

    Design principles:
    - High retrieval rate (>50%) → compress less aggressively
    - Full retrieval dominates → data is unique, skip compression
    - Search retrieval dominates → keep compressed, add search capability
    - Frequent queries → preserve fields mentioned in queries
    """

    # Thresholds for adjusting compression
    HIGH_RETRIEVAL_THRESHOLD = 0.5  # 50% retrieval = too aggressive
    MEDIUM_RETRIEVAL_THRESHOLD = 0.2  # 20% retrieval = acceptable
    MIN_SAMPLES_FOR_HINTS = 5  # Need at least 5 events to make recommendations

    def __init__(
        self,
        store: CompressionStore | None = None,
        enable_learning: bool = True,
        analysis_interval: float = 60.0,
    ):
        """Initialize feedback analyzer.

        Args:
            store: CompressionStore to analyze. If None, uses global store.
            enable_learning: Whether to update patterns from events.
            analysis_interval: Interval in seconds between re-analyzing store events.
        """
        self._store = store
        self._enable_learning = enable_learning
        self._lock = threading.Lock()

        # Learned patterns per tool
        self._tool_patterns: dict[str, LocalToolPattern] = {}

        # Time-based tracking
        self._last_analysis: float = 0.0
        self._analysis_interval: float = analysis_interval
        self._last_event_timestamp: float = (
            0.0  # Track last processed event to avoid double-counting
        )

        # Global statistics
        self._total_compressions: int = 0
        self._total_retrievals: int = 0

    @property
    def store(self) -> CompressionStore:
        """Get the compression store (lazy load global if not set)."""
        if self._store is None:
            from .compression_store import get_compression_store

            self._store = get_compression_store()
        return self._store

    def record_compression(
        self,
        tool_name: str | None,
        original_count: int,
        compressed_count: int,
        strategy: str | None = None,
        tool_signature_hash: str | None = None,
    ) -> None:
        """Record that a compression occurred.

        Called by SmartCrusher after compressing to track compression events.

        Args:
            tool_name: Name of the tool whose output was compressed.
            original_count: Original item count.
            compressed_count: Compressed item count.
            strategy: Compression strategy used (e.g., "SMART_SAMPLE", "TOP_N").
            tool_signature_hash: Hash from ToolSignature for correlation with TOIN.
        """
        if not self._enable_learning or not tool_name:
            return

        with self._lock:
            self._total_compressions += 1

            if tool_name not in self._tool_patterns:
                self._tool_patterns[tool_name] = LocalToolPattern(tool_name=tool_name)

            pattern = self._tool_patterns[tool_name]
            pattern.total_compressions += 1
            pattern.last_compression = time.time()

            # Track strategy usage
            if strategy:
                pattern.strategy_compressions[strategy] = (
                    pattern.strategy_compressions.get(strategy, 0) + 1
                )

                # CRITICAL FIX: When truncating strategy dicts, keep them in sync
                # to prevent desync between compressions and retrievals.
                # Both dicts must have the same keys for accurate retrieval rate calculation.
                if len(pattern.strategy_compressions) > 50:
                    self._truncate_strategy_dicts(pattern)

            # Track signature hash for TOIN correlation
            if tool_signature_hash:
                pattern.signature_hashes.add(tool_signature_hash)
                # CRITICAL FIX: Use deterministic truncation for signature_hashes
                # Sort lexicographically to ensure consistent behavior across runs
                if len(pattern.signature_hashes) > 100:
                    sorted_hashes = sorted(pattern.signature_hashes)[:100]
                    pattern.signature_hashes = set(sorted_hashes)

    def record_retrieval(
        self,
        event: RetrievalEvent,
        strategy: str | None = None,
    ) -> None:
        """Record a retrieval event for pattern learning.

        Called by CompressionStore when content is retrieved.

        Args:
            event: The retrieval event to record.
            strategy: Compression strategy that was used (for tracking success rates).
        """
        if not self._enable_learning:
            return

        tool_name = event.tool_name
        if not tool_name:
            return

        with self._lock:
            self._total_retrievals += 1

            if tool_name not in self._tool_patterns:
                self._tool_patterns[tool_name] = LocalToolPattern(tool_name=tool_name)

            pattern = self._tool_patterns[tool_name]
            pattern.total_retrievals += 1
            pattern.last_retrieval = time.time()

            if event.retrieval_type == "full":
                pattern.full_retrievals += 1
            else:
                pattern.search_retrievals += 1

            # Track strategy retrievals (for success rate calculation)
            if strategy:
                pattern.strategy_retrievals[strategy] = (
                    pattern.strategy_retrievals.get(strategy, 0) + 1
                )

                # CRITICAL FIX: When truncating strategy dicts, keep them in sync
                # to prevent desync between compressions and retrievals.
                if len(pattern.strategy_retrievals) > 50:
                    self._truncate_strategy_dicts(pattern)

            # Track query patterns
            if event.query:
                query_lower = event.query.lower()
                pattern.common_queries[query_lower] = pattern.common_queries.get(query_lower, 0) + 1

                # HIGH: Limit common_queries dict to prevent unbounded growth
                if len(pattern.common_queries) > 100:
                    sorted_queries = sorted(
                        pattern.common_queries.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:100]
                    pattern.common_queries = dict(sorted_queries)

                # Extract potential field names from query
                self._extract_field_hints(pattern, event.query)

    def _truncate_strategy_dicts(self, pattern: LocalToolPattern) -> None:
        """Truncate strategy_compressions and strategy_retrievals in sync.

        CRITICAL FIX: Both dicts must have the same keys for accurate retrieval
        rate calculation. When truncating, we keep the union of top strategies
        from both dicts, then truncate both to the same key set.
        """
        # Get top 40 strategies from each dict (using 40 to allow union to stay under 50)
        top_compressions = {
            k
            for k, _ in sorted(
                pattern.strategy_compressions.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:40]
        }
        top_retrievals = {
            k
            for k, _ in sorted(
                pattern.strategy_retrievals.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:40]
        }

        # Keep union of top strategies from both
        keys_to_keep = top_compressions | top_retrievals

        # Truncate both dicts to same keys
        pattern.strategy_compressions = {
            k: v for k, v in pattern.strategy_compressions.items() if k in keys_to_keep
        }
        pattern.strategy_retrievals = {
            k: v for k, v in pattern.strategy_retrievals.items() if k in keys_to_keep
        }

    def _extract_field_hints(self, pattern: LocalToolPattern, query: str) -> None:
        """Extract potential field names from search queries.

        Common patterns:
        - "field:value" or "field=value"
        - JSON field names like "status", "error", "id"
        """
        # Look for field:value patterns
        field_patterns = re.findall(r"(\w+)[=:]", query)
        for field_name in field_patterns:
            pattern.queried_fields[field_name] = pattern.queried_fields.get(field_name, 0) + 1

        # Look for common JSON field names
        common_fields = [
            "id",
            "name",
            "status",
            "error",
            "message",
            "type",
            "code",
            "result",
            "value",
            "data",
            "items",
            "count",
        ]
        query_lower = query.lower()
        for common_field in common_fields:
            if common_field in query_lower:
                pattern.queried_fields[common_field] = (
                    pattern.queried_fields.get(common_field, 0) + 1
                )

        # HIGH: Limit queried_fields dict to prevent unbounded growth
        if len(pattern.queried_fields) > 50:
            sorted_fields = sorted(
                pattern.queried_fields.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:50]
            pattern.queried_fields = dict(sorted_fields)

    def get_compression_hints(
        self,
        tool_name: str | None,
    ) -> CompressionHints:
        """Get compression hints for a specific tool based on learned patterns.

        Args:
            tool_name: Name of the tool to get hints for.

        Returns:
            CompressionHints with recommended settings.
        """
        hints = CompressionHints()

        if not tool_name:
            hints.reason = "No tool name provided, using defaults"
            return hints

        with self._lock:
            pattern = self._tool_patterns.get(tool_name)

            if pattern is None:
                hints.reason = f"No pattern data for {tool_name}, using defaults"
                return hints

            # Need minimum samples for reliable hints
            if pattern.total_compressions < self.MIN_SAMPLES_FOR_HINTS:
                hints.reason = (
                    f"Insufficient data ({pattern.total_compressions} samples), "
                    f"need {self.MIN_SAMPLES_FOR_HINTS}"
                )
                return hints

            # Calculate hints based on retrieval rate
            retrieval_rate = pattern.retrieval_rate

            if retrieval_rate > self.HIGH_RETRIEVAL_THRESHOLD:
                # High retrieval = compress less aggressively
                if pattern.full_retrieval_rate > 0.8:
                    # Almost all retrievals are full → skip compression
                    hints.skip_compression = True
                    hints.reason = (
                        f"Very high full retrieval rate ({pattern.full_retrieval_rate:.0%}), "
                        f"recommending skip compression"
                    )
                else:
                    # Mix of full and search → increase items
                    hints.max_items = 50
                    hints.suggested_items = 40
                    hints.aggressiveness = 0.3
                    hints.reason = (
                        f"High retrieval rate ({retrieval_rate:.0%}), "
                        f"recommending less aggressive compression"
                    )

            elif retrieval_rate > self.MEDIUM_RETRIEVAL_THRESHOLD:
                # Medium retrieval = slightly less aggressive
                hints.max_items = 30
                hints.suggested_items = 25
                hints.aggressiveness = 0.5
                hints.reason = (
                    f"Medium retrieval rate ({retrieval_rate:.0%}), "
                    f"recommending moderate compression"
                )

            else:
                # Low retrieval = current compression is working
                hints.max_items = 15
                hints.suggested_items = 10
                hints.aggressiveness = 0.7
                hints.reason = (
                    f"Low retrieval rate ({retrieval_rate:.0%}), current compression is effective"
                )

            # Add field preservation hints based on common queries
            if pattern.queried_fields:
                # Get top 5 most queried fields
                sorted_fields = sorted(
                    pattern.queried_fields.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
                hints.preserve_fields = [f for f, _ in sorted_fields]

            # Recommend the best strategy based on local retrieval patterns
            best = pattern.best_strategy()
            if best:
                hints.recommended_strategy = best

            return hints

    def get_all_patterns(self) -> dict[str, LocalToolPattern]:
        """Get all learned tool patterns.

        Returns:
            Dict mapping tool names to their patterns.
            HIGH FIX: Returns deep copies to prevent external mutation of internal state.
        """
        import copy as copy_module

        with self._lock:
            # Deep copy to prevent external code from modifying internal state
            return copy_module.deepcopy(self._tool_patterns)

    def get_stats(self) -> dict[str, Any]:
        """Get feedback statistics for monitoring.

        Returns:
            Dict with feedback statistics.
        """
        with self._lock:
            return {
                "total_compressions": self._total_compressions,
                "total_retrievals": self._total_retrievals,
                "global_retrieval_rate": (
                    self._total_retrievals / self._total_compressions
                    if self._total_compressions > 0
                    else 0.0
                ),
                "tools_tracked": len(self._tool_patterns),
                "tool_patterns": {
                    name: {
                        "compressions": p.total_compressions,
                        "retrievals": p.total_retrievals,
                        "retrieval_rate": p.retrieval_rate,
                        "full_rate": p.full_retrieval_rate,
                        "search_rate": p.search_rate,
                        "common_queries": list(p.common_queries.keys())[:5],
                        "queried_fields": list(p.queried_fields.keys())[:5],
                    }
                    for name, p in self._tool_patterns.items()
                },
            }

    def analyze_from_store(self) -> None:
        """Analyze retrieval events from the store.

        This pulls recent events from CompressionStore and updates patterns.
        Useful for catching up after restart or periodic refresh.

        HIGH FIX: All timestamp reads/writes happen under lock to prevent race
        conditions where another thread could cause events to be missed or
        double-counted.
        """
        if not self._enable_learning:
            return

        # Rate limit analysis - check under lock for thread safety
        now = time.time()
        with self._lock:
            if now - self._last_analysis < self._analysis_interval:
                return
            # Mark that we're starting analysis (prevents concurrent analysis)
            self._last_analysis = now
            last_ts = self._last_event_timestamp

        # Fetch events outside lock (store has its own lock)
        events = self.store.get_retrieval_events(limit=1000)

        # Filter events to only process new ones (avoid double-counting)
        new_events = [e for e in events if e.timestamp > last_ts]

        if new_events:
            # Find the maximum timestamp from new events
            max_timestamp = max(e.timestamp for e in new_events)

            for event in new_events:
                self.record_retrieval(event)

            # Update the timestamp AFTER processing - under lock for atomicity
            with self._lock:
                # Only update if our max_timestamp is greater than current
                # (another thread may have processed newer events)
                if max_timestamp > self._last_event_timestamp:
                    self._last_event_timestamp = max_timestamp

    def clear(self) -> None:
        """Clear all learned patterns. Mainly for testing."""
        with self._lock:
            self._tool_patterns.clear()
            self._total_compressions = 0
            self._total_retrievals = 0
            self._last_analysis = 0.0
            self._last_event_timestamp = 0.0


# Global feedback instance (lazy initialization)
_compression_feedback: CompressionFeedback | None = None
_feedback_lock = threading.Lock()


def get_compression_feedback() -> CompressionFeedback:
    """Get the global compression feedback instance.

    Returns:
        Global CompressionFeedback instance.
    """
    global _compression_feedback

    if _compression_feedback is None:
        with _feedback_lock:
            if _compression_feedback is None:
                _compression_feedback = CompressionFeedback()

    return _compression_feedback


def reset_compression_feedback() -> None:
    """Reset the global compression feedback. Mainly for testing."""
    global _compression_feedback

    with _feedback_lock:
        if _compression_feedback is not None:
            _compression_feedback.clear()
        _compression_feedback = None
