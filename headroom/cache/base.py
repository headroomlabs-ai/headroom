"""
Base types and interfaces for cache optimization.

This module defines the core abstractions that all cache optimizers implement.
The design allows for provider-specific implementations while maintaining a
consistent interface for users.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable


class CacheStrategy(Enum):
    """Cache optimization strategy."""

    # Just stabilize prefix (move dates, normalize whitespace)
    PREFIX_STABILIZATION = "prefix_stabilization"

    # Insert explicit cache breakpoints (Anthropic)
    EXPLICIT_BREAKPOINTS = "explicit_breakpoints"

    # Manage separate cached content objects (Google)
    CACHED_CONTENT = "cached_content"

    # No optimization possible (provider doesn't support caching)
    NONE = "none"


class BreakpointLocation(Enum):
    """Where to insert cache breakpoints."""

    AFTER_SYSTEM = "after_system"
    AFTER_TOOLS = "after_tools"
    AFTER_EXAMPLES = "after_examples"
    CUSTOM = "custom"


@dataclass
class CacheBreakpoint:
    """
    Represents a cache breakpoint location.

    For Anthropic, this maps to cache_control blocks.
    For other providers, this is informational.
    """

    # Message index where breakpoint should be inserted
    message_index: int

    # Location type
    location: BreakpointLocation

    # For content arrays, index within the content
    content_index: int | None = None

    # Token count at this breakpoint
    tokens_at_breakpoint: int = 0

    # Reason for this breakpoint
    reason: str = ""


@dataclass
class CacheConfig:
    """Configuration for cache optimization."""

    # Whether to optimize at all
    enabled: bool = True

    # Strategy to use (auto-detected if None)
    strategy: CacheStrategy | None = None

    # Minimum tokens before caching makes sense
    min_cacheable_tokens: int = 1024

    # Maximum number of breakpoints (Anthropic limit is 4)
    max_breakpoints: int = 4

    # Patterns to extract and move to dynamic section
    date_patterns: list[str] = field(
        default_factory=lambda: [
            r"Today is \w+ \d{1,2},? \d{4}\.?",
            r"Current date: \d{4}-\d{2}-\d{2}",
            r"The current time is .+\.",
        ]
    )

    # Whether to normalize whitespace
    normalize_whitespace: bool = True

    # Collapse multiple blank lines
    collapse_blank_lines: bool = True

    # Separator between static and dynamic content
    dynamic_separator: str = "\n\n---\n\n"

    # Dynamic content detection tiers (for OpenAI prefix stabilization)
    # - "regex": Fast pattern matching (~0ms) - always recommended
    # - "ner": Named Entity Recognition via spaCy (~5-10ms) - catches names, money, etc.
    # - "semantic": Embedding similarity (~20-50ms) - catches volatile patterns
    # Default is regex-only for speed. Add tiers for better detection at cost of latency.
    dynamic_detection_tiers: list[Literal["regex", "ner", "semantic"]] = field(
        default_factory=lambda: ["regex"]
    )

    # For semantic caching
    semantic_cache_enabled: bool = False
    semantic_similarity_threshold: float = 0.95
    semantic_cache_ttl_seconds: int = 300


@dataclass
class CacheMetrics:
    """Metrics about cache optimization."""

    # Prefix analysis
    stable_prefix_tokens: int = 0
    stable_prefix_hash: str = ""

    # Breakpoint info
    breakpoints_inserted: int = 0
    breakpoint_locations: list[CacheBreakpoint] = field(default_factory=list)

    # Cache hit estimation
    prefix_changed_from_previous: bool = False
    previous_prefix_hash: str | None = None
    estimated_cache_hit: bool = False

    # Savings estimation
    estimated_savings_percent: float = 0.0
    cacheable_tokens: int = 0
    non_cacheable_tokens: int = 0

    # Provider-specific
    provider_cache_id: str | None = None  # For Google's CachedContent
    cache_ttl_remaining_seconds: int | None = None


@dataclass
class OptimizationContext:
    """Context for optimization request."""

    # Request tracking
    request_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # Provider info
    provider: str = ""
    model: str = ""

    # Query for relevance (used by semantic cache)
    query: str | None = None

    # Previous request info (for cache hit detection)
    previous_prefix_hash: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheResult:
    """Result of cache optimization."""

    # Optimized messages
    messages: list[dict[str, Any]]

    # Whether this was a semantic cache hit
    semantic_cache_hit: bool = False

    # Cached response (if semantic cache hit)
    cached_response: Any | None = None

    # Optimization metrics
    metrics: CacheMetrics = field(default_factory=CacheMetrics)

    # Tokens before/after
    tokens_before: int = 0
    tokens_after: int = 0

    # Transforms applied
    transforms_applied: list[str] = field(default_factory=list)

    # Warnings
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class CacheOptimizer(Protocol):
    """
    Protocol for cache optimizers.

    All provider-specific optimizers must implement this interface.
    This allows for easy swapping of implementations and plugin registration.
    """

    @property
    def name(self) -> str:
        """Name of this optimizer."""
        ...

    @property
    def provider(self) -> str:
        """Provider this optimizer is for."""
        ...

    @property
    def strategy(self) -> CacheStrategy:
        """The caching strategy this optimizer uses."""
        ...

    def optimize(
        self,
        messages: list[dict[str, Any]],
        context: OptimizationContext,
        config: CacheConfig | None = None,
    ) -> CacheResult:
        """
        Optimize messages for caching.

        Args:
            messages: The messages to optimize.
            context: Optimization context with request info.
            config: Optional configuration override.

        Returns:
            CacheResult with optimized messages and metrics.
        """
        ...

    def get_metrics(self) -> CacheMetrics:
        """Get aggregated metrics from this optimizer."""
        ...

    def estimate_savings(
        self,
        messages: list[dict[str, Any]],
        context: OptimizationContext,
    ) -> float:
        """
        Estimate potential savings from optimization.

        Returns:
            Estimated savings as a percentage (0-100).
        """
        ...


class BaseCacheOptimizer(ABC):
    """
    Abstract base class for cache optimizers.

    Provides common functionality for all optimizers.
    """

    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()
        self._metrics_history: list[CacheMetrics] = []
        self._previous_prefix_hash: str | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this optimizer."""
        ...

    @property
    @abstractmethod
    def provider(self) -> str:
        """Provider this optimizer is for."""
        ...

    @property
    @abstractmethod
    def strategy(self) -> CacheStrategy:
        """The caching strategy this optimizer uses."""
        ...

    @abstractmethod
    def optimize(
        self,
        messages: list[dict[str, Any]],
        context: OptimizationContext,
        config: CacheConfig | None = None,
    ) -> CacheResult:
        """Optimize messages for caching."""
        ...

    def get_metrics(self) -> CacheMetrics:
        """Get aggregated metrics."""
        if not self._metrics_history:
            return CacheMetrics()

        # Return most recent metrics
        return self._metrics_history[-1]

    def estimate_savings(
        self,
        messages: list[dict[str, Any]],
        context: OptimizationContext,
    ) -> float:
        """Estimate potential savings."""
        # Default implementation - subclasses can override
        result = self.optimize(messages, context)
        return result.metrics.estimated_savings_percent

    def _record_metrics(self, metrics: CacheMetrics) -> None:
        """Record metrics for history."""
        self._metrics_history.append(metrics)
        # Keep only last 100 entries
        if len(self._metrics_history) > 100:
            self._metrics_history = self._metrics_history[-100:]

    def _compute_prefix_hash(self, content: str) -> str:
        """Compute a short hash of content."""
        import hashlib

        return hashlib.md5(content.encode()).hexdigest()[:12]  # nosec B324

    def _extract_system_content(self, messages: list[dict[str, Any]]) -> str:
        """Extract content from system messages."""
        parts = []
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, list):
                    # Handle content blocks
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
        return "\n".join(parts)

    def _count_tokens_estimate(self, text: str) -> int:
        """Rough token count estimate (4 chars per token)."""
        return len(text) // 4
