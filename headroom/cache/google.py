"""
Google Cache Optimizer for CachedContent API.

Google's Gemini API offers explicit cached content management through
the `genai.caching.CachedContent` API. Key characteristics:

- Minimum 32K tokens required for caching
- 75% discount on cached input tokens
- Storage costs (pay per hour for cached content)
- User-defined TTL (default 1 hour)
- Returns cache_id for subsequent requests

This optimizer provides cache lifecycle management utilities without
making actual API calls - users integrate with the google-generativeai
package themselves.

Usage:
    optimizer = GoogleCacheOptimizer()

    # Check if content is cacheable
    analysis = optimizer.analyze_cacheability(messages, context)

    # Optimize and get cache recommendation
    result = optimizer.optimize(messages, context)

    # After user creates cache via Google API, register it
    optimizer.register_cache(
        cache_id="cached-content-xyz",
        content_hash=result.metrics.stable_prefix_hash,
        token_count=50000,
        expires_at=datetime.now() + timedelta(hours=1),
    )

    # Check if existing cache can be reused
    cache_info = optimizer.get_reusable_cache(content_hash)

    # Extend cache TTL
    optimizer.extend_cache_ttl(cache_id, additional_seconds=3600)

    # Clean up expired caches
    optimizer.cleanup_expired_caches()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .base import (
    BaseCacheOptimizer,
    CacheConfig,
    CacheMetrics,
    CacheResult,
    CacheStrategy,
    OptimizationContext,
)

logger = logging.getLogger(__name__)


# Google-specific constants
GOOGLE_MIN_CACHE_TOKENS = 32_768  # 32K tokens minimum
GOOGLE_CACHE_DISCOUNT = 0.75  # 75% discount on cached tokens
GOOGLE_DEFAULT_TTL_SECONDS = 3600  # 1 hour default
GOOGLE_MAX_TTL_SECONDS = 86400 * 7  # 7 days maximum


@dataclass
class CachedContentInfo:
    """
    Information about a cached content object.

    Tracks the lifecycle of a Google CachedContent resource.
    """

    # Google's cache identifier
    cache_id: str

    # Hash of the content for matching
    content_hash: str

    # Timestamps
    created_at: datetime
    expires_at: datetime

    # Token count in the cached content
    token_count: int

    # Optional model used (some caches are model-specific)
    model: str | None = None

    # Display name for the cached content
    display_name: str | None = None

    # Metadata for tracking
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if cache has expired."""
        return datetime.now() >= self.expires_at

    @property
    def ttl_remaining_seconds(self) -> int:
        """Seconds remaining until expiry."""
        remaining = (self.expires_at - datetime.now()).total_seconds()
        return max(0, int(remaining))

    @property
    def age_seconds(self) -> int:
        """Age of the cache in seconds."""
        return int((datetime.now() - self.created_at).total_seconds())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "cache_id": self.cache_id,
            "content_hash": self.content_hash,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "token_count": self.token_count,
            "model": self.model,
            "display_name": self.display_name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CachedContentInfo:
        """Deserialize from dictionary."""
        return cls(
            cache_id=data["cache_id"],
            content_hash=data["content_hash"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]),
            token_count=data["token_count"],
            model=data.get("model"),
            display_name=data.get("display_name"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CacheabilityAnalysis:
    """
    Analysis of whether content is suitable for Google caching.

    Provides detailed information about caching viability and
    potential savings.
    """

    # Whether content meets minimum threshold
    is_cacheable: bool

    # Token counts
    total_tokens: int
    cacheable_tokens: int

    # Shortfall if not cacheable
    tokens_below_minimum: int = 0

    # Estimated savings
    estimated_hourly_storage_cost_usd: float = 0.0
    estimated_savings_per_request_percent: float = 0.0

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    # Content hash for cache matching
    content_hash: str = ""


class GoogleCacheOptimizer(BaseCacheOptimizer):
    """
    Cache optimizer for Google's Gemini CachedContent API.

    This optimizer provides:
    1. Analysis of whether content meets Google's caching requirements
    2. Cache lifecycle management (register, lookup, extend, delete)
    3. Optimization recommendations
    4. Integration utilities for the google-generativeai SDK

    The optimizer does NOT make actual API calls - it provides the
    infrastructure for users to manage caches themselves.

    Example workflow:
        optimizer = GoogleCacheOptimizer()

        # Analyze content
        result = optimizer.optimize(messages, context)

        if result.metrics.cacheable_tokens >= GOOGLE_MIN_CACHE_TOKENS:
            # User creates cache via Google SDK
            cached_content = genai.caching.CachedContent.create(
                model="gemini-1.5-pro",
                contents=contents,
                ttl=timedelta(hours=1),
            )

            # Register with optimizer for tracking
            optimizer.register_cache(
                cache_id=cached_content.name,
                content_hash=result.metrics.stable_prefix_hash,
                token_count=result.metrics.cacheable_tokens,
                expires_at=datetime.now() + timedelta(hours=1),
            )

        # Later, check for reusable cache
        cache = optimizer.get_reusable_cache(content_hash)
        if cache:
            # Use cache.cache_id in API call
            pass
    """

    def __init__(self, config: CacheConfig | None = None):
        """
        Initialize Google cache optimizer.

        Args:
            config: Optional cache configuration
        """
        super().__init__(config)

        # Override minimum tokens for Google's requirements
        if self.config.min_cacheable_tokens < GOOGLE_MIN_CACHE_TOKENS:
            self.config.min_cacheable_tokens = GOOGLE_MIN_CACHE_TOKENS

        # Cache registry: content_hash -> CachedContentInfo
        self._cache_registry: dict[str, CachedContentInfo] = {}

        # Also index by cache_id for direct lookup
        self._cache_by_id: dict[str, CachedContentInfo] = {}

        # Statistics
        self._caches_created: int = 0
        self._caches_reused: int = 0
        self._caches_expired: int = 0

    @property
    def name(self) -> str:
        """Name of this optimizer."""
        return "google-cached-content"

    @property
    def provider(self) -> str:
        """Provider this optimizer is for."""
        return "google"

    @property
    def strategy(self) -> CacheStrategy:
        """The caching strategy this optimizer uses."""
        return CacheStrategy.CACHED_CONTENT

    def optimize(
        self,
        messages: list[dict[str, Any]],
        context: OptimizationContext,
        config: CacheConfig | None = None,
    ) -> CacheResult:
        """
        Optimize messages for Google caching.

        This method:
        1. Analyzes content for cacheability
        2. Checks for existing reusable caches
        3. Returns optimization metrics and recommendations

        Args:
            messages: The messages to optimize
            context: Optimization context
            config: Optional configuration override

        Returns:
            CacheResult with analysis and cache information
        """

        # Extract cacheable content (system messages + static context)
        cacheable_content = self._extract_cacheable_content(messages)
        content_hash = self._compute_prefix_hash(cacheable_content)

        # Estimate tokens
        total_tokens = self._count_tokens_estimate(self._messages_to_text(messages))
        cacheable_tokens = self._count_tokens_estimate(cacheable_content)

        # Check for existing cache
        existing_cache = self.get_reusable_cache(content_hash)

        # Build metrics
        metrics = CacheMetrics(
            stable_prefix_tokens=cacheable_tokens,
            stable_prefix_hash=content_hash,
            prefix_changed_from_previous=(
                context.previous_prefix_hash != content_hash
                if context.previous_prefix_hash
                else False
            ),
            previous_prefix_hash=context.previous_prefix_hash,
            cacheable_tokens=cacheable_tokens,
            non_cacheable_tokens=total_tokens - cacheable_tokens,
        )

        # Calculate estimated savings
        if cacheable_tokens >= GOOGLE_MIN_CACHE_TOKENS:
            metrics.estimated_savings_percent = GOOGLE_CACHE_DISCOUNT * 100
            metrics.estimated_cache_hit = existing_cache is not None

        # Add cache info if available
        if existing_cache:
            metrics.provider_cache_id = existing_cache.cache_id
            metrics.cache_ttl_remaining_seconds = existing_cache.ttl_remaining_seconds
            self._caches_reused += 1

        # Build warnings
        warnings: list[str] = []
        if cacheable_tokens < GOOGLE_MIN_CACHE_TOKENS:
            shortfall = GOOGLE_MIN_CACHE_TOKENS - cacheable_tokens
            warnings.append(
                f"Content has {cacheable_tokens:,} tokens, needs {shortfall:,} more "
                f"to meet Google's 32K minimum for caching"
            )

        if existing_cache and existing_cache.ttl_remaining_seconds < 300:
            warnings.append(
                f"Existing cache expires in {existing_cache.ttl_remaining_seconds}s - "
                f"consider extending TTL"
            )

        # Record metrics
        self._record_metrics(metrics)
        self._previous_prefix_hash = content_hash

        # Build transforms applied list
        transforms: list[str] = ["content_analysis"]
        if existing_cache:
            transforms.append("cache_lookup")

        return CacheResult(
            messages=messages,  # Messages unchanged - caching is separate
            semantic_cache_hit=False,
            metrics=metrics,
            tokens_before=total_tokens,
            tokens_after=total_tokens,  # Token count doesn't change
            transforms_applied=transforms,
            warnings=warnings,
        )

    def analyze_cacheability(
        self,
        messages: list[dict[str, Any]],
        context: OptimizationContext,
    ) -> CacheabilityAnalysis:
        """
        Analyze content for Google cache suitability.

        Provides detailed analysis including:
        - Whether content meets minimum requirements
        - Estimated costs and savings
        - Recommendations for improving cacheability

        Args:
            messages: Messages to analyze
            context: Optimization context

        Returns:
            CacheabilityAnalysis with detailed information
        """
        cacheable_content = self._extract_cacheable_content(messages)
        content_hash = self._compute_prefix_hash(cacheable_content)

        total_tokens = self._count_tokens_estimate(self._messages_to_text(messages))
        cacheable_tokens = self._count_tokens_estimate(cacheable_content)

        is_cacheable = cacheable_tokens >= GOOGLE_MIN_CACHE_TOKENS
        tokens_below_minimum = max(0, GOOGLE_MIN_CACHE_TOKENS - cacheable_tokens)

        # Build recommendations
        recommendations: list[str] = []

        if not is_cacheable:
            recommendations.append(
                f"Add {tokens_below_minimum:,} more tokens to static content to enable caching"
            )
            recommendations.append(
                "Consider adding detailed examples or documentation to system prompt"
            )
        else:
            recommendations.append(
                "Content is cacheable. Create cache with google-generativeai SDK"
            )

            # Storage cost estimation (rough - actual pricing varies)
            # Assuming ~$0.001 per 1000 tokens per hour (simplified)
            hourly_cost = (cacheable_tokens / 1000) * 0.001
            recommendations.append(f"Estimated storage cost: ~${hourly_cost:.4f}/hour")

            # Break-even analysis
            if hourly_cost > 0:
                # Assuming $0.01 per 1000 input tokens base price
                base_cost_per_request = (cacheable_tokens / 1000) * 0.01
                savings_per_request = base_cost_per_request * GOOGLE_CACHE_DISCOUNT
                break_even_requests = (
                    hourly_cost / savings_per_request if savings_per_request > 0 else float("inf")
                )
                recommendations.append(f"Break-even: ~{int(break_even_requests)} requests/hour")

        return CacheabilityAnalysis(
            is_cacheable=is_cacheable,
            total_tokens=total_tokens,
            cacheable_tokens=cacheable_tokens,
            tokens_below_minimum=tokens_below_minimum,
            estimated_savings_per_request_percent=(
                GOOGLE_CACHE_DISCOUNT * 100 if is_cacheable else 0.0
            ),
            recommendations=recommendations,
            content_hash=content_hash,
        )

    # -------------------------------------------------------------------------
    # Cache Registry Management
    # -------------------------------------------------------------------------

    def register_cache(
        self,
        cache_id: str,
        content_hash: str,
        token_count: int,
        expires_at: datetime,
        *,
        model: str | None = None,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CachedContentInfo:
        """
        Register a cache after creating it via Google's API.

        Call this after successfully creating a CachedContent resource
        to enable cache reuse detection.

        Args:
            cache_id: Google's cache identifier (e.g., "cachedContents/xyz")
            content_hash: Hash of cached content (from optimize() metrics)
            token_count: Number of tokens in cached content
            expires_at: When the cache expires
            model: Optional model the cache was created for
            display_name: Optional display name
            metadata: Optional additional metadata

        Returns:
            CachedContentInfo for the registered cache

        Example:
            # After creating cache via Google SDK
            cached_content = genai.caching.CachedContent.create(...)

            info = optimizer.register_cache(
                cache_id=cached_content.name,
                content_hash=result.metrics.stable_prefix_hash,
                token_count=result.metrics.cacheable_tokens,
                expires_at=datetime.now() + timedelta(hours=1),
            )
        """
        # Remove any existing cache with same content hash
        old_cache = self._cache_registry.get(content_hash)
        if old_cache:
            self._cache_by_id.pop(old_cache.cache_id, None)
            logger.debug(
                f"Replacing existing cache for hash {content_hash}: "
                f"{old_cache.cache_id} -> {cache_id}"
            )

        cache_info = CachedContentInfo(
            cache_id=cache_id,
            content_hash=content_hash,
            created_at=datetime.now(),
            expires_at=expires_at,
            token_count=token_count,
            model=model,
            display_name=display_name,
            metadata=metadata or {},
        )

        self._cache_registry[content_hash] = cache_info
        self._cache_by_id[cache_id] = cache_info
        self._caches_created += 1

        logger.info(
            f"Registered cache {cache_id} with {token_count:,} tokens, "
            f"expires in {cache_info.ttl_remaining_seconds}s"
        )

        return cache_info

    def get_reusable_cache(
        self,
        content_hash: str,
        *,
        min_ttl_seconds: int = 60,
    ) -> CachedContentInfo | None:
        """
        Check if a reusable cache exists for the given content.

        Args:
            content_hash: Hash of the content to look up
            min_ttl_seconds: Minimum remaining TTL to consider reusable

        Returns:
            CachedContentInfo if reusable cache exists, None otherwise
        """
        cache_info = self._cache_registry.get(content_hash)

        if cache_info is None:
            return None

        if cache_info.is_expired:
            self._remove_cache_internal(content_hash)
            return None

        if cache_info.ttl_remaining_seconds < min_ttl_seconds:
            logger.debug(
                f"Cache {cache_info.cache_id} has insufficient TTL "
                f"({cache_info.ttl_remaining_seconds}s < {min_ttl_seconds}s)"
            )
            return None

        return cache_info

    def get_cache_by_id(self, cache_id: str) -> CachedContentInfo | None:
        """
        Look up cache information by cache ID.

        Args:
            cache_id: Google's cache identifier

        Returns:
            CachedContentInfo if found, None otherwise
        """
        return self._cache_by_id.get(cache_id)

    def extend_cache_ttl(
        self,
        cache_id: str,
        new_expires_at: datetime,
    ) -> CachedContentInfo | None:
        """
        Update the expiry time for a cache after extending via Google API.

        Call this after successfully calling update() on the CachedContent
        to extend its TTL.

        Args:
            cache_id: Google's cache identifier
            new_expires_at: New expiry time

        Returns:
            Updated CachedContentInfo or None if not found

        Example:
            # After extending via Google SDK
            cached_content.update(ttl=timedelta(hours=2))

            optimizer.extend_cache_ttl(
                cache_id=cached_content.name,
                new_expires_at=datetime.now() + timedelta(hours=2),
            )
        """
        cache_info = self._cache_by_id.get(cache_id)
        if cache_info is None:
            logger.warning(f"Cannot extend unknown cache: {cache_id}")
            return None

        old_expires = cache_info.expires_at
        cache_info.expires_at = new_expires_at

        logger.info(f"Extended cache {cache_id} TTL from {old_expires} to {new_expires_at}")

        return cache_info

    def remove_cache(self, cache_id: str) -> bool:
        """
        Remove a cache from the registry.

        Call this after deleting the cache via Google API.

        Args:
            cache_id: Google's cache identifier

        Returns:
            True if cache was removed, False if not found
        """
        cache_info = self._cache_by_id.get(cache_id)
        if cache_info is None:
            return False

        self._cache_by_id.pop(cache_id, None)
        self._cache_registry.pop(cache_info.content_hash, None)

        logger.info(f"Removed cache {cache_id} from registry")
        return True

    def _remove_cache_internal(self, content_hash: str) -> None:
        """Remove cache by content hash (internal use)."""
        cache_info = self._cache_registry.pop(content_hash, None)
        if cache_info:
            self._cache_by_id.pop(cache_info.cache_id, None)
            self._caches_expired += 1

    def cleanup_expired_caches(self) -> list[str]:
        """
        Remove all expired caches from the registry.

        Returns:
            List of removed cache IDs (for user to delete via Google API)

        Example:
            expired_ids = optimizer.cleanup_expired_caches()
            for cache_id in expired_ids:
                # User deletes via Google SDK
                genai.caching.CachedContent.get(cache_id).delete()
        """
        expired_ids: list[str] = []

        # Find expired caches
        for content_hash, cache_info in list(self._cache_registry.items()):
            if cache_info.is_expired:
                expired_ids.append(cache_info.cache_id)
                self._remove_cache_internal(content_hash)

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired caches")

        return expired_ids

    def list_caches(
        self,
        *,
        include_expired: bool = False,
    ) -> list[CachedContentInfo]:
        """
        List all registered caches.

        Args:
            include_expired: Whether to include expired caches

        Returns:
            List of CachedContentInfo objects
        """
        caches = list(self._cache_registry.values())

        if not include_expired:
            caches = [c for c in caches if not c.is_expired]

        # Sort by expiry time
        caches.sort(key=lambda c: c.expires_at)

        return caches

    def get_statistics(self) -> dict[str, Any]:
        """
        Get cache usage statistics.

        Returns:
            Dictionary with cache statistics
        """
        active_caches = [c for c in self._cache_registry.values() if not c.is_expired]
        total_cached_tokens = sum(c.token_count for c in active_caches)

        return {
            "active_caches": len(active_caches),
            "total_cached_tokens": total_cached_tokens,
            "caches_created": self._caches_created,
            "caches_reused": self._caches_reused,
            "caches_expired": self._caches_expired,
            "cache_hit_rate": (
                self._caches_reused / (self._caches_reused + self._caches_created)
                if (self._caches_reused + self._caches_created) > 0
                else 0.0
            ),
        }

    # -------------------------------------------------------------------------
    # Cache Creation Helpers
    # -------------------------------------------------------------------------

    def prepare_cache_creation(
        self,
        messages: list[dict[str, Any]],
        context: OptimizationContext,
        ttl_seconds: int = GOOGLE_DEFAULT_TTL_SECONDS,
    ) -> dict[str, Any] | None:
        """
        Prepare parameters for creating a Google cache.

        Returns a dictionary with suggested parameters for
        genai.caching.CachedContent.create().

        Args:
            messages: Messages to cache
            context: Optimization context
            ttl_seconds: Desired TTL in seconds

        Returns:
            Dictionary with cache creation parameters, or None if not cacheable

        Example:
            params = optimizer.prepare_cache_creation(messages, context)
            if params:
                cached_content = genai.caching.CachedContent.create(**params)
        """
        analysis = self.analyze_cacheability(messages, context)

        if not analysis.is_cacheable:
            logger.debug(
                f"Content not cacheable: {analysis.tokens_below_minimum} tokens below minimum"
            )
            return None

        cacheable_content = self._extract_cacheable_content(messages)

        return {
            "contents": cacheable_content,
            "ttl": timedelta(seconds=min(ttl_seconds, GOOGLE_MAX_TTL_SECONDS)),
            "display_name": f"headroom-cache-{analysis.content_hash[:8]}",
            "_headroom_metadata": {
                "content_hash": analysis.content_hash,
                "token_count": analysis.cacheable_tokens,
                "created_by": "headroom",
            },
        }

    def build_request_with_cache(
        self,
        messages: list[dict[str, Any]],
        cache_id: str,
    ) -> dict[str, Any]:
        """
        Build request parameters using an existing cache.

        Returns a dictionary suggesting how to structure the API call
        when using cached content.

        Args:
            messages: Full message list
            cache_id: Cache ID to use

        Returns:
            Dictionary with suggested request structure
        """
        # Extract only the non-cached (dynamic) content
        dynamic_messages = self._extract_dynamic_messages(messages)

        return {
            "cached_content": cache_id,
            "contents": dynamic_messages,
            "_headroom_note": (
                "Use cached_content parameter with GenerativeModel to leverage the cache"
            ),
        }

    # -------------------------------------------------------------------------
    # Content Extraction Helpers
    # -------------------------------------------------------------------------

    def _extract_cacheable_content(self, messages: list[dict[str, Any]]) -> str:
        """
        Extract content suitable for caching.

        Includes:
        - System messages
        - Static context (tools, examples)

        Excludes:
        - Recent conversation turns
        - Dynamic content (dates, user-specific data)
        """
        cacheable_parts: list[str] = []

        for msg in messages:
            role = msg.get("role", "")

            # System messages are always cacheable
            if role == "system":
                content = self._extract_message_content(msg)
                if content:
                    cacheable_parts.append(content)

            # First few user/assistant turns with examples might be cacheable
            # but we're conservative - only include system by default

        return "\n\n".join(cacheable_parts)

    def _extract_dynamic_messages(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Extract messages that should NOT be cached.

        These are the conversation turns after the cached prefix.
        """
        dynamic: list[dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") != "system":
                dynamic.append(msg)

        return dynamic

    def _extract_message_content(self, message: dict[str, Any]) -> str:
        """Extract text content from a message."""
        content = message.get("content", "")

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts)

        return ""

    def _messages_to_text(self, messages: list[dict[str, Any]]) -> str:
        """Convert all messages to text for token counting."""
        parts = []
        for msg in messages:
            content = self._extract_message_content(msg)
            if content:
                parts.append(f"{msg.get('role', 'unknown')}: {content}")
        return "\n\n".join(parts)

    # -------------------------------------------------------------------------
    # Serialization for Persistence
    # -------------------------------------------------------------------------

    def export_cache_registry(self) -> list[dict[str, Any]]:
        """
        Export cache registry for persistence.

        Returns:
            List of cache info dictionaries
        """
        return [info.to_dict() for info in self._cache_registry.values()]

    def import_cache_registry(
        self,
        cache_data: list[dict[str, Any]],
        *,
        skip_expired: bool = True,
    ) -> int:
        """
        Import caches from persisted data.

        Args:
            cache_data: List of cache info dictionaries
            skip_expired: Whether to skip already-expired caches

        Returns:
            Number of caches imported
        """
        imported = 0

        for data in cache_data:
            try:
                cache_info = CachedContentInfo.from_dict(data)

                if skip_expired and cache_info.is_expired:
                    continue

                self._cache_registry[cache_info.content_hash] = cache_info
                self._cache_by_id[cache_info.cache_id] = cache_info
                imported += 1

            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to import cache entry: {e}")
                continue

        logger.info(f"Imported {imported} caches from persisted data")
        return imported
