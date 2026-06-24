"""
Semantic Cache Layer.

Provides query-level semantic caching using embedding similarity.
This is COMPLEMENTARY to provider prompt caching - it caches complete
responses for semantically similar queries.

How it works:
1. When a query comes in, compute its embedding
2. Search for similar queries in the cache (cosine similarity)
3. If similarity > threshold, return cached response
4. Otherwise, proceed with normal optimization

Key difference from Prompt Caching:
- Prompt Caching: Provider caches KV-cache for prefix (same prompt = faster)
- Semantic Caching: We cache responses for similar queries (similar query = cached answer)

Usage:
    from headroom.cache import SemanticCacheLayer, CacheOptimizerRegistry

    # Get provider optimizer
    provider_optimizer = CacheOptimizerRegistry.get("anthropic")

    # Wrap with semantic layer
    semantic = SemanticCacheLayer(
        provider_optimizer,
        similarity_threshold=0.95,
    )

    result = semantic.process(messages, context)
    if result.semantic_cache_hit:
        # Use result.cached_response directly
        pass
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from headroom.models.config import ML_MODEL_DEFAULTS

from .base import (
    BaseCacheOptimizer,
    CacheConfig,
    CacheMetrics,
    CacheResult,
    OptimizationContext,
)


@dataclass
class CacheEntry:
    """Entry in the semantic cache."""

    # Query embedding
    embedding: list[float]

    # Original query text
    query: str

    # Cached response
    response: Any

    # Metadata
    created_at: float
    last_accessed: float
    access_count: int = 1

    # Hash of the full messages for exact matching
    messages_hash: str = ""


@dataclass
class SemanticCacheConfig:
    """Configuration for semantic caching."""

    # Similarity threshold for cache hit (0.0 - 1.0)
    similarity_threshold: float = 0.95

    # Maximum entries in cache
    max_entries: int = 1000

    # TTL in seconds (0 = no expiry)
    ttl_seconds: int = 300

    # Whether to use exact hash matching as fallback
    use_exact_matching: bool = True

    # Embedding model (if using embeddings)
    embedding_model: str = field(default_factory=lambda: ML_MODEL_DEFAULTS.sentence_transformer)


class SemanticCache:
    """
    In-memory semantic cache with LRU eviction.

    Stores query embeddings and responses, supporting both
    semantic similarity search and exact hash matching.
    """

    def __init__(
        self,
        config: SemanticCacheConfig | None = None,
        embedding_fn: Callable[[str], list[float]] | None = None,
    ):
        """
        Initialize the semantic cache.

        Args:
            config: Cache configuration
            embedding_fn: Optional custom embedding function.
                         If not provided, uses simple hash-based matching.
        """
        self.config = config or SemanticCacheConfig()
        self._embedding_fn = embedding_fn

        # LRU cache: key -> CacheEntry
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Exact hash index: messages_hash -> key
        self._hash_index: dict[str, str] = {}

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(
        self,
        query: str,
        messages_hash: str | None = None,
    ) -> CacheEntry | None:
        """
        Look up a cached entry.

        Args:
            query: Query text to search for
            messages_hash: Optional exact hash for fast lookup

        Returns:
            CacheEntry if found, None otherwise
        """
        self._cleanup_expired()

        # Try exact hash match first
        if messages_hash and self.config.use_exact_matching:
            key = self._hash_index.get(messages_hash)
            if key and key in self._cache:
                entry = self._cache[key]
                self._touch(key)
                self._hits += 1
                return entry

        # Try semantic similarity if we have embedding function
        if self._embedding_fn:
            query_embedding = self._embedding_fn(query)
            best_match, best_similarity = self._find_similar(query_embedding)

            if best_similarity >= self.config.similarity_threshold:
                self._touch(best_match)
                self._hits += 1
                return self._cache[best_match]

        self._misses += 1
        return None

    def put(
        self,
        query: str,
        response: Any,
        messages_hash: str | None = None,
    ) -> str:
        """
        Store a response in the cache.

        Args:
            query: Query text
            response: Response to cache
            messages_hash: Optional exact hash for fast lookup

        Returns:
            Cache key for the entry
        """
        self._cleanup_expired()

        # Evict if at capacity
        while len(self._cache) >= self.config.max_entries:
            self._evict_oldest()

        # Generate embedding if available
        embedding: list[float] = []
        if self._embedding_fn:
            embedding = self._embedding_fn(query)

        # Create cache key
        key = self._generate_key(query)

        now = time.time()
        entry = CacheEntry(
            embedding=embedding,
            query=query,
            response=response,
            created_at=now,
            last_accessed=now,
            messages_hash=messages_hash or "",
        )

        self._cache[key] = entry

        # Index by hash for fast exact matching
        if messages_hash:
            self._hash_index[messages_hash] = key

        return key

    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry by key."""
        if key in self._cache:
            entry = self._cache.pop(key)
            if entry.messages_hash:
                self._hash_index.pop(entry.messages_hash, None)
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._hash_index.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0

        return {
            "entries": len(self._cache),
            "max_entries": self.config.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "evictions": self._evictions,
        }

    def _find_similar(
        self,
        query_embedding: list[float],
    ) -> tuple[str, float]:
        """Find the most similar cached entry."""
        best_key = ""
        best_similarity = -1.0

        for key, entry in self._cache.items():
            if not entry.embedding:
                continue

            similarity = self._cosine_similarity(query_embedding, entry.embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_key = key

        return best_key, best_similarity

    def _cosine_similarity(
        self,
        a: list[float],
        b: list[float],
    ) -> float:
        """Compute cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def _touch(self, key: str) -> None:
        """Update access time and move to end of LRU."""
        try:
            entry = self._cache.pop(key)
        except KeyError:
            return
        entry.last_accessed = time.time()
        entry.access_count += 1
        self._cache[key] = entry

    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) entry."""
        if self._cache:
            key, entry = self._cache.popitem(last=False)
            if entry.messages_hash:
                self._hash_index.pop(entry.messages_hash, None)
            self._evictions += 1

    def _cleanup_expired(self) -> None:
        """Remove expired entries."""
        if self.config.ttl_seconds <= 0:
            return

        now = time.time()
        expired = [
            key
            for key, entry in self._cache.items()
            if now - entry.created_at > self.config.ttl_seconds
        ]

        for key in expired:
            entry = self._cache.pop(key)
            if entry.messages_hash:
                self._hash_index.pop(entry.messages_hash, None)

    def _generate_key(self, query: str) -> str:
        """Generate a cache key for a query."""
        return hashlib.sha256(query.encode()).hexdigest()[:16]


class SemanticCacheLayer:
    """
    Layer that adds semantic caching on top of provider optimizers.

    This layer checks for semantically similar queries before
    delegating to the underlying provider optimizer.
    """

    def __init__(
        self,
        provider_optimizer: BaseCacheOptimizer,
        similarity_threshold: float = 0.95,
        max_entries: int = 1000,
        ttl_seconds: int = 300,
        embedding_fn: Callable[[str], list[float]] | None = None,
    ):
        """
        Initialize the semantic cache layer.

        Args:
            provider_optimizer: Underlying provider optimizer
            similarity_threshold: Similarity threshold for cache hits
            max_entries: Maximum cache entries
            ttl_seconds: Cache TTL in seconds
            embedding_fn: Optional embedding function
        """
        self.provider_optimizer = provider_optimizer

        cache_config = SemanticCacheConfig(
            similarity_threshold=similarity_threshold,
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
        )
        self.cache = SemanticCache(cache_config, embedding_fn)

    def process(
        self,
        messages: list[dict[str, Any]],
        context: OptimizationContext,
        config: CacheConfig | None = None,
    ) -> CacheResult:
        """
        Process messages through semantic cache and provider optimizer.

        Args:
            messages: Messages to process
            context: Optimization context
            config: Optional configuration override

        Returns:
            CacheResult with semantic_cache_hit=True if cache hit
        """
        # Extract query for semantic matching
        query = context.query or self._extract_query(messages)
        messages_hash = self._compute_messages_hash(messages)

        # Check semantic cache
        cached = self.cache.get(query, messages_hash)
        if cached:
            return CacheResult(
                messages=messages,
                semantic_cache_hit=True,
                cached_response=cached.response,
                metrics=CacheMetrics(
                    estimated_cache_hit=True,
                    estimated_savings_percent=100.0,
                ),
                transforms_applied=["semantic_cache_hit"],
            )

        # Delegate to provider optimizer
        result = self.provider_optimizer.optimize(messages, context, config)

        return result

    def store_response(
        self,
        messages: list[dict[str, Any]],
        response: Any,
        context: OptimizationContext | None = None,
    ) -> str:
        """
        Store a response in the semantic cache.

        Call this after receiving a response from the LLM to enable
        future cache hits.

        Args:
            messages: Original messages
            response: Response from LLM
            context: Optional context with query

        Returns:
            Cache key
        """
        query = (context.query if context else None) or self._extract_query(messages)
        messages_hash = self._compute_messages_hash(messages)

        return self.cache.put(query, response, messages_hash)

    def get_stats(self) -> dict[str, Any]:
        """Get combined statistics."""
        return {
            "semantic_cache": self.cache.get_stats(),
            "provider_optimizer": self.provider_optimizer.name,
        }

    def _extract_query(self, messages: list[dict[str, Any]]) -> str:
        """Extract the last user query from messages."""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_val = block.get("text", "")
                            return str(text_val) if text_val else ""
        return ""

    def _compute_messages_hash(self, messages: list[dict[str, Any]]) -> str:
        """Compute a hash of all messages."""
        import json

        try:
            content = json.dumps(messages, sort_keys=True)
            return hashlib.sha256(content.encode()).hexdigest()[:24]
        except (TypeError, ValueError):
            return ""
