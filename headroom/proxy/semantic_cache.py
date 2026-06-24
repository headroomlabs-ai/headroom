"""Semantic cache for the Headroom proxy.

Simple semantic cache based on message content hash with LRU eviction.

Extracted from server.py for maintainability.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..memory.tracker import ComponentStats

from headroom.proxy.models import CacheEntry


class SemanticCache:
    """Simple semantic cache based on message content hash.

    Uses OrderedDict for O(1) LRU eviction instead of list with O(n) pop(0).
    """

    def __init__(self, max_entries: int = 1000, ttl_seconds: int = 3600):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        # OrderedDict maintains insertion order and supports O(1) move_to_end/popitem
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    def _compute_key(self, messages: list[dict], model: str) -> str:
        """Compute cache key from messages and model."""
        # Normalize messages for consistent hashing
        normalized = json.dumps(
            {
                "model": model,
                "messages": messages,
            },
            sort_keys=True,
        )
        return hashlib.sha256(normalized.encode()).hexdigest()[:32]

    async def get(self, messages: list[dict], model: str) -> CacheEntry | None:
        """Get cached response if exists and not expired."""
        key = self._compute_key(messages, model)
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                return None

            # Check expiration
            age = (datetime.now() - entry.created_at).total_seconds()
            if age > entry.ttl_seconds:
                del self._cache[key]
                return None

            entry.hit_count += 1
            # Move to end for LRU (O(1) operation)
            self._cache.move_to_end(key)
            return entry

    async def set(
        self,
        messages: list[dict],
        model: str,
        response_body: bytes,
        response_headers: dict[str, str],
        tokens_saved: int = 0,
    ):
        """Cache a response."""
        key = self._compute_key(messages, model)

        async with self._lock:
            # If key already exists, remove it first to update position
            if key in self._cache:
                del self._cache[key]

            # Evict oldest entries if at capacity (LRU) - O(1) with popitem
            while len(self._cache) >= self.max_entries:
                self._cache.popitem(last=False)  # Remove oldest (first) entry

            self._cache[key] = CacheEntry(
                response_body=response_body,
                response_headers=response_headers,
                created_at=datetime.now(),
                ttl_seconds=self.ttl_seconds,
                tokens_saved_per_hit=tokens_saved,
            )

    async def stats(self) -> dict:
        """Get cache statistics."""
        async with self._lock:
            total_hits = sum(e.hit_count for e in self._cache.values())
            return {
                "entries": len(self._cache),
                "max_entries": self.max_entries,
                "total_hits": total_hits,
                "ttl_seconds": self.ttl_seconds,
            }

    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()

    def get_memory_stats(self) -> ComponentStats:
        """Get memory statistics for the MemoryTracker.

        Returns:
            ComponentStats with current memory usage.
        """
        from ..memory.tracker import ComponentStats

        # Take a snapshot of cache values under the lock to avoid iterating
        # over a dict that may be mutated concurrently by async coroutines.
        # The lock is an asyncio.Lock and cannot be acquired in a sync method,
        # so we do a single atomic copy of the values view instead.
        snapshot = list(self._cache.values())
        entry_count = len(snapshot)

        size_bytes = sys.getsizeof(self._cache)
        total_hits = 0

        for entry in snapshot:
            size_bytes += sys.getsizeof(entry)
            size_bytes += len(entry.response_body)
            size_bytes += sys.getsizeof(entry.response_headers)
            for k, v in entry.response_headers.items():
                size_bytes += len(k) + len(v)
            total_hits += entry.hit_count

        return ComponentStats(
            name="semantic_cache",
            entry_count=entry_count,
            size_bytes=size_bytes,
            budget_bytes=None,
            hits=total_hits,
            misses=0,  # Would need to track this separately
            evictions=0,  # Would need to track this separately
        )
