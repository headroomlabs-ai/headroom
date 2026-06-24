"""Thread-safe LRU cache for hot memories in Headroom Memory.

Provides O(1) get/set operations with configurable size limits
and automatic eviction of least-recently-used entries.

Implements the MemoryCache protocol with async methods.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Lock
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Memory


class LRUMemoryCache:
    """Thread-safe LRU (Least Recently Used) cache for Memory objects.

    Implements the MemoryCache protocol with async methods that wrap
    synchronous operations.

    Features:
    - O(1) get and set operations using OrderedDict
    - Automatic eviction of least-recently-used entries when at capacity
    - Thread-safe with Lock for concurrent access
    - Move-to-end on access to maintain LRU ordering
    - Batch operations for efficiency

    Usage:
        cache = LRUMemoryCache(max_size=1000)
        await cache.put(memory_obj)
        memory = await cache.get("mem-123")  # Returns Memory or None

    The cache uses an OrderedDict internally where:
    - Most recently used items are at the end
    - Least recently used items are at the beginning
    - On capacity overflow, the first (oldest) item is evicted
    """

    def __init__(self, max_size: int = 1000) -> None:
        """Initialize the LRU cache.

        Args:
            max_size: Maximum number of entries to store. When exceeded,
                      the least recently used entry is evicted.

        Raises:
            ValueError: If max_size is less than 1.
        """
        if max_size < 1:
            raise ValueError(f"max_size must be at least 1, got {max_size}")

        self._max_size = max_size
        self._cache: OrderedDict[str, Memory] = OrderedDict()
        self._lock = Lock()

    async def get(self, memory_id: str) -> Memory | None:
        """Get a memory from the cache.

        Moves the accessed item to the end (most recently used position).

        Args:
            memory_id: The memory ID to retrieve.

        Returns:
            The Memory object if found, None otherwise.
        """
        with self._lock:
            if memory_id not in self._cache:
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(memory_id)
            return self._cache[memory_id]

    async def get_batch(self, memory_ids: list[str]) -> dict[str, Memory]:
        """Get multiple memories from the cache.

        Moves all accessed items to the end in the order they were requested.

        Args:
            memory_ids: List of memory IDs to retrieve.

        Returns:
            Dict mapping memory IDs to Memory objects for all found in cache.
            IDs not in cache are omitted from the result.
        """
        with self._lock:
            result: dict[str, Memory] = {}
            for memory_id in memory_ids:
                if memory_id in self._cache:
                    # Move to end (most recently used)
                    self._cache.move_to_end(memory_id)
                    result[memory_id] = self._cache[memory_id]
            return result

    async def put(
        self,
        memory: Memory,
        ttl_seconds: int | None = None,
    ) -> None:
        """Put a memory in the cache.

        If the memory already exists, updates the value and moves to end.
        If at capacity, evicts the least recently used entry first.

        Args:
            memory: The Memory object to cache.
            ttl_seconds: Time-to-live in seconds. Currently ignored in this
                         basic LRU implementation (reserved for future use).
        """
        # Note: ttl_seconds is accepted but ignored in this basic LRU implementation.
        # A TTL-aware version would need a background cleanup thread or lazy expiration.
        _ = ttl_seconds  # Explicitly ignore

        with self._lock:
            key = memory.id
            if key in self._cache:
                # Update existing entry and move to end
                self._cache[key] = memory
                self._cache.move_to_end(key)
            else:
                # Add new entry
                self._cache[key] = memory

                # Evict oldest if at capacity
                while len(self._cache) > self._max_size:
                    # popitem(last=False) removes the first (oldest) item
                    self._cache.popitem(last=False)

    async def put_batch(
        self,
        memories: list[Memory],
        ttl_seconds: int | None = None,
    ) -> None:
        """Put multiple memories in the cache.

        Args:
            memories: List of Memory objects to cache.
            ttl_seconds: Time-to-live in seconds. Currently ignored in this
                         basic LRU implementation (reserved for future use).
        """
        # Note: ttl_seconds is accepted but ignored in this basic LRU implementation.
        _ = ttl_seconds  # Explicitly ignore

        with self._lock:
            for memory in memories:
                key = memory.id
                if key in self._cache:
                    self._cache[key] = memory
                    self._cache.move_to_end(key)
                else:
                    self._cache[key] = memory

            # Evict oldest entries if over capacity
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    async def invalidate(self, memory_id: str) -> bool:
        """Invalidate (remove) a memory from cache.

        Args:
            memory_id: The memory ID to remove.

        Returns:
            True if the memory was in cache, False otherwise.
        """
        with self._lock:
            if memory_id in self._cache:
                del self._cache[memory_id]
                return True
            return False

    async def invalidate_batch(self, memory_ids: list[str]) -> int:
        """Invalidate multiple memories from cache.

        Args:
            memory_ids: List of memory IDs to invalidate.

        Returns:
            Number of memories that were in cache.
        """
        with self._lock:
            count = 0
            for memory_id in memory_ids:
                if memory_id in self._cache:
                    del self._cache[memory_id]
                    count += 1
            return count

    async def invalidate_scope(
        self,
        user_id: str,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> int:
        """Invalidate all cached memories at or below a scope.

        Args:
            user_id: Required user scope.
            session_id: If provided, invalidate session and below.
            agent_id: If provided, invalidate agent and below.

        Returns:
            Number of memories invalidated.
        """
        with self._lock:
            # Find all matching memory IDs
            to_remove = []
            for memory_id, memory in self._cache.items():
                if memory.user_id != user_id:
                    continue
                if session_id is not None and memory.session_id != session_id:
                    continue
                if agent_id is not None and memory.agent_id != agent_id:
                    continue
                to_remove.append(memory_id)

            # Remove them
            for memory_id in to_remove:
                del self._cache[memory_id]

            return len(to_remove)

    async def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        """Get the current number of entries in the cache.

        Returns:
            Number of entries currently stored.
        """
        with self._lock:
            return len(self._cache)

    @property
    def max_size(self) -> int | None:
        """Get the maximum cache size.

        Returns:
            Maximum number of entries allowed.
        """
        return self._max_size

    def contains(self, memory_id: str) -> bool:
        """Check if a memory exists in the cache without affecting LRU order.

        Args:
            memory_id: The memory ID to check.

        Returns:
            True if the memory exists, False otherwise.
        """
        with self._lock:
            return memory_id in self._cache

    def keys(self) -> list[str]:
        """Get all keys in the cache.

        Returns:
            List of keys in LRU order (oldest first, newest last).
        """
        with self._lock:
            return list(self._cache.keys())

    def stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dict with size, max_size, and utilization percentage.
        """
        with self._lock:
            current_size = len(self._cache)
            return {
                "size": current_size,
                "max_size": self._max_size,
                "utilization": (current_size / self._max_size) * 100 if self._max_size > 0 else 0.0,
            }
