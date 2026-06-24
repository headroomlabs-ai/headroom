"""Batch context storage for CCR post-processing.

When batches are submitted, we store the request context (messages, tools, model)
so that when results are retrieved, we can handle CCR tool calls and make
continuation API calls.

This module provides:
1. BatchContext: Data class for stored batch context
2. BatchContextStore: TTL-based cache for batch contexts
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..memory.tracker import ComponentStats

logger = logging.getLogger(__name__)

# Default TTL for batch contexts (24 hours - batches can take a while)
DEFAULT_BATCH_CONTEXT_TTL = 86400

# Maximum contexts to store (prevent memory issues)
MAX_BATCH_CONTEXTS = 10000


@dataclass
class BatchRequestContext:
    """Context for a single request within a batch."""

    custom_id: (
        str  # The request ID within the batch (custom_id for Anthropic, metadata.key for Google)
    )
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    model: str = ""
    system_instruction: str | None = None  # For Google format

    # Provider-specific extras
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchContext:
    """Context for an entire batch submission.

    Stores all request contexts so we can handle CCR tool calls
    when results are retrieved.
    """

    batch_id: str
    provider: str  # "anthropic", "openai", "google"
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0

    # Map of custom_id -> BatchRequestContext
    requests: dict[str, BatchRequestContext] = field(default_factory=dict)

    # API configuration for continuation calls
    api_key: str | None = None
    api_base_url: str | None = None

    def __post_init__(self) -> None:
        if self.expires_at == 0:
            self.expires_at = self.created_at + DEFAULT_BATCH_CONTEXT_TTL

    @property
    def is_expired(self) -> bool:
        """Check if this context has expired."""
        return time.time() > self.expires_at

    def add_request(self, request: BatchRequestContext) -> None:
        """Add a request context to this batch."""
        self.requests[request.custom_id] = request

    def get_request(self, custom_id: str) -> BatchRequestContext | None:
        """Get a request context by custom_id."""
        return self.requests.get(custom_id)


class BatchContextStore:
    """Thread-safe store for batch contexts.

    Stores batch submission contexts with TTL so that when results
    are retrieved, we can handle CCR tool calls.

    Features:
    - TTL-based expiration
    - Automatic cleanup of expired entries
    - Thread-safe operations
    - Memory limits

    Usage:
        store = BatchContextStore()

        # On batch submit
        context = BatchContext(batch_id="batch_123", provider="anthropic")
        for req in batch_requests:
            context.add_request(BatchRequestContext(
                custom_id=req["custom_id"],
                messages=req["params"]["messages"],
                tools=req["params"].get("tools"),
                model=req["params"]["model"],
            ))
        store.store(context)

        # On batch results retrieval
        context = store.get("batch_123")
        if context:
            for result in results:
                req_ctx = context.get_request(result["custom_id"])
                # ... handle CCR tool calls using req_ctx
    """

    def __init__(
        self,
        ttl: int = DEFAULT_BATCH_CONTEXT_TTL,
        max_contexts: int = MAX_BATCH_CONTEXTS,
    ) -> None:
        self._contexts: dict[str, BatchContext] = {}
        self._ttl = ttl
        self._max_contexts = max_contexts
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None

    async def store(self, context: BatchContext) -> None:
        """Store a batch context.

        Args:
            context: The batch context to store.
        """
        async with self._lock:
            # Enforce memory limit
            if len(self._contexts) >= self._max_contexts:
                # Remove oldest entries
                await self._cleanup_oldest()

            # Set expiration
            context.expires_at = time.time() + self._ttl
            self._contexts[context.batch_id] = context

            logger.debug(
                f"Stored batch context {context.batch_id} with "
                f"{len(context.requests)} requests (provider={context.provider})"
            )

    async def get(self, batch_id: str) -> BatchContext | None:
        """Get a batch context by ID.

        Args:
            batch_id: The batch ID to look up.

        Returns:
            The batch context, or None if not found or expired.
        """
        async with self._lock:
            context = self._contexts.get(batch_id)

            if context is None:
                return None

            if context.is_expired:
                del self._contexts[batch_id]
                logger.debug(f"Batch context {batch_id} expired and removed")
                return None

            return context

    async def remove(self, batch_id: str) -> bool:
        """Remove a batch context.

        Args:
            batch_id: The batch ID to remove.

        Returns:
            True if removed, False if not found.
        """
        async with self._lock:
            if batch_id in self._contexts:
                del self._contexts[batch_id]
                return True
            return False

    async def cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed.
        """
        async with self._lock:
            now = time.time()
            expired = [batch_id for batch_id, ctx in self._contexts.items() if ctx.expires_at < now]

            for batch_id in expired:
                del self._contexts[batch_id]

            if expired:
                logger.debug(f"Cleaned up {len(expired)} expired batch contexts")

            return len(expired)

    async def _cleanup_oldest(self) -> None:
        """Remove oldest entries to make room for new ones."""
        # Sort by creation time, remove oldest 10%
        if not self._contexts:
            return

        sorted_entries = sorted(
            self._contexts.items(),
            key=lambda x: x[1].created_at,
        )

        to_remove = max(1, len(sorted_entries) // 10)
        for batch_id, _ in sorted_entries[:to_remove]:
            del self._contexts[batch_id]

        logger.debug(f"Cleaned up {to_remove} oldest batch contexts")

    async def stats(self) -> dict[str, Any]:
        """Get store statistics.

        Thread-safe: acquires lock before accessing contexts dict to prevent
        RuntimeError from concurrent modification during iteration.
        """
        async with self._lock:
            return {
                "total_contexts": len(self._contexts),
                "max_contexts": self._max_contexts,
                "ttl_seconds": self._ttl,
                "providers": self._count_by_provider_locked(),
            }

    def _count_by_provider_locked(self) -> dict[str, int]:
        """Count contexts by provider. Must be called with lock held."""
        counts: dict[str, int] = {}
        for ctx in self._contexts.values():
            counts[ctx.provider] = counts.get(ctx.provider, 0) + 1
        return counts

    def get_memory_stats(self) -> ComponentStats:
        """Get memory statistics for the MemoryTracker.

        Thread-safe: takes a snapshot of contexts dict to prevent RuntimeError
        from concurrent modification during iteration. Dict copy is atomic in CPython.

        Returns:
            ComponentStats with current memory usage.
        """
        import sys

        from ..memory.tracker import ComponentStats

        # Take atomic snapshot to prevent RuntimeError during iteration
        # dict.copy() is atomic in CPython due to GIL
        contexts_snapshot = self._contexts.copy()

        # Calculate size
        size_bytes = sys.getsizeof(self._contexts)

        for batch_id, ctx in contexts_snapshot.items():
            size_bytes += len(batch_id)
            size_bytes += sys.getsizeof(ctx)

            # Add request contexts
            for req_id, req in ctx.requests.items():
                size_bytes += len(req_id)
                size_bytes += sys.getsizeof(req)
                # Messages can be large
                size_bytes += sys.getsizeof(req.messages)
                for msg in req.messages:
                    size_bytes += sys.getsizeof(msg)
                    for _k, v in msg.items():
                        if isinstance(v, str):
                            size_bytes += len(v)
                        elif isinstance(v, list):
                            size_bytes += sys.getsizeof(v)

                # Tools
                if req.tools:
                    size_bytes += sys.getsizeof(req.tools)

        return ComponentStats(
            name="batch_context_store",
            entry_count=len(self._contexts),
            size_bytes=size_bytes,
            budget_bytes=None,
            hits=0,
            misses=0,
            evictions=0,
        )


# Global store instance
_batch_context_store: BatchContextStore | None = None


def get_batch_context_store() -> BatchContextStore:
    """Get the global batch context store instance."""
    global _batch_context_store
    if _batch_context_store is None:
        _batch_context_store = BatchContextStore()
    return _batch_context_store


def reset_batch_context_store() -> None:
    """Reset the global batch context store (for testing)."""
    global _batch_context_store
    _batch_context_store = None
