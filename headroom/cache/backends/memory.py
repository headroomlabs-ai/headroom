"""In-memory storage backend for CompressionStore.

This is the default backend, providing fast access with no external dependencies.
Data is lost when the process exits.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from collections.abc import Collection
from typing import TYPE_CHECKING, Any

from ..context_clock import ContextTurnSnapshot, advance_context_state, resolve_context_conversation

if TYPE_CHECKING:
    from ..compression_store import CompressionEntry


class InMemoryBackend:
    """Thread-safe in-memory storage backend.

    This is the default backend for CompressionStore. It stores entries in a
    Python dict with thread-safe access via a lock.

    Characteristics:
    - Fast: O(1) get/set/delete operations
    - Volatile: Data lost on process exit
    - Thread-safe: All operations are protected by a lock
    - Memory-bound: Stores everything in RAM

    Usage:
        backend = InMemoryBackend()
        backend.set("abc123", entry)
        entry = backend.get("abc123")
    """

    def __init__(self) -> None:
        """Initialize the in-memory backend."""
        self._store: dict[str, CompressionEntry] = {}
        self._lock = threading.Lock()
        self._context_states: dict[str, tuple[dict[str, Any], float]] = {}

    def observe_context_turn(
        self, conversation_key: str, hash_keys: Collection[str], *, namespace_key: str | None = None
    ) -> ContextTurnSnapshot | None:
        with self._lock:
            now = time.time()
            events = {
                key: (entry.created_at, entry.created_at + entry.ttl, entry.event_identity)
                for key in set(hash_keys)
                if (entry := self._store.get(key)) is not None
                and entry.created_at + entry.ttl >= now
            }
            if not events:
                return None
            self._context_states = {
                key: value for key, value in self._context_states.items() if value[1] >= now
            }
            try:
                conversation_key = resolve_context_conversation(
                    conversation_key,
                    namespace_key,
                    {key: value[0] for key, value in self._context_states.items()},
                    events,
                )
                prior = self._context_states.get(conversation_key)
                state, snapshot, expires_at = advance_context_state(
                    None if prior is None else prior[0], events, self._store.keys(), now
                )
            except (ValueError, TypeError) as error:
                logging.getLogger(__name__).warning(
                    "CCR context clock invalid; proactive expansion skipped: %s", error
                )
                return None
            state["namespace"] = namespace_key if namespace_key is not None else "explicit"
            self._context_states[conversation_key] = (state, expires_at)
            return snapshot

    def get(self, hash_key: str) -> CompressionEntry | None:
        """Retrieve an entry by hash key.

        Args:
            hash_key: The unique hash identifying the entry.

        Returns:
            CompressionEntry if found, None otherwise.
        """
        with self._lock:
            return self._store.get(hash_key)

    def record_retrieval(self, hash_key: str, entry: CompressionEntry, query: str | None) -> None:
        """Update access metadata only if the retrieved event is still current."""
        with self._lock:
            current = self._store.get(hash_key)
            if current is None or current.event_identity != entry.event_identity:
                return
            current.record_access(query)
            entry.retrieval_count = current.retrieval_count
            entry.last_accessed = current.last_accessed
            entry.search_queries = current.search_queries.copy()

    def set_new(self, hash_key: str, entry: CompressionEntry) -> None:
        """Assign a fresh event timestamp under the shared backend lock."""
        with self._lock:
            previous = self._store.get(hash_key)
            if previous is not None and entry.created_at <= previous.created_at:
                entry.created_at = math.nextafter(previous.created_at, math.inf)
            self._store[hash_key] = entry

    def set(self, hash_key: str, entry: CompressionEntry) -> None:
        """Store an entry with the given hash key.

        Args:
            hash_key: The unique hash identifying the entry.
            entry: The CompressionEntry to store.
        """
        with self._lock:
            self._store[hash_key] = entry

    def delete(self, hash_key: str) -> bool:
        """Delete an entry by hash key.

        Args:
            hash_key: The unique hash identifying the entry.

        Returns:
            True if entry was deleted, False if it didn't exist.
        """
        with self._lock:
            if hash_key in self._store:
                del self._store[hash_key]
                return True
            return False

    def exists(self, hash_key: str) -> bool:
        """Check if an entry exists.

        Args:
            hash_key: The unique hash identifying the entry.

        Returns:
            True if entry exists, False otherwise.
        """
        with self._lock:
            return hash_key in self._store

    def clear(self) -> None:
        """Remove all entries from storage."""
        with self._lock:
            self._store.clear()
            self._context_states.clear()

    def count(self) -> int:
        """Get the number of entries in storage.

        Returns:
            Number of entries currently stored.
        """
        with self._lock:
            return len(self._store)

    def keys(self) -> list[str]:
        """Get all hash keys in storage.

        Returns:
            List of all hash keys.
        """
        with self._lock:
            return list(self._store.keys())

    def items(self) -> list[tuple[str, CompressionEntry]]:
        """Get all entries as (hash_key, entry) pairs.

        Returns:
            List of (hash_key, CompressionEntry) tuples.
        """
        with self._lock:
            return list(self._store.items())

    def get_stats(self) -> dict[str, Any]:
        """Get backend statistics.

        Returns:
            Dict with stats including entry_count and memory estimate.
        """
        with self._lock:
            entry_count = len(self._store)
            # Rough memory estimate
            bytes_used = sys.getsizeof(self._store)
            for entry in self._store.values():
                bytes_used += sys.getsizeof(entry)
                bytes_used += len(entry.original_content.encode("utf-8"))
                bytes_used += len(entry.compressed_content.encode("utf-8"))

            return {
                "backend_type": "memory",
                "entry_count": entry_count,
                "bytes_used": bytes_used,
            }
