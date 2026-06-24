"""In-memory storage backend for CompressionStore.

This is the default backend, providing fast access with no external dependencies.
Data is lost when the process exits.
"""

from __future__ import annotations

import sys
import threading
from typing import TYPE_CHECKING, Any

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

    def get(self, hash_key: str) -> CompressionEntry | None:
        """Retrieve an entry by hash key.

        Args:
            hash_key: The unique hash identifying the entry.

        Returns:
            CompressionEntry if found, None otherwise.
        """
        with self._lock:
            return self._store.get(hash_key)

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
