"""Base protocol for CompressionStore backends.

This protocol defines the minimal interface that storage backends must implement.
The interface is intentionally simple - it only handles CRUD operations on entries.
Higher-level concerns (search, feedback, eviction policies) are handled by CompressionStore.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..compression_store import CompressionEntry


@runtime_checkable
class CompressionStoreBackend(Protocol):
    """Protocol for CompressionStore storage backends.

    This protocol defines the minimal interface for pluggable storage backends.
    Implementations can use any storage mechanism: memory, MongoDB, Redis, etc.

    Design Principles:
    - Simple CRUD operations only
    - No business logic (search, feedback, eviction policies)
    - Thread-safety is implementation's responsibility
    - TTL handling can be delegated to backend or handled by CompressionStore

    Optional class attributes (read with ``getattr(backend, name, default)``,
    deliberately NOT part of the protocol so third-party backends stay valid
    under ``isinstance``):

    - ``is_process_local: bool`` (default False) -- True when entries live only
      in this process: not shared with other workers, gone on restart. Drives
      the retrieval-miss diagnostics, so a miss is not blamed on TTL when the
      real cause is that the entry was stored by a different worker.
    - ``writes_local_disk: bool`` (no default -- undeclared fails closed) --
      set it to ``False`` to declare that this backend keeps nothing on the
      host filesystem (a remote store: Redis, MongoDB, an HTTP service).
      ``--stateless`` / ``HEADROOM_STATELESS`` keeps such a backend, since the
      promise is about *this machine's* disk and an off-machine store is the
      only way to share retrieval across workers. A backend that does not
      declare it is swapped for the in-process one under stateless mode:
      verbatim tool-result originals are the payload at stake, and from the
      outside there is no way to tell where an undeclared backend puts them.
    - ``close() -> None`` -- release the backend's handles without destroying
      stored entries. Called when stateless mode swaps a backend out.

    Example implementation:
        class MyBackend:
            def get(self, hash_key: str) -> CompressionEntry | None:
                return self._storage.get(hash_key)

            def set(self, hash_key: str, entry: CompressionEntry) -> None:
                self._storage[hash_key] = entry

            # ... other methods
    """

    def get(self, hash_key: str) -> CompressionEntry | None:
        """Retrieve an entry by hash key.

        Args:
            hash_key: The unique hash identifying the entry.

        Returns:
            CompressionEntry if found, None otherwise.
            Does NOT check TTL - that's CompressionStore's responsibility.
        """
        ...

    def set(self, hash_key: str, entry: CompressionEntry) -> None:
        """Store an entry with the given hash key.

        Args:
            hash_key: The unique hash identifying the entry.
            entry: The CompressionEntry to store.

        Note:
            Overwrites any existing entry with the same key.
        """
        ...

    def delete(self, hash_key: str) -> bool:
        """Delete an entry by hash key.

        Args:
            hash_key: The unique hash identifying the entry.

        Returns:
            True if entry was deleted, False if it didn't exist.
        """
        ...

    def exists(self, hash_key: str) -> bool:
        """Check if an entry exists.

        Args:
            hash_key: The unique hash identifying the entry.

        Returns:
            True if entry exists, False otherwise.
            Does NOT check TTL - that's CompressionStore's responsibility.
        """
        ...

    def clear(self) -> None:
        """Remove all entries from storage."""
        ...

    def count(self) -> int:
        """Get the number of entries in storage.

        Returns:
            Number of entries currently stored.
        """
        ...

    def keys(self) -> list[str]:
        """Get all hash keys in storage.

        Returns:
            List of all hash keys.

        Note:
            For large stores, consider implementing an iterator version.
        """
        ...

    def items(self) -> list[tuple[str, CompressionEntry]]:
        """Get all entries as (hash_key, entry) pairs.

        Returns:
            List of (hash_key, CompressionEntry) tuples.

        Note:
            For large stores, consider implementing an iterator version.
        """
        ...

    def get_stats(self) -> dict[str, Any]:
        """Get backend-specific statistics.

        Returns:
            Dict with backend stats. Should include at minimum:
            - "entry_count": number of entries
            - "backend_type": name of the backend implementation

            Backends may include additional stats like:
            - "bytes_used": memory/storage used
            - "connection_status": for remote backends
        """
        ...
