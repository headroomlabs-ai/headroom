"""Storage backends for CompressionStore.

This module provides pluggable storage backends for CCR (Compress-Cache-Retrieve).
Backend selection depends on how the store is constructed:
- ``get_compression_store()`` (the proxy path) defaults to SQLite
  (restart-safe, shared across workers); ``HEADROOM_CCR_BACKEND=memory``
  forces in-memory, and other backends (Redis, MongoDB via entry points)
  can be selected by env.
- ``CompressionStore()`` constructed directly defaults to **in-memory**
  unless a backend is passed explicitly.

Usage:
    from headroom.cache.backends import SQLiteBackend, CompressionStoreBackend
    from headroom.cache.compression_store import CompressionStore, get_compression_store

    # Env-driven default (SQLite at ~/.headroom/ccr_store.db)
    store = get_compression_store()

    # Direct construction defaults to in-memory; pass a backend for persistence
    store = CompressionStore(backend=SQLiteBackend())

    # Use a custom backend
    class MyBackend:
        # Implement CompressionStoreBackend protocol
        ...
    store = CompressionStore(backend=MyBackend())
"""

from .base import CompressionStoreBackend
from .memory import InMemoryBackend
from .sqlite import SQLiteBackend

__all__ = [
    "CompressionStoreBackend",
    "InMemoryBackend",
    "SQLiteBackend",
]
