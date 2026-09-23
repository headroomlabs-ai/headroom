"""Storage backends for CompressionStore.

This module provides pluggable storage backends for CCR (Compress-Cache-Retrieve).
Backend selection depends on how the store is constructed:
- ``get_compression_store()`` (the proxy path) defaults to SQLite
  (restart-safe, shared across workers); ``HEADROOM_CCR_BACKEND=memory``
  forces in-memory, and other backends (Redis, MongoDB via entry points)
  can be selected by env.
- ``CompressionStore()`` constructed directly defaults to **in-memory**
  unless a backend is passed explicitly.

Stateless mode (``--stateless`` / ``HEADROOM_STATELESS``) overrides all of
that: every store handed out by ``get_compression_store()``, and every store
installed with ``set_request_compression_store()``, is checked and swapped to
an in-process backend unless the backend declares ``writes_local_disk =
False``. A custom backend that stores off-machine should set that class
attribute; see ``base.CompressionStoreBackend``.

Usage:
    from headroom.cache.backends import SQLiteBackend, CompressionStoreBackend
    from headroom.cache.compression_store import CompressionStore, get_compression_store

    # Env-driven default (SQLite at workspace_dir()/ccr_store.db)
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
