"""Memory adapters for Headroom's hierarchical memory system.

This module provides concrete implementations of the memory system's ports:
- SQLiteMemoryStore: SQLite-based memory persistence
- FTS5TextIndex: SQLite FTS5 full-text search index
- HNSWVectorIndex: HNSW-based vector index using hnswlib (optional)
- SQLiteVectorIndex: SQLite-based vector index using sqlite-vec (optional, recommended)
- LRUMemoryCache: Thread-safe LRU cache for hot memories
- InMemoryGraphStore: In-memory graph store for knowledge graphs
- SQLiteGraphStore: SQLite-based graph store (bounded memory, persistent)
- LocalEmbedder: sentence-transformers embedding (local, optional)
- OpenAIEmbedder: OpenAI API embedding (cloud, optional)
- OllamaEmbedder: Ollama API embedding (local server, optional)

Note: Some adapters require optional dependencies. Import errors are
deferred until the adapter is actually used.
"""

# Core adapters (no external dependencies beyond sqlite3)
from headroom.memory.adapters.cache import LRUMemoryCache
from headroom.memory.adapters.fts5 import FTS5TextIndex
from headroom.memory.adapters.graph import InMemoryGraphStore
from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.adapters.sqlite_graph import SQLiteGraphStore

# Check for optional dependencies availability
# Note: We don't import from hnsw.py here because hnswlib may crash with
# "Illegal instruction" on CPUs without required instructions (e.g., AVX).
# Instead, we check lazily when HNSWVectorIndex is actually used.
# HNSW_AVAILABLE is handled through __getattr__ to ensure lazy checking.

# Lazy imports for optional adapters
_HNSW_AVAILABLE: bool | None = None  # Internal cache for HNSW_AVAILABLE
_SQLITE_VEC_AVAILABLE: bool | None = None  # Internal cache for SQLITE_VEC_AVAILABLE
_HNSWVectorIndex = None
_SQLiteVectorIndex = None
_LocalEmbedder = None
_OpenAIEmbedder = None
_OllamaEmbedder = None


def __getattr__(name: str) -> type | bool:
    """Lazy import for optional adapters."""
    global _HNSWVectorIndex, _SQLiteVectorIndex
    global _LocalEmbedder, _OpenAIEmbedder, _OllamaEmbedder
    global _HNSW_AVAILABLE, _SQLITE_VEC_AVAILABLE

    if name == "HNSW_AVAILABLE":
        # Lazily check hnswlib availability
        if _HNSW_AVAILABLE is None:
            from headroom.memory.adapters.hnsw import _check_hnswlib_available

            _HNSW_AVAILABLE = _check_hnswlib_available()
        return _HNSW_AVAILABLE

    if name == "SQLITE_VEC_AVAILABLE":
        # Lazily check sqlite-vec availability
        if _SQLITE_VEC_AVAILABLE is None:
            from headroom.memory.adapters.sqlite_vector import is_sqlite_vec_available

            _SQLITE_VEC_AVAILABLE = is_sqlite_vec_available()
        return _SQLITE_VEC_AVAILABLE

    if name == "HNSWVectorIndex":
        if _HNSWVectorIndex is None:
            from headroom.memory.adapters.hnsw import HNSWVectorIndex

            _HNSWVectorIndex = HNSWVectorIndex
        return _HNSWVectorIndex

    if name == "SQLiteVectorIndex":
        if _SQLiteVectorIndex is None:
            from headroom.memory.adapters.sqlite_vector import SQLiteVectorIndex

            _SQLiteVectorIndex = SQLiteVectorIndex
        return _SQLiteVectorIndex

    if name == "LocalEmbedder":
        if _LocalEmbedder is None:
            from headroom.memory.adapters.embedders import LocalEmbedder

            _LocalEmbedder = LocalEmbedder
        return _LocalEmbedder

    if name == "OpenAIEmbedder":
        if _OpenAIEmbedder is None:
            from headroom.memory.adapters.embedders import OpenAIEmbedder

            _OpenAIEmbedder = OpenAIEmbedder
        return _OpenAIEmbedder

    if name == "OllamaEmbedder":
        if _OllamaEmbedder is None:
            from headroom.memory.adapters.embedders import OllamaEmbedder

            _OllamaEmbedder = OllamaEmbedder
        return _OllamaEmbedder

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Core adapters (always available)
    "FTS5TextIndex",
    "InMemoryGraphStore",
    "LRUMemoryCache",
    "SQLiteGraphStore",
    "SQLiteMemoryStore",
    # Optional adapters (lazy-loaded)
    "HNSWVectorIndex",
    "SQLiteVectorIndex",
    "LocalEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    # Availability flags
    "HNSW_AVAILABLE",
    "SQLITE_VEC_AVAILABLE",
]
