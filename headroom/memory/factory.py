"""Factory module for creating memory system components.

Provides a unified factory function that creates all memory system components
from a single configuration object, ensuring consistent initialization
and proper wiring between components.
"""

from __future__ import annotations

import threading
from importlib.metadata import entry_points
from pathlib import Path
from typing import TYPE_CHECKING, Any

from headroom.memory.config import (
    EmbedderBackend,
    MemoryConfig,
    StoreBackend,
    TextBackend,
    VectorBackend,
)

if TYPE_CHECKING:
    from headroom.memory.ports import Embedder, MemoryCache, MemoryStore, TextIndex, VectorIndex


# Extension groups for memory backends registered via setuptools entry points.
_MEMORY_STORE_GROUP = "headroom.memory_store"
_MEMORY_VECTOR_GROUP = "headroom.memory_vector"
_MEMORY_TEXT_GROUP = "headroom.memory_text"

# Process-wide embedder cache keyed by (backend, model). Embedders are
# stateless with respect to the memory store, so a single instance can
# safely serve every per-project ``LocalBackend`` created by the
# BackendRouter. Without this cache, opening N project DBs would load
# the sentence-transformers / ONNX model N times.
_EMBEDDER_CACHE: dict[tuple[str, str], Embedder] = {}
_EMBEDDER_CACHE_LOCK = threading.Lock()


def _load_external_backend(
    group: str,
    name: str | None,
    field_name: str,
    config: MemoryConfig,
) -> Any:
    """Load a memory backend registered via setuptools entry points.

    Mirrors the pattern used by
    `headroom.cache.compression_store._create_default_ccr_backend`.
    """
    if not name:
        raise ValueError(
            f"{field_name} is required when backend is EXTERNAL; "
            f"set it to the entry-point name registered under '{group}'."
        )
    ep = next((e for e in entry_points(group=group) if e.name == name), None)
    if ep is None:
        raise ValueError(
            f"No entry point registered under '{group}' with name '{name}'. "
            f"Install the package that provides it."
        )
    return ep.load()(config)


async def create_memory_system(
    config: MemoryConfig | None = None,
) -> tuple[MemoryStore, VectorIndex, TextIndex, Embedder, MemoryCache | None]:
    """Create a complete memory system from configuration.

    This factory function creates and initializes all memory system components
    based on the provided configuration. Components are created in dependency
    order to ensure proper initialization.

    Args:
        config: Memory system configuration. If None, uses default configuration.

    Returns:
        A tuple of (store, vector_index, text_index, embedder, cache) where:
        - store: The memory persistence backend
        - vector_index: The vector similarity search index
        - text_index: The full-text search index
        - embedder: The text embedding generator
        - cache: The memory cache (or None if caching is disabled)

    Raises:
        ValueError: If an unknown backend type is specified in the config.

    Example:
        config = MemoryConfig(
            embedder_backend=EmbedderBackend.LOCAL,
            cache_max_size=2000,
        )
        store, vector, text, embedder, cache = await create_memory_system(config)
    """
    config = config or MemoryConfig()

    # Create store
    store = _create_store(config)

    # Create embedder (needed by vector index for text queries)
    embedder = _create_embedder(config)

    # Create vector index
    vector_index = _create_vector_index(config)

    # Create text index
    text_index = _create_text_index(config)

    # Create cache (optional)
    cache = _create_cache(config) if config.cache_enabled else None

    return store, vector_index, text_index, embedder, cache


def _create_store(config: MemoryConfig) -> MemoryStore:
    """Create a memory store backend.

    Args:
        config: Memory system configuration.

    Returns:
        A MemoryStore implementation based on config.store_backend.

    Raises:
        ValueError: If the store backend is not supported.
    """
    if config.store_backend == StoreBackend.SQLITE:
        from headroom.memory.adapters.sqlite import SQLiteMemoryStore

        return SQLiteMemoryStore(config.db_path)

    if config.store_backend == StoreBackend.EXTERNAL:
        return _load_external_backend(  # type: ignore[no-any-return]
            _MEMORY_STORE_GROUP,
            config.store_backend_name,
            "store_backend_name",
            config,
        )

    raise ValueError(f"Unknown store backend: {config.store_backend}")


def _create_embedder(config: MemoryConfig) -> Embedder:
    """Create or return a cached embedder backend.

    The embedder is shared across every ``LocalBackend`` instance that
    requests the same ``(embedder_backend, embedder_model)`` pair. This
    matters for the per-project storage router, which can open many
    backends in the same process and must not pay the
    sentence-transformers / ONNX model-load cost more than once.

    Args:
        config: Memory system configuration.

    Returns:
        An Embedder implementation based on config.embedder_backend.

    Raises:
        ValueError: If the embedder backend is not supported.
    """

    # Validate inputs ahead of the cache. The cache key is
    # ``(backend, model)`` and intentionally does NOT include the API
    # key — but that means a cached OpenAI embedder would shadow the
    # config-validation step for a subsequent caller who forgot to pass
    # ``openai_api_key``. Run the validation up front instead.
    if config.embedder_backend == EmbedderBackend.OPENAI and not config.openai_api_key:
        raise ValueError("openai_api_key is required for OpenAI embedder")

    key = (
        config.embedder_backend.value
        if hasattr(config.embedder_backend, "value")
        else str(config.embedder_backend),
        config.embedder_model or "",
    )

    with _EMBEDDER_CACHE_LOCK:
        cached = _EMBEDDER_CACHE.get(key)
        if cached is not None:
            return cached

        if config.embedder_backend == EmbedderBackend.LOCAL:
            from headroom.memory.adapters.embedders import LocalEmbedder

            embedder: Embedder = LocalEmbedder(model_name=config.embedder_model)

        elif config.embedder_backend == EmbedderBackend.ONNX:
            from headroom.memory.adapters.embedders import OnnxLocalEmbedder

            embedder = OnnxLocalEmbedder()

        elif config.embedder_backend == EmbedderBackend.OPENAI:
            from headroom.memory.adapters.embedders import OpenAIEmbedder

            embedder = OpenAIEmbedder(
                api_key=config.openai_api_key,
                model_name=config.embedder_model,
            )

        elif config.embedder_backend == EmbedderBackend.OLLAMA:
            from headroom.memory.adapters.embedders import OllamaEmbedder

            embedder = OllamaEmbedder(
                base_url=config.ollama_base_url,
                model_name=config.embedder_model,
            )
        else:
            raise ValueError(f"Unknown embedder backend: {config.embedder_backend}")

        _EMBEDDER_CACHE[key] = embedder
        return embedder


def _reset_embedder_cache_for_tests() -> None:
    """Clear the process-wide embedder cache. Test-only seam."""

    with _EMBEDDER_CACHE_LOCK:
        _EMBEDDER_CACHE.clear()


def _create_vector_index(config: MemoryConfig) -> VectorIndex:
    """Create a vector index backend.

    Args:
        config: Memory system configuration.

    Returns:
        A VectorIndex implementation based on config.vector_backend.

    Raises:
        ValueError: If the vector backend is not supported or unavailable.
    """
    backend = config.vector_backend

    if backend == VectorBackend.EXTERNAL:
        return _load_external_backend(  # type: ignore[no-any-return]
            _MEMORY_VECTOR_GROUP,
            config.vector_backend_name,
            "vector_backend_name",
            config,
        )

    # AUTO: prefer SQLITE_VEC → HNSW → fail with helpful message
    if backend == VectorBackend.AUTO:
        from headroom.memory.adapters import HNSW_AVAILABLE, SQLITE_VEC_AVAILABLE

        if SQLITE_VEC_AVAILABLE:
            backend = VectorBackend.SQLITE_VEC
        elif HNSW_AVAILABLE:
            backend = VectorBackend.HNSW
        else:
            raise ValueError(
                "No vector index backend available for memory. Install one:\n"
                "  pip install sqlite-vec   (recommended, lightweight)\n"
                "  pip install hnswlib      (alternative)\n"
                "Or install the full proxy bundle: pip install headroom-ai[proxy]"
            )

    if backend == VectorBackend.SQLITE_VEC:
        from headroom.memory.adapters import SQLITE_VEC_AVAILABLE

        if not SQLITE_VEC_AVAILABLE:
            raise ValueError(
                "sqlite-vec is not available. Install with: pip install sqlite-vec\n"
                "Or use vector_backend=VectorBackend.HNSW"
            )

        from headroom.memory.adapters.sqlite_vector import SQLiteVectorIndex

        # Derive vector db path from main db path if not specified
        if config.vector_db_path:
            vector_db_path = config.vector_db_path
        else:
            # "memory.db" -> "memory_vectors.db"
            vector_db_path = config.db_path.parent / f"{config.db_path.stem}_vectors.db"

        return SQLiteVectorIndex(
            dimension=config.vector_dimension,
            db_path=vector_db_path,
            page_cache_size_kb=config.vector_cache_size_kb,
        )

    if backend == VectorBackend.HNSW:
        from headroom.memory.adapters import HNSW_AVAILABLE

        if not HNSW_AVAILABLE:
            raise ValueError(
                "hnswlib is not available. Install with: pip install hnswlib\n"
                "Or use vector_backend=VectorBackend.SQLITE_VEC"
            )

        from headroom.memory.adapters.hnsw import HNSWVectorIndex

        # Derive persistent save path from the main DB path so the HNSW
        # index survives across process restarts (critical for cross-agent
        # interop: memories saved by Codex MCP must be searchable by Claude).
        hnsw_save_path: str | Path | None = None
        if config.db_path:
            hnsw_save_path = config.db_path.parent / f"{config.db_path.stem}_hnsw"

        return HNSWVectorIndex(
            dimension=config.vector_dimension,
            ef_construction=config.hnsw_ef_construction,
            m=config.hnsw_m,
            ef_search=config.hnsw_ef_search,
            max_entries=config.hnsw_max_entries,
            save_path=hnsw_save_path,
            auto_save=True,
        )

    raise ValueError(f"Unknown vector backend: {config.vector_backend}")


def _create_text_index(config: MemoryConfig) -> TextIndex:
    """Create a text index backend.

    Args:
        config: Memory system configuration.

    Returns:
        A TextIndex implementation based on config.text_backend.

    Raises:
        ValueError: If the text backend is not supported.
    """
    if config.text_backend == TextBackend.FTS5:
        from headroom.memory.adapters.fts5 import FTS5TextIndex

        # FTS5TextIndex has a compatible interface but different method signatures
        return FTS5TextIndex(db_path=config.db_path)  # type: ignore[return-value]

    if config.text_backend == TextBackend.EXTERNAL:
        return _load_external_backend(  # type: ignore[no-any-return]
            _MEMORY_TEXT_GROUP,
            config.text_backend_name,
            "text_backend_name",
            config,
        )

    raise ValueError(f"Unknown text backend: {config.text_backend}")


def _create_cache(config: MemoryConfig) -> MemoryCache:
    """Create a memory cache.

    Args:
        config: Memory system configuration.

    Returns:
        A MemoryCache implementation.
    """
    from headroom.memory.adapters.cache import LRUMemoryCache

    # LRUMemoryCache implements MemoryCache protocol
    return LRUMemoryCache(max_size=config.cache_max_size)  # type: ignore[return-value]
