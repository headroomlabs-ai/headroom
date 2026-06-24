"""Memory backends for Headroom's hierarchical memory system.

This module provides backend adapters for the memory system:
- LocalBackend: Fully local using SQLite + HNSW + InMemoryGraph (default)
- Mem0Backend: Graph + vector memory via Mem0 (Neo4j + Qdrant)
- Mem0SystemAdapter: Adapter to use Mem0Backend with MemorySystem tools
- DirectMem0Adapter: Optimized adapter that bypasses Mem0's LLM extraction

LocalBackend is always available. Mem0-based backends require optional
dependencies and imports are deferred until the backend is actually used.

Performance comparison:
    Mem0SystemAdapter:   3-4 LLM calls per memory_save (Mem0 extracts internally)
    DirectMem0Adapter:   0 LLM calls when using pre-extracted facts/entities
                         (embeddings only, main LLM does extraction in one pass)
"""

# LocalBackend is always available (no optional dependencies)
from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

# Lazy imports for optional backends
_Mem0Backend = None
_Mem0Config = None
_Mem0SystemAdapter = None
_DirectMem0Adapter = None
_DirectMem0Config = None


def __getattr__(name: str) -> type:
    """Lazy import for optional backends."""
    global _Mem0Backend, _Mem0Config, _Mem0SystemAdapter, _DirectMem0Adapter, _DirectMem0Config

    if name == "Mem0Backend":
        if _Mem0Backend is None:
            from headroom.memory.backends.mem0 import Mem0Backend

            _Mem0Backend = Mem0Backend
        return _Mem0Backend

    if name == "Mem0Config":
        if _Mem0Config is None:
            from headroom.memory.backends.mem0 import Mem0Config

            _Mem0Config = Mem0Config
        return _Mem0Config

    if name == "Mem0SystemAdapter":
        if _Mem0SystemAdapter is None:
            from headroom.memory.backends.mem0_system_adapter import Mem0SystemAdapter

            _Mem0SystemAdapter = Mem0SystemAdapter
        return _Mem0SystemAdapter

    if name == "DirectMem0Adapter":
        if _DirectMem0Adapter is None:
            from headroom.memory.backends.direct_mem0 import DirectMem0Adapter

            _DirectMem0Adapter = DirectMem0Adapter
        return _DirectMem0Adapter

    if name == "DirectMem0Config":
        if _DirectMem0Config is None:
            from headroom.memory.backends.direct_mem0 import Mem0Config as DirectMem0Config

            _DirectMem0Config = DirectMem0Config
        return _DirectMem0Config

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Local backend (always available)
    "LocalBackend",
    "LocalBackendConfig",
    # Mem0 backend (optional dependencies)
    "Mem0Backend",
    "Mem0Config",
    "Mem0SystemAdapter",
    # Direct Mem0 adapter (optimized, bypasses LLM extraction)
    "DirectMem0Adapter",
    "DirectMem0Config",
]
