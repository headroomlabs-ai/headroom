"""Headroom Memory - Simple, zero-config memory for AI applications.

Quick Start (No Docker Required):
    from headroom.memory import Memory

    # Create memory instance - works out of the box!
    memory = Memory()

    # Save memories
    await memory.save("User prefers dark mode and uses Python", user_id="alice")

    # Search memories
    results = await memory.search("What programming language?", user_id="alice")
    for r in results:
        print(r.content, r.score)

Production Mode (with Docker):
    # Start services: docker compose up -d qdrant neo4j
    memory = Memory(backend="qdrant-neo4j")

    # Same API, production-grade backends
    await memory.save("User works at Netflix", user_id="alice")

Backends:
    - "local" (default): SQLite + HNSW + InMemoryGraph. No setup required.
    - "qdrant-neo4j": Qdrant + Neo4j. Requires Docker services.

Advanced Usage - LLM Wrapper:
    from openai import OpenAI
    from headroom.memory import with_memory

    client = with_memory(OpenAI(), user_id="alice")
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "I prefer Python"}]
    )
    # Memory automatically extracted and stored!

Advanced Usage - Tool-based:
    from openai import OpenAI
    from headroom.memory import with_memory_tools, LocalBackend

    client = with_memory_tools(
        OpenAI(),
        backend=LocalBackend(),
        user_id="alice",
        optimized=True,  # LLM extracts facts/entities in ONE call
    )
"""

# =============================================================================
# Configuration
# =============================================================================
# =============================================================================
# Graph adapters
# =============================================================================
from headroom.memory.adapters.graph import InMemoryGraphStore

# =============================================================================
# Backend implementations (lazy imports for optional dependencies)
# =============================================================================
# LocalBackend is always available (no optional dependencies)
from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

# =============================================================================
# Memory Bridge (markdown <-> Headroom bidirectional sync)
# =============================================================================
from headroom.memory.bridge import ImportStats, MemoryBridge, SyncStats
from headroom.memory.bridge_config import BridgeConfig, MarkdownFormat
from headroom.memory.config import (
    EmbedderBackend,
    MemoryConfig,
    StoreBackend,
    TextBackend,
    VectorBackend,
)

# =============================================================================
# Core orchestrator
# =============================================================================
from headroom.memory.core import HierarchicalMemory

# =============================================================================
# Simple API (recommended for most users)
# =============================================================================
from headroom.memory.easy import Memory, MemoryResult

# =============================================================================
# Factory
# =============================================================================
from headroom.memory.factory import create_memory_system

# =============================================================================
# Data models (internal)
# =============================================================================
from headroom.memory.models import Memory as MemoryModel
from headroom.memory.models import ScopeLevel

# =============================================================================
# Protocol interfaces (ports)
# =============================================================================
from headroom.memory.ports import (
    # Core protocols
    Embedder,
    # Graph dataclasses
    Entity,
    # Graph protocol
    GraphStore,
    MemoryCache,
    # Filter dataclasses
    MemoryFilter,
    # Memory search result
    MemorySearchResult,
    MemoryStore,
    Relationship,
    Subgraph,
    TextFilter,
    TextIndex,
    # Search result dataclasses
    TextSearchResult,
    VectorFilter,
    VectorIndex,
    VectorSearchResult,
)

# =============================================================================
# Memory system orchestrator
# =============================================================================
from headroom.memory.system import MemoryBackend, MemorySystem

# =============================================================================
# Memory tools for LLM function calling
# =============================================================================
from headroom.memory.tools import (
    MEMORY_TOOLS,
    MEMORY_TOOLS_OPTIMIZED,
    get_memory_tools,
    get_memory_tools_optimized,
)

# =============================================================================
# Wrapper for LLM clients (main user-facing API)
# =============================================================================
from headroom.memory.wrapper import MemoryWrapper, with_memory

# =============================================================================
# Tool-based wrapper for LLM clients
# =============================================================================
from headroom.memory.wrapper_tools import MemoryToolsWrapper, with_memory_tools

# Lazy imports for optional backends to avoid ImportError if dependencies not installed
_Mem0Backend = None
_Mem0Config = None
_DirectMem0Adapter = None
_DirectMem0Config = None


def __getattr__(name: str) -> type:
    """Lazy import for optional backend components."""
    global _Mem0Backend, _Mem0Config, _DirectMem0Adapter, _DirectMem0Config

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
    # =========================================================================
    # Simple API (recommended for most users)
    # =========================================================================
    "Memory",  # Zero-config memory class
    "MemoryResult",  # Search result dataclass
    # =========================================================================
    # LLM Wrapper API
    # =========================================================================
    "with_memory",
    "MemoryWrapper",
    # Tool-based wrapper
    "with_memory_tools",
    "MemoryToolsWrapper",
    # =========================================================================
    # Core orchestrator
    # =========================================================================
    "HierarchicalMemory",
    # =========================================================================
    # Data models (internal)
    # =========================================================================
    "MemoryModel",  # Internal memory model (renamed from Memory)
    "ScopeLevel",
    # =========================================================================
    # Protocol interfaces (ports)
    # =========================================================================
    "MemoryStore",
    "VectorIndex",
    "TextIndex",
    "Embedder",
    "MemoryCache",
    "GraphStore",
    # =========================================================================
    # Filter dataclasses
    # =========================================================================
    "MemoryFilter",
    "VectorFilter",
    "TextFilter",
    # =========================================================================
    # Search result dataclasses
    # =========================================================================
    "VectorSearchResult",
    "TextSearchResult",
    "MemorySearchResult",
    # =========================================================================
    # Graph dataclasses
    # =========================================================================
    "Entity",
    "Relationship",
    "Subgraph",
    # =========================================================================
    # Configuration
    # =========================================================================
    "MemoryConfig",
    "StoreBackend",
    "VectorBackend",
    "TextBackend",
    "EmbedderBackend",
    # =========================================================================
    # Factory
    # =========================================================================
    "create_memory_system",
    # =========================================================================
    # Memory tools for LLM function calling
    # =========================================================================
    "MEMORY_TOOLS",
    "MEMORY_TOOLS_OPTIMIZED",
    "get_memory_tools",
    "get_memory_tools_optimized",
    # =========================================================================
    # Memory system orchestrator
    # =========================================================================
    "MemorySystem",
    "MemoryBackend",
    # =========================================================================
    # Graph adapters
    # =========================================================================
    "InMemoryGraphStore",
    # =========================================================================
    # Backend implementations
    # =========================================================================
    # Local backend (always available)
    "LocalBackend",
    "LocalBackendConfig",
    # Mem0 backend (optional dependencies - lazy loaded)
    "Mem0Backend",
    "Mem0Config",
    # DirectMem0Adapter - optimized Mem0 adapter that bypasses internal LLM calls
    # Use with optimized=True in with_memory_tools() for best performance
    "DirectMem0Adapter",
    "DirectMem0Config",
    # =========================================================================
    # Memory Bridge (markdown <-> Headroom bidirectional sync)
    # =========================================================================
    "MemoryBridge",
    "BridgeConfig",
    "MarkdownFormat",
    "ImportStats",
    "SyncStats",
]
