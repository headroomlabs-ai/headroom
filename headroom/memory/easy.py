"""Simple, zero-config memory API for developers.

This module provides the easiest way to use Headroom's memory system.
No Docker required - works out of the box with embedded databases.

Usage:
    from headroom.memory import Memory

    # Create memory instance (no setup required)
    memory = Memory()

    # Save memories
    await memory.save(
        "User prefers dark mode and uses Python",
        user_id="alice",
    )

    # Search memories
    results = await memory.search(
        "What programming language?",
        user_id="alice",
    )
    for r in results:
        print(r.content, r.score)

    # For production (requires Docker: docker compose up -d qdrant neo4j)
    memory = Memory(backend="qdrant-neo4j")

Backends:
    - "local" (default): SQLite + HNSW + InMemoryGraph. No setup required.
    - "qdrant-neo4j": Qdrant + Neo4j. Requires Docker services.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from headroom.memory.ports import MemorySearchResult


@dataclass
class MemoryResult:
    """A single memory search result."""

    content: str
    score: float
    id: str
    metadata: dict[str, Any]

    @classmethod
    def from_search_result(cls, result: MemorySearchResult) -> MemoryResult:
        """Create from internal MemorySearchResult."""
        return cls(
            content=result.memory.content,
            score=result.score,
            id=result.memory.id,
            metadata=result.memory.metadata,
        )


class Memory:
    """Simple, zero-config memory API.

    Works out of the box with no external dependencies.
    Just create an instance and start saving/searching.

    Args:
        backend: Which backend to use:
            - "local" (default): Embedded SQLite + HNSW. No Docker needed.
            - "qdrant-neo4j": External Qdrant + Neo4j. Requires Docker.
        db_path: Path for local database (only for "local" backend).
            Defaults to ~/.headroom/memory.db
        qdrant_url: Full Qdrant URL (only for "qdrant-neo4j" backend). When set,
            takes precedence over ``qdrant_host``/``qdrant_port``. Useful for
            hosted Qdrant (Qdrant Cloud) and non-default container stacks.
            Defaults to the ``HEADROOM_QDRANT_URL`` env var if unset.
        qdrant_host: Qdrant host (only for "qdrant-neo4j" backend). Defaults
            to the ``HEADROOM_QDRANT_HOST`` env var or ``localhost``.
        qdrant_port: Qdrant port (only for "qdrant-neo4j" backend). Defaults
            to the ``HEADROOM_QDRANT_PORT`` env var or ``6333``.
        qdrant_api_key: API key for hosted Qdrant. Defaults to
            ``HEADROOM_QDRANT_API_KEY`` if unset.
        neo4j_uri: Neo4j URI (only for "qdrant-neo4j" backend).

    Examples:
        # Simplest usage - no config needed
        memory = Memory()
        await memory.save("User likes Python", user_id="alice")

        # With custom database path
        memory = Memory(db_path="./my_app.db")

        # Production mode with local Docker services
        memory = Memory(backend="qdrant-neo4j")

        # Hosted Qdrant via URL + API key (or set HEADROOM_QDRANT_URL /
        # HEADROOM_QDRANT_API_KEY in the environment)
        memory = Memory(
            backend="qdrant-neo4j",
            qdrant_url="https://xyz.cloud.qdrant.io:6333",
            qdrant_api_key="...",
        )
    """

    def __init__(
        self,
        backend: str = "local",
        db_path: str | Path | None = None,
        qdrant_url: str | None = None,
        qdrant_host: str | None = None,
        qdrant_port: int | None = None,
        qdrant_api_key: str | None = None,
        neo4j_uri: str = "neo4j://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "password",
    ) -> None:
        from headroom.memory import qdrant_env

        self._backend_type = backend
        self._backend: Any = None
        self._initialized = False

        # Config for local backend
        if db_path is None:
            # Default: workspace memory.db (respects HEADROOM_WORKSPACE_DIR)
            from headroom import paths as _paths

            default_db = _paths.memory_db_path()
            default_db.parent.mkdir(parents=True, exist_ok=True)
            db_path = default_db
        self._db_path = Path(db_path)

        # Config for qdrant-neo4j backend.
        # ``None`` sentinels fall back to HEADROOM_QDRANT_* env vars so that
        # ``Memory(backend="qdrant-neo4j")`` picks up hosted/custom Qdrant
        # deployments without any code changes. Explicit values always win.
        self._qdrant_url = qdrant_url if qdrant_url is not None else qdrant_env.qdrant_env_url()
        self._qdrant_host = qdrant_host if qdrant_host is not None else qdrant_env.qdrant_env_host()
        self._qdrant_port = qdrant_port if qdrant_port is not None else qdrant_env.qdrant_env_port()
        self._qdrant_api_key = (
            qdrant_api_key if qdrant_api_key is not None else qdrant_env.qdrant_env_api_key()
        )
        self._neo4j_uri = neo4j_uri
        self._neo4j_user = neo4j_user
        self._neo4j_password = neo4j_password

    async def _ensure_initialized(self) -> None:
        """Initialize the backend on first use."""
        if self._initialized:
            return

        if self._backend_type == "local":
            from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

            config = LocalBackendConfig(db_path=str(self._db_path))
            self._backend = LocalBackend(config)

        elif self._backend_type == "qdrant-neo4j":
            try:
                from headroom.memory.backends.direct_mem0 import (
                    DirectMem0Adapter,
                    Mem0Config,
                )

                mem0_config = Mem0Config(
                    qdrant_url=self._qdrant_url,
                    qdrant_host=self._qdrant_host,
                    qdrant_port=self._qdrant_port,
                    qdrant_api_key=self._qdrant_api_key,
                    neo4j_uri=self._neo4j_uri,
                    neo4j_user=self._neo4j_user,
                    neo4j_password=self._neo4j_password,
                    enable_graph=True,
                )
                self._backend = DirectMem0Adapter(mem0_config)
            except ImportError as e:
                raise ImportError(
                    "qdrant-neo4j backend requires additional packages. "
                    "Install with: pip install 'headroom-ai[memory-stack]'\n"
                    "And start Docker services: docker compose up -d qdrant neo4j"
                ) from e
        else:
            raise ValueError(f"Unknown backend: {self._backend_type}")

        self._initialized = True

    async def save(
        self,
        content: str,
        user_id: str,
        importance: float = 0.5,
        facts: list[str] | None = None,
        entities: list[dict[str, str]] | None = None,
        relationships: list[dict[str, str]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Save a memory.

        Args:
            content: The memory content to save.
            user_id: User identifier for scoping memories.
            importance: Importance score 0.0-1.0 (default 0.5).
            facts: Optional pre-extracted facts for better search.
            entities: Optional entities [{"entity": "name", "entity_type": "type"}].
            relationships: Optional relationships [{"source": "a", "relationship": "knows", "destination": "b"}].
            metadata: Optional additional metadata.

        Returns:
            The memory ID.

        Example:
            # Simple save
            memory_id = await memory.save(
                "User prefers dark mode",
                user_id="alice",
            )

            # With pre-extraction for better accuracy
            memory_id = await memory.save(
                "Alice works at Netflix using Python",
                user_id="alice",
                facts=["Alice works at Netflix", "Alice uses Python"],
                entities=[
                    {"entity": "Netflix", "entity_type": "organization"},
                    {"entity": "Python", "entity_type": "technology"},
                ],
                relationships=[
                    {"source": "Alice", "relationship": "works_at", "destination": "Netflix"},
                ],
            )
        """
        await self._ensure_initialized()

        result = await self._backend.save_memory(
            content=content,
            user_id=user_id,
            importance=importance,
            facts=facts,
            extracted_entities=entities,
            extracted_relationships=relationships,
            metadata=metadata,
        )

        return str(result.id)

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        include_graph: bool = True,
    ) -> list[MemoryResult]:
        """Search memories by semantic similarity.

        Args:
            query: Natural language search query.
            user_id: User identifier to scope the search.
            top_k: Maximum number of results (default 10).
            include_graph: Whether to expand results via knowledge graph (default True).

        Returns:
            List of MemoryResult objects sorted by relevance.

        Example:
            results = await memory.search(
                "What programming language does the user prefer?",
                user_id="alice",
            )
            for r in results:
                print(f"{r.score:.2f}: {r.content}")
        """
        await self._ensure_initialized()

        results = await self._backend.search_memories(
            query=query,
            user_id=user_id,
            top_k=top_k,
            include_related=include_graph,
        )

        return [MemoryResult.from_search_result(r) for r in results]

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: The memory ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        await self._ensure_initialized()
        return bool(await self._backend.delete_memory(memory_id))

    async def clear(self, user_id: str) -> int:
        """Clear all memories for a user.

        Args:
            user_id: User identifier.

        Returns:
            Number of memories deleted.
        """
        await self._ensure_initialized()

        if hasattr(self._backend, "clear_user"):
            return int(await self._backend.clear_user(user_id))
        else:
            # Fallback for backends without clear_user
            return 0

    async def close(self) -> None:
        """Close the memory backend and release resources."""
        if self._backend and hasattr(self._backend, "close"):
            await self._backend.close()
        self._initialized = False

    @property
    def backend_type(self) -> str:
        """Get the backend type being used."""
        return self._backend_type

    def __repr__(self) -> str:
        return f"Memory(backend={self._backend_type!r})"
