"""Mem0 backend adapter for Headroom's hierarchical memory system.

Provides integration with Mem0's graph and vector memory capabilities:
- Graph database for relationships (Neo4j)
- Vector database for semantic search (Qdrant)
- Automatic entity extraction
- Relationship inference

Supports both local mode (embedded services) and cloud mode (Mem0 API).
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from headroom.memory import qdrant_env
from headroom.memory.models import Memory
from headroom.memory.ports import MemoryFilter, VectorFilter, VectorSearchResult


@dataclass
class Mem0Config:
    """Configuration for Mem0 backend.

    Qdrant connection fields default to values read from ``HEADROOM_QDRANT_*``
    environment variables (see :mod:`headroom.memory.qdrant_env`). Passing an
    explicit value to the constructor always wins over the environment.

    Attributes:
        mode: Operating mode - "local" for embedded or "cloud" for Mem0 API.
        api_key: API key for Mem0 cloud mode.
        neo4j_uri: Neo4j connection URI for local mode.
        neo4j_user: Neo4j username for local mode.
        neo4j_password: Neo4j password for local mode.
        qdrant_url: Full Qdrant URL (e.g. ``https://xyz.cloud.qdrant.io:6333``).
            When set, takes precedence over ``qdrant_host``/``qdrant_port``.
        qdrant_host: Qdrant host for local mode.
        qdrant_port: Qdrant port for local mode.
        qdrant_api_key: API key for hosted Qdrant (e.g. Qdrant Cloud).
        qdrant_https: Force HTTPS on/off. ``None`` lets the Qdrant client decide.
        qdrant_prefer_grpc: Use gRPC transport instead of HTTP.
        qdrant_grpc_port: gRPC port (only used when ``qdrant_prefer_grpc`` is True).
        llm_model: LLM model for entity extraction.
        embedder_model: Embedding model for vector search.
        collection_name: Name of the collection/namespace in Mem0.
    """

    mode: str = "local"  # "local" or "cloud"

    # Cloud mode settings
    api_key: str | None = None

    # Local mode settings - Neo4j and Qdrant config
    neo4j_uri: str = "neo4j://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    # Qdrant settings (defaults resolve from HEADROOM_QDRANT_* env vars)
    qdrant_url: str | None = field(default_factory=qdrant_env.qdrant_env_url)
    qdrant_host: str = field(default_factory=qdrant_env.qdrant_env_host)
    qdrant_port: int = field(default_factory=qdrant_env.qdrant_env_port)
    qdrant_api_key: str | None = field(default_factory=qdrant_env.qdrant_env_api_key)
    qdrant_https: bool | None = field(default_factory=qdrant_env.qdrant_env_https)
    qdrant_prefer_grpc: bool = field(default_factory=qdrant_env.qdrant_env_prefer_grpc)
    qdrant_grpc_port: int = field(default_factory=qdrant_env.qdrant_env_grpc_port)

    # Common settings
    llm_model: str = "gpt-4o-mini"  # For entity extraction
    embedder_model: str = "text-embedding-3-small"
    enable_graph: bool = True  # Set to False to disable graph storage (vector-only)

    # Collection settings
    collection_name: str = "headroom_memories"


class Mem0Backend:
    """
    Mem0 backend implementation for Headroom memory system.

    Mem0 provides:
    - Graph database for relationships (Neo4j)
    - Vector database for semantic search (Qdrant)
    - Automatic entity extraction
    - Relationship inference

    This adapter maps Mem0's API to Headroom's MemoryBackend interface:
    - mem0.add() -> save_memory()
    - mem0.search() -> search_memories()
    - mem0.update() -> update_memory()
    - mem0.delete() -> delete_memory()
    - mem0.get() -> get_memory()

    Usage:
        config = Mem0Config(mode="local")
        backend = Mem0Backend(config)
        await backend.save_memory(memory)
        results = await backend.search_memories(query="user preferences")
    """

    def __init__(self, config: Mem0Config | None = None) -> None:
        """Initialize the Mem0 backend.

        Args:
            config: Configuration for Mem0. If None, uses default local config.
        """
        self._config = config or Mem0Config()
        self._client: Any = None
        self._initialized = False

    async def _ensure_client(self) -> Any:
        """Ensure Mem0 client is initialized.

        Returns:
            The initialized Mem0 Memory client.

        Raises:
            ImportError: If mem0 package is not installed.
        """
        if self._client is None:
            try:
                from mem0 import Memory as Mem0Memory
            except ImportError:
                raise ImportError(
                    "mem0 package not installed. Install with: pip install 'headroom-ai[memory-stack]'"
                ) from None

            if self._config.mode == "cloud":
                if not self._config.api_key:
                    raise ValueError("api_key is required for cloud mode")
                # Cloud mode - use API key
                self._client = await asyncio.to_thread(Mem0Memory, api_key=self._config.api_key)
            else:
                # Local mode with configuration
                qdrant_provider_cfg: dict[str, Any] = {
                    "collection_name": self._config.collection_name,
                }
                if self._config.qdrant_url:
                    qdrant_provider_cfg["url"] = self._config.qdrant_url
                else:
                    qdrant_provider_cfg["host"] = self._config.qdrant_host
                    qdrant_provider_cfg["port"] = self._config.qdrant_port
                if self._config.qdrant_api_key:
                    qdrant_provider_cfg["api_key"] = self._config.qdrant_api_key

                config: dict[str, Any] = {
                    "vector_store": {
                        "provider": "qdrant",
                        "config": qdrant_provider_cfg,
                    },
                    "llm": {
                        "provider": "openai",
                        "config": {
                            "model": self._config.llm_model,
                        },
                    },
                    "embedder": {
                        "provider": "openai",
                        "config": {
                            "model": self._config.embedder_model,
                        },
                    },
                }

                # Optionally enable graph storage (Neo4j)
                if self._config.enable_graph:
                    config["graph_store"] = {
                        "provider": "neo4j",
                        "config": {
                            "url": self._config.neo4j_uri,
                            "username": self._config.neo4j_user,
                            "password": self._config.neo4j_password,
                        },
                    }

                self._client = await asyncio.to_thread(Mem0Memory.from_config, config)

            self._initialized = True

        return self._client

    def _build_mem0_metadata(self, memory: Memory) -> dict[str, Any]:
        """Convert Memory object to Mem0 metadata dict.

        Args:
            memory: The Memory object to convert.

        Returns:
            Dict of metadata for Mem0.
        """
        metadata = {
            "headroom_id": memory.id,
            "user_id": memory.user_id,
            "importance": memory.importance,
            "created_at": memory.created_at.isoformat(),
            "valid_from": memory.valid_from.isoformat(),
            "access_count": memory.access_count,
            "entity_refs": memory.entity_refs,
        }

        if memory.session_id:
            metadata["session_id"] = memory.session_id
        if memory.agent_id:
            metadata["agent_id"] = memory.agent_id
        if memory.turn_id:
            metadata["turn_id"] = memory.turn_id
        if memory.valid_until:
            metadata["valid_until"] = memory.valid_until.isoformat()
        if memory.supersedes:
            metadata["supersedes"] = memory.supersedes
        if memory.superseded_by:
            metadata["superseded_by"] = memory.superseded_by
        if memory.promoted_from:
            metadata["promoted_from"] = memory.promoted_from
        if memory.promotion_chain:
            metadata["promotion_chain"] = memory.promotion_chain
        if memory.last_accessed:
            metadata["last_accessed"] = memory.last_accessed.isoformat()
        if memory.metadata:
            metadata["custom_metadata"] = memory.metadata

        return metadata

    def _mem0_result_to_memory(self, result: dict[str, Any]) -> Memory:
        """Convert Mem0 result dict to Memory object.

        Args:
            result: The Mem0 result dict.

        Returns:
            A Memory object.
        """
        metadata = result.get("metadata", {})

        # Extract fields from metadata, with defaults
        memory_id = metadata.get("headroom_id", result.get("id", str(uuid.uuid4())))
        user_id = metadata.get("user_id", "")
        importance = metadata.get("importance", 0.5)

        # Parse timestamps
        created_at_str = metadata.get("created_at")
        created_at = (
            datetime.fromisoformat(created_at_str)
            if created_at_str
            else datetime.now(timezone.utc).replace(tzinfo=None)
        )

        valid_from_str = metadata.get("valid_from")
        valid_from = (
            datetime.fromisoformat(valid_from_str)
            if valid_from_str
            else datetime.now(timezone.utc).replace(tzinfo=None)
        )

        valid_until_str = metadata.get("valid_until")
        valid_until = datetime.fromisoformat(valid_until_str) if valid_until_str else None

        last_accessed_str = metadata.get("last_accessed")
        last_accessed = datetime.fromisoformat(last_accessed_str) if last_accessed_str else None

        # Get content from Mem0 result
        content = result.get("memory", result.get("content", ""))

        return Memory(
            id=memory_id,
            content=content,
            user_id=user_id,
            session_id=metadata.get("session_id"),
            agent_id=metadata.get("agent_id"),
            turn_id=metadata.get("turn_id"),
            created_at=created_at,
            valid_from=valid_from,
            valid_until=valid_until,
            importance=importance,
            supersedes=metadata.get("supersedes"),
            superseded_by=metadata.get("superseded_by"),
            promoted_from=metadata.get("promoted_from"),
            promotion_chain=metadata.get("promotion_chain", []),
            access_count=metadata.get("access_count", 0),
            last_accessed=last_accessed,
            entity_refs=metadata.get("entity_refs", []),
            embedding=None,  # Mem0 manages embeddings internally
            metadata=metadata.get("custom_metadata", {}),
        )

    def _build_mem0_filters(self, filter: MemoryFilter | VectorFilter) -> dict[str, Any]:
        """Build Mem0 filter dict from MemoryFilter or VectorFilter.

        Args:
            filter: The filter to convert.

        Returns:
            Dict of filters for Mem0 search.
        """
        filters: dict[str, Any] = {}

        if filter.user_id:
            filters["user_id"] = filter.user_id
        if hasattr(filter, "session_id") and filter.session_id:
            filters["session_id"] = filter.session_id
        if hasattr(filter, "agent_id") and filter.agent_id:
            filters["agent_id"] = filter.agent_id

        return filters

    async def save_memory(self, memory: Memory) -> str:
        """Save a memory to Mem0.

        Maps to mem0.add().

        Args:
            memory: The memory to save.

        Returns:
            The memory ID (may be Mem0's ID or our original ID).
        """
        client = await self._ensure_client()
        metadata = self._build_mem0_metadata(memory)

        # Mem0's add method takes messages or data
        # We'll pass the content and metadata
        result = await asyncio.to_thread(
            client.add,
            memory.content,
            user_id=memory.user_id,
            metadata=metadata,
        )

        # Mem0 returns a list of results from add operation
        if isinstance(result, dict) and "results" in result:
            results = result["results"]
            if results and len(results) > 0:
                return str(results[0].get("id", memory.id))
        elif isinstance(result, list) and len(result) > 0:
            return str(result[0].get("id", memory.id))

        return memory.id

    async def save_memory_batch(self, memories: list[Memory]) -> list[str]:
        """Save multiple memories to Mem0.

        Args:
            memories: List of memories to save.

        Returns:
            List of memory IDs.
        """
        ids = []
        for memory in memories:
            mem_id = await self.save_memory(memory)
            ids.append(mem_id)
        return ids

    async def search_memories(
        self,
        query: str,
        user_id: str | None = None,
        filter: VectorFilter | None = None,
        limit: int = 10,
    ) -> list[VectorSearchResult]:
        """Search for memories using semantic search.

        Maps to mem0.search().

        Args:
            query: The search query text.
            user_id: Optional user ID to scope the search.
            filter: Optional VectorFilter for additional filtering.
            limit: Maximum number of results to return.

        Returns:
            List of VectorSearchResult objects.
        """
        client = await self._ensure_client()

        # Build search kwargs
        search_kwargs: dict[str, Any] = {
            "query": query,
            "limit": limit,
        }

        if user_id:
            search_kwargs["user_id"] = user_id
        elif filter and filter.user_id:
            search_kwargs["user_id"] = filter.user_id

        # Add filters if provided
        if filter:
            filters = self._build_mem0_filters(filter)
            if filters:
                search_kwargs["filters"] = filters

        # Execute search
        results = await asyncio.to_thread(client.search, **search_kwargs)

        # Convert results to VectorSearchResult objects
        search_results: list[VectorSearchResult] = []

        # Handle different response formats from Mem0
        result_list = results if isinstance(results, list) else results.get("results", [])

        for rank, result in enumerate(result_list, start=1):
            memory = self._mem0_result_to_memory(result)
            similarity = result.get("score", result.get("similarity", 0.0))

            search_results.append(
                VectorSearchResult(
                    memory=memory,
                    similarity=float(similarity),
                    rank=rank,
                )
            )

        return search_results

    async def update_memory(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Update a memory in Mem0.

        Maps to mem0.update().

        Args:
            memory_id: The ID of the memory to update.
            content: New content for the memory.
            metadata: New metadata to merge.

        Returns:
            True if the memory was updated, False if not found.
        """
        client = await self._ensure_client()

        try:
            update_kwargs: dict[str, Any] = {"memory_id": memory_id}
            if content:
                update_kwargs["data"] = content
            if metadata:
                update_kwargs["metadata"] = metadata

            await asyncio.to_thread(client.update, **update_kwargs)
            return True
        except Exception:
            # Mem0 may raise an exception if memory not found
            return False

    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory from Mem0.

        Maps to mem0.delete().

        Args:
            memory_id: The ID of the memory to delete.

        Returns:
            True if the memory was deleted, False if not found.
        """
        client = await self._ensure_client()

        try:
            await asyncio.to_thread(client.delete, memory_id=memory_id)
            return True
        except Exception:
            return False

    async def delete_memory_batch(self, memory_ids: list[str]) -> int:
        """Delete multiple memories from Mem0.

        Args:
            memory_ids: List of memory IDs to delete.

        Returns:
            Number of memories deleted.
        """
        deleted = 0
        for memory_id in memory_ids:
            if await self.delete_memory(memory_id):
                deleted += 1
        return deleted

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Get a memory by ID from Mem0.

        Maps to mem0.get().

        Args:
            memory_id: The ID of the memory to retrieve.

        Returns:
            The Memory object if found, None otherwise.
        """
        client = await self._ensure_client()

        try:
            result = await asyncio.to_thread(client.get, memory_id=memory_id)

            if result is None:
                return None

            # Handle different response formats
            if isinstance(result, dict):
                return self._mem0_result_to_memory(result)
            elif isinstance(result, list) and len(result) > 0:
                return self._mem0_result_to_memory(result[0])

            return None
        except Exception:
            return None

    async def get_all_memories(
        self,
        user_id: str,
        limit: int = 100,
    ) -> list[Memory]:
        """Get all memories for a user.

        Maps to mem0.get_all().

        Args:
            user_id: The user ID to get memories for.
            limit: Maximum number of memories to return.

        Returns:
            List of Memory objects.
        """
        client = await self._ensure_client()

        try:
            results = await asyncio.to_thread(client.get_all, user_id=user_id, limit=limit)

            # Handle different response formats
            result_list = results if isinstance(results, list) else results.get("results", [])

            return [self._mem0_result_to_memory(r) for r in result_list]
        except Exception:
            return []

    async def query(self, filter: MemoryFilter) -> list[Memory]:
        """Query memories using a MemoryFilter.

        This maps the MemoryFilter to Mem0's search capabilities.

        Args:
            filter: The filter criteria.

        Returns:
            List of matching Memory objects.
        """
        client = await self._ensure_client()

        # If no user_id, we can't query Mem0 effectively
        if not filter.user_id:
            return []

        try:
            # Get all memories for the user and filter locally
            # Mem0 doesn't support all our filter options natively
            results = await asyncio.to_thread(
                client.get_all,
                user_id=filter.user_id,
                limit=filter.limit or 100,
            )

            result_list = results if isinstance(results, list) else results.get("results", [])

            memories = [self._mem0_result_to_memory(r) for r in result_list]

            # Apply local filtering for fields Mem0 doesn't support
            filtered = []
            for memory in memories:
                # Session filter
                if filter.session_id and memory.session_id != filter.session_id:
                    continue

                # Agent filter
                if filter.agent_id and memory.agent_id != filter.agent_id:
                    continue

                # Turn filter
                if filter.turn_id and memory.turn_id != filter.turn_id:
                    continue

                # Importance filters
                if filter.min_importance is not None and memory.importance < filter.min_importance:
                    continue
                if filter.max_importance is not None and memory.importance > filter.max_importance:
                    continue

                # Temporal filters
                if filter.created_after is not None and memory.created_at < filter.created_after:
                    continue
                if filter.created_before is not None and memory.created_at > filter.created_before:
                    continue

                # Superseded filter
                if not filter.include_superseded and memory.valid_until is not None:
                    continue

                # Entity refs filter
                if filter.entity_refs:
                    if not any(ref in memory.entity_refs for ref in filter.entity_refs):
                        continue

                filtered.append(memory)

            # Apply sorting
            if filter.order_by == "importance":
                filtered.sort(key=lambda m: m.importance, reverse=filter.order_desc)
            elif filter.order_by == "access_count":
                filtered.sort(key=lambda m: m.access_count, reverse=filter.order_desc)
            elif filter.order_by == "last_accessed":
                filtered.sort(
                    key=lambda m: m.last_accessed or datetime.min,
                    reverse=filter.order_desc,
                )
            else:  # created_at
                filtered.sort(key=lambda m: m.created_at, reverse=filter.order_desc)

            # Apply offset and limit
            start = filter.offset
            end = start + filter.limit if filter.limit else None
            return filtered[start:end]

        except Exception:
            return []

    def supports_graph(self) -> bool:
        """Check if this backend supports graph operations.

        Returns:
            True, as Mem0 uses Neo4j for graph storage.
        """
        return True

    def supports_vector_search(self) -> bool:
        """Check if this backend supports vector search.

        Returns:
            True, as Mem0 uses Qdrant for vector search.
        """
        return True

    async def get_related_memories(
        self,
        memory_id: str,
        limit: int = 10,
    ) -> list[Memory]:
        """Get memories related to the given memory via graph relationships.

        This leverages Mem0's graph capabilities to find related memories
        based on extracted entities and relationships.

        Args:
            memory_id: The ID of the memory to find relations for.
            limit: Maximum number of related memories to return.

        Returns:
            List of related Memory objects.
        """
        await self._ensure_client()

        # First get the memory to find its user_id
        memory = await self.get_memory(memory_id)
        if not memory:
            return []

        try:
            # Search for related memories using the memory content as query
            results = await self.search_memories(
                query=memory.content,
                user_id=memory.user_id,
                limit=limit + 1,  # +1 to account for the original memory
            )

            # Filter out the original memory
            related = [r.memory for r in results if r.memory.id != memory_id]
            return related[:limit]
        except Exception:
            return []

    async def close(self) -> None:
        """Close the Mem0 client and release resources."""
        self._client = None
        self._initialized = False
