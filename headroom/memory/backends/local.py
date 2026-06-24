"""Local backend adapter for Headroom's hierarchical memory system.

Provides a fully local memory backend using embedded databases:
- SQLite for memory storage
- SQLite-vec for vector search (bounded, persistent) - preferred
- HNSW for vector search (fallback if sqlite-vec unavailable)
- FTS5 for text search
- SQLite graph for relationships (bounded memory, persistent)

No network calls required, fast startup, suitable for development and
single-process production deployments.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from headroom.memory.adapters.graph_models import Entity, Relationship, Subgraph
from headroom.memory.models import Memory
from headroom.memory.ports import MemorySearchResult
from headroom.models.config import ML_MODEL_DEFAULTS

if TYPE_CHECKING:
    from headroom.memory.adapters.graph import InMemoryGraphStore
    from headroom.memory.adapters.sqlite_graph import SQLiteGraphStore
    from headroom.memory.core import HierarchicalMemory

logger = logging.getLogger(__name__)


@dataclass
class LocalBackendConfig:
    """Configuration for local backend.

    Attributes:
        db_path: Path to the SQLite database file for memories.
        graph_db_path: Path to the SQLite database file for graph. If None,
                      derives from db_path (e.g., "memory.db" -> "memory_graph.db").
        embedder_model: Name of the sentence-transformers model for embeddings.
        vector_dimension: Dimension of embedding vectors (must match embedder model).
        graph_persist: If True, use SQLiteGraphStore (bounded, persistent).
                      If False, use InMemoryGraphStore (unbounded, volatile).
        graph_cache_size_kb: SQLite page cache size for graph store in KB.
                            Higher = more memory, faster queries. Default: 8192 (8MB).
        cache_enabled: Whether to enable memory caching.
        cache_max_size: Maximum number of entries in the cache.
    """

    db_path: str = "memory.db"
    graph_db_path: str | None = None  # Derived from db_path if not specified
    embedder_backend: str = "local"  # "local" (sentence-transformers), "openai", "ollama"
    embedder_model: str = field(default_factory=lambda: ML_MODEL_DEFAULTS.sentence_transformer)
    vector_dimension: int = field(
        default_factory=lambda: ML_MODEL_DEFAULTS.sentence_transformer_dim
    )
    openai_api_key: str | None = None  # Required when embedder_backend="openai"
    ollama_base_url: str = "http://localhost:11434"  # For embedder_backend="ollama"
    graph_persist: bool = True  # Use SQLiteGraphStore (bounded, persistent)
    graph_cache_size_kb: int = 8192  # 8MB default
    cache_enabled: bool = True
    cache_max_size: int = 1000


class LocalBackend:
    """
    Local backend using embedded databases.

    This backend provides a fully local memory system with:
    - SQLite for memory storage (MemoryStore)
    - SQLite-vec for vector search (VectorIndex) - bounded, persistent
    - FTS5 for text search (TextIndex)
    - SQLite graph for relationships (GraphStore) - bounded, persistent

    All operations are performed locally with no network calls,
    making it suitable for:
    - Development and testing
    - Single-process applications
    - Privacy-sensitive deployments
    - Offline operation

    Usage:
        config = LocalBackendConfig(
            db_path="my_memory.db",
            embedder_model="all-MiniLM-L6-v2",
        )
        backend = LocalBackend(config)

        # Save a memory with entities and relationships
        memory = await backend.save_memory(
            content="Alice works at Acme Corp",
            user_id="user123",
            importance=0.8,
            entities=["Alice", "Acme Corp"],
            relationships=[{"source": "Alice", "target": "Acme Corp", "type": "works_at"}],
        )

        # Search with graph expansion
        results = await backend.search_memories(
            query="Where does Alice work?",
            user_id="user123",
            include_related=True,
        )
    """

    def __init__(self, config: LocalBackendConfig | None = None) -> None:
        """Initialize the local backend.

        Args:
            config: Configuration for the backend. Uses defaults if None.
        """
        self._config = config or LocalBackendConfig()
        self._initialized = False
        self._hierarchical_memory: HierarchicalMemory | None = None
        self._graph: InMemoryGraphStore | SQLiteGraphStore | None = None

    async def _ensure_initialized(self) -> None:
        """Ensure the backend is initialized with all components.

        Creates the HierarchicalMemory system and graph store on first use.
        Uses SQLiteGraphStore (bounded, persistent) when graph_persist=True,
        or InMemoryGraphStore (unbounded, volatile) when graph_persist=False.
        """
        if not self._initialized:
            from headroom.memory import HierarchicalMemory, MemoryConfig
            from headroom.memory.config import EmbedderBackend

            # Map string embedder_backend to enum
            embedder_backend_map = {
                "local": EmbedderBackend.LOCAL,
                "onnx": EmbedderBackend.ONNX,
                "openai": EmbedderBackend.OPENAI,
                "ollama": EmbedderBackend.OLLAMA,
            }
            embedder_backend = embedder_backend_map.get(
                self._config.embedder_backend, EmbedderBackend.LOCAL
            )

            mem_config = MemoryConfig(
                db_path=Path(self._config.db_path),
                embedder_backend=embedder_backend,
                embedder_model=self._config.embedder_model,
                vector_dimension=self._config.vector_dimension,
                openai_api_key=self._config.openai_api_key,
                ollama_base_url=self._config.ollama_base_url,
                cache_enabled=self._config.cache_enabled,
                cache_max_size=self._config.cache_max_size,
            )

            self._hierarchical_memory = await HierarchicalMemory.create(mem_config)

            # Choose graph store based on config
            if self._config.graph_persist:
                from headroom.memory.adapters.sqlite_graph import SQLiteGraphStore

                # Derive graph db path from main db path if not specified
                if self._config.graph_db_path:
                    graph_db_path = self._config.graph_db_path
                else:
                    # "memory.db" -> "memory_graph.db"
                    db_path = Path(self._config.db_path)
                    graph_db_path = str(db_path.parent / f"{db_path.stem}_graph{db_path.suffix}")

                self._graph = SQLiteGraphStore(
                    db_path=graph_db_path,
                    page_cache_size_kb=self._config.graph_cache_size_kb,
                )
                logger.info(
                    f"LocalBackend: Using SQLiteGraphStore at {graph_db_path} "
                    f"(cache: {self._config.graph_cache_size_kb}KB)"
                )
            else:
                from headroom.memory.adapters.graph import InMemoryGraphStore

                self._graph = InMemoryGraphStore()
                logger.info("LocalBackend: Using InMemoryGraphStore (unbounded)")

            self._initialized = True

    # =========================================================================
    # Core Memory Operations
    # =========================================================================

    async def save_memory(
        self,
        content: str,
        user_id: str,
        importance: float = 0.5,
        entities: list[str] | None = None,
        relationships: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        turn_id: str | None = None,
        # Pre-extraction fields (for optimized mode)
        facts: list[str] | None = None,
        extracted_entities: list[dict[str, str]] | None = None,
        extracted_relationships: list[dict[str, str]] | None = None,
    ) -> Memory:
        """Save a memory with optional entities and relationships.

        Creates a memory, stores it via HierarchicalMemory, and optionally
        adds entities and relationships to the knowledge graph.

        Supports two modes:
        1. Standard mode: Pass content, entities, and relationships (inferred types)
        2. Pre-extraction mode: Pass facts, extracted_entities, extracted_relationships
           (from optimized tool schema with explicit types)

        Args:
            content: The memory content/text.
            user_id: User identifier (required).
            importance: Importance score (0.0 - 1.0).
            entities: List of entity names to add to the graph (simple format).
            relationships: List of relationship dicts with keys:
                - source: Source entity name
                - target: Target entity name
                - type: Relationship type (e.g., "works_at", "knows")
            metadata: Additional metadata.
            session_id: Optional session identifier.
            agent_id: Optional agent identifier.
            turn_id: Optional turn identifier.
            facts: Pre-extracted discrete facts (optimized mode).
                If provided, each fact is stored as a separate memory.
            extracted_entities: Pre-extracted entities with types (optimized mode).
                Format: [{"entity": "name", "entity_type": "type"}]
            extracted_relationships: Pre-extracted relationships (optimized mode).
                Format: [{"source": "entity1", "relationship": "type", "destination": "entity2"}]

        Returns:
            The created Memory object (for the main content, or first fact if facts provided).
        """
        await self._ensure_initialized()
        assert self._hierarchical_memory is not None
        assert self._graph is not None

        # Determine if using pre-extraction mode
        has_pre_extraction = bool(facts or extracted_entities or extracted_relationships)

        # Merge entity names from both simple and typed formats
        all_entity_names: list[str] = list(entities) if entities else []
        entity_types: dict[str, str] = {}

        if extracted_entities:
            for ent in extracted_entities:
                name = ent.get("entity", "")
                if name and name not in all_entity_names:
                    all_entity_names.append(name)
                if name:
                    entity_types[name.lower()] = ent.get("entity_type", "unknown")

        # Merge relationships from both formats
        all_relationships: list[dict[str, Any]] = list(relationships) if relationships else []

        if extracted_relationships:
            for rel in extracted_relationships:
                # Convert from optimized format to standard format
                all_relationships.append(
                    {
                        "source": rel.get("source", ""),
                        "target": rel.get("destination", ""),  # Note: "destination" in optimized
                        "type": rel.get("relationship", "related_to"),
                    }
                )

        # Prepare base metadata
        base_metadata = metadata or {}
        if has_pre_extraction:
            base_metadata["_pre_extracted"] = True
            if facts:
                base_metadata["_fact_count"] = len(facts)

        # Store memories
        memories_created: list[Memory] = []

        if facts:
            # Store each fact as a separate memory (like DirectMem0Adapter)
            for i, fact in enumerate(facts):
                fact_metadata = {**base_metadata, "_fact_index": i}
                memory = await self._hierarchical_memory.add(
                    content=fact,
                    user_id=user_id,
                    session_id=session_id,
                    agent_id=agent_id,
                    turn_id=turn_id,
                    importance=importance,
                    entity_refs=all_entity_names,
                    metadata=fact_metadata,
                )
                memories_created.append(memory)
        else:
            # Store single memory with content
            memory = await self._hierarchical_memory.add(
                content=content,
                user_id=user_id,
                session_id=session_id,
                agent_id=agent_id,
                turn_id=turn_id,
                importance=importance,
                entity_refs=all_entity_names,
                metadata=base_metadata,
            )
            memories_created.append(memory)

        # Get primary memory (first created) for graph linking
        primary_memory = memories_created[0]

        # Add entities to graph
        if all_entity_names:
            entity_id_map: dict[str, str] = {}

            for entity_name in all_entity_names:
                # Check if entity already exists
                existing = await self._graph.get_entity_by_name(user_id, entity_name)
                if existing:
                    entity_id_map[entity_name.lower()] = existing.id
                else:
                    # Create new entity with type if available
                    entity_type = entity_types.get(entity_name.lower(), "unknown")
                    entity = Entity(
                        id=str(uuid.uuid4()),
                        user_id=user_id,
                        name=entity_name,
                        entity_type=entity_type,
                        metadata={"source_memory_id": primary_memory.id},
                    )
                    await self._graph.add_entity(entity)
                    entity_id_map[entity_name.lower()] = entity.id

            # Add relationships to graph
            if all_relationships:
                for rel in all_relationships:
                    source_name = rel.get("source", "").lower()
                    target_name = rel.get("target", "").lower()
                    rel_type = rel.get("type", "related_to")

                    source_id = entity_id_map.get(source_name)
                    target_id = entity_id_map.get(target_name)

                    if source_id and target_id:
                        relationship = Relationship(
                            id=str(uuid.uuid4()),
                            user_id=user_id,
                            source_id=source_id,
                            target_id=target_id,
                            relation_type=rel_type,
                            metadata={"source_memory_id": primary_memory.id},
                        )
                        await self._graph.add_relationship(relationship)

        # Return primary memory
        return primary_memory

    async def search_memories(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        entities: list[str] | None = None,
        include_related: bool = True,
        min_similarity: float = 0.0,
        session_id: str | None = None,
    ) -> list[MemorySearchResult]:
        """Search memories with optional graph expansion.

        Performs vector search via HierarchicalMemory and optionally
        expands results via the knowledge graph.

        Args:
            query: Natural language search query.
            user_id: User identifier to scope the search.
            top_k: Maximum number of results to return.
            entities: Optional filter by related entities.
            include_related: If True, expand results via knowledge graph.
            min_similarity: Minimum cosine similarity threshold.
            session_id: Optional session filter to isolate memories by session.

        Returns:
            List of MemorySearchResult objects with scores and related entities.
        """
        await self._ensure_initialized()
        assert self._hierarchical_memory is not None
        assert self._graph is not None

        # Perform vector search
        vector_results = await self._hierarchical_memory.search(
            query=query,
            user_id=user_id,
            session_id=session_id,
            top_k=top_k * 2 if include_related else top_k,  # Over-fetch for deduplication
            min_similarity=min_similarity,
        )

        # Convert to MemorySearchResult and collect entity refs
        results: list[MemorySearchResult] = []
        seen_memory_ids: set[str] = set()
        all_entity_refs: set[str] = set()

        for vr in vector_results:
            if vr.memory.id in seen_memory_ids:
                continue

            seen_memory_ids.add(vr.memory.id)
            all_entity_refs.update(vr.memory.entity_refs)

            results.append(
                MemorySearchResult(
                    memory=vr.memory,
                    score=vr.similarity,
                    related_entities=list(vr.memory.entity_refs),
                    related_memories=[],
                )
            )

        # Graph expansion if requested
        if include_related and all_entity_refs:
            # Find entities in graph by name
            entity_ids: list[str] = []
            for entity_name in all_entity_refs:
                entity = await self._graph.get_entity_by_name(user_id, entity_name)
                if entity:
                    entity_ids.append(entity.id)

            if entity_ids:
                # Query subgraph with 1-2 hops
                subgraph = await self._graph.query_subgraph(
                    entity_ids=entity_ids,
                    max_hops=2,
                )

                # Get memory IDs linked to discovered entities
                related_memory_ids: set[str] = set()
                for entity in subgraph.entities:
                    if entity.metadata and "source_memory_id" in entity.metadata:
                        related_memory_ids.add(entity.metadata["source_memory_id"])
                for rel in subgraph.relationships:
                    if rel.metadata and "source_memory_id" in rel.metadata:
                        related_memory_ids.add(rel.metadata["source_memory_id"])

                # Fetch related memories not already in results
                new_memory_ids = related_memory_ids - seen_memory_ids
                for mem_id in new_memory_ids:
                    memory = await self._hierarchical_memory.get(mem_id)
                    if memory and memory.user_id == user_id:
                        # Filter by session_id if specified (security: prevent session leakage)
                        if session_id is not None and memory.session_id != session_id:
                            continue
                        # Add with lower score since it's from graph expansion
                        results.append(
                            MemorySearchResult(
                                memory=memory,
                                score=0.5,  # Default score for graph-expanded results
                                related_entities=list(memory.entity_refs),
                                related_memories=[],
                            )
                        )
                        seen_memory_ids.add(mem_id)

        # Filter by specified entities if provided
        if entities:
            entities_lower = {e.lower() for e in entities}
            results = [
                r
                for r in results
                if any(ref.lower() in entities_lower for ref in r.related_entities)
            ]

        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def update_memory(
        self,
        memory_id: str,
        new_content: str,
        reason: str | None = None,
        user_id: str | None = None,
    ) -> Memory:
        """Update a memory with new content (creates versioned history).

        Uses HierarchicalMemory.supersede() to create a new version while
        preserving the old version for historical queries.

        Args:
            memory_id: ID of the memory to update.
            new_content: The new content for the memory.
            reason: Reason for the update (stored for audit trail).
            user_id: User ID for validation (optional).

        Returns:
            The new Memory that supersedes the old one.

        Raises:
            ValueError: If the memory is not found.
        """
        # Note: reason and user_id are accepted for protocol compliance
        # but not yet used in the underlying implementation
        _ = reason
        _ = user_id
        await self._ensure_initialized()
        assert self._hierarchical_memory is not None

        # Use supersede for versioned updates
        new_memory = await self._hierarchical_memory.supersede(
            old_memory_id=memory_id,
            new_content=new_content,
        )

        return new_memory

    async def delete_memory(
        self,
        memory_id: str,
        reason: str | None = None,
        user_id: str | None = None,
    ) -> bool:
        """Delete a memory.

        Removes the memory from storage and indexes. Also cleans up
        graph references if the memory was the source of entities
        or relationships.

        Args:
            memory_id: ID of the memory to delete.
            reason: Reason for deletion (stored for audit trail).
            user_id: User ID for validation (optional).

        Returns:
            True if the memory was deleted, False if not found.
        """
        # Note: reason and user_id are accepted for protocol compliance
        # but not yet used in the underlying implementation
        _ = reason
        _ = user_id
        await self._ensure_initialized()
        assert self._hierarchical_memory is not None
        assert self._graph is not None

        # Get memory to find its user_id for graph cleanup
        memory = await self._hierarchical_memory.get(memory_id)
        if memory is None:
            return False

        # Delete from HierarchicalMemory
        deleted = await self._hierarchical_memory.delete(memory_id)

        if deleted:
            # Clean up graph references
            # Find and delete entities that were created from this memory
            user_entities = await self._graph.get_entities_for_user(memory.user_id)
            for entity in user_entities:
                if entity.metadata.get("source_memory_id") == memory_id:
                    await self._graph.delete_entity(entity.id)

        return deleted

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Get a memory by ID.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            The Memory if found, None otherwise.
        """
        await self._ensure_initialized()
        assert self._hierarchical_memory is not None

        return await self._hierarchical_memory.get(memory_id)

    # =========================================================================
    # Capability Properties
    # =========================================================================

    @property
    def supports_graph(self) -> bool:
        """Whether this backend supports knowledge graph operations.

        Returns:
            True, as this backend uses SQLiteGraphStore (default) or InMemoryGraphStore.
        """
        return True

    @property
    def supports_vector_search(self) -> bool:
        """Whether this backend supports semantic vector search.

        Returns:
            True, as this backend uses HNSW vector index.
        """
        return True

    @property
    def supports_text_search(self) -> bool:
        """Whether this backend supports full-text search.

        Returns:
            True, as this backend uses FTS5 text index.
        """
        return True

    # =========================================================================
    # Graph Operations
    # =========================================================================

    async def get_graph(self) -> InMemoryGraphStore | SQLiteGraphStore:
        """Get the underlying graph store.

        Returns:
            The graph store instance (SQLiteGraphStore if graph_persist=True,
            InMemoryGraphStore otherwise).
        """
        await self._ensure_initialized()
        assert self._graph is not None
        return self._graph

    async def query_subgraph(
        self,
        entity_names: list[str],
        user_id: str,
        max_hops: int = 2,
    ) -> Subgraph:
        """Query a subgraph starting from named entities.

        Args:
            entity_names: Starting entity names for the traversal.
            user_id: User identifier for entity lookup.
            max_hops: Maximum number of hops from starting entities.

        Returns:
            Subgraph containing reachable entities and relationships.
        """
        await self._ensure_initialized()
        assert self._graph is not None

        # Find entity IDs by name
        entity_ids: list[str] = []
        for name in entity_names:
            entity = await self._graph.get_entity_by_name(user_id, name)
            if entity:
                entity_ids.append(entity.id)

        if not entity_ids:
            return Subgraph(entities=[], relationships=[], root_entity_ids=[])

        return await self._graph.query_subgraph(
            entity_ids=entity_ids,
            max_hops=max_hops,
        )

    # =========================================================================
    # Additional Convenience Methods
    # =========================================================================

    async def get_user_memories(
        self,
        user_id: str,
        limit: int = 100,
    ) -> list[Memory]:
        """Get all memories for a user.

        Args:
            user_id: User identifier.
            limit: Maximum number of memories to return.

        Returns:
            List of memories for the user.
        """
        await self._ensure_initialized()
        assert self._hierarchical_memory is not None

        return await self._hierarchical_memory.get_user_memories(
            user_id=user_id,
            limit=limit,
        )

    async def clear_user(self, user_id: str) -> int:
        """Clear all memories and graph data for a user.

        Args:
            user_id: User identifier.

        Returns:
            Number of memories deleted.
        """
        await self._ensure_initialized()
        assert self._hierarchical_memory is not None
        assert self._graph is not None

        # Clear memories
        count = await self._hierarchical_memory.clear_scope(user_id=user_id)

        # Clear graph
        await self._graph.clear_user(user_id)

        return count

    async def close(self) -> None:
        """Close the backend and release resources."""
        # Close HierarchicalMemory to release httpx clients in embedders
        if self._hierarchical_memory is not None:
            await self._hierarchical_memory.close()
        self._hierarchical_memory = None
        self._graph = None
        self._initialized = False

    # =========================================================================
    # Text Search
    # =========================================================================

    async def text_search(
        self,
        query: str,
        user_id: str,
        limit: int = 100,
    ) -> list[MemorySearchResult]:
        """Full-text search for memories.

        Uses FTS5 index for keyword matching with BM25 ranking.

        Args:
            query: Search query text.
            user_id: User identifier to scope the search.
            limit: Maximum number of results.

        Returns:
            List of MemorySearchResult objects.
        """
        await self._ensure_initialized()
        assert self._hierarchical_memory is not None

        # Build text filter and call text index directly
        # (HierarchicalMemory.text_search has a signature mismatch with the protocol)
        from headroom.memory.ports import TextFilter

        text_filter = TextFilter(
            query=query,
            user_id=user_id,
            limit=limit,
        )

        # Use the protocol-compliant search_memories method on the text index
        text_index = self._hierarchical_memory.text_index
        text_results = await text_index.search_memories(text_filter)  # type: ignore[attr-defined]

        # Convert to MemorySearchResult
        return [
            MemorySearchResult(
                memory=tr.memory,
                score=tr.score,
                related_entities=list(tr.memory.entity_refs),
                related_memories=[],
            )
            for tr in text_results
        ]

    async def hybrid_search(
        self,
        query: str,
        user_id: str,
        top_k: int = 10,
        vector_weight: float = 0.5,
        text_weight: float = 0.5,
        min_similarity: float = 0.0,
    ) -> list[MemorySearchResult]:
        """Hybrid search combining vector similarity and text matching.

        Performs both semantic (vector) and keyword (BM25) search,
        then merges results with weighted score combination.

        Args:
            query: Search query text.
            user_id: User identifier to scope the search.
            top_k: Maximum number of results to return.
            vector_weight: Weight for vector similarity scores (0-1).
            text_weight: Weight for text match scores (0-1).
            min_similarity: Minimum similarity threshold for vector results.

        Returns:
            List of MemorySearchResult objects sorted by combined score.
        """
        await self._ensure_initialized()

        # Fetch more candidates than needed for better coverage
        fetch_k = top_k * 3

        # Perform vector search
        vector_results = await self.search_memories(
            query=query,
            user_id=user_id,
            top_k=fetch_k,
            include_related=False,
            min_similarity=min_similarity,
        )

        # Perform text search
        text_results = await self.text_search(
            query=query,
            user_id=user_id,
            limit=fetch_k,
        )

        # Normalize scores and merge
        # Vector scores are already 0-1 (cosine similarity)
        # Text scores need normalization
        max_text_score = max((r.score for r in text_results), default=1.0) or 1.0

        # Build score maps
        vector_scores: dict[str, float] = {r.memory.id: r.score for r in vector_results}
        text_scores: dict[str, float] = {
            r.memory.id: r.score / max_text_score for r in text_results
        }

        # Merge all unique memories
        all_memories: dict[str, MemorySearchResult] = {}
        for r in vector_results:
            all_memories[r.memory.id] = r
        for r in text_results:
            if r.memory.id not in all_memories:
                all_memories[r.memory.id] = r

        # Calculate combined scores
        combined_results: list[tuple[float, MemorySearchResult]] = []
        for memory_id, result in all_memories.items():
            v_score = vector_scores.get(memory_id, 0.0)
            t_score = text_scores.get(memory_id, 0.0)
            combined = vector_weight * v_score + text_weight * t_score
            combined_results.append((combined, result))

        # Sort by combined score and return top_k
        combined_results.sort(key=lambda x: x[0], reverse=True)
        return [
            MemorySearchResult(
                memory=r.memory,
                score=score,
                related_entities=r.related_entities,
                related_memories=r.related_memories,
            )
            for score, r in combined_results[:top_k]
        ]
