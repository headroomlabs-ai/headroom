"""Protocol interfaces for pluggable memory system components."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from headroom.memory.models import Memory, ScopeLevel

if TYPE_CHECKING:
    import numpy as np

# =============================================================================
# Filter Dataclasses
# =============================================================================


@dataclass
class MemoryFilter:
    """Filter criteria for memory store queries."""

    # Scope filters
    user_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    turn_id: str | None = None
    scope_levels: list[ScopeLevel] | None = None

    # Temporal filters
    created_after: datetime | None = None
    created_before: datetime | None = None
    valid_at: datetime | None = None  # Point-in-time query
    include_superseded: bool = False  # Include historical versions

    # Importance filters
    min_importance: float | None = None
    max_importance: float | None = None

    # Entity filters
    entity_refs: list[str] | None = None  # Any of these entities

    # Lineage filters
    has_supersedes: bool | None = None
    has_promoted_from: bool | None = None

    # Pagination
    limit: int | None = None
    offset: int = 0

    # Sorting
    order_by: str = "created_at"  # created_at, importance, access_count, last_accessed
    order_desc: bool = True

    # Metadata filters
    metadata_filters: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorFilter:
    """Filter criteria for vector similarity searches."""

    # Required: query vector or text (one must be provided)
    query_vector: np.ndarray | None = None
    query_text: str | None = None  # Will be embedded if vector not provided

    # Search parameters
    top_k: int = 10
    min_similarity: float = 0.0  # Minimum cosine similarity threshold

    # Scope filters (inherited from MemoryFilter)
    user_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    scope_levels: list[ScopeLevel] | None = None

    # Temporal filters
    valid_at: datetime | None = None
    include_superseded: bool = False

    # Entity filters
    entity_refs: list[str] | None = None

    # Metadata filters
    metadata_filters: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextFilter:
    """Filter criteria for full-text searches."""

    # Required: search query
    query: str = ""

    # Search mode
    match_mode: str = "contains"  # contains, prefix, exact, fuzzy, regex
    case_sensitive: bool = False

    # Result parameters
    limit: int = 100

    # Scope filters (inherited from MemoryFilter)
    user_id: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    scope_levels: list[ScopeLevel] | None = None

    # Temporal filters
    valid_at: datetime | None = None
    include_superseded: bool = False

    # Metadata filters
    metadata_filters: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Search Result Dataclasses
# =============================================================================


@dataclass
class VectorSearchResult:
    """Result from a vector similarity search."""

    memory: Memory
    similarity: float  # Cosine similarity score (0.0 - 1.0)
    rank: int  # Position in results (1-indexed)

    def __lt__(self, other: VectorSearchResult) -> bool:
        """Enable sorting by similarity (descending)."""
        return self.similarity > other.similarity


@dataclass
class TextSearchResult:
    """Result from a full-text search."""

    memory: Memory
    score: float  # Relevance score (implementation-specific)
    rank: int  # Position in results (1-indexed)
    highlights: list[str] = field(default_factory=list)  # Matching snippets
    matched_terms: list[str] = field(default_factory=list)  # Terms that matched

    def __lt__(self, other: TextSearchResult) -> bool:
        """Enable sorting by score (descending)."""
        return self.score > other.score


# =============================================================================
# Graph Entity Dataclasses
# =============================================================================


@dataclass
class Entity:
    """Represents an entity node in the knowledge graph."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    entity_type: str = ""
    user_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Relationship:
    """Represents a directed relationship between two entities in the knowledge graph."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_entity_id: str = ""
    target_entity_id: str = ""
    relation_type: str = ""
    user_id: str = ""
    memory_id: str | None = None  # Optional link to a Memory that sourced this relationship
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Subgraph:
    """A subset of the knowledge graph containing entities and their relationships."""

    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)

    def to_context(self) -> str:
        """
        Convert the subgraph to a text representation suitable for LLM context.

        Returns:
            A formatted string describing the entities and their relationships.
        """
        if not self.entities and not self.relationships:
            return ""

        lines: list[str] = []

        # Build entity lookup for relationship formatting
        entity_map = {e.id: e for e in self.entities}

        # Format entities
        if self.entities:
            lines.append("Entities:")
            for entity in self.entities:
                entity_line = f"  - {entity.name} ({entity.entity_type})"
                if entity.metadata:
                    meta_str = ", ".join(f"{k}={v}" for k, v in entity.metadata.items())
                    entity_line += f" [{meta_str}]"
                lines.append(entity_line)

        # Format relationships
        if self.relationships:
            lines.append("")
            lines.append("Relationships:")
            for rel in self.relationships:
                source_name = entity_map.get(
                    rel.source_entity_id, Entity(name=rel.source_entity_id)
                ).name
                target_name = entity_map.get(
                    rel.target_entity_id, Entity(name=rel.target_entity_id)
                ).name
                rel_line = f"  - {source_name} --[{rel.relation_type}]--> {target_name}"
                if rel.weight != 1.0:
                    rel_line += f" (weight={rel.weight})"
                lines.append(rel_line)

        return "\n".join(lines)


@dataclass
class MemorySearchResult:
    """Unified search result combining memory with graph context."""

    memory: Memory
    score: float
    related_entities: list[str] = field(default_factory=list)
    related_memories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "memory_id": self.memory.id,
            "content": self.memory.content,
            "importance": self.memory.importance,
            "entities": self.memory.entity_refs,
            "created_at": self.memory.created_at.isoformat(),
            "score": self.score,
            "related_entities": self.related_entities,
            "related_memories": self.related_memories,
        }


# =============================================================================
# Protocol Interfaces
# =============================================================================


@runtime_checkable
class MemoryStore(Protocol):
    """
    Protocol for memory persistence backends.

    Implementations handle CRUD operations and filtering for Memory objects.
    Examples: SQLite, PostgreSQL, DynamoDB, Redis, in-memory.
    """

    async def save(self, memory: Memory) -> None:
        """
        Save or update a memory.

        If a memory with the same ID exists, it will be updated.

        Args:
            memory: The memory to save.
        """
        ...

    async def save_batch(self, memories: list[Memory]) -> None:
        """
        Save multiple memories in a single operation.

        Args:
            memories: List of memories to save.
        """
        ...

    async def get(self, memory_id: str) -> Memory | None:
        """
        Retrieve a memory by ID.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            The memory if found, None otherwise.
        """
        ...

    async def get_batch(self, memory_ids: list[str]) -> list[Memory]:
        """
        Retrieve multiple memories by ID.

        Args:
            memory_ids: List of memory IDs to retrieve.

        Returns:
            List of found memories (may be shorter than input if some not found).
        """
        ...

    async def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            True if the memory was deleted, False if not found.
        """
        ...

    async def delete_batch(self, memory_ids: list[str]) -> int:
        """
        Delete multiple memories by ID.

        Args:
            memory_ids: List of memory IDs to delete.

        Returns:
            Number of memories actually deleted.
        """
        ...

    async def query(self, filter: MemoryFilter) -> list[Memory]:
        """
        Query memories matching the given filter.

        Args:
            filter: Filter criteria for the query.

        Returns:
            List of matching memories.
        """
        ...

    async def count(self, filter: MemoryFilter) -> int:
        """
        Count memories matching the given filter.

        Args:
            filter: Filter criteria for the count.

        Returns:
            Number of matching memories.
        """
        ...

    async def supersede(
        self,
        old_memory_id: str,
        new_memory: Memory,
        supersede_time: datetime | None = None,
    ) -> Memory:
        """
        Supersede an existing memory with a new version.

        This creates a temporal chain: the old memory's valid_until is set,
        and the new memory's supersedes field points to the old one.

        Args:
            old_memory_id: ID of the memory to supersede.
            new_memory: The new memory that replaces it.
            supersede_time: When the supersession occurred (defaults to now).

        Returns:
            The saved new memory with lineage fields populated.
        """
        ...

    async def get_history(
        self,
        memory_id: str,
        include_future: bool = False,
    ) -> list[Memory]:
        """
        Get the full history chain for a memory.

        Follows the supersedes/superseded_by chain to return all versions.

        Args:
            memory_id: ID of any memory in the chain.
            include_future: Whether to include memories that superseded this one.

        Returns:
            List of memories in temporal order (oldest first).
        """
        ...

    async def clear_scope(
        self,
        user_id: str,
        session_id: str | None = None,
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> int:
        """
        Clear all memories at or below a scope level.

        Args:
            user_id: Required user scope.
            session_id: If provided, clear session and below.
            agent_id: If provided, clear agent and below.
            turn_id: If provided, clear only that turn.

        Returns:
            Number of memories deleted.
        """
        ...


@runtime_checkable
class VectorIndex(Protocol):
    """
    Protocol for vector similarity search backends.

    Implementations handle embedding storage and similarity search.
    Examples: FAISS, Annoy, Pinecone, Weaviate, Qdrant.
    """

    async def index(self, memory: Memory) -> None:
        """
        Index a memory's embedding for similarity search.

        The memory must have an embedding set.

        Args:
            memory: The memory to index.

        Raises:
            ValueError: If the memory has no embedding.
        """
        ...

    async def index_batch(self, memories: list[Memory]) -> int:
        """
        Index multiple memories' embeddings.

        Memories without embeddings are skipped.

        Args:
            memories: List of memories to index.

        Returns:
            Number of memories actually indexed.
        """
        ...

    async def remove(self, memory_id: str) -> bool:
        """
        Remove a memory from the vector index.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            True if removed, False if not found.
        """
        ...

    async def remove_batch(self, memory_ids: list[str]) -> int:
        """
        Remove multiple memories from the vector index.

        Args:
            memory_ids: List of memory IDs to remove.

        Returns:
            Number of memories actually removed.
        """
        ...

    async def search(self, filter: VectorFilter) -> list[VectorSearchResult]:
        """
        Search for similar memories using vector similarity.

        Args:
            filter: Vector search filter with query and constraints.

        Returns:
            List of search results sorted by similarity (descending).
        """
        ...

    async def update_embedding(self, memory_id: str, embedding: np.ndarray) -> bool:
        """
        Update the embedding for an indexed memory.

        Args:
            memory_id: The unique identifier of the memory.
            embedding: The new embedding vector.

        Returns:
            True if updated, False if memory not found in index.
        """
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension this index expects."""
        ...

    @property
    def size(self) -> int:
        """Return the number of vectors currently indexed."""
        ...


@runtime_checkable
class TextIndex(Protocol):
    """
    Protocol for full-text search backends.

    Implementations handle text indexing and keyword search.
    Examples: SQLite FTS5, Elasticsearch, Tantivy, in-memory.
    """

    async def index(self, memory: Memory) -> None:
        """
        Index a memory's content for full-text search.

        Args:
            memory: The memory to index.
        """
        ...

    async def index_batch(self, memories: list[Memory]) -> int:
        """
        Index multiple memories for full-text search.

        Args:
            memories: List of memories to index.

        Returns:
            Number of memories actually indexed.
        """
        ...

    async def remove(self, memory_id: str) -> bool:
        """
        Remove a memory from the text index.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            True if removed, False if not found.
        """
        ...

    async def remove_batch(self, memory_ids: list[str]) -> int:
        """
        Remove multiple memories from the text index.

        Args:
            memory_ids: List of memory IDs to remove.

        Returns:
            Number of memories actually removed.
        """
        ...

    async def search(self, filter: TextFilter) -> list[TextSearchResult]:
        """
        Search for memories using full-text search.

        Args:
            filter: Text search filter with query and constraints.

        Returns:
            List of search results sorted by relevance.
        """
        ...

    async def update_content(self, memory_id: str, content: str) -> bool:
        """
        Update the indexed content for a memory.

        Args:
            memory_id: The unique identifier of the memory.
            content: The new content to index.

        Returns:
            True if updated, False if memory not found in index.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """
    Protocol for text embedding generation.

    Implementations convert text to dense vector representations.
    Examples: OpenAI embeddings, sentence-transformers, Cohere.
    """

    async def embed(self, text: str) -> np.ndarray:
        """
        Generate an embedding for a single text.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector as a numpy array.
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        ...

    @property
    def dimension(self) -> int:
        """Return the dimension of generated embeddings."""
        ...

    @property
    def model_name(self) -> str:
        """Return the name/identifier of the embedding model."""
        ...

    @property
    def max_tokens(self) -> int:
        """Return the maximum number of tokens the model can process."""
        ...


@runtime_checkable
class MemoryCache(Protocol):
    """
    Protocol for memory caching layer.

    Implementations provide fast access to frequently-used memories.
    Examples: LRU cache, Redis, in-memory dict with TTL.
    """

    async def get(self, memory_id: str) -> Memory | None:
        """
        Get a memory from cache.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            The cached memory if found, None otherwise.
        """
        ...

    async def get_batch(self, memory_ids: list[str]) -> dict[str, Memory]:
        """
        Get multiple memories from cache.

        Args:
            memory_ids: List of memory IDs to retrieve.

        Returns:
            Dict mapping found memory IDs to their memories.
        """
        ...

    async def put(self, memory: Memory, ttl_seconds: int | None = None) -> None:
        """
        Put a memory in cache.

        Args:
            memory: The memory to cache.
            ttl_seconds: Optional time-to-live in seconds.
        """
        ...

    async def put_batch(
        self,
        memories: list[Memory],
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Put multiple memories in cache.

        Args:
            memories: List of memories to cache.
            ttl_seconds: Optional time-to-live in seconds.
        """
        ...

    async def invalidate(self, memory_id: str) -> bool:
        """
        Invalidate (remove) a memory from cache.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            True if the memory was in cache, False otherwise.
        """
        ...

    async def invalidate_batch(self, memory_ids: list[str]) -> int:
        """
        Invalidate multiple memories from cache.

        Args:
            memory_ids: List of memory IDs to invalidate.

        Returns:
            Number of memories that were in cache.
        """
        ...

    async def invalidate_scope(
        self,
        user_id: str,
        session_id: str | None = None,
        agent_id: str | None = None,
    ) -> int:
        """
        Invalidate all cached memories at or below a scope.

        Args:
            user_id: Required user scope.
            session_id: If provided, invalidate session and below.
            agent_id: If provided, invalidate agent and below.

        Returns:
            Number of memories invalidated.
        """
        ...

    async def clear(self) -> None:
        """Clear all entries from the cache."""
        ...

    @property
    def size(self) -> int:
        """Return the current number of cached entries."""
        ...

    @property
    def max_size(self) -> int | None:
        """Return the maximum cache size, or None if unbounded."""
        ...


@runtime_checkable
class GraphStore(Protocol):
    """
    Protocol for knowledge graph storage backends.

    Implementations handle entity and relationship storage and graph traversal.
    Examples: Neo4j, NetworkX, SQLite with adjacency tables, in-memory.
    """

    async def add_entity(self, entity: Entity) -> None:
        """
        Add an entity to the graph.

        If an entity with the same ID exists, it will be updated.

        Args:
            entity: The entity to add.
        """
        ...

    async def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship between two entities.

        If a relationship with the same ID exists, it will be updated.

        Args:
            relationship: The relationship to add.
        """
        ...

    async def get_entity(self, entity_id: str) -> Entity | None:
        """
        Retrieve an entity by ID.

        Args:
            entity_id: The unique identifier of the entity.

        Returns:
            The entity if found, None otherwise.
        """
        ...

    async def get_entity_by_name(
        self,
        name: str,
        user_id: str,
        entity_type: str | None = None,
    ) -> Entity | None:
        """
        Retrieve an entity by name within a user's graph.

        Args:
            name: The name of the entity.
            user_id: The user scope for the lookup.
            entity_type: Optional entity type filter.

        Returns:
            The entity if found, None otherwise.
        """
        ...

    async def get_relationships(
        self,
        entity_id: str,
        relation_types: list[str] | None = None,
        direction: str = "both",
    ) -> list[Relationship]:
        """
        Get relationships connected to an entity.

        Args:
            entity_id: The entity to get relationships for.
            relation_types: Optional filter for specific relationship types.
            direction: "outgoing", "incoming", or "both" (default).

        Returns:
            List of relationships matching the criteria.
        """
        ...

    async def query_subgraph(
        self,
        entity_ids: list[str],
        hops: int = 1,
        relation_types: list[str] | None = None,
    ) -> Subgraph:
        """
        Extract a subgraph around the given entities.

        Args:
            entity_ids: Starting entity IDs for the subgraph extraction.
            hops: Number of relationship hops to traverse (default 1).
            relation_types: Optional filter for specific relationship types.

        Returns:
            A Subgraph containing the entities and relationships within the specified hops.
        """
        ...

    async def find_path(
        self,
        source_entity_id: str,
        target_entity_id: str,
        max_hops: int = 3,
    ) -> list[Relationship] | None:
        """
        Find a path between two entities.

        Args:
            source_entity_id: The starting entity ID.
            target_entity_id: The target entity ID.
            max_hops: Maximum number of hops to search (default 3).

        Returns:
            List of relationships forming the path, or None if no path exists.
        """
        ...

    async def delete_entity(self, entity_id: str) -> bool:
        """
        Delete an entity and its associated relationships.

        Args:
            entity_id: The unique identifier of the entity.

        Returns:
            True if the entity was deleted, False if not found.
        """
        ...

    async def delete_relationship(self, relationship_id: str) -> bool:
        """
        Delete a relationship by ID.

        Args:
            relationship_id: The unique identifier of the relationship.

        Returns:
            True if the relationship was deleted, False if not found.
        """
        ...

    async def clear_user(self, user_id: str) -> int:
        """
        Clear all entities and relationships for a user.

        Args:
            user_id: The user scope to clear.

        Returns:
            Number of entities deleted.
        """
        ...
