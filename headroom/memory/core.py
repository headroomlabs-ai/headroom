"""Core HierarchicalMemory orchestrator for Headroom.

This module provides the main HierarchicalMemory class that coordinates
all memory system components: store, vector index, text index, embedder,
and cache. It implements the high-level memory operations with automatic
embedding, indexing, caching, and memory bubbling.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any

from headroom.memory.config import MemoryConfig
from headroom.memory.factory import create_memory_system
from headroom.memory.models import Memory, ScopeLevel
from headroom.memory.ports import MemoryFilter, TextFilter, VectorFilter

if TYPE_CHECKING:
    from headroom.memory.ports import (
        Embedder,
        MemoryCache,
        MemoryStore,
        TextIndex,
        TextSearchResult,
        VectorIndex,
        VectorSearchResult,
    )

logger = logging.getLogger(__name__)


class HierarchicalMemory:
    """Main orchestrator for the hierarchical memory system.

    HierarchicalMemory coordinates all memory system components to provide
    a unified API for memory operations. It handles:
    - Automatic embedding generation
    - Multi-index updates (store, vector, text)
    - Cache management
    - Memory bubbling (promoting important memories up the hierarchy)
    - Hierarchical scoping (user -> session -> agent -> turn)
    - Temporal queries (point-in-time, supersession)

    Usage:
        # Create with default configuration
        memory = await HierarchicalMemory.create()

        # Or with custom configuration
        config = MemoryConfig(embedder_backend=EmbedderBackend.OPENAI)
        memory = await HierarchicalMemory.create(config)

        # Add a memory
        await memory.add(
            content="User prefers Python over JavaScript",
            user_id="alice",
            importance=0.9,
        )

        # Search semantically
        results = await memory.search("programming language preferences", user_id="alice")

        # Full-text search
        results = await memory.text_search("Python", user_id="alice")

        # Query with filters
        memories = await memory.query(MemoryFilter(
            user_id="alice",
            min_importance=0.8,
        ))
    """

    def __init__(
        self,
        store: MemoryStore,
        vector_index: VectorIndex,
        text_index: TextIndex,
        embedder: Embedder,
        cache: MemoryCache | None = None,
        config: MemoryConfig | None = None,
    ) -> None:
        """Initialize HierarchicalMemory with components.

        Prefer using the create() factory method instead of direct initialization.

        Args:
            store: Memory persistence backend.
            vector_index: Vector similarity search index.
            text_index: Full-text search index.
            embedder: Text embedding generator.
            cache: Optional memory cache.
            config: Configuration (for bubbling settings, etc.).
        """
        self._store = store
        self._vector_index = vector_index
        self._text_index = text_index
        self._embedder = embedder
        self._cache = cache
        self._config = config or MemoryConfig()

    @classmethod
    async def create(cls, config: MemoryConfig | None = None) -> HierarchicalMemory:
        """Create a HierarchicalMemory instance from configuration.

        This is the recommended way to create a HierarchicalMemory instance.
        It creates all necessary components based on the configuration.

        Args:
            config: Memory system configuration. Uses defaults if not provided.

        Returns:
            Fully initialized HierarchicalMemory instance.

        Example:
            memory = await HierarchicalMemory.create()
            # Or with config
            memory = await HierarchicalMemory.create(MemoryConfig(
                embedder_backend=EmbedderBackend.OPENAI,
                openai_api_key="sk-...",
            ))
        """
        config = config or MemoryConfig()
        store, vector_index, text_index, embedder, cache = await create_memory_system(config)
        return cls(store, vector_index, text_index, embedder, cache, config)

    # =========================================================================
    # Memory Creation
    # =========================================================================

    async def add(
        self,
        content: str,
        user_id: str,
        session_id: str | None = None,
        agent_id: str | None = None,
        turn_id: str | None = None,
        importance: float = 0.5,
        entity_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        auto_embed: bool = True,
        auto_bubble: bool | None = None,
    ) -> Memory:
        """Add a new memory to the system.

        Creates a memory with the specified content and scope, generates
        embeddings, and indexes it for search. Optionally bubbles important
        memories up the hierarchy.

        Args:
            content: The memory content/text.
            user_id: User identifier (required - top of hierarchy).
            session_id: Session identifier (optional).
            agent_id: Agent identifier (optional).
            turn_id: Turn identifier (optional).
            importance: Importance score (0.0 - 1.0).
            entity_refs: List of entity references.
            metadata: Additional metadata.
            auto_embed: Whether to generate embedding automatically.
            auto_bubble: Whether to bubble up (uses config default if None).

        Returns:
            The created and stored Memory object.

        Example:
            memory = await system.add(
                content="User prefers dark mode",
                user_id="alice",
                session_id="sess-123",
                importance=0.8,
            )
        """
        # Create memory object
        memory = Memory(
            content=content,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            turn_id=turn_id,
            importance=importance,
            entity_refs=entity_refs or [],
            metadata=metadata or {},
        )

        # Generate embedding if requested
        if auto_embed:
            embedding = await self._embedder.embed(content)
            memory.embedding = embedding

        # Save to store
        await self._store.save(memory)

        # Index for vector search
        if memory.embedding is not None:
            await self._vector_index.index(memory)

        # Index for text search
        await self._index_for_text_search(memory)

        # Update cache
        if self._cache is not None:
            await self._cache.put(memory)

        # Handle bubbling
        should_bubble = auto_bubble if auto_bubble is not None else self._config.auto_bubble
        if should_bubble:
            await self._maybe_bubble(memory)

        logger.debug(f"Added memory {memory.id} at scope {memory.scope_level.value}")
        return memory

    async def add_batch(
        self,
        memories_data: list[dict[str, Any]],
        auto_embed: bool = True,
    ) -> list[Memory]:
        """Add multiple memories in a batch operation.

        More efficient than calling add() multiple times due to batch
        embedding and batch database operations.

        Args:
            memories_data: List of dicts with memory parameters
                          (content, user_id, etc.).
            auto_embed: Whether to generate embeddings automatically.

        Returns:
            List of created Memory objects.

        Example:
            memories = await system.add_batch([
                {"content": "Fact 1", "user_id": "alice"},
                {"content": "Fact 2", "user_id": "alice"},
            ])
        """
        # Create Memory objects
        memories: list[Memory] = []
        for data in memories_data:
            memory = Memory(
                content=data["content"],
                user_id=data["user_id"],
                session_id=data.get("session_id"),
                agent_id=data.get("agent_id"),
                turn_id=data.get("turn_id"),
                importance=data.get("importance", 0.5),
                entity_refs=data.get("entity_refs", []),
                metadata=data.get("metadata", {}),
            )
            memories.append(memory)

        # Batch embed
        if auto_embed:
            texts = [m.content for m in memories]
            embeddings = await self._embedder.embed_batch(texts)
            for memory, embedding in zip(memories, embeddings):
                memory.embedding = embedding

        # Batch save
        await self._store.save_batch(memories)

        # Batch index for vector search
        memories_with_embeddings = [m for m in memories if m.embedding is not None]
        if memories_with_embeddings:
            await self._vector_index.index_batch(memories_with_embeddings)

        # Index for text search
        for memory in memories:
            await self._index_for_text_search(memory)

        # Update cache
        if self._cache is not None:
            await self._cache.put_batch(memories)

        logger.debug(f"Added batch of {len(memories)} memories")
        return memories

    # =========================================================================
    # Memory Retrieval
    # =========================================================================

    async def get(self, memory_id: str) -> Memory | None:
        """Get a memory by ID.

        Checks cache first, then falls back to store.

        Args:
            memory_id: The unique memory identifier.

        Returns:
            The Memory if found, None otherwise.
        """
        # Check cache first
        if self._cache is not None:
            cached = await self._cache.get(memory_id)
            if cached is not None:
                return cached

        # Fall back to store
        memory = await self._store.get(memory_id)

        # Update cache on hit
        if memory is not None and self._cache is not None:
            await self._cache.put(memory)

        return memory

    async def query(self, filter: MemoryFilter) -> list[Memory]:
        """Query memories with filtering.

        Args:
            filter: Filter criteria for the query.

        Returns:
            List of matching memories.

        Example:
            memories = await system.query(MemoryFilter(
                user_id="alice",
                min_importance=0.7,
                limit=10,
            ))
        """
        return await self._store.query(filter)

    async def count(self, filter: MemoryFilter) -> int:
        """Count memories matching filter criteria.

        Args:
            filter: Filter criteria.

        Returns:
            Number of matching memories.
        """
        return await self._store.count(filter)

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search(
        self,
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        top_k: int = 10,
        min_similarity: float = 0.0,
        scope_levels: list[ScopeLevel] | None = None,
        include_superseded: bool = False,
    ) -> list[VectorSearchResult]:
        """Semantic search for similar memories.

        Uses vector similarity to find memories semantically similar
        to the query text.

        Args:
            query: Search query text.
            user_id: Filter by user.
            session_id: Filter by session.
            agent_id: Filter by agent.
            top_k: Maximum number of results.
            min_similarity: Minimum cosine similarity threshold.
            scope_levels: Filter by scope levels.
            include_superseded: Include superseded memories.

        Returns:
            List of VectorSearchResult sorted by similarity.

        Example:
            results = await system.search(
                "programming preferences",
                user_id="alice",
                top_k=5,
            )
            for result in results:
                print(f"{result.similarity:.2f}: {result.memory.content}")
        """
        # Embed query
        query_vector = await self._embedder.embed(query)

        # Build filter
        vector_filter = VectorFilter(
            query_vector=query_vector,
            top_k=top_k,
            min_similarity=min_similarity,
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            scope_levels=scope_levels,
            include_superseded=include_superseded,
        )

        return await self._vector_index.search(vector_filter)

    async def text_search(
        self,
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[TextSearchResult]:
        """Full-text search for memories.

        Uses keyword matching with BM25 ranking to find memories
        containing the search terms.

        Args:
            query: Search query text.
            user_id: Filter by user.
            session_id: Filter by session.
            limit: Maximum number of results.

        Returns:
            List of TextSearchResult sorted by relevance.

        Example:
            results = await system.text_search("Python", user_id="alice")
        """
        text_filter = TextFilter(
            query=query,
            user_id=user_id,
            session_id=session_id,
            limit=limit,
        )

        return await self._text_index.search(text_filter)

    # =========================================================================
    # Memory Updates
    # =========================================================================

    async def update(
        self,
        memory_id: str,
        content: str | None = None,
        importance: float | None = None,
        entity_refs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        re_embed: bool = True,
    ) -> Memory | None:
        """Update an existing memory.

        Updates the specified fields and re-indexes if content changes.

        Args:
            memory_id: ID of memory to update.
            content: New content (triggers re-embedding if re_embed=True).
            importance: New importance score.
            entity_refs: New entity references.
            metadata: New or updated metadata (merged with existing).
            re_embed: Whether to regenerate embedding on content change.

        Returns:
            Updated Memory, or None if not found.
        """
        memory = await self._store.get(memory_id)
        if memory is None:
            return None

        content_changed = False

        if content is not None and content != memory.content:
            memory.content = content
            content_changed = True

        if importance is not None:
            memory.importance = importance

        if entity_refs is not None:
            memory.entity_refs = entity_refs

        if metadata is not None:
            memory.metadata.update(metadata)

        # Re-embed if content changed
        if content_changed and re_embed:
            memory.embedding = await self._embedder.embed(memory.content)

        # Save updates
        await self._store.save(memory)

        # Update indexes
        if content_changed:
            if memory.embedding is not None:
                await self._vector_index.index(memory)
            await self._index_for_text_search(memory)

        # Invalidate and re-cache
        if self._cache is not None:
            await self._cache.invalidate(memory_id)
            await self._cache.put(memory)

        return memory

    async def supersede(
        self,
        old_memory_id: str,
        new_content: str,
        supersede_time: datetime | None = None,
        auto_embed: bool = True,
    ) -> Memory:
        """Supersede an existing memory with a new version.

        Creates a temporal chain where the old memory's validity ends
        and the new memory begins. Both are kept for historical queries.

        Args:
            old_memory_id: ID of memory to supersede.
            new_content: Content for the new memory.
            supersede_time: When the supersession occurred (default: now).
            auto_embed: Whether to embed the new content.

        Returns:
            The new Memory that supersedes the old one.

        Raises:
            ValueError: If old memory not found.

        Example:
            # User's preference changed
            new_mem = await system.supersede(
                old_memory.id,
                "User now prefers JavaScript over Python",
            )
        """
        # Get old memory
        old_memory = await self._store.get(old_memory_id)
        if old_memory is None:
            raise ValueError(f"Memory {old_memory_id} not found")

        # Create new memory with same scope
        new_memory = Memory(
            content=new_content,
            user_id=old_memory.user_id,
            session_id=old_memory.session_id,
            agent_id=old_memory.agent_id,
            turn_id=old_memory.turn_id,
            importance=old_memory.importance,
            entity_refs=old_memory.entity_refs.copy(),
            metadata=old_memory.metadata.copy(),
        )

        # Embed new content
        if auto_embed:
            new_memory.embedding = await self._embedder.embed(new_content)

        # Perform supersession in store
        new_memory = await self._store.supersede(old_memory_id, new_memory, supersede_time)

        # Update indexes
        if new_memory.embedding is not None:
            await self._vector_index.index(new_memory)
        await self._index_for_text_search(new_memory)

        # Update cache
        if self._cache is not None:
            await self._cache.invalidate(old_memory_id)
            await self._cache.put(new_memory)

        logger.debug(f"Superseded memory {old_memory_id} with {new_memory.id}")
        return new_memory

    async def get_history(
        self,
        memory_id: str,
        include_future: bool = False,
    ) -> list[Memory]:
        """Get the full history chain for a memory.

        Follows supersession links to return all versions of a memory.

        Args:
            memory_id: ID of any memory in the chain.
            include_future: Whether to include memories that superseded this one.

        Returns:
            List of memories in temporal order (oldest first).
        """
        return await self._store.get_history(memory_id, include_future)

    # =========================================================================
    # Memory Deletion
    # =========================================================================

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory from all indexes.

        Args:
            memory_id: ID of memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        # Delete from store
        deleted = await self._store.delete(memory_id)

        if deleted:
            # Remove from indexes
            await self._vector_index.remove(memory_id)
            await self._text_index.remove(memory_id)

            # Invalidate cache
            if self._cache is not None:
                await self._cache.invalidate(memory_id)

        return deleted

    async def clear_scope(
        self,
        user_id: str,
        session_id: str | None = None,
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> int:
        """Clear all memories at or below a scope level.

        Args:
            user_id: Required user scope.
            session_id: If provided, clear session and below.
            agent_id: If provided, clear agent and below.
            turn_id: If provided, clear only that turn.

        Returns:
            Number of memories deleted.
        """
        # Get IDs of memories to clear
        filter = MemoryFilter(
            user_id=user_id,
            session_id=session_id,
            agent_id=agent_id,
            turn_id=turn_id,
            include_superseded=True,  # Clear all versions
        )
        memories = await self._store.query(filter)
        memory_ids = [m.id for m in memories]

        if not memory_ids:
            return 0

        # Clear from store
        count = await self._store.clear_scope(user_id, session_id, agent_id, turn_id)

        # Clear from indexes
        await self._vector_index.remove_batch(memory_ids)
        for mid in memory_ids:
            await self._text_index.remove(mid)

        # Clear from cache
        if self._cache is not None:
            await self._cache.invalidate_scope(user_id, session_id, agent_id)

        logger.debug(f"Cleared {count} memories at scope user={user_id}, session={session_id}")
        return count

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def remember(
        self,
        content: str,
        user_id: str,
        session_id: str | None = None,
        importance: float = 0.5,
    ) -> Memory:
        """Convenience method to quickly add a memory.

        Shorthand for add() with common parameters.

        Args:
            content: What to remember.
            user_id: Who it's for.
            session_id: Optional session context.
            importance: How important (0.0 - 1.0).

        Returns:
            The created Memory.

        Example:
            await system.remember("Likes coffee", user_id="alice", importance=0.7)
        """
        return await self.add(
            content=content,
            user_id=user_id,
            session_id=session_id,
            importance=importance,
        )

    async def recall(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> list[Memory]:
        """Convenience method to recall relevant memories.

        Performs semantic search and returns just the Memory objects.

        Args:
            query: What to recall.
            user_id: Whose memories to search.
            top_k: Maximum memories to return.

        Returns:
            List of relevant Memory objects.

        Example:
            memories = await system.recall("coffee preferences", user_id="alice")
        """
        results = await self.search(query, user_id=user_id, top_k=top_k)
        return [r.memory for r in results]

    async def get_user_memories(
        self,
        user_id: str,
        limit: int = 100,
        include_sessions: bool = True,
    ) -> list[Memory]:
        """Get all memories for a user.

        Args:
            user_id: User identifier.
            limit: Maximum memories to return.
            include_sessions: If True, include session-level memories.
                            If False, only return user-level memories.

        Returns:
            List of memories for the user.
        """
        filter = MemoryFilter(
            user_id=user_id,
            limit=limit,
        )

        if not include_sessions:
            filter.scope_levels = [ScopeLevel.USER]

        return await self._store.query(filter)

    async def get_session_memories(
        self,
        user_id: str,
        session_id: str,
        limit: int = 100,
    ) -> list[Memory]:
        """Get all memories for a session.

        Args:
            user_id: User identifier.
            session_id: Session identifier.
            limit: Maximum memories to return.

        Returns:
            List of memories for the session.
        """
        filter = MemoryFilter(
            user_id=user_id,
            session_id=session_id,
            limit=limit,
        )
        return await self._store.query(filter)

    # =========================================================================
    # Internal Methods
    # =========================================================================

    async def _index_for_text_search(self, memory: Memory) -> None:
        """Index a memory for full-text search.

        Uses the protocol-compliant async method on the text index.
        """
        # Use the async index_memory method which is protocol-compliant
        await self._text_index.index_memory(memory)  # type: ignore[attr-defined]

    async def _maybe_bubble(self, memory: Memory) -> None:
        """Maybe bubble a memory up the hierarchy based on importance.

        Bubbling creates a copy of the memory at a higher scope level.
        Only happens if the memory meets bubbling criteria (high importance).
        """
        # Only bubble high-importance memories
        if memory.importance < self._config.bubble_threshold:
            return

        # Only bubble from session or lower scopes
        current_scope = memory.scope_level
        if current_scope == ScopeLevel.USER:
            return  # Already at highest scope

        # Target scope is USER level for high-importance memories
        target_scope = ScopeLevel.USER

        # Create bubbled memory at user scope
        bubbled = Memory(
            content=memory.content,
            user_id=memory.user_id,
            session_id=None,  # Bubble to user level
            agent_id=None,
            turn_id=None,
            importance=memory.importance,
            entity_refs=memory.entity_refs.copy(),
            metadata=memory.metadata.copy(),
            embedding=memory.embedding.copy() if memory.embedding is not None else None,
            promoted_from=memory.id,
            promotion_chain=memory.promotion_chain + [memory.id],
        )

        # Save bubbled memory
        await self._store.save(bubbled)

        # Index bubbled memory
        if bubbled.embedding is not None:
            await self._vector_index.index(bubbled)
        await self._index_for_text_search(bubbled)

        logger.debug(
            f"Bubbled memory {memory.id} from {current_scope.value} to {target_scope.value}"
        )

    def _scope_level_value(self, level: ScopeLevel) -> int:
        """Get numeric value for scope level (lower = broader scope)."""
        return {
            ScopeLevel.USER: 0,
            ScopeLevel.SESSION: 1,
            ScopeLevel.AGENT: 2,
            ScopeLevel.TURN: 3,
        }[level]

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def store(self) -> MemoryStore:
        """Access the underlying memory store."""
        return self._store

    @property
    def vector_index(self) -> VectorIndex:
        """Access the underlying vector index."""
        return self._vector_index

    @property
    def text_index(self) -> TextIndex:
        """Access the underlying text index."""
        return self._text_index

    @property
    def embedder(self) -> Embedder:
        """Access the underlying embedder."""
        return self._embedder

    @property
    def cache(self) -> MemoryCache | None:
        """Access the underlying cache (may be None)."""
        return self._cache

    @property
    def config(self) -> MemoryConfig:
        """Access the configuration."""
        return self._config

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def close(self) -> None:
        """Close all resources held by the memory system.

        This should be called when done using the memory system to properly
        clean up resources like HTTP clients used by embedders.
        """
        # Close embedder if it has a close method (e.g., API-based embedders)
        if hasattr(self._embedder, "close"):
            await self._embedder.close()

        # Close store if it has a close method
        if hasattr(self._store, "close"):
            await self._store.close()

        # Close vector index if it has a close method
        if hasattr(self._vector_index, "close"):
            await self._vector_index.close()

        # Close text index if it has a close method
        if hasattr(self._text_index, "close"):
            await self._text_index.close()

        # Close cache if it has a close method
        if self._cache is not None and hasattr(self._cache, "close"):
            await self._cache.close()

        logger.debug("HierarchicalMemory closed")

    async def __aenter__(self) -> HierarchicalMemory:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit - closes resources."""
        await self.close()
