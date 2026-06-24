"""Adapter to make Mem0Backend compatible with MemorySystem's MemoryBackend protocol.

This adapter bridges the Mem0Backend interface to the MemoryBackend protocol
required by MemorySystem, enabling Mem0 to be used with the memory tools system.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any

from headroom.memory.backends.mem0 import Mem0Backend, Mem0Config
from headroom.memory.models import Memory
from headroom.memory.ports import MemorySearchResult


def _utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class Mem0SystemAdapter:
    """Adapter that makes Mem0Backend conform to MemorySystem's MemoryBackend protocol.

    This adapter wraps Mem0Backend and provides the interface expected by
    MemorySystem, enabling LLM-driven memory tools (memory_save, memory_search,
    memory_update, memory_delete) to work with Mem0's graph and vector capabilities.

    Usage:
        from headroom.memory.backends.mem0_system_adapter import Mem0SystemAdapter, Mem0Config
        from headroom.memory.system import MemorySystem

        # Create adapter with Mem0 configuration
        config = Mem0Config(mode="local", enable_graph=True)
        adapter = Mem0SystemAdapter(config)

        # Use with MemorySystem
        memory_system = MemorySystem(adapter, user_id="alice")
        tools = memory_system.get_tools()

        # Process tool calls
        result = await memory_system.process_tool_call(
            "memory_save",
            {"content": "User prefers Python", "importance": 0.8}
        )
    """

    def __init__(self, config: Mem0Config | None = None) -> None:
        """Initialize the adapter.

        Args:
            config: Configuration for Mem0. If None, uses default local config.
        """
        self._backend = Mem0Backend(config)
        self._config = config or Mem0Config()

    async def save_memory(
        self,
        content: str,
        user_id: str,
        importance: float,
        entities: list[str] | None = None,
        relationships: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Save a new memory to Mem0.

        Note: Mem0 uses an LLM to extract facts from the content. The actual
        stored memory may be rephrased from the original content. If no facts
        are extracted, the memory won't be stored.

        Args:
            content: The memory content to store.
            user_id: User identifier for scoping.
            importance: Importance score (0.0 - 1.0).
            entities: List of entity references.
            relationships: List of relationship dicts with source, relation, target.
            session_id: Optional session identifier.
            metadata: Optional additional metadata.

        Returns:
            The created Memory object with Mem0's assigned ID and content.
        """
        # Ensure Mem0 client is ready
        client = await self._backend._ensure_client()

        # Build metadata for Mem0
        mem0_metadata: dict[str, Any] = {
            "importance": importance,
            "session_id": session_id,
            "entities": entities or [],
        }
        if relationships:
            mem0_metadata["relationships"] = relationships
        if metadata:
            mem0_metadata.update(metadata)

        # Call Mem0's add() directly - it extracts facts via LLM
        result = await asyncio.to_thread(
            client.add,
            content,
            user_id=user_id,
            metadata=mem0_metadata,
        )

        # Parse Mem0's response
        now = _utcnow()

        if isinstance(result, dict) and "results" in result:
            results = result["results"]
            if results and len(results) > 0:
                # Mem0 extracted and stored a memory
                first_result = results[0]
                mem0_id = first_result.get("id", str(uuid.uuid4()))
                # Mem0 rephrases the content - use what was actually stored
                stored_content = first_result.get("memory", content)

                return Memory(
                    id=mem0_id,
                    content=stored_content,
                    user_id=user_id,
                    session_id=session_id,
                    importance=importance,
                    entity_refs=entities or [],
                    metadata=metadata or {},
                    created_at=now,
                    valid_from=now,
                )

        # If Mem0 didn't extract anything (duplicate or no facts),
        # create a fallback memory object with generated ID
        # Note: This memory wasn't actually stored in Mem0
        return Memory(
            id=str(uuid.uuid4()),
            content=content,
            user_id=user_id,
            session_id=session_id,
            importance=importance,
            entity_refs=entities or [],
            metadata={**(metadata or {}), "_mem0_status": "not_extracted"},
            created_at=now,
            valid_from=now,
        )

    async def search_memories(
        self,
        query: str,
        user_id: str,
        entities: list[str] | None = None,
        include_related: bool = False,
        top_k: int = 10,
        session_id: str | None = None,
    ) -> list[MemorySearchResult]:
        """Search memories by semantic similarity.

        Args:
            query: Natural language search query.
            user_id: User identifier for scoping.
            entities: Filter to memories mentioning these entities.
            include_related: Whether to include related memories.
            top_k: Maximum number of results.
            session_id: Optional session filter.

        Returns:
            List of MemorySearchResult ordered by relevance.
        """
        # Search via Mem0Backend
        vector_results = await self._backend.search_memories(
            query=query,
            user_id=user_id,
            limit=top_k,
        )

        # Convert VectorSearchResult to MemorySearchResult
        results: list[MemorySearchResult] = []
        for vr in vector_results:
            # Filter by entities if provided
            if entities:
                # Check if any of the search entities are in the memory's entity refs
                memory_entities = vr.memory.entity_refs or []
                if not any(e in memory_entities for e in entities):
                    # Also check if entity appears in content (case-insensitive)
                    content_lower = vr.memory.content.lower()
                    if not any(e.lower() in content_lower for e in entities):
                        continue

            # Filter by session if provided
            if session_id and vr.memory.session_id != session_id:
                continue

            # Get related entities from memory
            related_entities = vr.memory.entity_refs or []

            # Find related memory IDs if requested
            related_memories: list[str] = []
            if include_related and related_entities:
                # Search for memories that share entities
                try:
                    related_results = await self._backend.search_memories(
                        query=" ".join(related_entities[:3]),  # Use top entities as query
                        user_id=user_id,
                        limit=5,
                    )
                    related_memories = [
                        r.memory.id for r in related_results if r.memory.id != vr.memory.id
                    ]
                except Exception:
                    pass

            results.append(
                MemorySearchResult(
                    memory=vr.memory,
                    score=vr.similarity,
                    related_entities=related_entities,
                    related_memories=related_memories,
                )
            )

        return results[:top_k]

    async def update_memory(
        self,
        memory_id: str,
        new_content: str,
        reason: str | None = None,
        user_id: str | None = None,
    ) -> Memory:
        """Update an existing memory with new content.

        Creates a new version while preserving history (supersession).

        Args:
            memory_id: ID of the memory to update.
            new_content: New content to replace existing.
            reason: Reason for the update (for audit trail).
            user_id: User ID for validation (optional).

        Returns:
            The updated Memory object.

        Raises:
            ValueError: If memory not found.
        """
        # Get the existing memory using Mem0 client directly
        client = await self._backend._ensure_client()

        try:
            existing_data = await asyncio.to_thread(client.get, memory_id=memory_id)
        except Exception:
            existing_data = None

        if not existing_data:
            raise ValueError(f"Memory not found: {memory_id}")

        # Extract user_id from existing data
        existing_user_id = existing_data.get("user_id", "")

        # Validate user if provided
        if user_id and existing_user_id and existing_user_id != user_id:
            raise ValueError("Cannot update memories belonging to other users")

        # Build update metadata
        now = _utcnow()
        update_metadata: dict[str, Any] = {}
        if reason:
            update_metadata["update_reason"] = reason
            update_metadata["updated_at"] = now.isoformat()

        # Update via Mem0
        try:
            await asyncio.to_thread(
                client.update,
                memory_id=memory_id,
                data=new_content,
            )
        except Exception as e:
            raise ValueError(f"Failed to update memory: {e}") from e

        # Fetch the updated memory
        try:
            updated_data = await asyncio.to_thread(client.get, memory_id=memory_id)
        except Exception:
            updated_data = None

        if updated_data:
            return Memory(
                id=memory_id,
                content=updated_data.get("memory", new_content),
                user_id=updated_data.get("user_id", existing_user_id),
                importance=0.5,  # Mem0 doesn't store importance
                created_at=now,
                valid_from=now,
            )
        else:
            # Return a constructed memory object
            return Memory(
                id=memory_id,
                content=new_content,
                user_id=existing_user_id,
                importance=0.5,
                created_at=now,
                valid_from=now,
            )

    async def delete_memory(
        self,
        memory_id: str,
        reason: str | None = None,
        user_id: str | None = None,
    ) -> bool:
        """Delete a memory from the backend.

        Args:
            memory_id: ID of the memory to delete.
            reason: Reason for deletion (for audit trail).
            user_id: User ID for validation (optional).

        Returns:
            True if deleted, False if not found.
        """
        client = await self._backend._ensure_client()

        # Validate user if needed
        if user_id:
            try:
                existing = await asyncio.to_thread(client.get, memory_id=memory_id)
                if existing and existing.get("user_id") and existing["user_id"] != user_id:
                    return False
            except Exception:
                pass

        try:
            await asyncio.to_thread(client.delete, memory_id=memory_id)
            return True
        except Exception:
            return False

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Retrieve a specific memory by ID.

        Args:
            memory_id: The memory identifier.

        Returns:
            The Memory if found, None otherwise.
        """
        client = await self._backend._ensure_client()

        try:
            result = await asyncio.to_thread(client.get, memory_id=memory_id)

            if result is None:
                return None

            # Convert Mem0 result to Memory object
            return Memory(
                id=result.get("id", memory_id),
                content=result.get("memory", ""),
                user_id=result.get("user_id", ""),
                importance=0.5,  # Mem0 doesn't track importance
                created_at=_utcnow(),
                valid_from=_utcnow(),
                metadata=result.get("metadata") or {},
            )
        except Exception:
            return None

    @property
    def supports_graph(self) -> bool:
        """Whether this backend supports graph/relationship queries."""
        return self._config.enable_graph

    @property
    def supports_vector_search(self) -> bool:
        """Whether this backend supports vector similarity search."""
        return True

    async def close(self) -> None:
        """Close the backend and release resources."""
        await self._backend.close()
