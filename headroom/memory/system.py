"""MemorySystem orchestrator for LLM-driven memory operations.

This module provides the MemorySystem class that bridges LLM tool calls
to the underlying memory backend. It handles tool call dispatch, argument
validation, and response formatting for seamless integration with
function-calling LLMs.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from headroom.memory.models import Memory
from headroom.memory.ports import MemorySearchResult
from headroom.memory.tools import MEMORY_TOOLS, MEMORY_TOOLS_OPTIMIZED

logger = logging.getLogger(__name__)


# =============================================================================
# Memory Backend Protocol
# =============================================================================


@runtime_checkable
class MemoryBackend(Protocol):
    """Protocol defining the interface for memory storage backends.

    This protocol abstracts the underlying storage implementation, allowing
    the MemorySystem to work with different backends (SQLite, PostgreSQL,
    vector databases, etc.) without modification.
    """

    async def save_memory(
        self,
        content: str,
        user_id: str,
        importance: float,
        entities: list[str] | None = None,
        relationships: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        # Pre-extraction fields for optimized storage
        facts: list[str] | None = None,
        extracted_entities: list[dict[str, str]] | None = None,
        extracted_relationships: list[dict[str, str]] | None = None,
    ) -> Memory:
        """Save a new memory to the backend.

        Args:
            content: The memory content to store.
            user_id: User identifier for scoping.
            importance: Importance score (0.0 - 1.0).
            entities: List of entity references.
            relationships: List of relationship dicts with source, relation, target.
            session_id: Optional session identifier.
            metadata: Optional additional metadata.
            facts: Pre-extracted discrete facts (for optimized storage).
            extracted_entities: Pre-extracted entities with types (for graph).
            extracted_relationships: Pre-extracted relationships (for graph).

        Returns:
            The created Memory object.
        """
        ...

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
        ...

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
        ...

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
        ...

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Retrieve a specific memory by ID.

        Args:
            memory_id: The memory identifier.

        Returns:
            The Memory if found, None otherwise.
        """
        ...

    @property
    def supports_graph(self) -> bool:
        """Whether this backend supports graph/relationship queries."""
        ...

    @property
    def supports_vector_search(self) -> bool:
        """Whether this backend supports vector similarity search."""
        ...

    async def close(self) -> None:
        """Close the backend and release resources."""
        ...


# =============================================================================
# Memory System Orchestrator
# =============================================================================


class MemorySystem:
    """Orchestrator for LLM-driven memory operations.

    MemorySystem provides a high-level interface for LLMs to interact with
    the memory system via function calling. It handles:
    - Tool call dispatch and argument validation
    - Response formatting for LLM consumption
    - User and session scoping
    - Error handling and graceful degradation

    Usage:
        # Initialize with a backend
        backend = await create_memory_backend(config)
        memory_system = MemorySystem(backend, user_id="alice")

        # Get available tools for the LLM
        tools = memory_system.get_tools()

        # Process a tool call from the LLM
        result = await memory_system.process_tool_call(
            "memory_save",
            {
                "content": "User prefers Python",
                "importance": 0.8,
            }
        )
    """

    def __init__(
        self,
        backend: MemoryBackend,
        user_id: str,
        session_id: str | None = None,
    ) -> None:
        """Initialize the MemorySystem.

        Args:
            backend: The memory storage backend.
            user_id: User identifier for scoping all operations.
            session_id: Optional session identifier for session-scoped memories.
        """
        self._backend = backend
        self._user_id = user_id
        self._session_id = session_id

    # =========================================================================
    # Tool Retrieval
    # =========================================================================

    def get_tools(self, optimized: bool = False) -> list[dict[str, Any]]:
        """Get the memory tool definitions for LLM function calling.

        Args:
            optimized: If True, return tools with pre-extraction fields.
                       Use with DirectMem0Adapter for best performance.
                       The main LLM should extract facts/entities/relationships
                       when calling memory_save to bypass backend LLM extraction.

        Returns:
            List of tool definitions in OpenAI function calling format.
        """
        if optimized:
            return MEMORY_TOOLS_OPTIMIZED.copy()
        return MEMORY_TOOLS.copy()

    # =========================================================================
    # Tool Call Dispatch
    # =========================================================================

    async def process_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a tool call from the LLM.

        Dispatches the tool call to the appropriate handler method and
        formats the response for the LLM.

        Args:
            tool_name: Name of the tool to invoke.
            arguments: Tool arguments from the LLM.

        Returns:
            Dict with success status, message, and relevant data.

        Example:
            result = await system.process_tool_call(
                "memory_search",
                {"query": "programming preferences", "top_k": 5}
            )
            # result = {
            #     "success": True,
            #     "message": "Found 3 relevant memories",
            #     "memories": [...]
            # }
        """
        handlers = {
            "memory_save": self._handle_save,
            "memory_search": self._handle_search,
            "memory_update": self._handle_update,
            "memory_delete": self._handle_delete,
        }

        handler = handlers.get(tool_name)
        if handler is None:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}",
                "message": f"Tool '{tool_name}' is not a valid memory tool. "
                f"Available tools: {list(handlers.keys())}",
            }

        try:
            return await handler(arguments)
        except Exception as e:
            logger.exception(f"Error processing tool call {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to execute {tool_name}: {e}",
            }

    # =========================================================================
    # Tool Handlers
    # =========================================================================

    async def handle_memory_save(
        self,
        content: str,
        importance: float,
        entities: list[str] | None = None,
        relationships: list[dict[str, str]] | None = None,
        metadata: dict[str, Any] | None = None,
        # Pre-extraction fields for optimized storage
        facts: list[str] | None = None,
        extracted_entities: list[dict[str, str]] | None = None,
        extracted_relationships: list[dict[str, str]] | None = None,
        # Async control
        background: bool | None = None,
    ) -> dict[str, Any]:
        """Save a new memory.

        Args:
            content: The information to remember.
            importance: Importance score (0.0 - 1.0).
            entities: Optional list of entity references.
            relationships: Optional list of entity relationships.
            metadata: Optional additional metadata.
            facts: Pre-extracted discrete facts (for optimized storage).
            extracted_entities: Pre-extracted entities with types (for graph).
            extracted_relationships: Pre-extracted relationships (for graph).
            background: If True, save in background and return immediately.

        Returns:
            Dict with success status, message, and memory details.
            If background=True, includes task_id for status tracking.
        """
        # Validate importance
        if not 0.0 <= importance <= 1.0:
            return {
                "success": False,
                "error": f"Invalid importance: {importance}",
                "message": "Importance must be between 0.0 and 1.0",
            }

        # Build kwargs for save_memory - only include pre-extraction fields if supported
        save_kwargs: dict[str, Any] = {
            "content": content,
            "user_id": self._user_id,
            "importance": importance,
            "entities": entities,
            "relationships": relationships,
            "session_id": self._session_id,
            "metadata": metadata,
        }

        # Add pre-extraction fields if provided (for DirectMem0Adapter)
        if facts is not None:
            save_kwargs["facts"] = facts
        if extracted_entities is not None:
            save_kwargs["extracted_entities"] = extracted_entities
        if extracted_relationships is not None:
            save_kwargs["extracted_relationships"] = extracted_relationships
        if background is not None:
            save_kwargs["background"] = background

        # Log optimization usage
        has_pre_extraction = bool(facts or extracted_entities or extracted_relationships)
        if has_pre_extraction:
            logger.info("Using pre-extracted data for optimized storage")

        # Save the memory
        try:
            memory = await self._backend.save_memory(**save_kwargs)
        except TypeError:
            # Backend doesn't support pre-extraction fields, use basic call
            logger.debug("Backend doesn't support pre-extraction, falling back to basic save")
            memory = await self._backend.save_memory(
                content=content,
                user_id=self._user_id,
                importance=importance,
                entities=entities,
                relationships=relationships,
                session_id=self._session_id,
                metadata=metadata,
            )

        logger.info(f"Saved memory {memory.id}: {content[:50]}...")

        result: dict[str, Any] = {
            "success": True,
            "message": f"Saved memory with ID {memory.id}",
            "memory_id": memory.id,
            "content": memory.content,
            "importance": memory.importance,
        }

        # Include optimization info if pre-extraction was used
        if has_pre_extraction:
            result["optimized"] = True
            if facts:
                result["fact_count"] = len(facts)

        # Include async/background info if save was async
        if memory.metadata and memory.metadata.get("_async"):
            result["async"] = True
            result["task_id"] = memory.metadata.get("_task_id")
            result["status"] = memory.metadata.get("_status", "processing")
            result["message"] = f"Memory save queued (task: {result['task_id']})"

        return result

    async def handle_memory_search(
        self,
        query: str,
        entities: list[str] | None = None,
        include_related: bool = False,
        top_k: int = 10,
    ) -> dict[str, Any]:
        """Search for relevant memories.

        Args:
            query: Natural language search query.
            entities: Optional entity filter.
            include_related: Whether to include related memories.
            top_k: Maximum number of results.

        Returns:
            Dict with success status, message, and search results.
        """
        # Validate top_k
        top_k = max(1, min(top_k, 50))

        # Search memories
        results = await self._backend.search_memories(
            query=query,
            user_id=self._user_id,
            entities=entities,
            include_related=include_related,
            top_k=top_k,
            session_id=self._session_id,
        )

        if not results:
            return {
                "success": True,
                "message": "No relevant memories found",
                "memories": [],
                "count": 0,
            }

        # Format results for LLM
        formatted_memories = [r.to_dict() for r in results]

        logger.info(f"Search '{query[:30]}...' returned {len(results)} results")

        return {
            "success": True,
            "message": f"Found {len(results)} relevant memories",
            "memories": formatted_memories,
            "count": len(results),
        }

    async def handle_memory_update(
        self,
        memory_id: str,
        new_content: str,
        reason: str,
    ) -> dict[str, Any]:
        """Update an existing memory.

        Args:
            memory_id: ID of the memory to update.
            new_content: New content to replace existing.
            reason: Reason for the update.

        Returns:
            Dict with success status, message, and updated memory details.
        """
        # Verify memory exists
        existing = await self._backend.get_memory(memory_id)
        if existing is None:
            return {
                "success": False,
                "error": f"Memory not found: {memory_id}",
                "message": f"No memory exists with ID {memory_id}. "
                "Use memory_search to find the correct memory ID.",
            }

        # Verify ownership
        if existing.user_id != self._user_id:
            return {
                "success": False,
                "error": "Permission denied",
                "message": "Cannot update memories belonging to other users.",
            }

        # Update the memory
        try:
            updated = await self._backend.update_memory(
                memory_id=memory_id,
                new_content=new_content,
                reason=reason,
                user_id=self._user_id,
            )
        except ValueError as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to update memory: {e}",
            }

        logger.info(f"Updated memory {memory_id}: {reason}")

        return {
            "success": True,
            "message": f"Updated memory {memory_id}",
            "memory_id": updated.id,
            "old_content": existing.content,
            "new_content": updated.content,
            "reason": reason,
        }

    async def handle_memory_delete(
        self,
        memory_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Delete a memory.

        Args:
            memory_id: ID of the memory to delete.
            reason: Reason for deletion.

        Returns:
            Dict with success status and message.
        """
        # Verify memory exists
        existing = await self._backend.get_memory(memory_id)
        if existing is None:
            return {
                "success": False,
                "error": f"Memory not found: {memory_id}",
                "message": f"No memory exists with ID {memory_id}. "
                "Use memory_search to find the correct memory ID.",
            }

        # Verify ownership
        if existing.user_id != self._user_id:
            return {
                "success": False,
                "error": "Permission denied",
                "message": "Cannot delete memories belonging to other users.",
            }

        # Delete the memory
        deleted = await self._backend.delete_memory(
            memory_id=memory_id,
            reason=reason,
            user_id=self._user_id,
        )

        if not deleted:
            return {
                "success": False,
                "error": "Delete failed",
                "message": f"Failed to delete memory {memory_id}",
            }

        logger.info(f"Deleted memory {memory_id}: {reason}")

        return {
            "success": True,
            "message": f"Deleted memory {memory_id}",
            "memory_id": memory_id,
            "deleted_content": existing.content,
            "reason": reason,
        }

    # =========================================================================
    # Private Dispatch Helpers
    # =========================================================================

    async def _handle_save(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Internal dispatcher for memory_save."""
        return await self.handle_memory_save(
            content=arguments["content"],
            importance=arguments["importance"],
            entities=arguments.get("entities"),
            relationships=arguments.get("relationships"),
            metadata=arguments.get("metadata"),
            # Pre-extraction fields for optimized storage
            facts=arguments.get("facts"),
            extracted_entities=arguments.get("extracted_entities"),
            extracted_relationships=arguments.get("extracted_relationships"),
            # Async control
            background=arguments.get("background"),
        )

    async def _handle_search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Internal dispatcher for memory_search."""
        return await self.handle_memory_search(
            query=arguments["query"],
            entities=arguments.get("entities"),
            include_related=arguments.get("include_related", False),
            top_k=arguments.get("top_k", 10),
        )

    async def _handle_update(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Internal dispatcher for memory_update."""
        return await self.handle_memory_update(
            memory_id=arguments["memory_id"],
            new_content=arguments["new_content"],
            reason=arguments.get("reason", "Updated by user"),
        )

    async def _handle_delete(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Internal dispatcher for memory_delete."""
        return await self.handle_memory_delete(
            memory_id=arguments["memory_id"],
            reason=arguments.get("reason", "Deleted by user"),
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def user_id(self) -> str:
        """Get the current user ID."""
        return self._user_id

    @property
    def session_id(self) -> str | None:
        """Get the current session ID."""
        return self._session_id

    @property
    def backend(self) -> MemoryBackend:
        """Get the underlying backend."""
        return self._backend

    @property
    def supports_graph(self) -> bool:
        """Whether the backend supports graph queries."""
        return self._backend.supports_graph

    @property
    def supports_vector_search(self) -> bool:
        """Whether the backend supports vector search."""
        return self._backend.supports_vector_search

    # =========================================================================
    # Async/Background Task Management
    # =========================================================================

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get the status of a background save task.

        Only available when using backends that support async saves
        (e.g., DirectMem0Adapter with async_writes=True).

        Args:
            task_id: The task ID returned from an async save.

        Returns:
            Dict with status and result if completed.
            Returns {"status": "not_supported"} if backend doesn't support async.
        """
        if hasattr(self._backend, "get_task_status"):
            result: dict[str, Any] = self._backend.get_task_status(task_id)
            return result
        return {"status": "not_supported", "message": "Backend doesn't support async tasks"}

    def get_pending_tasks(self) -> list[str]:
        """Get list of pending background task IDs.

        Returns:
            List of task IDs, or empty list if not supported.
        """
        if hasattr(self._backend, "get_pending_tasks"):
            tasks: list[str] = self._backend.get_pending_tasks()
            return tasks
        return []

    async def wait_for_task(self, task_id: str, timeout: float = 30.0) -> dict[str, Any]:
        """Wait for a background task to complete.

        Args:
            task_id: The task ID to wait for.
            timeout: Maximum seconds to wait.

        Returns:
            Task result or timeout error.
        """
        if hasattr(self._backend, "wait_for_task"):
            task_result: dict[str, Any] = await self._backend.wait_for_task(task_id, timeout)
            return task_result
        return {"status": "not_supported", "message": "Backend doesn't support async tasks"}

    async def flush_pending(self, timeout: float = 60.0) -> dict[str, Any]:
        """Wait for all pending background tasks to complete.

        Useful before shutting down or when you need to ensure all saves
        have been persisted.

        Args:
            timeout: Maximum seconds to wait for all tasks.

        Returns:
            Summary of completed and failed tasks.
        """
        if hasattr(self._backend, "flush_pending"):
            flush_result: dict[str, Any] = await self._backend.flush_pending(timeout)
            return flush_result
        return {"completed": 0, "failed": 0, "pending": 0}
