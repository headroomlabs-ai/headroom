"""Memory wrapper - the main API for Headroom Memory.

One-line integration with zero-latency inline extraction:
    from headroom import with_memory
    client = with_memory(OpenAI(), user_id="alice")

This uses the Letta/MemGPT approach - memories are extracted inline
as part of the LLM response, not in a separate API call.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from pathlib import Path
from typing import Any

from headroom.memory.config import EmbedderBackend, MemoryConfig
from headroom.memory.core import HierarchicalMemory
from headroom.memory.inline_extractor import (
    inject_memory_instruction,
    parse_response_with_memory,
)
from headroom.memory.models import Memory

logger = logging.getLogger(__name__)


class MemoryWrapper:
    """Wraps an LLM client to add automatic memory with zero extra latency.

    Uses inline extraction (Letta-style) - memories are extracted as part
    of the LLM response, not in a separate API call.

    Intercepts chat completions to:
    1. BEFORE: Inject relevant memories into user message (semantic search)
    2. DURING: Memory instruction in system prompt
    3. AFTER: Parse response to extract and store memories

    The original system prompt is preserved for caching.

    Usage:
        client = with_memory(OpenAI(), user_id="alice")
        response = client.chat.completions.create(...)
    """

    def __init__(
        self,
        client: Any,
        user_id: str,
        db_path: str | Path = "headroom_memory.db",
        top_k: int = 5,
        session_id: str | None = None,
        agent_id: str | None = None,
        embedder_backend: EmbedderBackend = EmbedderBackend.LOCAL,
        openai_api_key: str | None = None,
        _memory: HierarchicalMemory | None = None,  # For testing
    ):
        """Initialize the memory wrapper.

        Args:
            client: LLM client (OpenAI, Anthropic, etc.)
            user_id: User identifier for memory isolation
            db_path: Path to SQLite database
            top_k: Number of memories to inject
            session_id: Optional session ID for session-scoped memories
            agent_id: Optional agent ID for agent-scoped memories
            embedder_backend: Which embedder to use (LOCAL or OPENAI)
            openai_api_key: API key if using OpenAI embeddings
            _memory: Override memory system (for testing)
        """
        self._client = client
        self._user_id = user_id
        self._session_id = session_id
        self._agent_id = agent_id
        self._top_k = top_k
        self._db_path = Path(db_path)

        # Initialize memory system (async, so we defer)
        self._memory = _memory
        self._memory_config = MemoryConfig(
            db_path=self._db_path,
            embedder_backend=embedder_backend,
            openai_api_key=openai_api_key,
        )
        self._initialized = _memory is not None

        # Create wrapped chat interface
        self.chat = _WrappedChat(self)

    def _ensure_initialized(self) -> None:
        """Ensure memory system is initialized (sync wrapper for async init)."""
        if not self._initialized:
            # Run async initialization in sync context
            loop = asyncio.new_event_loop()
            try:
                self._memory = loop.run_until_complete(
                    HierarchicalMemory.create(self._memory_config)
                )
                self._initialized = True
            finally:
                loop.close()

    @property
    def memory(self) -> _MemoryAPI:
        """Direct access to memory operations."""
        self._ensure_initialized()
        assert self._memory is not None
        return _MemoryAPI(
            self._memory,
            self._user_id,
            self._session_id,
            self._agent_id,
        )

    def _inject_memories(self, messages: list[dict]) -> list[dict]:
        """Inject relevant memories into messages.

        Uses semantic search to find relevant memories.
        Memories are prepended to the FIRST user message to preserve
        system prompt caching.

        Args:
            messages: Original messages list

        Returns:
            New messages list with memories injected
        """
        self._ensure_initialized()
        assert self._memory is not None

        # Find the last user message for search context
        user_content = None
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        if not user_content:
            return messages

        # Search for relevant memories (async -> sync)
        loop = asyncio.new_event_loop()
        try:
            memories = loop.run_until_complete(
                self._memory.search(
                    query=str(user_content),
                    user_id=self._user_id,
                    session_id=self._session_id,
                    top_k=self._top_k,
                )
            )
        finally:
            loop.close()

        if not memories:
            return messages

        # Build context block (search returns VectorSearchResult with .memory attr)
        context_lines = ["<context>"]
        for result in memories:
            context_lines.append(f"- {result.memory.content}")
        context_lines.append("</context>")
        context_block = "\n".join(context_lines)

        # Find the first user message and prepend context
        new_messages = copy.deepcopy(messages)
        for msg in new_messages:
            if msg.get("role") == "user":
                original = msg.get("content", "")
                msg["content"] = f"{context_block}\n\n{original}"
                break

        return new_messages

    def _store_memories(self, memories: list[dict[str, Any]]) -> None:
        """Store extracted memories.

        Args:
            memories: List of memory dicts from inline extraction
        """
        self._ensure_initialized()
        assert self._memory is not None

        loop = asyncio.new_event_loop()
        try:
            for mem in memories:
                content = mem.get("content", "")

                if content:
                    loop.run_until_complete(
                        self._memory.add(
                            content=content,
                            user_id=self._user_id,
                            session_id=self._session_id,
                            agent_id=self._agent_id,
                            importance=0.7,  # Default importance for extracted memories
                        )
                    )
        finally:
            loop.close()


class _WrappedChat:
    """Wrapped chat interface that intercepts completions."""

    def __init__(self, wrapper: MemoryWrapper):
        self._wrapper = wrapper
        self.completions = _WrappedCompletions(wrapper)


class _WrappedCompletions:
    """Wrapped completions with inline memory extraction."""

    def __init__(self, wrapper: MemoryWrapper):
        self._wrapper = wrapper

    def create(self, **kwargs: Any) -> Any:
        """Create a chat completion with memory injection and inline extraction.

        Flow:
        1. Search for relevant memories (semantic)
        2. Inject memories into user message
        3. Add memory extraction instruction to system prompt
        4. Forward to LLM
        5. Parse response to extract memories
        6. Store extracted memories
        7. Return clean response (without memory block)

        All kwargs are passed through to the underlying client.
        """
        messages = kwargs.get("messages", [])

        # 1. Inject relevant memories into user message
        enhanced_messages = self._wrapper._inject_memories(messages)

        # 2. Add memory extraction instruction to system prompt
        enhanced_messages = inject_memory_instruction(enhanced_messages, short=True)
        kwargs["messages"] = enhanced_messages

        # 3. Forward to LLM
        response = self._wrapper._client.chat.completions.create(**kwargs)

        # 4. Parse response and extract memories
        raw_content = response.choices[0].message.content
        parsed = parse_response_with_memory(raw_content)

        # 5. Store extracted memories
        if parsed.memories:
            self._wrapper._store_memories(parsed.memories)
            logger.debug(f"Extracted and stored {len(parsed.memories)} memories")

        # 6. Return clean response (modify in place)
        response.choices[0].message.content = parsed.content

        return response


class _MemoryAPI:
    """Direct API for memory operations."""

    def __init__(
        self,
        memory: HierarchicalMemory,
        user_id: str,
        session_id: str | None = None,
        agent_id: str | None = None,
    ):
        self._memory = memory
        self._user_id = user_id
        self._session_id = session_id
        self._agent_id = agent_id

    def _run_async(self, coro: Any) -> Any:
        """Run async coroutine in sync context."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def search(self, query: str, top_k: int = 5) -> list[Memory]:
        """Semantic search for memories.

        Args:
            query: Search query
            top_k: Max results

        Returns:
            Matching memories
        """
        results = self._run_async(
            self._memory.search(
                query=query,
                user_id=self._user_id,
                session_id=self._session_id,
                top_k=top_k,
            )
        )
        # Extract Memory objects from VectorSearchResult
        return [r.memory for r in results]

    def add(
        self,
        content: str,
        importance: float = 0.5,
    ) -> Memory:
        """Manually add a memory.

        Args:
            content: Memory content
            importance: 0.0-1.0

        Returns:
            The created memory
        """
        result: Memory = self._run_async(
            self._memory.add(
                content=content,
                user_id=self._user_id,
                session_id=self._session_id,
                agent_id=self._agent_id,
                importance=importance,
            )
        )
        return result

    def get_all(self) -> list[Memory]:
        """Get all memories for this user."""
        from headroom.memory.ports import MemoryFilter

        filter = MemoryFilter(user_id=self._user_id)
        memories: list[Memory] = self._run_async(self._memory.query(filter))
        return memories

    def clear(self) -> int:
        """Clear all memories for this user."""
        count: int = self._run_async(self._memory.clear_scope(user_id=self._user_id))
        return count

    def stats(self) -> dict:
        """Get memory statistics."""
        memories = self.get_all()

        return {
            "total": len(memories),
        }


def with_memory(
    client: Any,
    user_id: str,
    db_path: str | Path = "headroom_memory.db",
    top_k: int = 5,
    session_id: str | None = None,
    agent_id: str | None = None,
    embedder_backend: EmbedderBackend = EmbedderBackend.LOCAL,
    openai_api_key: str | None = None,
    **kwargs: Any,
) -> MemoryWrapper:
    """Wrap an LLM client to add automatic memory with zero extra latency.

    Uses inline extraction (Letta-style) - memories are extracted as part
    of the LLM response, not in a separate API call.

    Args:
        client: LLM client (OpenAI, Anthropic, Mistral, Groq, etc.)
        user_id: User identifier for memory isolation
        db_path: Path to SQLite database (default: headroom_memory.db)
        top_k: Number of memories to inject per request (default: 5)
        session_id: Optional session ID for session-scoped memories
        agent_id: Optional agent ID for agent-scoped memories
        embedder_backend: Which embedder to use (LOCAL or OPENAI)
        openai_api_key: API key if using OpenAI embeddings
        **kwargs: Additional arguments passed to MemoryWrapper

    Returns:
        Wrapped client with automatic memory

    Example:
        from openai import OpenAI
        from headroom import with_memory

        client = with_memory(OpenAI(), user_id="alice")

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "I prefer Python"}]
        )
        # Memory automatically extracted INLINE (zero extra latency!)

        # Later...
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "What language should I use?"}]
        )
        # Memory about Python preference automatically injected!
    """
    return MemoryWrapper(
        client=client,
        user_id=user_id,
        db_path=db_path,
        top_k=top_k,
        session_id=session_id,
        agent_id=agent_id,
        embedder_backend=embedder_backend,
        openai_api_key=openai_api_key,
        **kwargs,
    )
