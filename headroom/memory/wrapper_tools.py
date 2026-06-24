"""LLM client wrapper that adds memory tools.

Instead of auto-injecting memories, this wrapper:
1. Adds memory tools to every request
2. Intercepts tool calls and handles memory operations
3. Returns results with tool responses

Usage:
    from openai import OpenAI
    from headroom.memory import with_memory_tools, LocalBackend

    client = with_memory_tools(
        OpenAI(),
        backend=LocalBackend(),
        user_id="alice",
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Remember that I like Python"}]
    )
    # LLM will call memory_save tool if it decides to save this

Optimized Mode:
    For best performance with Mem0-backed storage, use optimized=True:

    client = with_memory_tools(
        OpenAI(),
        backend=DirectMem0Adapter(config),
        user_id="alice",
        optimized=True,  # Enables pre-extraction
    )

    This will:
    1. Use enhanced tool schemas with pre-extraction fields (facts, entities, relationships)
    2. Inject extraction system prompt so LLM extracts structured data
    3. Bypass Mem0's internal LLM calls - 0 backend LLM calls vs 3-4!
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
from typing import Any, TypeVar

from headroom.memory.extraction import EXTRACTION_SYSTEM_PROMPT
from headroom.memory.system import MemoryBackend, MemorySystem
from headroom.memory.tools import get_memory_tools, get_memory_tools_optimized, get_tool_names

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MemoryToolsWrapper:
    """Wrapper that adds memory tools to an OpenAI-compatible client.

    This wrapper takes a different approach from `with_memory`:
    - Instead of inline extraction, it provides explicit memory tools
    - The LLM decides when to save/search/update/delete memories
    - Tool calls are intercepted and processed automatically

    Optimized Mode (Letta-style):
        When optimized=True, the wrapper enables a more efficient memory flow:
        - Uses enhanced tool schemas with pre-extraction fields
        - Injects extraction system prompt so LLM extracts facts/entities/relationships
        - Backend can bypass internal LLM extraction (0 calls vs 3-4 with Mem0)

    Attributes:
        memory: Access to the underlying MemorySystem for manual operations
    """

    def __init__(
        self,
        client: T,
        backend: MemoryBackend,
        user_id: str,
        session_id: str | None = None,
        auto_handle_tools: bool = True,
        optimized: bool = False,
        inject_extraction_prompt: bool = True,
    ):
        """Initialize the memory tools wrapper.

        Args:
            client: OpenAI-compatible client (OpenAI, Azure, etc.)
            backend: Memory backend (LocalBackend, Mem0Backend, etc.)
            user_id: User identifier for scoping memory operations
            session_id: Optional session identifier
            auto_handle_tools: If True, automatically process memory tool calls
                and store results on the response object
            optimized: If True, use enhanced tool schemas with pre-extraction
                fields (facts, entities, relationships). When the LLM provides
                these fields, the backend can bypass internal LLM extraction.
                Use with DirectMem0Adapter for best performance.
            inject_extraction_prompt: If True (and optimized=True), inject the
                extraction system prompt into messages so the LLM knows to
                extract structured data when calling memory_save.
        """
        self._client: Any = client
        self._memory = MemorySystem(backend, user_id, session_id)
        self._auto_handle = auto_handle_tools
        self._optimized = optimized
        self._inject_extraction_prompt = inject_extraction_prompt and optimized

    @property
    def memory(self) -> MemorySystem:
        """Access the underlying MemorySystem for manual operations.

        Use this to directly call memory operations:
            client.memory.get_tools()
            await client.memory.process_tool_call("memory_save", {...})
        """
        return self._memory

    @property
    def chat(self) -> MemoryToolsChatCompletions:
        """Access the wrapped chat interface."""
        return MemoryToolsChatCompletions(
            self._client.chat,
            self._memory,
            self._auto_handle,
            self._optimized,
            self._inject_extraction_prompt,
        )

    def __getattr__(self, name: str) -> Any:
        """Proxy other attributes to underlying client.

        This allows accessing other client features like:
            client.models.list()
            client.embeddings.create(...)
        """
        return getattr(self._client, name)


class MemoryToolsChatCompletions:
    """Proxies chat.completions with memory tools injection."""

    def __init__(
        self,
        chat: Any,
        memory: MemorySystem,
        auto_handle: bool,
        optimized: bool = False,
        inject_extraction_prompt: bool = False,
    ):
        self._chat = chat
        self._memory = memory
        self._auto_handle = auto_handle
        self._optimized = optimized
        self._inject_extraction_prompt = inject_extraction_prompt

    @property
    def completions(self) -> MemoryToolsCompletions:
        """Access the wrapped completions interface."""
        return MemoryToolsCompletions(
            self._chat.completions,
            self._memory,
            self._auto_handle,
            self._optimized,
            self._inject_extraction_prompt,
        )


class MemoryToolsCompletions:
    """Proxies completions.create with memory tools injection and handling."""

    def __init__(
        self,
        completions: Any,
        memory: MemorySystem,
        auto_handle: bool,
        optimized: bool = False,
        inject_extraction_prompt: bool = False,
    ):
        self._completions = completions
        self._memory = memory
        self._auto_handle = auto_handle
        self._optimized = optimized
        self._inject_extraction_prompt = inject_extraction_prompt

    def _run_async(self, coro: Any) -> Any:
        """Run async coroutine in sync context."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None:
            # We're in an async context - create a new thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        else:
            # No running loop - safe to use asyncio.run
            return asyncio.run(coro)

    def _process_memory_tool_calls(
        self,
        response: Any,
        memory_tool_names: set[str],
    ) -> dict[str, Any]:
        """Process memory tool calls from the response.

        Args:
            response: The API response object
            memory_tool_names: Set of memory tool names to handle

        Returns:
            Dict mapping tool_call.id to result
        """
        results: dict[str, Any] = {}

        if not hasattr(response, "choices") or not response.choices:
            return results

        message = response.choices[0].message
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return results

        for tool_call in message.tool_calls:
            if tool_call.function.name in memory_tool_names:
                try:
                    args = json.loads(tool_call.function.arguments)
                    result = self._run_async(
                        self._memory.process_tool_call(
                            tool_call.function.name,
                            args,
                        )
                    )
                    results[tool_call.id] = result
                    logger.debug(
                        f"Processed memory tool {tool_call.function.name}: "
                        f"{result.get('message', 'success')}"
                    )
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse tool arguments for {tool_call.function.name}: {e}"
                    )
                    results[tool_call.id] = {
                        "success": False,
                        "error": f"Invalid JSON arguments: {e}",
                        "message": "Failed to parse tool call arguments",
                    }
                except Exception as e:
                    logger.error(f"Error processing memory tool {tool_call.function.name}: {e}")
                    results[tool_call.id] = {
                        "success": False,
                        "error": str(e),
                        "message": f"Failed to execute {tool_call.function.name}",
                    }

        return results

    def _prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prepare messages, optionally injecting extraction prompt.

        Args:
            messages: Original messages list

        Returns:
            Messages with extraction prompt injected if optimized mode enabled.
        """
        if not self._inject_extraction_prompt:
            return messages

        # Deep copy to avoid mutating original
        messages = copy.deepcopy(messages)

        # Find or create system message
        system_idx = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "system":
                system_idx = i
                break

        extraction_instruction = f"\n\n{EXTRACTION_SYSTEM_PROMPT}"

        if system_idx is not None:
            # Append to existing system message
            messages[system_idx]["content"] += extraction_instruction
        else:
            # Insert new system message at the beginning
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": EXTRACTION_SYSTEM_PROMPT.strip(),
                },
            )

        return messages

    def create(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a chat completion with memory tools.

        Memory tools are automatically added to the tools list.
        If auto_handle_tools is enabled, memory tool calls are processed
        and results are stored on the response object.

        In optimized mode (optimized=True):
        - Uses enhanced tool schemas with pre-extraction fields
        - Injects extraction system prompt so LLM extracts facts/entities
        - Enables backends to bypass internal LLM extraction

        Args:
            messages: List of message dicts
            tools: Optional list of additional tools (memory tools will be merged)
            **kwargs: Additional arguments passed to the underlying API

        Returns:
            API response with optional _memory_tool_results attribute
            containing processed memory tool results.

        Example:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Remember I like Python"}]
            )

            # If LLM called memory tools and auto_handle is enabled:
            if hasattr(response, '_memory_tool_results'):
                for tool_id, result in response._memory_tool_results.items():
                    print(f"Tool {tool_id}: {result['message']}")
        """
        # Get memory tools - use optimized version if enabled
        if self._optimized:
            memory_tools = get_memory_tools_optimized()
        else:
            memory_tools = get_memory_tools()

        all_tools = memory_tools + (tools or [])

        # Prepare messages (inject extraction prompt if optimized)
        prepared_messages = self._prepare_messages(messages)

        # Make the API call
        response = self._completions.create(
            messages=prepared_messages,
            tools=all_tools,
            **kwargs,
        )

        # Process memory tool calls if auto_handle is enabled
        if self._auto_handle:
            memory_tool_names = set(get_tool_names())
            results = self._process_memory_tool_calls(response, memory_tool_names)
            if results:
                response._memory_tool_results = results

        return response

    async def acreate(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Async version of create.

        Create a chat completion with memory tools asynchronously.

        In optimized mode (optimized=True):
        - Uses enhanced tool schemas with pre-extraction fields
        - Injects extraction system prompt so LLM extracts facts/entities
        - Enables backends to bypass internal LLM extraction

        Args:
            messages: List of message dicts
            tools: Optional list of additional tools (memory tools will be merged)
            **kwargs: Additional arguments passed to the underlying API

        Returns:
            API response with optional _memory_tool_results attribute.
        """
        # Get memory tools - use optimized version if enabled
        if self._optimized:
            memory_tools = get_memory_tools_optimized()
        else:
            memory_tools = get_memory_tools()

        all_tools = memory_tools + (tools or [])

        # Prepare messages (inject extraction prompt if optimized)
        prepared_messages = self._prepare_messages(messages)

        # Make the async API call
        # Try async method first, fall back to sync if not available
        if hasattr(self._completions, "acreate"):
            response = await self._completions.acreate(
                messages=prepared_messages,
                tools=all_tools,
                **kwargs,
            )
        elif hasattr(self._completions, "create") and asyncio.iscoroutinefunction(
            self._completions.create
        ):
            response = await self._completions.create(
                messages=prepared_messages,
                tools=all_tools,
                **kwargs,
            )
        else:
            # Fall back to sync in executor
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._completions.create(
                    messages=prepared_messages,
                    tools=all_tools,
                    **kwargs,
                ),
            )

        # Process memory tool calls if auto_handle is enabled
        if self._auto_handle:
            memory_tool_names = set(get_tool_names())
            results = await self._aprocess_memory_tool_calls(response, memory_tool_names)
            if results:
                response._memory_tool_results = results

        return response

    async def _aprocess_memory_tool_calls(
        self,
        response: Any,
        memory_tool_names: set[str],
    ) -> dict[str, Any]:
        """Async version of _process_memory_tool_calls.

        Args:
            response: The API response object
            memory_tool_names: Set of memory tool names to handle

        Returns:
            Dict mapping tool_call.id to result
        """
        results: dict[str, Any] = {}

        if not hasattr(response, "choices") or not response.choices:
            return results

        message = response.choices[0].message
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return results

        for tool_call in message.tool_calls:
            if tool_call.function.name in memory_tool_names:
                try:
                    args = json.loads(tool_call.function.arguments)
                    result = await self._memory.process_tool_call(
                        tool_call.function.name,
                        args,
                    )
                    results[tool_call.id] = result
                    logger.debug(
                        f"Processed memory tool {tool_call.function.name}: "
                        f"{result.get('message', 'success')}"
                    )
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse tool arguments for {tool_call.function.name}: {e}"
                    )
                    results[tool_call.id] = {
                        "success": False,
                        "error": f"Invalid JSON arguments: {e}",
                        "message": "Failed to parse tool call arguments",
                    }
                except Exception as e:
                    logger.error(f"Error processing memory tool {tool_call.function.name}: {e}")
                    results[tool_call.id] = {
                        "success": False,
                        "error": str(e),
                        "message": f"Failed to execute {tool_call.function.name}",
                    }

        return results


def with_memory_tools(
    client: T,
    backend: MemoryBackend,
    user_id: str,
    session_id: str | None = None,
    auto_handle_tools: bool = True,
    optimized: bool = False,
    inject_extraction_prompt: bool = True,
) -> MemoryToolsWrapper:
    """Wrap an OpenAI-compatible client with memory tools.

    This wrapper adds memory tools to every chat completion request,
    allowing the LLM to autonomously manage memories through function calling.

    Unlike `with_memory` which uses inline extraction, this approach:
    - Gives the LLM explicit control over memory operations
    - Uses standard function calling (works with any compatible model)
    - Provides more transparency about what's being saved

    Optimized Mode (Letta-style):
        When optimized=True, enables efficient memory extraction:
        - Uses enhanced tool schemas with pre-extraction fields (facts, entities, relationships)
        - Injects extraction system prompt so LLM extracts structured data
        - Backend can bypass internal LLM extraction (0 calls vs 3-4 with Mem0!)

        Use with DirectMem0Adapter for best performance.

    Args:
        client: OpenAI-compatible client (OpenAI, Azure, Anthropic, etc.)
        backend: Memory backend to use (LocalBackend, DirectMem0Adapter, etc.)
        user_id: User identifier for scoping memory operations
        session_id: Optional session identifier
        auto_handle_tools: If True, automatically process memory tool calls
            and store results on the response object. Set to False if you
            want to handle tool calls manually.
        optimized: If True, use enhanced tool schemas with pre-extraction
            fields. When the LLM provides facts/entities/relationships,
            the backend can bypass internal LLM extraction for significant
            performance improvement. Use with DirectMem0Adapter or LocalBackend.
        inject_extraction_prompt: If True (and optimized=True), inject the
            extraction system prompt into messages so the LLM knows to
            extract structured data when calling memory_save.

    Returns:
        Wrapped client with memory tools enabled

    Example:
        from openai import OpenAI
        from headroom.memory import with_memory_tools
        from headroom.memory.backends import LocalBackend

        # Standard mode - basic memory tools
        client = with_memory_tools(
            OpenAI(),
            backend=LocalBackend(),
            user_id="alice",
        )

        # Optimized mode - pre-extraction for better performance
        client = with_memory_tools(
            OpenAI(),
            backend=LocalBackend(),
            user_id="alice",
            optimized=True,  # Enable pre-extraction
        )

        # Use normally - memory tools are automatically available
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "I work at Netflix using Python"}]
        )

        # In optimized mode, LLM will extract and include:
        # - facts: ["Works at Netflix", "Uses Python"]
        # - extracted_entities: [{"entity": "Netflix", "entity_type": "organization"}]
        # - extracted_relationships: [{"source": "user", "relationship": "works_at", "destination": "Netflix"}]

        # Results are available on response._memory_tool_results
        if hasattr(response, '_memory_tool_results'):
            for tool_id, result in response._memory_tool_results.items():
                print(f"Tool {tool_id}: {result['message']}")
    """
    return MemoryToolsWrapper(
        client,
        backend=backend,
        user_id=user_id,
        session_id=session_id,
        auto_handle_tools=auto_handle_tools,
        optimized=optimized,
        inject_extraction_prompt=inject_extraction_prompt,
    )
