"""Memory integration handler for the proxy server.

This module provides memory capabilities for the Headroom proxy:
1. MemoryHandler - Unified handler for memory operations
   - inject_tools() - Add memory tools to requests
   - search_and_format_context() - Search memories, format for injection
   - has_memory_tool_calls() - Detect memory tool usage in response
   - handle_memory_tool_calls() - Execute tools, return results

Usage:
    config = MemoryConfig(enabled=True, backend="local")
    handler = MemoryHandler(config)

    # Inject tools into request
    tools, was_injected = handler.inject_tools(existing_tools, "anthropic")

    # Search and inject context
    context = await handler.search_and_format_context(user_id, messages)

    # Handle tool calls in response
    if handler.has_memory_tool_calls(response, "anthropic"):
        results = await handler.handle_memory_tool_calls(response, user_id, "anthropic")
"""

from __future__ import annotations

import asyncio
import enum
import inspect
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from headroom.memory import qdrant_env

if TYPE_CHECKING:
    from headroom.memory.backends.local import LocalBackend

logger = logging.getLogger(__name__)


class MemoryMode(str, enum.Enum):
    """Memory injection mode (PR-B6).

    AUTO_TAIL (default): Memory retrieval runs at request entry; results are
    appended to the latest user message tail (the live zone). The cache hot
    zone (system prompt / instructions / frozen prefix) is never mutated —
    invariant I2 from PR-A2.

    TOOL: Auto-injection is disabled entirely. The model calls the
    ``memory_search`` tool explicitly when it wants memory; retrieval runs in
    the tool execution path, not the prompt-construction path. Memory becomes
    opt-in (and visible to the model) rather than implicit.

    See REALIGNMENT/04-phase-B-live-zone.md PR-B6 for the rationale.
    """

    AUTO_TAIL = "auto_tail"
    TOOL = "tool"


# Memory tool names for detection (Headroom's custom tools)
MEMORY_TOOL_NAMES = {"memory_save", "memory_search", "memory_update", "memory_delete"}

# Anthropic's native memory tool name
NATIVE_MEMORY_TOOL_NAME = "memory"

# Beta header required for native memory tool
NATIVE_MEMORY_BETA_HEADER = "context-management-2025-06-27"

# Native memory tool type
NATIVE_MEMORY_TOOL_TYPE = "memory_20250818"

# Maximum time to wait for a single backend initialization (one-shot).
# Applies to MemoryHandler._ensure_initialized. On timeout, _initialized
# stays False so that subsequent requests retry instead of deadlocking.
# See wiki/plans/2026-04-17-fix-codex-proxy-resilience-plan.md "Risks" row 7.
STARTUP_INIT_TIMEOUT_SECONDS = 30.0


@dataclass
class MemoryConfig:
    """Configuration for memory handler.

    Qdrant connection fields default to values read from ``HEADROOM_QDRANT_*``
    environment variables (see :mod:`headroom.memory.qdrant_env`). Passing an
    explicit value to the constructor always wins over the environment.
    """

    enabled: bool = False
    backend: Literal["local", "qdrant-neo4j"] = "local"
    db_path: str = "headroom_memory.db"
    inject_tools: bool = True
    inject_context: bool = True
    top_k: int = 10
    min_similarity: float = 0.3
    # PR-B6: Memory injection mode. AUTO_TAIL (default) auto-appends retrieved
    # memory to the latest user message tail. TOOL disables auto-injection;
    # the model must call ``memory_search`` to retrieve. Configurable per
    # deployment via ``ProxyConfig.memory_mode``.
    mode: MemoryMode = MemoryMode.AUTO_TAIL
    # Native memory tool (Anthropic's built-in memory_20250818)
    use_native_tool: bool = False
    native_memory_dir: str = ""  # Directory for native memory files (default: ~/.headroom/memories)
    # Qdrant+Neo4j config (Qdrant defaults resolve from HEADROOM_QDRANT_* env vars)
    qdrant_url: str | None = field(default_factory=qdrant_env.qdrant_env_url)
    qdrant_host: str = field(default_factory=qdrant_env.qdrant_env_host)
    qdrant_port: int = field(default_factory=qdrant_env.qdrant_env_port)
    qdrant_api_key: str | None = field(default_factory=qdrant_env.qdrant_env_api_key)
    neo4j_uri: str = "neo4j://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    # Memory Bridge (bidirectional markdown <-> Headroom sync)
    bridge_enabled: bool = False
    bridge_md_paths: list[str] = field(default_factory=list)
    bridge_md_format: str = "auto"
    bridge_auto_import: bool = False
    bridge_export_path: str = ""


class MemoryHandler:
    """Unified handler for memory operations in the proxy.

    Responsibilities:
    1. Initialize and manage memory backend
    2. Inject memory tools into requests
    3. Search and inject relevant memories as context
    4. Handle memory tool calls in responses

    Supports two modes:
    - Custom tools: Headroom's memory_save, memory_search, etc. (default)
    - Native tool: Anthropic's memory_20250818 built-in tool (experimental)
    """

    # Cosine similarity thresholds for dedup
    DEDUP_AUTO_THRESHOLD = 0.92  # Auto-supersede (same fact, different wording)
    DEDUP_HINT_THRESHOLD = 0.75  # Suggest merge to LLM (related, possibly duplicate)

    def __init__(self, config: MemoryConfig, agent_type: str = "unknown") -> None:
        self.config = config
        self.agent_type = agent_type
        self._backend: LocalBackend | Any = None
        self._initialized = False
        # Async singleflight guard for backend init. Ensures concurrent first
        # callers land on one init (double-checked pattern inside
        # _ensure_initialized). Not used by the legacy sync _initialized flag
        # on its own because that flag isn't atomic across await points.
        self._init_lock: asyncio.Lock | None = None
        self._memory_tools: list[dict[str, Any]] | None = None
        # Native memory tool directory
        self._native_memory_dir: Path | None = None
        if config.use_native_tool:
            self._init_native_memory_dir()
        # Memory Bridge
        self._bridge: Any = None  # MemoryBridge, lazy imported

    def _get_init_lock(self) -> asyncio.Lock:
        """Lazily create the init lock bound to the running event loop.

        Avoids ``DeprecationWarning: There is no current event loop`` when
        ``MemoryHandler`` is constructed before the loop is set up.
        """
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        return self._init_lock

    async def _close_backend_instance(self, backend: Any, *, reason: str) -> None:
        """Best-effort close for a partially initialized backend."""
        close = getattr(backend, "close", None)
        if not callable(close):
            return
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.warning(
                "Memory: failed to close backend during %s cleanup: %s",
                reason,
                exc,
            )

    def _init_native_memory_dir(self) -> None:
        """Initialize native memory directory."""
        if self.config.native_memory_dir:
            self._native_memory_dir = Path(self.config.native_memory_dir)
        else:
            # Default: workspace memories directory (respects HEADROOM_WORKSPACE_DIR)
            from headroom import paths as _paths

            self._native_memory_dir = _paths.native_memory_dir()

        # Create directory if it doesn't exist
        self._native_memory_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Memory: Native memory directory: {self._native_memory_dir}")

    def get_beta_headers(self) -> dict[str, str]:
        """Get beta headers required for native memory tool.

        Returns:
            Dict with beta headers to add to request, or empty dict.
        """
        if self.config.use_native_tool and self.config.inject_tools:
            return {"anthropic-beta": NATIVE_MEMORY_BETA_HEADER}
        return {}

    async def _ensure_initialized(self) -> None:
        """Lazy initialization of memory backend.

        Singleflight via ``self._init_lock`` with double-checked pattern:
        concurrent first callers await the same load rather than triggering
        N parallel backend inits. Wrapped in ``asyncio.wait_for`` with a
        configurable timeout (``STARTUP_INIT_TIMEOUT_SECONDS``); on timeout
        ``self._initialized`` stays ``False`` so a later request can retry
        (fail-open contract — no exception propagates to request handlers).
        """
        # Fast path: already initialized, no lock contention.
        if self._initialized:
            return

        if not self.config.enabled:
            return

        lock = self._get_init_lock()

        async def _do_init() -> None:
            async with lock:
                # Double-check after acquiring the lock — another task may
                # have completed the init while we were waiting.
                if self._initialized:
                    return
                await self._init_backend_locked()

        try:
            await asyncio.wait_for(_do_init(), timeout=STARTUP_INIT_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            # Fail-open: leave _initialized=False so subsequent calls retry.
            # CRITICAL: also null the backend — _init_backend_locked may have
            # already assigned ``self._backend`` before its own await raised /
            # was cancelled by wait_for. Callers that do
            # ``if self.memory_handler._backend:`` must not see a
            # truthy-but-broken backend.
            existing_backend = self._backend
            if existing_backend is not None:
                await self._close_backend_instance(existing_backend, reason="timeout")
            self._backend = None
            self._initialized = False
            logger.error(
                "Memory: backend initialization timed out after "
                f"{STARTUP_INIT_TIMEOUT_SECONDS}s "
                f"(backend={self.config.backend}). "
                "Subsequent requests will retry."
            )
            return
        except asyncio.CancelledError:
            # External cancellation (shutdown / task cancelled).
            # CancelledError is BaseException — the TimeoutError branch
            # above does NOT catch it, and caller ``except Exception``
            # blocks don't either, so it propagates unconditionally.
            # Reset state so any later retry starts clean, then re-raise:
            # cancellation is a signal, not an error to swallow.
            existing_backend = self._backend
            if existing_backend is not None:
                await self._close_backend_instance(existing_backend, reason="cancellation")
            self._backend = None
            self._initialized = False
            logger.info(f"Memory: backend initialization cancelled (backend={self.config.backend})")
            raise

    async def _init_backend_locked(self) -> None:
        """Actual backend-init body. Must be called with ``_init_lock`` held."""
        if self.config.backend == "local":
            from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

            # Auto-detect embedder: ONNX (default, ~86MB, no torch) → local (if torch available)
            embedder_backend = "onnx"
            embedder_model = "all-MiniLM-L6-v2"
            vector_dimension = 384

            # Check if ONNX runtime is available (should be — it's in proxy deps)
            try:
                import onnxruntime  # noqa: F401
            except ImportError:
                # Fall back to sentence-transformers (requires torch)
                embedder_backend = "local"
                logger.info(
                    "Memory: onnxruntime not available, falling back to sentence-transformers"
                )

            backend_config = LocalBackendConfig(
                db_path=self.config.db_path,
                embedder_backend=embedder_backend,
                embedder_model=embedder_model,
                vector_dimension=vector_dimension,
            )
            self._backend = LocalBackend(backend_config)
            await self._backend._ensure_initialized()
            logger.info(
                f"Memory: Initialized LocalBackend at {self.config.db_path} "
                f"(embedder: {embedder_backend})"
            )

        elif self.config.backend == "qdrant-neo4j":
            try:
                from headroom.memory.backends.direct_mem0 import (
                    DirectMem0Adapter,
                    Mem0Config,
                )

                mem0_config = Mem0Config(
                    qdrant_url=self.config.qdrant_url,
                    qdrant_host=self.config.qdrant_host,
                    qdrant_port=self.config.qdrant_port,
                    qdrant_api_key=self.config.qdrant_api_key,
                    neo4j_uri=self.config.neo4j_uri,
                    neo4j_user=self.config.neo4j_user,
                    neo4j_password=self.config.neo4j_password,
                    enable_graph=True,
                )
                self._backend = DirectMem0Adapter(mem0_config)
                await self._backend.ensure_initialized()
                qdrant_target = (
                    self.config.qdrant_url or f"{self.config.qdrant_host}:{self.config.qdrant_port}"
                )
                logger.info(f"Memory: Initialized Qdrant+Neo4j backend ({qdrant_target})")
            except ImportError as e:
                logger.error(
                    f"Memory: Failed to import qdrant-neo4j dependencies: {e}. "
                    "Install with: pip install 'headroom-ai[memory-stack]'"
                )
                raise
        else:
            raise ValueError(f"Unknown memory backend: {self.config.backend}")

        self._initialized = True

        # Auto-import from Memory Bridge if configured
        if self.config.bridge_enabled and self.config.bridge_auto_import:
            await self._init_and_import_bridge()

    async def _init_and_import_bridge(self) -> None:
        """Initialize the Memory Bridge and run auto-import."""
        if self._bridge is not None:
            return
        try:
            from headroom.memory.bridge import MemoryBridge
            from headroom.memory.bridge_config import BridgeConfig, MarkdownFormat

            bridge_config = BridgeConfig(
                md_paths=[Path(p) for p in self.config.bridge_md_paths],
                md_format=MarkdownFormat(self.config.bridge_md_format),
                auto_import_on_startup=True,
                export_path=Path(self.config.bridge_export_path)
                if self.config.bridge_export_path
                else None,
            )
            self._bridge = MemoryBridge(bridge_config, self._backend)
            stats = await self._bridge.import_from_markdown()
            logger.info(
                f"Memory Bridge: Auto-imported {stats.sections_imported} sections "
                f"({stats.sections_skipped_duplicate} duplicates skipped)"
            )
        except Exception as e:
            logger.warning(f"Memory Bridge: Auto-import failed: {e}")

    def _get_memory_tools(self) -> list[dict[str, Any]]:
        """Get memory tool definitions (cached)."""
        if self._memory_tools is None:
            from headroom.memory.tools import get_memory_tools_optimized

            self._memory_tools = get_memory_tools_optimized()
        return self._memory_tools

    def compute_memory_tool_definitions(
        self,
        provider: str = "anthropic",
    ) -> list[dict[str, Any]]:
        """Return the memory tool definitions for ``provider`` (pure, no I/O).

        Replaces the building half of ``inject_tools`` so the proxy
        injection path can route through ``SessionToolTracker`` (PR-A7).
        Honors ``self.config.use_native_tool`` for Anthropic so the
        native ``memory_20250818`` tool flows through the same sticky
        codepath as the custom ``memory_save`` / ``memory_search`` set.

        The returned list is a fresh list of dicts. Order is stable
        (matches ``_get_memory_tools()`` order) so the canonical bytes
        are deterministic across calls.
        """
        if not self.config.inject_tools:
            return []

        if self.config.use_native_tool and provider == "anthropic":
            return [
                {
                    "type": NATIVE_MEMORY_TOOL_TYPE,
                    "name": NATIVE_MEMORY_TOOL_NAME,
                }
            ]

        out: list[dict[str, Any]] = []
        for memory_tool in self._get_memory_tools():
            tool_name = memory_tool["function"]["name"]
            if provider == "anthropic":
                out.append(
                    {
                        "name": tool_name,
                        "description": memory_tool["function"]["description"],
                        "input_schema": memory_tool["function"]["parameters"],
                    }
                )
            else:
                # OpenAI format — return a fresh shallow copy so callers
                # can mutate without surprise. dict() is sufficient: the
                # nested schema is treated as immutable downstream.
                out.append(dict(memory_tool))
        return out

    def inject_tools(
        self,
        tools: list[dict[str, Any]] | None,
        provider: str = "anthropic",
    ) -> tuple[list[dict[str, Any]], bool]:
        """Inject memory tools into tools list.

        Args:
            tools: Existing tools list (may be None).
            provider: Provider for tool format ("anthropic" or "openai").

        Returns:
            Tuple of (updated_tools, was_injected).

        NOTE (PR-A7): The proxy now wires injection through
        ``apply_session_sticky_memory_tools`` so tool list bytes stay
        cache-stable across turns. This method remains as the
        non-session-aware fallback for tests / callers that don't have
        a session_id (e.g. diagnostic shadow runs).
        """
        if not self.config.inject_tools:
            return tools or [], False

        tools = list(tools) if tools else []

        # Use native memory tool if configured
        if self.config.use_native_tool:
            return self._inject_native_tool(tools)

        # Check which tools are already present
        existing_names: set[str] = set()
        for tool in tools:
            name = tool.get("name") or tool.get("function", {}).get("name")
            if name:
                existing_names.add(name)

        # Add missing memory tools
        was_injected = False
        for memory_tool in self._get_memory_tools():
            tool_name = memory_tool["function"]["name"]
            if tool_name in existing_names:
                continue

            # Convert to provider format
            if provider == "anthropic":
                tools.append(
                    {
                        "name": tool_name,
                        "description": memory_tool["function"]["description"],
                        "input_schema": memory_tool["function"]["parameters"],
                    }
                )
            else:
                # OpenAI format
                tools.append(memory_tool)

            was_injected = True

        return tools, was_injected

    def _inject_native_tool(self, tools: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
        """Inject Anthropic's native memory tool (memory_20250818).

        This uses Anthropic's built-in memory tool format which may be
        allowed by Claude Code subscription credentials (unlike custom tools).

        Returns:
            Tuple of (updated_tools, was_injected).
        """
        # Check if native memory tool already present
        for tool in tools:
            if tool.get("type") == NATIVE_MEMORY_TOOL_TYPE:
                return tools, False
            if tool.get("name") == NATIVE_MEMORY_TOOL_NAME:
                return tools, False

        # Add native memory tool
        native_tool = {
            "type": NATIVE_MEMORY_TOOL_TYPE,
            "name": NATIVE_MEMORY_TOOL_NAME,
        }
        tools.append(native_tool)

        logger.info(
            f"Memory: Injected native memory tool ({NATIVE_MEMORY_TOOL_TYPE}). "
            f"Beta header required: {NATIVE_MEMORY_BETA_HEADER}"
        )
        return tools, True

    async def search_and_format_context(
        self,
        user_id: str,
        messages: list[dict[str, Any]],
    ) -> str | None:
        """Search memories and format as context injection.

        Args:
            user_id: User identifier for memory scoping.
            messages: Conversation messages (used to extract query).

        Returns:
            Formatted context string, or None if no relevant memories.

        PR-B6: When ``self.config.mode == MemoryMode.TOOL``, this method
        returns ``None`` unconditionally so the proxy never auto-injects.
        The model must call ``memory_search`` explicitly to retrieve. This
        is the single chokepoint that gates auto-injection across all
        provider handlers (Anthropic /v1/messages, OpenAI /v1/chat/completions,
        OpenAI /v1/responses).
        """
        if not self.config.inject_context:
            return None

        # PR-B6: Tool mode disables auto-injection. The model calls
        # ``memory_search`` to retrieve when it wants to. Log the skip
        # decision so cache-affecting routing remains observable.
        if self.config.mode == MemoryMode.TOOL:
            logger.info(
                "event=memory_mode_skip mode=tool user_id=%s reason=tool_mode_no_auto_injection",
                user_id,
            )
            return None

        await self._ensure_initialized()
        if not self._backend:
            return None

        # Extract query from last user message
        query = self._extract_user_query(messages)
        if not query:
            logger.debug("Memory: No user query found for context search")
            return None

        try:
            # Search memories
            results = await self._backend.search_memories(
                query=query,
                user_id=user_id,
                top_k=self.config.top_k,
                include_related=True,
            )

            if not results:
                logger.debug(f"Memory: No memories found for user {user_id}")
                return None

            # Filter by minimum similarity
            filtered_results = [r for r in results if r.score >= self.config.min_similarity]

            if not filtered_results:
                logger.debug(
                    f"Memory: {len(results)} memories found but none above threshold "
                    f"{self.config.min_similarity}"
                )
                return None

            # Format as context
            memory_lines = []
            for i, result in enumerate(filtered_results, 1):
                memory_lines.append(f"{i}. {result.memory.content}")
                if hasattr(result, "related_entities") and result.related_entities:
                    entities_str = ", ".join(result.related_entities[:3])
                    memory_lines.append(f"   (Related: {entities_str})")

        except Exception as e:
            logger.warning(f"Memory: Search failed for user {user_id}: {e}")
            return None

        if not memory_lines:
            return None

        context = f"""## Relevant Memories for This User

The following information was previously saved about this user:

{chr(10).join(memory_lines)}

Use this context to provide personalized and contextually relevant responses."""

        logger.info(
            f"Memory: Injecting {len(memory_lines)} memories "
            f"({len(context)} chars) for user {user_id}"
        )
        return context

    @staticmethod
    def _append_to_latest_user_tail(
        messages: list[dict[str, Any]],
        context_text: str,
        *,
        provider: Literal["anthropic", "openai"] = "anthropic",
        frozen_message_count: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Append memory context to the live-zone tail (latest user message).

        PR-B6 canonical entry point for memory tail injection. Replaces the
        retired ``_inject_to_system_or_instructions`` path (deleted in PR-A2).
        The cache hot zone — system / instructions / frozen prefix — is never
        mutated by this helper.

        Args:
            messages: Provider-shaped message list. For Anthropic this is the
                Messages API ``messages`` array. For OpenAI Chat Completions
                this is ``body["messages"]``.
            context_text: Pre-formatted memory context block. Empty/missing
                returns the input unchanged.
            provider: ``"anthropic"`` or ``"openai"``. Selects the correct
                tail-append helper for the provider's content shape.
            frozen_message_count: For Anthropic: the cache-frozen prefix
                length. The latest user message must lie outside this prefix
                to be eligible for mutation. Ignored for OpenAI Chat
                Completions (which does not have a frozen-prefix concept on
                this path).

        Returns:
            ``(new_messages, bytes_appended)``. ``bytes_appended == 0`` means
            no eligible user text block was found; the message list is
            returned unchanged.

        Determinism: the bytes appended are byte-identical for the same
        ``context_text`` across runs. The caller is responsible for ensuring
        ``context_text`` itself is deterministic (i.e. that the upstream
        vector search produced the same results in the same order).
        """
        if not messages or not context_text:
            return messages, 0

        if provider == "anthropic":
            # Late import to avoid circular: AnthropicHandlerMixin lives in
            # headroom.proxy.handlers.anthropic which imports MemoryHandler.
            from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

            new_messages = AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
                messages,
                context_text,
                frozen_message_count=frozen_message_count,
            )
            if new_messages is messages:
                return messages, 0
            return new_messages, len(context_text)

        if provider == "openai":
            from headroom.proxy.helpers import append_text_to_latest_user_chat_message

            return append_text_to_latest_user_chat_message(messages, context_text)

        raise ValueError(f"Unknown provider {provider!r}; expected 'anthropic' or 'openai'")

    def _extract_user_query(self, messages: list[dict[str, Any]]) -> str:
        """Extract the user query from the last user message."""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue

            content = msg.get("content", "")

            if isinstance(content, str):
                return content[:500]  # Limit query length

            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = str(block.get("text", ""))
                        if text:
                            return text[:500]

        return ""

    def has_memory_tool_calls(
        self,
        response: dict[str, Any],
        provider: str = "anthropic",
    ) -> bool:
        """Check if response contains memory tool calls."""
        tool_calls = self._extract_tool_calls(response, provider)
        for tc in tool_calls:
            name = tc.get("name") or tc.get("function", {}).get("name")
            # Check for both custom and native memory tools
            if name in MEMORY_TOOL_NAMES or name == NATIVE_MEMORY_TOOL_NAME:
                return True
        return False

    def _extract_tool_calls(
        self,
        response: dict[str, Any],
        provider: str,
    ) -> list[dict[str, Any]]:
        """Extract tool calls from response based on provider format."""
        if provider == "anthropic":
            content = response.get("content", [])
            if isinstance(content, list):
                return [block for block in content if block.get("type") == "tool_use"]
            return []

        elif provider == "openai":
            # Chat Completions format: choices[0].message.tool_calls
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                tc_list = list(message.get("tool_calls", []) or [])
                if tc_list:
                    return tc_list

            # Responses API format: output[] with type=function_call
            output = response.get("output", [])
            if isinstance(output, list):
                fc_items = [
                    item
                    for item in output
                    if isinstance(item, dict) and item.get("type") == "function_call"
                ]
                if fc_items:
                    return fc_items

            return []

        return []

    async def handle_memory_tool_calls(
        self,
        response: dict[str, Any],
        user_id: str,
        provider: str = "anthropic",
    ) -> list[dict[str, Any]]:
        """Execute memory tool calls and return results.

        Args:
            response: The API response containing tool calls.
            user_id: User identifier for memory operations.
            provider: Provider format ("anthropic" or "openai").

        Returns:
            List of tool results in provider format.
        """
        tool_calls = self._extract_tool_calls(response, provider)
        results: list[dict[str, Any]] = []

        for tc in tool_calls:
            tool_name = tc.get("name") or tc.get("function", {}).get("name")
            tool_id = tc.get("id") or tc.get("call_id", "")

            # Parse input data
            if provider == "anthropic":
                input_data = tc.get("input", {})
            else:
                # Chat Completions format: function.arguments
                # Responses API format: arguments (top-level string)
                args_str = tc.get("arguments") or tc.get("function", {}).get("arguments") or "{}"
                try:
                    input_data = json.loads(args_str)
                except json.JSONDecodeError:
                    input_data = {}

            # Handle native memory tool
            if tool_name == NATIVE_MEMORY_TOOL_NAME:
                result_content = await self._execute_native_memory_tool(input_data, user_id)
            elif tool_name in MEMORY_TOOL_NAMES:
                # Custom memory tools need backend
                await self._ensure_initialized()
                if not self._backend:
                    continue
                result_content = await self._execute_memory_tool(
                    tool_name, input_data, user_id, provider
                )
            else:
                continue

            # Format result based on provider
            if provider == "anthropic":
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_id,
                        "content": result_content,
                    }
                )
            else:
                results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": result_content,
                    }
                )

            logger.info(f"Memory: Executed {tool_name} for user {user_id}")

        return results

    async def _execute_memory_tool(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        user_id: str,
        provider: str = "anthropic",
    ) -> str:
        """Execute a memory tool and return result string."""
        try:
            if tool_name == "memory_save":
                return await self._execute_save(input_data, user_id, provider)
            elif tool_name == "memory_search":
                return await self._execute_search(input_data, user_id)
            elif tool_name == "memory_update":
                return await self._execute_update(input_data, user_id, provider)
            elif tool_name == "memory_delete":
                return await self._execute_delete(input_data, user_id)
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})

        except Exception as e:
            logger.error(f"Memory: Tool {tool_name} failed: {e}")
            return json.dumps({"status": "error", "error": str(e)})

    async def _execute_save(
        self, input_data: dict[str, Any], user_id: str, provider: str = "anthropic"
    ) -> str:
        """Execute memory_save tool with provenance, dedup hints, and async background dedup."""
        content = input_data.get("content", "")
        if not content:
            return json.dumps({"status": "error", "error": "content is required"})

        # Extract parameters
        importance = input_data.get("importance", 0.5)
        facts = input_data.get("facts")
        entities = input_data.get("entities")
        extracted_entities = input_data.get("extracted_entities")
        relationships = input_data.get("relationships")
        extracted_relationships = input_data.get("extracted_relationships")

        # Agent provenance metadata
        provenance_metadata = {
            "source_agent": self.agent_type,
            "source_provider": provider,
            "created_via": "tool_call",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }

        # Save to backend
        memory = await self._backend.save_memory(
            content=content,
            user_id=user_id,
            importance=importance,
            facts=facts,
            entities=entities,
            extracted_entities=extracted_entities,
            relationships=relationships,
            extracted_relationships=extracted_relationships,
            metadata=provenance_metadata,
        )

        # Search for similar existing memories (for hints + async dedup)
        similar_memories = []
        try:
            results = await self._backend.search_memories(
                query=content,
                user_id=user_id,
                top_k=5,
            )
            # Exclude the memory we just saved
            similar_memories = [r for r in results if r.memory.id != memory.id]
        except Exception as e:
            logger.debug(f"Memory: Similar search failed during save: {e}")

        # Build response with dedup hints for the LLM
        result: dict[str, Any] = {
            "status": "saved",
            "memory_id": memory.id,
            "content": memory.content[:100] + "..."
            if len(memory.content) > 100
            else memory.content,
        }

        # Enriched hint: if similar memory exists, suggest merge to the LLM
        if similar_memories and similar_memories[0].score >= self.DEDUP_HINT_THRESHOLD:
            top = similar_memories[0]
            source_info = ""
            src = top.memory.metadata.get("source_agent", "")
            if src:
                source_info = f", saved by {src}"
            result["note"] = (
                f"Similar memory exists (id: {top.memory.id}, "
                f"{top.score:.0%} match{source_info}): "
                f"'{top.memory.content[:120]}'. "
                f"Call memory_update('{top.memory.id}', '<merged content>') to consolidate, "
                f"or ignore if these are distinct facts."
            )

        # Async background dedup: auto-supersede obvious duplicates
        if similar_memories:
            asyncio.create_task(self._background_dedup(memory.id, similar_memories, user_id))

        logger.info(
            f"Memory: Saved '{content[:60]}' for user {user_id} "
            f"(agent={self.agent_type}, provider={provider}, "
            f"similar={len(similar_memories)})"
        )

        return json.dumps(result)

    async def _background_dedup(
        self,
        new_memory_id: str,
        similar_results: list[Any],
        user_id: str,
    ) -> None:
        """Auto-supersede obvious duplicates in background (fire-and-forget).

        If an existing memory has >0.92 cosine similarity to the new one,
        mark the older one as superseded. This runs asynchronously and
        never blocks the tool response.
        """
        try:
            for result in similar_results:
                if result.score < self.DEDUP_AUTO_THRESHOLD:
                    continue
                if result.memory.id == new_memory_id:
                    continue

                old = result.memory
                # Skip if already superseded
                if old.metadata.get("superseded_by"):
                    continue

                # Mark old memory as superseded by deleting it
                # (update_memory creates a new version — for dedup we just remove the duplicate)
                if hasattr(self._backend, "delete_memory"):
                    await self._backend.delete_memory(old.id)
                    logger.info(
                        f"Memory dedup: removed '{old.content[:50]}' "
                        f"(superseded by {new_memory_id}, {result.score:.2f} cosine, "
                        f"agent={old.metadata.get('source_agent', '?')})"
                    )
        except Exception as e:
            logger.warning(f"Memory background dedup failed: {e}")

    async def _execute_search(self, input_data: dict[str, Any], user_id: str) -> str:
        """Execute memory_search tool."""
        query = input_data.get("query", "")
        if not query:
            return json.dumps({"status": "error", "error": "query is required"})

        top_k = input_data.get("top_k", 10)
        include_related = input_data.get("include_related", True)
        entities_filter = input_data.get("entities")

        results = await self._backend.search_memories(
            query=query,
            user_id=user_id,
            top_k=top_k,
            include_related=include_related,
            entities=entities_filter,
        )

        return json.dumps(
            {
                "status": "found",
                "count": len(results),
                "memories": [
                    {
                        "id": r.memory.id,
                        "content": r.memory.content,
                        "score": round(r.score, 3),
                        "entities": (
                            r.related_entities[:5]
                            if hasattr(r, "related_entities") and r.related_entities
                            else []
                        ),
                    }
                    for r in results
                ],
            }
        )

    async def _execute_update(
        self, input_data: dict[str, Any], user_id: str, provider: str = "anthropic"
    ) -> str:
        """Execute memory_update tool with edit history tracking."""
        memory_id = input_data.get("memory_id", "")
        new_content = input_data.get("new_content", "")

        if not memory_id:
            return json.dumps({"status": "error", "error": "memory_id is required"})
        if not new_content:
            return json.dumps({"status": "error", "error": "new_content is required"})

        reason = input_data.get("reason")

        # Build edit history entry
        edit_entry = {
            "agent": self.agent_type,
            "provider": provider,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
        }

        # Check if backend has update_memory method
        if hasattr(self._backend, "update_memory"):
            # Try to get old memory for history
            old_content = ""
            try:
                old_results = await self._backend.search_memories(
                    query=memory_id, user_id=user_id, top_k=1
                )
                if old_results:
                    old_content = old_results[0].memory.content[:200]
                    edit_entry["previous_content"] = old_content
            except Exception:
                pass

            memory = await self._backend.update_memory(
                memory_id=memory_id,
                new_content=new_content,
                reason=f"Updated by {self.agent_type} via {provider}: {reason or 'no reason'}",
                user_id=user_id,
            )
            logger.info(
                f"Memory: Updated {memory_id} by {self.agent_type} "
                f"(provider={provider}, reason={reason})"
            )
            return json.dumps({"status": "updated", "memory_id": memory.id})
        else:
            # Fallback: delete old, save new
            await self._backend.delete_memory(memory_id)
            memory = await self._backend.save_memory(
                content=new_content,
                user_id=user_id,
                importance=0.5,
                metadata={
                    "source_agent": self.agent_type,
                    "source_provider": provider,
                    "created_via": "tool_call_update_fallback",
                    "supersedes_id": memory_id,
                },
            )
            return json.dumps(
                {
                    "status": "updated",
                    "memory_id": memory.id,
                    "note": "Replaced via delete+save",
                }
            )

    async def _execute_delete(self, input_data: dict[str, Any], user_id: str) -> str:
        """Execute memory_delete tool."""
        memory_id = input_data.get("memory_id", "")
        if not memory_id:
            return json.dumps({"status": "error", "error": "memory_id is required"})

        deleted = await self._backend.delete_memory(memory_id)

        return json.dumps(
            {
                "status": "deleted" if deleted else "not_found",
                "memory_id": memory_id,
            }
        )

    # =========================================================================
    # Native Memory Tool (Anthropic's memory_20250818)
    # =========================================================================
    #
    # HYBRID ARCHITECTURE:
    # Claude uses Anthropic's native memory tool interface (file operations),
    # but we translate these to our semantic vector store backend.
    #
    # This gives us:
    # - Native tool format (subscription-safe, approved by Anthropic)
    # - Semantic search (our vector embeddings under the hood)
    # - Best of both worlds
    #
    # Translation mapping:
    #   view /memories              → Show overview + search instructions
    #   view /memories/search/X     → Semantic search for X
    #   view /memories/recent       → Recent memories
    #   view /memories/<path>       → Find memory by path/topic
    #   create /memories/<path>     → Save to vector store (path as tag)
    #   delete /memories/<path>     → Delete from vector store
    #   str_replace                 → Update memory content
    # =========================================================================

    async def _execute_native_memory_tool(self, input_data: dict[str, Any], user_id: str) -> str:
        """Execute Anthropic's native memory tool with semantic backend.

        This is a TRANSLATION LAYER: Claude thinks it's doing file operations,
        but we're actually using our semantic vector store.

        Commands:
        - view: Semantic search or list memories
        - create: Save to vector store
        - str_replace: Update memory content
        - insert: Append to memory
        - delete: Remove from vector store
        - rename: Update memory tags/path
        """
        # Ensure our semantic backend is initialized
        await self._ensure_initialized()

        command = input_data.get("command", "")

        try:
            if command == "view":
                return await self._native_view_semantic(input_data, user_id)
            elif command == "create":
                return await self._native_create_semantic(input_data, user_id)
            elif command == "str_replace":
                return await self._native_update_semantic(input_data, user_id)
            elif command == "insert":
                return await self._native_append_semantic(input_data, user_id)
            elif command == "delete":
                return await self._native_delete_semantic(input_data, user_id)
            elif command == "rename":
                return await self._native_rename_semantic(input_data, user_id)
            else:
                return f"Error: Unknown command '{command}'"
        except Exception as e:
            logger.error(f"Memory: Native tool error: {e}")
            return f"Error: {e}"

    def _resolve_native_path(self, path: str, user_id: str) -> Path:
        """Resolve path within user's memory directory safely.

        Prevents path traversal attacks by ensuring path stays within
        the user's memory directory.
        """
        assert self._native_memory_dir is not None

        # User-scoped memory directory
        user_dir = self._native_memory_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)

        # Normalize path (remove /memories prefix if present)
        if path.startswith("/memories"):
            path = path[len("/memories") :]
        if path.startswith("/"):
            path = path[1:]

        # Resolve and validate
        resolved = (user_dir / path).resolve()

        # Security: ensure path is within user directory
        try:
            resolved.relative_to(user_dir.resolve())
        except ValueError:
            raise ValueError(f"Path traversal detected: {path}") from None

        return resolved

    def _native_view(self, input_data: dict[str, Any], user_id: str) -> str:
        """View directory contents or file contents."""
        path = input_data.get("path", "/memories")
        view_range = input_data.get("view_range")

        resolved = self._resolve_native_path(path, user_id)

        if not resolved.exists():
            return f"The path {path} does not exist. Please provide a valid path."

        if resolved.is_dir():
            # List directory contents
            lines = [
                f"Here're the files and directories up to 2 levels deep in {path}, "
                "excluding hidden items and node_modules:"
            ]

            def get_size(p: Path) -> str:
                if p.is_file():
                    size = p.stat().st_size
                    if size < 1024:
                        return f"{size}B"
                    elif size < 1024 * 1024:
                        return f"{size / 1024:.1f}K"
                    else:
                        return f"{size / (1024 * 1024):.1f}M"
                return "4.0K"  # Default for directories

            def list_recursive(p: Path, rel_path: str, depth: int) -> None:
                if depth > 2:
                    return
                if p.name.startswith(".") or p.name == "node_modules":
                    return

                lines.append(f"{get_size(p)}\t{rel_path}")

                if p.is_dir() and depth < 2:
                    try:
                        for child in sorted(p.iterdir()):
                            child_rel = (
                                f"{rel_path}/{child.name}"
                                if rel_path != path
                                else f"{path}/{child.name}"
                            )
                            list_recursive(child, child_rel, depth + 1)
                    except PermissionError:
                        pass

            list_recursive(resolved, path, 0)
            return "\n".join(lines)

        else:
            # Read file contents with line numbers
            try:
                content = resolved.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = resolved.read_text(encoding="latin-1")

            lines_content = content.split("\n")

            if len(lines_content) > 999999:
                return f"File {path} exceeds maximum line limit of 999,999 lines."

            # Apply view_range if specified
            start_line = 1
            end_line = len(lines_content)
            if view_range and len(view_range) >= 2:
                start_line = max(1, view_range[0])
                end_line = min(len(lines_content), view_range[1])

            result_lines = [f"Here's the content of {path} with line numbers:"]
            for i, line in enumerate(lines_content[start_line - 1 : end_line], start=start_line):
                result_lines.append(f"{i:6d}\t{line}")

            return "\n".join(result_lines)

    def _native_create(self, input_data: dict[str, Any], user_id: str) -> str:
        """Create a new file."""
        path = input_data.get("path", "")
        file_text = input_data.get("file_text", "")

        if not path:
            return "Error: path is required"

        resolved = self._resolve_native_path(path, user_id)

        if resolved.exists():
            return f"Error: File {path} already exists"

        # Create parent directories if needed
        resolved.parent.mkdir(parents=True, exist_ok=True)

        resolved.write_text(file_text, encoding="utf-8")
        logger.info(f"Memory: Native create: {path} for user {user_id}")

        return f"File created successfully at: {path}"

    def _native_str_replace(self, input_data: dict[str, Any], user_id: str) -> str:
        """Replace text in a file."""
        path = input_data.get("path", "")
        old_str = input_data.get("old_str", "")
        new_str = input_data.get("new_str", "")

        if not path:
            return "Error: path is required"
        if not old_str:
            return "Error: old_str is required"

        resolved = self._resolve_native_path(path, user_id)

        if not resolved.exists():
            return f"Error: The path {path} does not exist. Please provide a valid path."

        if resolved.is_dir():
            return f"Error: The path {path} does not exist. Please provide a valid path."

        content = resolved.read_text(encoding="utf-8")

        # Check for occurrences
        occurrences = content.count(old_str)
        if occurrences == 0:
            return f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
        if occurrences > 1:
            # Find line numbers
            lines = content.split("\n")
            found_lines = []
            for i, line in enumerate(lines, 1):
                if old_str in line:
                    found_lines.append(str(i))
            return (
                f"No replacement was performed. Multiple occurrences of old_str `{old_str}` "
                f"in lines: {', '.join(found_lines)}. Please ensure it is unique"
            )

        # Perform replacement
        new_content = content.replace(old_str, new_str, 1)
        resolved.write_text(new_content, encoding="utf-8")

        # Show snippet around the change
        lines = new_content.split("\n")
        for i, line in enumerate(lines):
            if new_str in line:
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                snippet_lines = ["The memory file has been edited."]
                for j in range(start, end):
                    snippet_lines.append(f"{j + 1:6d}\t{lines[j]}")
                return "\n".join(snippet_lines)

        return "The memory file has been edited."

    def _native_insert(self, input_data: dict[str, Any], user_id: str) -> str:
        """Insert text at a specific line."""
        path = input_data.get("path", "")
        insert_line = input_data.get("insert_line", 0)
        insert_text = input_data.get("insert_text", "")

        if not path:
            return "Error: path is required"

        resolved = self._resolve_native_path(path, user_id)

        if not resolved.exists():
            return f"Error: The path {path} does not exist"

        if resolved.is_dir():
            return f"Error: The path {path} does not exist"

        content = resolved.read_text(encoding="utf-8")
        lines = content.split("\n")
        n_lines = len(lines)

        if insert_line < 0 or insert_line > n_lines:
            return (
                f"Error: Invalid `insert_line` parameter: {insert_line}. "
                f"It should be within the range of lines of the file: [0, {n_lines}]"
            )

        # Insert at specified line
        lines.insert(insert_line, insert_text.rstrip("\n"))

        resolved.write_text("\n".join(lines), encoding="utf-8")

        return f"The file {path} has been edited."

    def _native_delete_file(self, input_data: dict[str, Any], user_id: str) -> str:
        """Delete a file or directory."""
        path = input_data.get("path", "")

        if not path:
            return "Error: path is required"

        resolved = self._resolve_native_path(path, user_id)

        if not resolved.exists():
            return f"Error: The path {path} does not exist"

        import shutil

        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()

        logger.info(f"Memory: Native delete: {path} for user {user_id}")
        return f"Successfully deleted {path}"

    def _native_rename(self, input_data: dict[str, Any], user_id: str) -> str:
        """Rename or move a file/directory."""
        old_path = input_data.get("old_path", "")
        new_path = input_data.get("new_path", "")

        if not old_path:
            return "Error: old_path is required"
        if not new_path:
            return "Error: new_path is required"

        resolved_old = self._resolve_native_path(old_path, user_id)
        resolved_new = self._resolve_native_path(new_path, user_id)

        if not resolved_old.exists():
            return f"Error: The path {old_path} does not exist"

        if resolved_new.exists():
            return f"Error: The destination {new_path} already exists"

        # Create parent directory if needed
        resolved_new.parent.mkdir(parents=True, exist_ok=True)

        resolved_old.rename(resolved_new)

        logger.info(f"Memory: Native rename: {old_path} -> {new_path} for user {user_id}")
        return f"Successfully renamed {old_path} to {new_path}"

    # =========================================================================
    # Semantic Translation Methods (Native Tool → Vector Store)
    # =========================================================================

    async def _native_view_semantic(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle VIEW command with semantic search capabilities.

        Path patterns:
        - /memories              → Overview + search instructions
        - /memories/search/X     → Semantic search for X
        - /memories/recent       → Recent memories (last 10)
        - /memories/all          → List all memories (paginated)
        - /memories/<topic>      → Search by topic/path
        """
        path = input_data.get("path", "/memories")

        # Normalize path
        if path.startswith("/memories"):
            subpath = path[len("/memories") :].lstrip("/")
        else:
            subpath = path.lstrip("/")

        # CASE 1: /memories/search/<query> → Semantic search
        if subpath.startswith("search/"):
            query = subpath[len("search/") :]
            if not query:
                return "Error: Please provide a search query. Example: view /memories/search/food preferences"
            return await self._semantic_search(query, user_id)

        # CASE 2: /memories/recent → Recent memories
        if subpath == "recent":
            return await self._get_recent_memories(user_id, limit=10)

        # CASE 3: /memories/all → List all (paginated)
        if subpath == "all":
            return await self._list_all_memories(user_id, limit=20)

        # CASE 4: /memories (root) → Overview with instructions
        if not subpath or subpath == "":
            return await self._get_memory_overview(user_id)

        # CASE 5: /memories/<something> → Search by topic
        # Treat the path as a search query
        return await self._semantic_search(subpath.replace("/", " ").replace("_", " "), user_id)

    async def _semantic_search(self, query: str, user_id: str, top_k: int = 5) -> str:
        """Perform semantic search and format results."""
        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            results = await self._backend.search_memories(
                query=query,
                user_id=user_id,
                top_k=top_k,
                include_related=True,
            )

            if not results:
                return f"No memories found matching '{query}'.\n\nTip: Try a broader search term, or use 'view /memories/recent' to see recent memories."

            lines = [f"Found {len(results)} memories matching '{query}':\n"]
            for i, r in enumerate(results, 1):
                score_pct = int(r.score * 100)
                content_preview = r.memory.content[:200]
                if len(r.memory.content) > 200:
                    content_preview += "..."

                lines.append(f"{i:6d}\t[{score_pct}% match] {content_preview}")

                # Show related entities if available
                if hasattr(r, "related_entities") and r.related_entities:
                    entities = ", ".join(r.related_entities[:3])
                    lines.append(f"      \t   Related: {entities}")
                lines.append("")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Memory: Semantic search failed: {e}")
            return f"Error searching memories: {e}"

    async def _get_recent_memories(self, user_id: str, limit: int = 10) -> str:
        """Get most recent memories."""
        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            # Use a generic query to get recent items
            # Most backends will return by recency when query is broad
            results = await self._backend.search_memories(
                query="recent memories",
                user_id=user_id,
                top_k=limit,
            )

            if not results:
                return "No memories stored yet.\n\nTo save a memory, use: create /memories/<topic>.txt with your content"

            lines = ["Recent memories:\n"]
            for i, r in enumerate(results, 1):
                content_preview = r.memory.content[:150]
                if len(r.memory.content) > 150:
                    content_preview += "..."
                # Format timestamp if available
                timestamp = ""
                if hasattr(r.memory, "created_at") and r.memory.created_at:
                    timestamp = f" ({r.memory.created_at})"
                lines.append(f"{i:6d}\t{content_preview}{timestamp}")
            lines.append("")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Memory: Get recent failed: {e}")
            return f"Error getting recent memories: {e}"

    async def _list_all_memories(self, user_id: str, limit: int = 20) -> str:
        """List all memories (paginated)."""
        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            # Get all memories with a broad search
            results = await self._backend.search_memories(
                query="*",  # Broad query
                user_id=user_id,
                top_k=limit,
            )

            if not results:
                return "No memories stored yet."

            lines = [f"Showing up to {limit} memories:\n"]
            for i, r in enumerate(results, 1):
                content_preview = r.memory.content[:100]
                if len(r.memory.content) > 100:
                    content_preview += "..."
                lines.append(f"{i:6d}\t{content_preview}")

            if len(results) >= limit:
                lines.append(f"\n(Showing first {limit}. Use search to find specific memories.)")

            return "\n".join(lines)

        except Exception as e:
            logger.error(f"Memory: List all failed: {e}")
            return f"Error listing memories: {e}"

    async def _get_memory_overview(self, user_id: str) -> str:
        """Get memory directory overview with search instructions."""
        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            # Get count of memories
            results = await self._backend.search_memories(
                query="*",
                user_id=user_id,
                top_k=100,  # Just to get a count
            )
            count = len(results) if results else 0

            # Get a few recent as preview
            preview_lines = []
            if results:
                for r in results[:3]:
                    preview = r.memory.content[:60]
                    if len(r.memory.content) > 60:
                        preview += "..."
                    preview_lines.append(f"  • {preview}")

            overview = f"""Here're the files and directories up to 2 levels deep in /memories:
4.0K\t/memories

📁 Memory System ({count} memories stored)

To SEARCH memories (semantic):
  view /memories/search/<your query>
  Example: view /memories/search/food preferences
  Example: view /memories/search/work projects

To see RECENT memories:
  view /memories/recent

To see ALL memories:
  view /memories/all

To SAVE a new memory:
  create /memories/<topic>.txt "your content here"
  Example: create /memories/preferences.txt "User likes pizza"
"""

            if preview_lines:
                overview += "\nRecent memories:\n" + "\n".join(preview_lines)

            return overview

        except Exception as e:
            logger.error(f"Memory: Overview failed: {e}")
            # Return basic help even on error
            return """📁 Memory System

To SEARCH memories: view /memories/search/<query>
To see RECENT: view /memories/recent
To SAVE: create /memories/<topic>.txt "content"
"""

    async def _native_create_semantic(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle CREATE command - save to semantic vector store."""
        path = input_data.get("path", "")
        file_text = input_data.get("file_text", "")

        if not path:
            return "Error: path is required"
        if not file_text:
            return "Error: file_text is required (the memory content)"

        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            # Extract topic from path for metadata
            topic = (
                path.replace("/memories/", "")
                .replace("/", "_")
                .replace(".txt", "")
                .replace(".md", "")
            )

            # Save to our semantic backend
            memory = await self._backend.save_memory(
                content=file_text,
                user_id=user_id,
                importance=0.5,
                metadata={"virtual_path": path, "topic": topic},
            )

            logger.info(f"Memory: Semantic create: {path} -> id={memory.id} for user {user_id}")
            return f"File created successfully at: {path}"

        except Exception as e:
            logger.error(f"Memory: Semantic create failed: {e}")
            return f"Error: {e}"

    async def _native_update_semantic(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle STR_REPLACE command - update memory content."""
        path = input_data.get("path", "")
        old_str = input_data.get("old_str", "")
        new_str = input_data.get("new_str", "")

        if not path:
            return "Error: path is required"
        if not old_str:
            return "Error: old_str is required"

        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            # Search for memory containing old_str
            results = await self._backend.search_memories(
                query=old_str,
                user_id=user_id,
                top_k=5,
            )

            # Find exact match
            matching_memory = None
            for r in results:
                if old_str in r.memory.content:
                    matching_memory = r.memory
                    break

            if not matching_memory:
                return f"No replacement was performed, old_str `{old_str}` did not appear verbatim in memories."

            # Check for multiple occurrences
            if matching_memory.content.count(old_str) > 1:
                return f"No replacement was performed. Multiple occurrences of old_str `{old_str}`. Please ensure it is unique."

            # Perform replacement
            new_content = matching_memory.content.replace(old_str, new_str, 1)

            # Update via delete + create (or update if backend supports it)
            if hasattr(self._backend, "update_memory"):
                await self._backend.update_memory(
                    memory_id=matching_memory.id,
                    new_content=new_content,
                    user_id=user_id,
                )
            else:
                await self._backend.delete_memory(matching_memory.id)
                await self._backend.save_memory(
                    content=new_content,
                    user_id=user_id,
                    importance=0.5,
                )

            # Show snippet around the change
            lines = new_content.split("\n")
            snippet = "\n".join(f"{i + 1:6d}\t{line}" for i, line in enumerate(lines[:5]))

            logger.info(f"Memory: Semantic update for user {user_id}")
            return f"The memory file has been edited.\n{snippet}"

        except Exception as e:
            logger.error(f"Memory: Semantic update failed: {e}")
            return f"Error: {e}"

    async def _native_append_semantic(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle INSERT command - append to memory or create new."""
        path = input_data.get("path", "")
        insert_text = input_data.get("insert_text", "")
        _insert_line = input_data.get("insert_line", 0)  # Unused in semantic mode

        if not path:
            return "Error: path is required"
        if not insert_text:
            return "Error: insert_text is required"

        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            # For semantic backend, append is just creating a new memory
            # with the additional context
            topic = path.replace("/memories/", "").replace("/", "_").replace(".txt", "")

            await self._backend.save_memory(
                content=insert_text,
                user_id=user_id,
                importance=0.5,
                metadata={"virtual_path": path, "topic": topic, "appended": True},
            )

            logger.info(f"Memory: Semantic append: {path} for user {user_id}")
            return f"The file {path} has been edited."

        except Exception as e:
            logger.error(f"Memory: Semantic append failed: {e}")
            return f"Error: {e}"

    async def _native_delete_semantic(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle DELETE command - remove from vector store."""
        path = input_data.get("path", "")

        if not path:
            return "Error: path is required"

        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            # Search for memories with this path
            topic = (
                path.replace("/memories/", "")
                .replace("/", " ")
                .replace("_", " ")
                .replace(".txt", "")
            )

            results = await self._backend.search_memories(
                query=topic,
                user_id=user_id,
                top_k=10,
            )

            if not results:
                return f"Error: The path {path} does not exist"

            # Delete matching memories
            deleted_count = 0
            for r in results:
                # Check if metadata matches path
                metadata = getattr(r.memory, "metadata", {}) or {}
                if metadata.get("virtual_path") == path or r.score > 0.8:
                    await self._backend.delete_memory(r.memory.id)
                    deleted_count += 1

            if deleted_count == 0:
                return f"Error: The path {path} does not exist"

            logger.info(
                f"Memory: Semantic delete: {path} ({deleted_count} memories) for user {user_id}"
            )
            return f"Successfully deleted {path}"

        except Exception as e:
            logger.error(f"Memory: Semantic delete failed: {e}")
            return f"Error: {e}"

    async def _native_rename_semantic(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle RENAME command - update memory path/topic."""
        old_path = input_data.get("old_path", "")
        new_path = input_data.get("new_path", "")

        if not old_path:
            return "Error: old_path is required"
        if not new_path:
            return "Error: new_path is required"

        if not self._backend:
            return "Error: Memory backend not initialized"

        try:
            # Search for memories with old path
            old_topic = (
                old_path.replace("/memories/", "")
                .replace("/", " ")
                .replace("_", " ")
                .replace(".txt", "")
            )

            results = await self._backend.search_memories(
                query=old_topic,
                user_id=user_id,
                top_k=10,
            )

            if not results:
                return f"Error: The path {old_path} does not exist"

            # Update metadata for matching memories (re-save with new path)
            new_topic = new_path.replace("/memories/", "").replace("/", "_").replace(".txt", "")
            renamed_count = 0

            for r in results:
                metadata = getattr(r.memory, "metadata", {}) or {}
                if metadata.get("virtual_path") == old_path or r.score > 0.8:
                    # Delete old and create with new path
                    await self._backend.delete_memory(r.memory.id)
                    await self._backend.save_memory(
                        content=r.memory.content,
                        user_id=user_id,
                        importance=getattr(r.memory, "importance", 0.5),
                        metadata={"virtual_path": new_path, "topic": new_topic},
                    )
                    renamed_count += 1

            if renamed_count == 0:
                return f"Error: The path {old_path} does not exist"

            logger.info(f"Memory: Semantic rename: {old_path} -> {new_path} for user {user_id}")
            return f"Successfully renamed {old_path} to {new_path}"

        except Exception as e:
            logger.error(f"Memory: Semantic rename failed: {e}")
            return f"Error: {e}"

    @property
    def backend(self) -> Any:
        """Expose the backend for external components (e.g., TrafficLearner)."""
        return self._backend

    @property
    def initialized(self) -> bool:
        """Whether the backend has been initialized."""
        return self._initialized

    async def ensure_initialized(self) -> None:
        """Initialize the configured backend so readiness checks can be accurate."""
        await self._ensure_initialized()

    async def warmup_embedder(self) -> bool:
        """Force one warm-up embed call so the ONNX graph is compiled now.

        Returns ``True`` if the embedder was exercised successfully,
        ``False`` otherwise. Best-effort — all errors are swallowed and
        logged, never raised, so startup cannot be blocked by embedder
        cold-start failures.

        Only meaningful for the ``local`` backend (the ONNX/sentence
        embedder warm-up is what we want to preempt). Qdrant/Neo4j is a
        no-op because Mem0 handles its own embedder lifecycle upstream.
        """
        if not self._initialized or self._backend is None:
            return False

        try:
            hm = getattr(self._backend, "_hierarchical_memory", None)
            if hm is None:
                return False
            embedder = getattr(hm, "_embedder", None) or getattr(hm, "embedder", None)
            if embedder is None:
                return False
            if not hasattr(embedder, "embed"):
                return False
            await embedder.embed("warmup")
            logger.info("Memory: embedder warm-up encode complete")
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Memory: embedder warm-up failed (non-fatal): {exc}")
            return False

    def health_status(self) -> dict[str, Any]:
        """Return a lightweight health snapshot for readiness endpoints."""
        return {
            "enabled": self.config.enabled,
            "backend": self.config.backend,
            "initialized": self._initialized,
            "native_tool": self.config.use_native_tool,
            "bridge_enabled": self.config.bridge_enabled,
        }

    async def close(self) -> None:
        """Close the memory backend."""
        if self._backend is not None:
            await self._close_backend_instance(self._backend, reason="handler close")
        self._backend = None
        self._initialized = False
        logger.info("Memory: Handler closed")
