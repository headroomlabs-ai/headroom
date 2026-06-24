"""Memory tool adapter for multi-provider support.

This module provides a unified adapter for memory tools across different LLM providers.
It handles provider detection, tool injection, and tool call execution with appropriate
format conversions for each provider.

Supported providers:
- Anthropic: Native memory_20250818 tool and custom tools
- OpenAI: Function calling format
- Gemini: Function calling format
- Generic: Fallback for unknown providers

Usage:
    config = MemoryToolAdapterConfig(enabled=True)
    adapter = MemoryToolAdapter(config)

    # Detect provider from request
    provider = adapter.detect_provider(request_headers, model_name)

    # Inject tools
    tools, beta_headers = adapter.inject_tools(existing_tools, provider)

    # Handle tool calls in response
    if adapter.has_memory_tool_calls(response, provider):
        results = await adapter.handle_tool_calls(response, user_id, provider)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from headroom.memory.backends.local import LocalBackend

logger = logging.getLogger(__name__)

# =============================================================================
# Provider Types
# =============================================================================

Provider = Literal["anthropic", "openai", "gemini", "generic"]

# =============================================================================
# Tool Names
# =============================================================================

# Custom memory tool names (Headroom's tools)
MEMORY_TOOL_NAMES = {"memory_save", "memory_search", "memory_update", "memory_delete"}

# Anthropic's native memory tool
NATIVE_MEMORY_TOOL_NAME = "memory"
NATIVE_MEMORY_TOOL_TYPE = "memory_20250818"

# Beta header for Anthropic's native memory tool
ANTHROPIC_BETA_HEADER = "context-management-2025-06-27"

# =============================================================================
# Tool Schemas - Anthropic Native Tool
# =============================================================================

ANTHROPIC_NATIVE_TOOL: dict[str, Any] = {
    "type": NATIVE_MEMORY_TOOL_TYPE,
    "name": NATIVE_MEMORY_TOOL_NAME,
}

# =============================================================================
# Tool Schemas - Anthropic Custom Tools
# =============================================================================

ANTHROPIC_CUSTOM_TOOLS: list[dict[str, Any]] = [
    {
        "name": "memory_save",
        "description": """Save important information to long-term memory for future reference.

Use this tool when you encounter information that should be remembered across conversations:
- User preferences (e.g., "prefers Python over JavaScript")
- Personal facts (e.g., "works at Acme Corp", "has a dog named Max")
- Project context (e.g., "working on a CLI tool", "using React 18")
- Decisions made (e.g., "chose PostgreSQL for the database")
- Important relationships (e.g., "Alice is Bob's manager")

DO NOT save: transient info, sensitive data (passwords, keys), redundant info.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to remember. Be specific and self-contained.",
                },
                "importance": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Importance score from 0.0 (low) to 1.0 (critical).",
                },
                "facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Pre-extracted discrete facts for efficient storage.",
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Entity names referenced in this memory.",
                },
                "extracted_entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "entity": {"type": "string"},
                            "entity_type": {"type": "string"},
                        },
                        "required": ["entity", "entity_type"],
                    },
                    "description": "Pre-extracted entities with types.",
                },
                "extracted_relationships": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "relationship": {"type": "string"},
                            "destination": {"type": "string"},
                        },
                        "required": ["source", "relationship", "destination"],
                    },
                    "description": "Pre-extracted relationships for graph storage.",
                },
            },
            "required": ["content", "importance"],
        },
    },
    {
        "name": "memory_search",
        "description": """Search stored memories to recall relevant information.

Use this tool to retrieve previously saved information before responding to questions about:
- User preferences or past decisions
- Personal or professional context
- Previously discussed topics or projects
- Relationships between people, systems, or concepts

Search BEFORE saving to avoid duplicates.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query.",
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to memories mentioning these entities.",
                },
                "include_related": {
                    "type": "boolean",
                    "description": "Also retrieve connected memories.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum number of memories to retrieve (default 10).",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_update",
        "description": """Update an existing memory with corrected or evolved information.

Use when:
- User provides a correction to stored information
- Information has changed over time
- Adding detail or clarification to an existing memory""",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The unique ID of the memory to update.",
                },
                "new_content": {
                    "type": "string",
                    "description": "The updated content.",
                },
                "reason": {
                    "type": "string",
                    "description": "Explanation for the update.",
                },
            },
            "required": ["memory_id", "new_content"],
        },
    },
    {
        "name": "memory_delete",
        "description": """Delete a memory that is no longer relevant or was stored in error.

Use when:
- User explicitly asks to forget something
- Information is outdated and no longer applicable
- A memory was saved in error""",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The unique ID of the memory to delete.",
                },
                "reason": {
                    "type": "string",
                    "description": "Explanation for the deletion.",
                },
            },
            "required": ["memory_id"],
        },
    },
]

# =============================================================================
# Tool Schemas - OpenAI Function Calling Format
# =============================================================================

OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "memory_save",
            "description": """Save important information to long-term memory for future reference.

Use this tool when you encounter information that should be remembered across conversations:
- User preferences, personal facts, project context, decisions, relationships

DO NOT save: transient info, sensitive data (passwords, keys), redundant info.""",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to remember. Be specific and self-contained.",
                    },
                    "importance": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Importance score from 0.0 (low) to 1.0 (critical).",
                    },
                    "facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Pre-extracted discrete facts.",
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Entity names referenced in this memory.",
                    },
                    "extracted_entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity": {"type": "string"},
                                "entity_type": {"type": "string"},
                            },
                            "required": ["entity", "entity_type"],
                        },
                        "description": "Pre-extracted entities with types.",
                    },
                    "extracted_relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "relationship": {"type": "string"},
                                "destination": {"type": "string"},
                            },
                            "required": ["source", "relationship", "destination"],
                        },
                        "description": "Pre-extracted relationships.",
                    },
                },
                "required": ["content", "importance"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search",
            "description": "Search stored memories to recall relevant information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query.",
                    },
                    "entities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter to memories mentioning these entities.",
                    },
                    "include_related": {
                        "type": "boolean",
                        "description": "Also retrieve connected memories.",
                    },
                    "top_k": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Maximum number of memories to retrieve.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_update",
            "description": "Update an existing memory with corrected or evolved information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The unique ID of the memory to update.",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "The updated content.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Explanation for the update.",
                    },
                },
                "required": ["memory_id", "new_content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": "Delete a memory that is no longer relevant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The unique ID of the memory to delete.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Explanation for the deletion.",
                    },
                },
                "required": ["memory_id"],
            },
        },
    },
]

# =============================================================================
# Tool Schemas - Gemini Function Calling Format
# =============================================================================

# Gemini uses a similar format to OpenAI but with slight differences
GEMINI_TOOLS: list[dict[str, Any]] = [
    {
        "name": "memory_save",
        "description": """Save important information to long-term memory for future reference.

Use this tool when you encounter information that should be remembered across conversations:
- User preferences, personal facts, project context, decisions, relationships

DO NOT save: transient info, sensitive data (passwords, keys), redundant info.""",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to remember. Be specific and self-contained.",
                },
                "importance": {
                    "type": "number",
                    "description": "Importance score from 0.0 (low) to 1.0 (critical).",
                },
                "facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Pre-extracted discrete facts.",
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Entity names referenced in this memory.",
                },
            },
            "required": ["content", "importance"],
        },
    },
    {
        "name": "memory_search",
        "description": "Search stored memories to recall relevant information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query.",
                },
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to memories mentioning these entities.",
                },
                "include_related": {
                    "type": "boolean",
                    "description": "Also retrieve connected memories.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum number of memories to retrieve.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_update",
        "description": "Update an existing memory with corrected or evolved information.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The unique ID of the memory to update.",
                },
                "new_content": {
                    "type": "string",
                    "description": "The updated content.",
                },
                "reason": {
                    "type": "string",
                    "description": "Explanation for the update.",
                },
            },
            "required": ["memory_id", "new_content"],
        },
    },
    {
        "name": "memory_delete",
        "description": "Delete a memory that is no longer relevant.",
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "The unique ID of the memory to delete.",
                },
                "reason": {
                    "type": "string",
                    "description": "Explanation for the deletion.",
                },
            },
            "required": ["memory_id"],
        },
    },
]


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class MemoryToolAdapterConfig:
    """Configuration for the memory tool adapter.

    Attributes:
        enabled: Whether memory features are enabled.
        use_native_tool: Use Anthropic's native memory_20250818 tool (Anthropic only).
        inject_tools: Whether to inject memory tools into requests.
        inject_context: Whether to inject memory context into requests.
        db_path: Path to the local memory database.
        top_k: Number of memories to retrieve in searches.
        min_similarity: Minimum similarity score for memory retrieval.
    """

    enabled: bool = False
    use_native_tool: bool = True  # Default to native for Anthropic (subscription-safe)
    inject_tools: bool = True
    inject_context: bool = True
    db_path: str = "headroom_memory.db"
    top_k: int = 10
    min_similarity: float = 0.3


# =============================================================================
# Memory Tool Adapter
# =============================================================================


class MemoryToolAdapter:
    """Adapter for memory tools across different LLM providers.

    This adapter provides a unified interface for:
    1. Detecting the LLM provider from requests
    2. Injecting memory tools in provider-specific formats
    3. Providing required beta headers
    4. Detecting memory tool calls in responses
    5. Handling tool calls with the semantic backend

    Example:
        adapter = MemoryToolAdapter(config)
        provider = adapter.detect_provider(headers, model)
        tools, headers = adapter.inject_tools(existing_tools, provider)

        # Later, when processing response
        if adapter.has_memory_tool_calls(response, provider):
            results = await adapter.handle_tool_calls(response, user_id, provider)
    """

    def __init__(self, config: MemoryToolAdapterConfig) -> None:
        """Initialize the adapter.

        Args:
            config: Configuration for the adapter.
        """
        self.config = config
        self._backend: LocalBackend | Any = None
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Lazy initialization of the semantic backend.

        Imports and initializes the LocalBackend from memory_handler
        to provide semantic search and storage capabilities.
        """
        if self._initialized:
            return

        if not self.config.enabled:
            return

        from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

        backend_config = LocalBackendConfig(db_path=self.config.db_path)
        self._backend = LocalBackend(backend_config)
        await self._backend._ensure_initialized()

        self._initialized = True
        logger.info(f"MemoryToolAdapter: Initialized backend at {self.config.db_path}")

    def detect_provider(
        self,
        request_headers: dict[str, str] | None = None,
        model_name: str | None = None,
    ) -> Provider:
        """Detect the LLM provider from request headers and model name.

        Detection priority:
        1. Explicit headers (x-api-key for Anthropic, authorization for OpenAI)
        2. Model name patterns (claude-*, gpt-*, gemini-*)
        3. Fallback to generic

        Args:
            request_headers: HTTP headers from the request (optional).
            model_name: Name of the model being used (optional).

        Returns:
            The detected provider.
        """
        headers = request_headers or {}
        model = (model_name or "").lower()

        # Check headers for provider hints
        if "x-api-key" in headers or "anthropic-version" in headers:
            return "anthropic"

        if headers.get("authorization", "").startswith("Bearer sk-"):
            # OpenAI uses sk-* API keys
            return "openai"

        # Check model name patterns
        if model.startswith("claude"):
            return "anthropic"

        if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
            return "openai"

        if model.startswith("gemini") or "gemma" in model:
            return "gemini"

        # Fallback to generic
        return "generic"

    def inject_tools(
        self,
        tools: list[dict[str, Any]] | None,
        provider: Provider,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Inject memory tools into the tools list for the given provider.

        Args:
            tools: Existing tools list (may be None).
            provider: The LLM provider to format tools for.

        Returns:
            Tuple of (updated_tools, beta_headers).
            beta_headers contains any required headers (e.g., anthropic-beta).
        """
        if not self.config.inject_tools:
            return tools or [], {}

        tools = list(tools) if tools else []
        beta_headers: dict[str, str] = {}

        # Get existing tool names
        existing_names = self._get_existing_tool_names(tools)

        # Handle Anthropic native tool
        if provider == "anthropic" and self.config.use_native_tool:
            if NATIVE_MEMORY_TOOL_NAME not in existing_names:
                tools.append(ANTHROPIC_NATIVE_TOOL.copy())
                beta_headers["anthropic-beta"] = ANTHROPIC_BETA_HEADER
                logger.info("MemoryToolAdapter: Injected native memory tool for Anthropic")
            return tools, beta_headers

        # Handle custom tools by provider
        if provider == "anthropic":
            tools, was_injected = self._inject_anthropic_tools(tools, existing_names)
        elif provider == "openai":
            tools, was_injected = self._inject_openai_tools(tools, existing_names)
        elif provider == "gemini":
            tools, was_injected = self._inject_gemini_tools(tools, existing_names)
        else:
            # Generic fallback uses OpenAI format
            tools, was_injected = self._inject_openai_tools(tools, existing_names)

        if was_injected:
            logger.info(f"MemoryToolAdapter: Injected custom tools for {provider}")

        return tools, beta_headers

    def _get_existing_tool_names(self, tools: list[dict[str, Any]]) -> set[str]:
        """Extract tool names from existing tools list."""
        names: set[str] = set()
        for tool in tools:
            # Anthropic format
            if "name" in tool:
                names.add(tool["name"])
            # OpenAI format
            if "function" in tool and "name" in tool["function"]:
                names.add(tool["function"]["name"])
        return names

    def _inject_anthropic_tools(
        self,
        tools: list[dict[str, Any]],
        existing_names: set[str],
    ) -> tuple[list[dict[str, Any]], bool]:
        """Inject Anthropic-formatted custom memory tools."""
        was_injected = False
        for memory_tool in ANTHROPIC_CUSTOM_TOOLS:
            if memory_tool["name"] not in existing_names:
                tools.append(memory_tool.copy())
                was_injected = True
        return tools, was_injected

    def _inject_openai_tools(
        self,
        tools: list[dict[str, Any]],
        existing_names: set[str],
    ) -> tuple[list[dict[str, Any]], bool]:
        """Inject OpenAI-formatted memory tools."""
        was_injected = False
        for memory_tool in OPENAI_TOOLS:
            tool_name = memory_tool["function"]["name"]
            if tool_name not in existing_names:
                tools.append(memory_tool.copy())
                was_injected = True
        return tools, was_injected

    def _inject_gemini_tools(
        self,
        tools: list[dict[str, Any]],
        existing_names: set[str],
    ) -> tuple[list[dict[str, Any]], bool]:
        """Inject Gemini-formatted memory tools."""
        was_injected = False
        for memory_tool in GEMINI_TOOLS:
            if memory_tool["name"] not in existing_names:
                tools.append(memory_tool.copy())
                was_injected = True
        return tools, was_injected

    def get_beta_headers(self, provider: Provider) -> dict[str, str]:
        """Get any required beta headers for the provider.

        Args:
            provider: The LLM provider.

        Returns:
            Dict of header name -> value for any required beta headers.
        """
        if provider == "anthropic" and self.config.use_native_tool:
            return {"anthropic-beta": ANTHROPIC_BETA_HEADER}
        return {}

    def has_memory_tool_calls(
        self,
        response: dict[str, Any],
        provider: Provider,
    ) -> bool:
        """Check if the response contains memory tool calls.

        Args:
            response: The API response from the LLM.
            provider: The LLM provider.

        Returns:
            True if response contains memory tool calls.
        """
        tool_calls = self._extract_tool_calls(response, provider)
        for tc in tool_calls:
            name = self._get_tool_name(tc, provider)
            if name in MEMORY_TOOL_NAMES or name == NATIVE_MEMORY_TOOL_NAME:
                return True
        return False

    def _extract_tool_calls(
        self,
        response: dict[str, Any],
        provider: Provider,
    ) -> list[dict[str, Any]]:
        """Extract tool calls from response based on provider format."""
        if provider == "anthropic":
            content = response.get("content", [])
            if isinstance(content, list):
                return [block for block in content if block.get("type") == "tool_use"]
            return []

        elif provider == "openai":
            choices = response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                return list(message.get("tool_calls", []) or [])
            return []

        elif provider == "gemini":
            # Gemini format: candidates[0].content.parts[*].functionCall
            candidates = response.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                return [p for p in parts if "functionCall" in p]
            return []

        # Generic fallback - try both formats
        tool_calls = []

        # Try Anthropic format
        content = response.get("content", [])
        if isinstance(content, list):
            tool_calls.extend([block for block in content if block.get("type") == "tool_use"])

        # Try OpenAI format
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            tool_calls.extend(list(message.get("tool_calls", []) or []))

        return tool_calls

    def _get_tool_name(self, tool_call: dict[str, Any], provider: Provider) -> str:
        """Get the tool name from a tool call."""
        if provider == "anthropic":
            return str(tool_call.get("name", ""))
        elif provider == "openai":
            return str(tool_call.get("function", {}).get("name", ""))
        elif provider == "gemini":
            func_call = tool_call.get("functionCall", {})
            return str(func_call.get("name", ""))
        else:
            # Generic - try both
            return str(tool_call.get("name", "") or tool_call.get("function", {}).get("name", ""))

    def _get_tool_id(self, tool_call: dict[str, Any], provider: Provider) -> str:
        """Get the tool call ID."""
        if provider == "anthropic":
            return str(tool_call.get("id", ""))
        elif provider == "openai":
            return str(tool_call.get("id", ""))
        elif provider == "gemini":
            # Gemini doesn't use IDs in the same way
            return str(tool_call.get("functionCall", {}).get("name", ""))
        else:
            return str(tool_call.get("id", ""))

    def _get_tool_input(
        self,
        tool_call: dict[str, Any],
        provider: Provider,
    ) -> dict[str, Any]:
        """Get the tool input/arguments from a tool call."""
        if provider == "anthropic":
            result = tool_call.get("input", {})
            return dict(result) if isinstance(result, dict) else {}
        elif provider == "openai":
            args_str = tool_call.get("function", {}).get("arguments", "{}")
            try:
                parsed = json.loads(args_str)
                return dict(parsed) if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}
        elif provider == "gemini":
            result = tool_call.get("functionCall", {}).get("args", {})
            return dict(result) if isinstance(result, dict) else {}
        else:
            # Generic - try both
            if "input" in tool_call:
                result = tool_call["input"]
                return dict(result) if isinstance(result, dict) else {}
            args_str = tool_call.get("function", {}).get("arguments", "{}")
            try:
                parsed = json.loads(args_str)
                return dict(parsed) if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                return {}

    async def handle_tool_calls(
        self,
        response: dict[str, Any],
        user_id: str,
        provider: Provider,
    ) -> list[dict[str, Any]]:
        """Handle memory tool calls and return results in provider format.

        Args:
            response: The API response containing tool calls.
            user_id: User identifier for memory operations.
            provider: The LLM provider.

        Returns:
            List of tool results in provider-appropriate format.
        """
        await self._ensure_initialized()

        tool_calls = self._extract_tool_calls(response, provider)
        results: list[dict[str, Any]] = []

        for tc in tool_calls:
            tool_name = self._get_tool_name(tc, provider)
            tool_id = self._get_tool_id(tc, provider)
            input_data = self._get_tool_input(tc, provider)

            # Skip non-memory tools
            if tool_name not in MEMORY_TOOL_NAMES and tool_name != NATIVE_MEMORY_TOOL_NAME:
                continue

            # Execute the tool
            if tool_name == NATIVE_MEMORY_TOOL_NAME:
                result_content = await self._execute_native_tool(input_data, user_id)
            else:
                result_content = await self._execute_custom_tool(tool_name, input_data, user_id)

            # Format result for provider
            result = self._format_tool_result(tool_id, result_content, provider)
            results.append(result)

            logger.info(f"MemoryToolAdapter: Executed {tool_name} for user {user_id}")

        return results

    def _format_tool_result(
        self,
        tool_id: str,
        content: str,
        provider: Provider,
    ) -> dict[str, Any]:
        """Format a tool result for the given provider."""
        if provider == "anthropic":
            return {
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content,
            }
        elif provider == "openai":
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": content,
            }
        elif provider == "gemini":
            return {
                "functionResponse": {
                    "name": tool_id,
                    "response": {"result": content},
                }
            }
        else:
            # Generic uses OpenAI format
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": content,
            }

    async def _execute_native_tool(
        self,
        input_data: dict[str, Any],
        user_id: str,
    ) -> str:
        """Execute Anthropic's native memory tool.

        This translates native memory commands to our semantic backend:
        - view: semantic search or list memories
        - create: save to vector store
        - str_replace: update memory
        - delete: remove from vector store
        """
        if not self._backend:
            return "Error: Memory backend not initialized"

        command = input_data.get("command", "")

        try:
            if command == "view":
                return await self._native_view(input_data, user_id)
            elif command == "create":
                return await self._native_create(input_data, user_id)
            elif command == "str_replace":
                return await self._native_update(input_data, user_id)
            elif command == "delete":
                return await self._native_delete(input_data, user_id)
            else:
                return f"Error: Unknown command '{command}'"
        except Exception as e:
            logger.error(f"MemoryToolAdapter: Native tool error: {e}")
            return f"Error: {e}"

    async def _native_view(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle VIEW command - semantic search or list memories."""
        path = input_data.get("path", "/memories")

        # Normalize path
        if path.startswith("/memories"):
            subpath = path[len("/memories") :].lstrip("/")
        else:
            subpath = path.lstrip("/")

        # Search pattern: /memories/search/<query>
        if subpath.startswith("search/"):
            query = subpath[len("search/") :]
            if not query:
                return "Error: Please provide a search query"
            return await self._semantic_search(query, user_id)

        # Recent: /memories/recent
        if subpath == "recent":
            return await self._semantic_search("recent memories", user_id, top_k=10)

        # Root: /memories
        if not subpath:
            return await self._get_memory_overview(user_id)

        # Treat path as search topic
        return await self._semantic_search(
            subpath.replace("/", " ").replace("_", " "),
            user_id,
        )

    async def _native_create(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle CREATE command - save to vector store."""
        path = input_data.get("path", "")
        file_text = input_data.get("file_text", "")

        if not file_text:
            return "Error: file_text is required"

        topic = path.replace("/memories/", "").replace("/", "_").replace(".txt", "")

        memory = await self._backend.save_memory(
            content=file_text,
            user_id=user_id,
            importance=0.5,
            metadata={"virtual_path": path, "topic": topic},
        )

        logger.info(f"MemoryToolAdapter: Created memory {memory.id} for {user_id}")
        return f"File created successfully at: {path}"

    async def _native_update(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle STR_REPLACE command - update memory content."""
        old_str = input_data.get("old_str", "")
        new_str = input_data.get("new_str", "")

        if not old_str:
            return "Error: old_str is required"

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
            return "No replacement performed, old_str not found in memories"

        # Perform replacement
        new_content = matching_memory.content.replace(old_str, new_str, 1)

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

        return "The memory has been edited."

    async def _native_delete(self, input_data: dict[str, Any], user_id: str) -> str:
        """Handle DELETE command - remove from vector store."""
        path = input_data.get("path", "")
        topic = path.replace("/memories/", "").replace("/", " ").replace("_", " ")

        results = await self._backend.search_memories(
            query=topic,
            user_id=user_id,
            top_k=10,
        )

        if not results:
            return f"Error: The path {path} does not exist"

        deleted_count = 0
        for r in results:
            metadata = getattr(r.memory, "metadata", {}) or {}
            if metadata.get("virtual_path") == path or r.score > 0.8:
                await self._backend.delete_memory(r.memory.id)
                deleted_count += 1

        if deleted_count == 0:
            return f"Error: The path {path} does not exist"

        return f"Successfully deleted {path}"

    async def _semantic_search(
        self,
        query: str,
        user_id: str,
        top_k: int = 5,
    ) -> str:
        """Perform semantic search and format results."""
        results = await self._backend.search_memories(
            query=query,
            user_id=user_id,
            top_k=top_k,
            include_related=True,
        )

        if not results:
            return f"No memories found matching '{query}'"

        lines = [f"Found {len(results)} memories matching '{query}':\n"]
        for i, r in enumerate(results, 1):
            score_pct = int(r.score * 100)
            content_preview = r.memory.content[:200]
            if len(r.memory.content) > 200:
                content_preview += "..."
            lines.append(f"{i}. [{score_pct}% match] {content_preview}")

        return "\n".join(lines)

    async def _get_memory_overview(self, user_id: str) -> str:
        """Get memory overview with search instructions."""
        results = await self._backend.search_memories(
            query="*",
            user_id=user_id,
            top_k=100,
        )
        count = len(results) if results else 0

        return f"""Memory System ({count} memories stored)

To SEARCH: view /memories/search/<query>
To see RECENT: view /memories/recent
To SAVE: create /memories/<topic>.txt "content"
"""

    async def _execute_custom_tool(
        self,
        tool_name: str,
        input_data: dict[str, Any],
        user_id: str,
    ) -> str:
        """Execute a custom memory tool."""
        if not self._backend:
            return json.dumps({"error": "Memory backend not initialized"})

        try:
            if tool_name == "memory_save":
                return await self._execute_save(input_data, user_id)
            elif tool_name == "memory_search":
                return await self._execute_search(input_data, user_id)
            elif tool_name == "memory_update":
                return await self._execute_update(input_data, user_id)
            elif tool_name == "memory_delete":
                return await self._execute_delete(input_data, user_id)
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as e:
            logger.error(f"MemoryToolAdapter: Tool {tool_name} failed: {e}")
            return json.dumps({"status": "error", "error": str(e)})

    async def _execute_save(self, input_data: dict[str, Any], user_id: str) -> str:
        """Execute memory_save tool."""
        content = input_data.get("content", "")
        if not content:
            return json.dumps({"status": "error", "error": "content is required"})

        importance = input_data.get("importance", 0.5)
        facts = input_data.get("facts")
        entities = input_data.get("entities")
        extracted_entities = input_data.get("extracted_entities")
        extracted_relationships = input_data.get("extracted_relationships")

        memory = await self._backend.save_memory(
            content=content,
            user_id=user_id,
            importance=importance,
            facts=facts,
            entities=entities,
            extracted_entities=extracted_entities,
            relationships=extracted_relationships,
            extracted_relationships=extracted_relationships,
        )

        return json.dumps(
            {
                "status": "saved",
                "memory_id": memory.id,
                "content": memory.content[:100] + "..."
                if len(memory.content) > 100
                else memory.content,
            }
        )

    async def _execute_search(self, input_data: dict[str, Any], user_id: str) -> str:
        """Execute memory_search tool."""
        query = input_data.get("query", "")
        if not query:
            return json.dumps({"status": "error", "error": "query is required"})

        top_k = input_data.get("top_k", self.config.top_k)
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

    async def _execute_update(self, input_data: dict[str, Any], user_id: str) -> str:
        """Execute memory_update tool."""
        memory_id = input_data.get("memory_id", "")
        new_content = input_data.get("new_content", "")

        if not memory_id:
            return json.dumps({"status": "error", "error": "memory_id is required"})
        if not new_content:
            return json.dumps({"status": "error", "error": "new_content is required"})

        reason = input_data.get("reason")

        if hasattr(self._backend, "update_memory"):
            memory = await self._backend.update_memory(
                memory_id=memory_id,
                new_content=new_content,
                reason=reason,
                user_id=user_id,
            )
            return json.dumps({"status": "updated", "memory_id": memory.id})
        else:
            # Fallback: delete old, save new
            await self._backend.delete_memory(memory_id)
            memory = await self._backend.save_memory(
                content=new_content,
                user_id=user_id,
                importance=0.5,
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

    async def close(self) -> None:
        """Close the backend connection."""
        if self._backend and hasattr(self._backend, "close"):
            await self._backend.close()
        self._backend = None
        self._initialized = False
        logger.info("MemoryToolAdapter: Closed")
