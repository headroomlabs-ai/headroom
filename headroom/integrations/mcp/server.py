"""Headroom MCP integration helpers for compressing tool outputs.

This module currently provides these MCP integration surfaces:

1. HeadroomMCPCompressor - core compression logic for MCP tool results
2. compress_tool_result() - standalone helper for host applications
3. HeadroomMCPClientWrapper - client wrapper that compresses tool outputs
4. create_headroom_mcp_proxy() - configuration helper for custom proxy setups

The key insight: MCP tool outputs are the PERFECT use case for Headroom.
They're often large (100s-1000s of items), structured (JSON), and contain
mostly low-relevance data with a few critical items (errors, matches).

Example - Custom proxy configuration:
    ```python
    # Build config for your own MCP proxy/server wrapper
    proxy_config = create_headroom_mcp_proxy(
        upstream_servers=[
            ("slack", slack_server),
            ("database", db_server),
            ("github", github_server),
        ],
        config=HeadroomConfig(),
    )
    ```

Example - Standalone Function:
    ```python
    # In your MCP host application
    result = await mcp_client.call_tool("search_logs", {"service": "api"})

    # Compress before adding to context
    compressed = compress_tool_result(
        content=result,
        tool_name="search_logs",
        tool_args={"service": "api"},
        user_query="find errors in api service",
    )
    ```

Example - Middleware (for MCP client libraries):
    ```python
    # Wrap your MCP client's transport
    middleware = HeadroomMCPMiddleware(config)
    client = MCPClient(transport=middleware.wrap(base_transport))
    ```
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from headroom.config import HeadroomConfig, SmartCrusherConfig
from headroom.providers.openai import OpenAIProvider
from headroom.transforms.smart_crusher import SmartCrusher


@dataclass
class MCPCompressionResult:
    """Result of compressing an MCP tool output."""

    original_content: str
    compressed_content: str
    original_tokens: int
    compressed_tokens: int
    tokens_saved: int
    compression_ratio: float
    items_before: int | None
    items_after: int | None
    errors_preserved: int
    was_compressed: bool
    tool_name: str
    context_used: str


@dataclass
class MCPToolProfile:
    """Configuration profile for a specific MCP tool.

    Different tools may need different compression strategies:
    - Slack search: High error preservation, relevance to query
    - Database query: Schema detection, anomaly preservation
    - File listing: Minimal compression (paths are important)
    """

    tool_name_pattern: str  # Regex pattern to match tool names
    enabled: bool = True
    max_items: int = 20
    min_tokens_to_compress: int = 500
    preserve_error_keywords: set[str] = field(
        default_factory=lambda: {"error", "failed", "exception", "critical", "fatal"}
    )
    always_keep_fields: set[str] = field(default_factory=set)  # Fields to never drop


# Default profiles for common MCP servers
DEFAULT_MCP_PROFILES: list[MCPToolProfile] = [
    # Slack - preserve errors and messages matching query
    MCPToolProfile(
        tool_name_pattern=r".*slack.*",
        max_items=25,
        preserve_error_keywords={"error", "failed", "exception", "bug", "issue", "broken"},
    ),
    # Database - preserve errors and anomalies
    MCPToolProfile(
        tool_name_pattern=r".*database.*|.*sql.*|.*query.*",
        max_items=30,
        preserve_error_keywords={"error", "null", "failed", "exception", "violation"},
    ),
    # GitHub - preserve errors and high-priority issues
    MCPToolProfile(
        tool_name_pattern=r".*github.*|.*git.*",
        max_items=20,
        preserve_error_keywords={"error", "bug", "critical", "urgent", "blocker"},
    ),
    # Logs - preserve ALL errors
    MCPToolProfile(
        tool_name_pattern=r".*log.*|.*trace.*",
        max_items=40,  # Keep more for logs
        preserve_error_keywords={"error", "fatal", "critical", "exception", "failed", "panic"},
    ),
    # File system - minimal compression (paths matter)
    MCPToolProfile(
        tool_name_pattern=r".*file.*|.*fs.*|.*directory.*",
        max_items=50,
        min_tokens_to_compress=1000,  # Higher threshold
    ),
    # Generic fallback
    MCPToolProfile(
        tool_name_pattern=r".*",
        max_items=20,
    ),
]


class HeadroomMCPCompressor:
    """Core compression logic for MCP tool outputs.

    This class handles the actual compression of MCP tool results.
    It's used by both the proxy server and standalone functions.
    """

    def __init__(
        self,
        config: HeadroomConfig | None = None,
        profiles: list[MCPToolProfile] | None = None,
        token_counter: Callable[[str], int] | None = None,
    ):
        """Initialize MCP compressor.

        Args:
            config: Headroom configuration.
            profiles: Tool-specific compression profiles.
            token_counter: Function to count tokens. Uses tiktoken if None.
        """
        self.config = config or HeadroomConfig()
        self.profiles = profiles or DEFAULT_MCP_PROFILES

        # Initialize token counter
        if token_counter:
            self._count_tokens = token_counter
        else:
            provider = OpenAIProvider()
            counter = provider.get_token_counter("gpt-4o")
            self._count_tokens = counter.count_text

    def get_profile(self, tool_name: str) -> MCPToolProfile:
        """Get the compression profile for a tool."""
        for profile in self.profiles:
            if re.match(profile.tool_name_pattern, tool_name, re.IGNORECASE):
                return profile
        # Return last profile (generic fallback)
        return self.profiles[-1]

    def compress(
        self,
        content: str,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        user_query: str = "",
    ) -> MCPCompressionResult:
        """Compress MCP tool output.

        Args:
            content: Raw tool output (usually JSON string).
            tool_name: Name of the MCP tool (e.g., "mcp__slack__search").
            tool_args: Arguments passed to the tool (used for context).
            user_query: User's original query (for relevance scoring).

        Returns:
            MCPCompressionResult with compressed content and metrics.
        """
        profile = self.get_profile(tool_name)
        original_tokens = self._count_tokens(content)

        # Build context for relevance scoring
        context_parts = []
        if user_query:
            context_parts.append(user_query)
        if tool_args:
            context_parts.append(json.dumps(tool_args))
        context = " ".join(context_parts)

        # Check if compression is needed
        if not profile.enabled or original_tokens < profile.min_tokens_to_compress:
            return MCPCompressionResult(
                original_content=content,
                compressed_content=content,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                tokens_saved=0,
                compression_ratio=0.0,
                items_before=None,
                items_after=None,
                errors_preserved=0,
                was_compressed=False,
                tool_name=tool_name,
                context_used=context,
            )

        # Try to parse as JSON
        try:
            json.loads(content)
        except json.JSONDecodeError:
            # Not JSON, return as-is
            return MCPCompressionResult(
                original_content=content,
                compressed_content=content,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                tokens_saved=0,
                compression_ratio=0.0,
                items_before=None,
                items_after=None,
                errors_preserved=0,
                was_compressed=False,
                tool_name=tool_name,
                context_used=context,
            )

        # Find arrays to compress
        items_before = 0
        items_after = 0
        errors_preserved = 0

        # Create SmartCrusher with profile settings.
        #
        # The MCP integration emits JSON-shaped output that downstream
        # tool consumers parse and iterate. The PR4 lossless path
        # substitutes a CSV+schema STRING in place of arrays — great
        # for LLM prompts but wire-format-incompatible with consumers
        # that expect JSON arrays. So we keep the runtime MCP wrapper
        # on the lossy + CCR-Dropped path: the LLM sees row-level
        # subsets inline, with the full payload retrievable via CCR
        # cache. (Same retention semantics as Python's pre-PR4
        # SmartCrusher behavior.)
        smart_config = SmartCrusherConfig(
            enabled=True,
            min_tokens_to_crush=profile.min_tokens_to_compress,
            max_items_after_crush=profile.max_items,
        )
        crusher = SmartCrusher(config=smart_config, with_compaction=False)  # type: ignore[arg-type]

        # Build messages for SmartCrusher (it expects conversation format)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": context or f"Process {tool_name} results"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": tool_name, "arguments": json.dumps(tool_args or {})},
                    }
                ],
            },
            {"role": "tool", "content": content, "tool_call_id": "call_1"},
        ]

        # Create tokenizer wrapper
        class TokenizerWrapper:
            def __init__(self, count_fn: Any) -> None:
                self._count = count_fn

            def count_text(self, text: str) -> int:
                result = self._count(text)
                return int(result) if result is not None else 0

            def count_messages(self, messages: list[dict[str, Any]]) -> int:
                total = 0
                for msg in messages:
                    if msg.get("content"):
                        total += self._count(str(msg["content"]))
                return total

        tokenizer = TokenizerWrapper(self._count_tokens)

        # Apply SmartCrusher
        result = crusher.apply(messages, tokenizer=tokenizer)  # type: ignore[arg-type]
        compressed_content = result.messages[-1]["content"]

        # Remove any Headroom markers for clean output
        compressed_content = re.sub(r"\n<headroom:[^>]+>", "", compressed_content)

        # Count items and errors
        try:
            original_data = json.loads(content)
            compressed_data = json.loads(compressed_content)

            # Find the array in original
            for _key, value in original_data.items():
                if isinstance(value, list):
                    items_before = len(value)
                    break

            # Find the array in compressed
            for _key, value in compressed_data.items():
                if isinstance(value, list):
                    items_after = len(value)
                    # Count errors preserved
                    for item in value:
                        item_str = str(item).lower()
                        if any(kw in item_str for kw in profile.preserve_error_keywords):
                            errors_preserved += 1
                    break
        except (json.JSONDecodeError, AttributeError):
            pass

        compressed_tokens = self._count_tokens(compressed_content)
        tokens_saved = original_tokens - compressed_tokens
        compression_ratio = tokens_saved / original_tokens if original_tokens > 0 else 0.0

        return MCPCompressionResult(
            original_content=content,
            compressed_content=compressed_content,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            tokens_saved=tokens_saved,
            compression_ratio=compression_ratio,
            items_before=items_before,
            items_after=items_after,
            errors_preserved=errors_preserved,
            was_compressed=True,
            tool_name=tool_name,
            context_used=context,
        )


def compress_tool_result(
    content: str,
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
    user_query: str = "",
    config: HeadroomConfig | None = None,
) -> str:
    """Compress an MCP tool result (standalone function).

    This is the simplest way to use Headroom with MCP in your host application.

    Args:
        content: Raw tool output.
        tool_name: Name of the MCP tool.
        tool_args: Arguments passed to the tool.
        user_query: User's query for relevance scoring.
        config: Optional Headroom configuration.

    Returns:
        Compressed content string.

    Example:
        ```python
        # In your MCP host application
        result = await client.call_tool("search_logs", {"service": "api"})

        compressed = compress_tool_result(
            content=result,
            tool_name="search_logs",
            tool_args={"service": "api"},
            user_query="find errors",
        )

        messages.append({"role": "tool", "content": compressed})
        ```
    """
    compressor = HeadroomMCPCompressor(config=config)
    result = compressor.compress(
        content=content,
        tool_name=tool_name,
        tool_args=tool_args,
        user_query=user_query,
    )
    return result.compressed_content


def compress_tool_result_with_metrics(
    content: str,
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
    user_query: str = "",
    config: HeadroomConfig | None = None,
) -> MCPCompressionResult:
    """Compress an MCP tool result and return full metrics.

    Same as compress_tool_result but returns detailed metrics.

    Returns:
        MCPCompressionResult with all compression metrics.
    """
    compressor = HeadroomMCPCompressor(config=config)
    return compressor.compress(
        content=content,
        tool_name=tool_name,
        tool_args=tool_args,
        user_query=user_query,
    )


class HeadroomMCPClientWrapper:
    """Wrapper for MCP clients that automatically compresses tool results.

    This wraps an MCP client to transparently compress all tool outputs.

    Example:
        ```python
        from mcp import Client
        from headroom.integrations.mcp import HeadroomMCPClientWrapper

        # Wrap your MCP client
        base_client = Client(transport)
        client = HeadroomMCPClientWrapper(base_client)

        # Use normally - compression is automatic
        result = await client.call_tool("search", {"query": "errors"})
        ```
    """

    def __init__(
        self,
        client: Any,
        config: HeadroomConfig | None = None,
        user_query_extractor: Callable[[dict], str] | None = None,
    ):
        """Initialize wrapper.

        Args:
            client: The MCP client to wrap.
            config: Headroom configuration.
            user_query_extractor: Function to extract user query from context.
        """
        self._client = client
        self._compressor = HeadroomMCPCompressor(config=config)
        self._query_extractor = user_query_extractor or (lambda x: "")
        self._metrics: list[MCPCompressionResult] = []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Call an MCP tool and compress the result.

        Args:
            name: Tool name.
            arguments: Tool arguments.
            context: Optional context (for query extraction).

        Returns:
            Compressed tool result.
        """
        # Call the underlying client
        result = await self._client.call_tool(name, arguments)

        # Extract user query from context if available
        user_query = ""
        if context and self._query_extractor is not None:
            user_query = self._query_extractor(context)

        # Compress
        compression_result = self._compressor.compress(
            content=result,
            tool_name=name,
            tool_args=arguments,
            user_query=user_query,
        )

        self._metrics.append(compression_result)
        return compression_result.compressed_content

    def get_metrics(self) -> list[MCPCompressionResult]:
        """Get compression metrics for all tool calls."""
        return self._metrics.copy()

    def get_total_tokens_saved(self) -> int:
        """Get total tokens saved across all tool calls."""
        return sum(m.tokens_saved for m in self._metrics)

    def __getattr__(self, name: str) -> Any:
        """Forward all other attributes to the wrapped client."""
        return getattr(self._client, name)


# Type alias for MCP Server (will be properly typed when mcp package is used)
MCPServer = Any


def create_headroom_mcp_proxy(
    upstream_servers: list[tuple[str, MCPServer]],
    config: HeadroomConfig | None = None,
) -> dict[str, Any]:
    """Create configuration for a custom Headroom MCP proxy/server wrapper.

    This returns the compressor and upstream-server mapping needed by an
    application-defined MCP proxy/server wrapper. Headroom does not yet ship
    a ready-to-run ``HeadroomMCPProxy`` server implementation.

    Args:
        upstream_servers: List of (name, server) tuples.
        config: Headroom configuration.

    Returns:
        Configuration dict for a custom proxy/server wrapper.

    Example:
        ```python
        # In your MCP server setup
        proxy_config = create_headroom_mcp_proxy(
            upstream_servers=[
                ("slack", slack_server),
                ("database", db_server),
            ]
        )

        # Use proxy_config to initialize your proxy server
        ```
    """
    return {
        "upstream_servers": dict(upstream_servers),
        "compressor": HeadroomMCPCompressor(config=config),
        "config": config or HeadroomConfig(),
    }
