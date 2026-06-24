"""MCP (Model Context Protocol) integration for Headroom.

This package provides compression utilities for MCP tool results,
helping reduce context usage when tools return large outputs.

Example:
    from headroom.integrations.mcp import compress_tool_result

    # Compress large tool output
    result = compress_tool_result(
        tool_name="search",
        result=large_json_result,
        max_chars=5000,
    )
"""

from .server import (
    DEFAULT_MCP_PROFILES,
    HeadroomMCPClientWrapper,
    HeadroomMCPCompressor,
    MCPCompressionResult,
    MCPToolProfile,
    compress_tool_result,
    compress_tool_result_with_metrics,
    create_headroom_mcp_proxy,
)

__all__ = [
    "HeadroomMCPCompressor",
    "HeadroomMCPClientWrapper",
    "MCPCompressionResult",
    "MCPToolProfile",
    "compress_tool_result",
    "compress_tool_result_with_metrics",
    "create_headroom_mcp_proxy",
    "DEFAULT_MCP_PROFILES",
]
