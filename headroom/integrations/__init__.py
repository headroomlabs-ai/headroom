"""Headroom integrations with popular LLM frameworks.

Available integrations:

LangChain (pip install headroom[langchain]):
    - HeadroomChatModel: Drop-in wrapper for any LangChain chat model
    - HeadroomChatMessageHistory: Automatic conversation compression
    - HeadroomDocumentCompressor: Relevance-based document filtering
    - HeadroomToolWrapper: Tool output compression for agents
    - StreamingMetricsTracker: Token counting during streaming
    - HeadroomLangSmithCallbackHandler: LangSmith trace enrichment

Agno (pip install agno):
    - HeadroomAgnoModel: Drop-in wrapper for any Agno model
    - HeadroomPreHook/HeadroomPostHook: Agent-level hooks for tracking
    - create_headroom_hooks: Convenience function to create hook pairs

MCP (Model Context Protocol):
    - HeadroomMCPCompressor: Compress MCP tool results
    - compress_tool_result: Simple function for tool compression

Example:
    # LangChain integration
    from headroom.integrations import HeadroomChatModel
    # or explicitly:
    from headroom.integrations.langchain import HeadroomChatModel

    # Agno integration
    from headroom.integrations.agno import HeadroomAgnoModel
    # or explicitly:
    from headroom.integrations.agno import HeadroomAgnoModel

    # MCP integration
    from headroom.integrations import compress_tool_result
    # or explicitly:
    from headroom.integrations.mcp import compress_tool_result
"""

# Re-export from langchain subpackage for backwards compatibility
from .langchain import (
    # Retrievers
    CompressionMetrics,
    # Core
    HeadroomCallbackHandler,
    # Memory
    HeadroomChatMessageHistory,
    HeadroomChatModel,
    HeadroomDocumentCompressor,
    # LangSmith
    HeadroomLangSmithCallbackHandler,
    HeadroomRunnable,
    # Agents
    HeadroomToolWrapper,
    OptimizationMetrics,
    # Streaming
    StreamingMetrics,
    StreamingMetricsCallback,
    StreamingMetricsTracker,
    ToolCompressionMetrics,
    ToolMetricsCollector,
    # Provider Detection
    detect_provider,
    get_headroom_provider,
    get_model_name_from_langchain,
    get_tool_metrics,
    is_langsmith_available,
    is_langsmith_tracing_enabled,
    langchain_available,
    optimize_messages,
    reset_tool_metrics,
    track_async_streaming_response,
    track_streaming_response,
    wrap_tools_with_headroom,
)

# Re-export from mcp subpackage for backwards compatibility
from .mcp import (
    DEFAULT_MCP_PROFILES,
    HeadroomMCPClientWrapper,
    HeadroomMCPCompressor,
    MCPCompressionResult,
    MCPToolProfile,
    compress_tool_result,
    compress_tool_result_with_metrics,
    create_headroom_mcp_proxy,
)

# Re-export from agno subpackage (optional dependency)
try:
    from .agno import (
        HeadroomAgnoModel,
        HeadroomPostHook,
        HeadroomPreHook,
        agno_available,
        create_headroom_hooks,
        get_model_name_from_agno,
    )
    from .agno import OptimizationMetrics as AgnoOptimizationMetrics
    from .agno import get_headroom_provider as get_agno_provider
    from .agno import optimize_messages as optimize_agno_messages

    _AGNO_AVAILABLE = True
except ImportError:
    _AGNO_AVAILABLE = False

__all__ = [
    # LangChain Core
    "HeadroomChatModel",
    "HeadroomCallbackHandler",
    "HeadroomRunnable",
    "OptimizationMetrics",
    "optimize_messages",
    "langchain_available",
    # Provider Detection
    "detect_provider",
    "get_headroom_provider",
    "get_model_name_from_langchain",
    # Memory
    "HeadroomChatMessageHistory",
    # Retrievers
    "HeadroomDocumentCompressor",
    "CompressionMetrics",
    # Agents
    "HeadroomToolWrapper",
    "ToolCompressionMetrics",
    "ToolMetricsCollector",
    "wrap_tools_with_headroom",
    "get_tool_metrics",
    "reset_tool_metrics",
    # LangSmith
    "HeadroomLangSmithCallbackHandler",
    "is_langsmith_available",
    "is_langsmith_tracing_enabled",
    # Streaming
    "StreamingMetricsTracker",
    "StreamingMetricsCallback",
    "StreamingMetrics",
    "track_streaming_response",
    "track_async_streaming_response",
    # MCP
    "HeadroomMCPCompressor",
    "HeadroomMCPClientWrapper",
    "MCPCompressionResult",
    "MCPToolProfile",
    "compress_tool_result",
    "compress_tool_result_with_metrics",
    "create_headroom_mcp_proxy",
    "DEFAULT_MCP_PROFILES",
    # Agno
    "HeadroomAgnoModel",
    "HeadroomPreHook",
    "HeadroomPostHook",
    "agno_available",
    "create_headroom_hooks",
    "get_agno_provider",
    "get_model_name_from_agno",
    "AgnoOptimizationMetrics",
    "optimize_agno_messages",
]
