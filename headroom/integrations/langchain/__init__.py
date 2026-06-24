"""LangChain integration for Headroom.

This package provides seamless integration with LangChain, including:
- HeadroomChatModel: Drop-in wrapper for any LangChain chat model
- HeadroomChatMessageHistory: Automatic conversation compression
- HeadroomDocumentCompressor: Relevance-based document filtering
- HeadroomToolWrapper: Tool output compression for agents
- StreamingMetricsTracker: Token counting during streaming
- HeadroomLangSmithCallbackHandler: LangSmith trace enrichment
- compress_tool_messages: LangGraph pre-model hook for ToolMessage compression
- create_compress_tool_messages_node: LangGraph node factory

Example:
    from langchain_openai import ChatOpenAI
    from headroom.integrations.langchain import HeadroomChatModel

    # Wrap any LangChain model
    llm = HeadroomChatModel(ChatOpenAI(model="gpt-4o"))

    # Use like normal - optimization happens automatically
    response = llm.invoke("Hello!")

Install: pip install headroom[langchain]
"""

# Agent tool wrapping
from .agents import (
    HeadroomToolWrapper,
    ToolCompressionMetrics,
    ToolMetricsCollector,
    get_tool_metrics,
    reset_tool_metrics,
    wrap_tools_with_headroom,
)

# Core chat model wrapper
from .chat_model import (
    HeadroomCallbackHandler,
    HeadroomChatModel,
    HeadroomRunnable,
    OptimizationMetrics,
    langchain_available,
    optimize_messages,
)

# LangGraph integration
from .langgraph import (
    CompressToolMessagesConfig,
    CompressToolMessagesResult,
    ToolMessageCompressionMetrics,
    compress_tool_messages,
    create_compress_tool_messages_node,
)

# LangSmith integration
from .langsmith import (
    HeadroomLangSmithCallbackHandler,
    is_langsmith_available,
    is_langsmith_tracing_enabled,
)

# Memory integration
from .memory import HeadroomChatMessageHistory

# Provider auto-detection
from .providers import (
    detect_provider,
    get_headroom_provider,
    get_model_name_from_langchain,
)

# Retriever integration
from .retriever import CompressionMetrics, HeadroomDocumentCompressor

# Streaming metrics
from .streaming import (
    StreamingMetrics,
    StreamingMetricsCallback,
    StreamingMetricsTracker,
    track_async_streaming_response,
    track_streaming_response,
)

__all__ = [
    # Core
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
    # LangGraph
    "compress_tool_messages",
    "create_compress_tool_messages_node",
    "CompressToolMessagesConfig",
    "CompressToolMessagesResult",
    "ToolMessageCompressionMetrics",
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
]
