"""CCR (Compress-Cache-Retrieve) module for reversible compression.

This module provides tool injection and retrieval handling for the CCR architecture.
When tool outputs are compressed, the LLM can retrieve more data if needed.

Four key components:
1. Tool Injection: Proxy injects headroom_retrieve tool into requests
2. Response Handler: Intercepts responses, handles CCR tool calls automatically
3. Context Tracker: Tracks compressed content across turns, enables proactive expansion
4. Batch Processing: Handles CCR tool calls in batch API results (async processing)

Two distribution channels for the retrieval tool:
1. Tool Injection: Proxy injects tool into request when compression occurs
2. MCP Server: Standalone server exposes tool via MCP protocol

When MCP is configured, tool injection is skipped to avoid duplicates.

Batch API Support:
- On batch submit: Store request context (messages, tools) in BatchContextStore
- On batch results: Detect CCR tool calls, execute retrieval, make continuation calls
- Works with all providers: Anthropic, OpenAI, Google
"""

from .batch_processor import (
    BatchResultProcessor,
    BatchResultProcessorConfig,
    ProcessedBatchResult,
    process_batch_results,
)
from .batch_store import (
    BatchContext,
    BatchContextStore,
    BatchRequestContext,
    get_batch_context_store,
    reset_batch_context_store,
)
from .context_tracker import (
    CompressedContext,
    ContextTracker,
    ContextTrackerConfig,
    ExpansionRecommendation,
    get_context_tracker,
    reset_context_tracker,
)
from .response_handler import (
    CCRResponseHandler,
    CCRToolCall,
    CCRToolResult,
    ResponseHandlerConfig,
    StreamingCCRBuffer,
    StreamingCCRHandler,
)
from .tool_injection import (
    CCR_TOOL_NAME,
    CCRToolInjector,
    create_ccr_tool_definition,
    create_system_instructions,
    parse_tool_call,
)

# MCP server is optional (requires mcp package)
try:
    from .mcp_server import HeadroomMCPServer, create_ccr_mcp_server

    MCP_SERVER_AVAILABLE = True
except ImportError:
    HeadroomMCPServer = None  # type: ignore
    create_ccr_mcp_server = None  # type: ignore
    MCP_SERVER_AVAILABLE = False

__all__ = [
    # Tool injection
    "CCR_TOOL_NAME",
    "CCRToolInjector",
    "create_ccr_tool_definition",
    "create_system_instructions",
    "parse_tool_call",
    # Response handling
    "CCRResponseHandler",
    "CCRToolCall",
    "CCRToolResult",
    "ResponseHandlerConfig",
    "StreamingCCRBuffer",
    "StreamingCCRHandler",
    # Context tracking
    "CompressedContext",
    "ContextTracker",
    "ContextTrackerConfig",
    "ExpansionRecommendation",
    "get_context_tracker",
    "reset_context_tracker",
    # Batch processing
    "BatchContext",
    "BatchContextStore",
    "BatchRequestContext",
    "BatchResultProcessor",
    "BatchResultProcessorConfig",
    "ProcessedBatchResult",
    "get_batch_context_store",
    "process_batch_results",
    "reset_batch_context_store",
    # MCP server
    "HeadroomMCPServer",
    "create_ccr_mcp_server",
    "MCP_SERVER_AVAILABLE",
]
