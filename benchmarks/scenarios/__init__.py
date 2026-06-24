"""Benchmark scenario generators for Headroom SDK.

This package provides realistic data generators for benchmarking Headroom
transforms and relevance scorers.

Modules:
    tool_outputs: Generators for tool output data (search, logs, API responses)
    conversations: Generators for conversation history (agentic, RAG)
"""

from .conversations import (
    generate_agentic_conversation,
    generate_rag_conversation,
)
from .tool_outputs import (
    generate_api_responses,
    generate_database_rows,
    generate_log_entries,
    generate_search_results,
)

__all__ = [
    "generate_search_results",
    "generate_log_entries",
    "generate_api_responses",
    "generate_database_rows",
    "generate_agentic_conversation",
    "generate_rag_conversation",
]
