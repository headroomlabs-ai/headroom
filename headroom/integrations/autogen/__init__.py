"""AutoGen integration for Headroom.

This module provides tool output compression for AutoGen agents,
wrapping FunctionTool instances so their outputs are automatically
compressed before entering the agent's model context.

Components:
    - HeadroomToolWrapper: Wraps a single AutoGen FunctionTool with compression
    - wrap_tools_with_headroom: Wraps multiple tools at once
    - ToolCompressionMetrics: Per-invocation metrics dataclass
    - ToolMetricsCollector: Aggregates metrics across all invocations

Example:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_core.tools import FunctionTool
    from headroom.integrations.autogen import wrap_tools_with_headroom

    def search_db(query: str) -> str:
        return json.dumps(results)

    tool = FunctionTool(search_db, description="Search the database")
    wrapped = wrap_tools_with_headroom([tool])

    agent = AssistantAgent(name="researcher", tools=wrapped, ...)

Install: pip install headroom-ai autogen-agentchat
"""

from .agents import (
    HeadroomToolWrapper,
    ToolCompressionMetrics,
    ToolMetricsCollector,
    get_tool_metrics,
    reset_tool_metrics,
    wrap_tools_with_headroom,
)

__all__ = [
    "HeadroomToolWrapper",
    "ToolCompressionMetrics",
    "ToolMetricsCollector",
    "wrap_tools_with_headroom",
    "get_tool_metrics",
    "reset_tool_metrics",
]
