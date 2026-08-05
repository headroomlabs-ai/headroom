"""CrewAI integration for Headroom.

This module provides tool output compression for CrewAI agents,
wrapping BaseTool instances so their outputs are automatically
compressed before entering the agent's LLM context.

Components:
    - HeadroomToolWrapper: Wraps a single CrewAI BaseTool with compression
    - wrap_tools_with_headroom: Wraps multiple tools at once
    - ToolCompressionMetrics: Per-invocation metrics dataclass
    - ToolMetricsCollector: Aggregates metrics across all invocations

Example:
    from crewai import Agent, Crew, Task
    from crewai.tools.base_tool import tool
    from headroom.integrations.crewai import wrap_tools_with_headroom

    @tool
    def search_db(query: str) -> str:
        \"\"\"Search the database.\"\"\"
        return json.dumps(results)

    wrapped = wrap_tools_with_headroom([search_db])
    agent = Agent(role="Researcher", tools=wrapped, ...)

Install: pip install headroom-ai crewai
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
