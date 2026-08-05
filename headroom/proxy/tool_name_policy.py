"""Tool-definition name extraction policy used by proxy injection helpers."""

from __future__ import annotations

from typing import Any


def extract_tool_name(tool_definition: dict[str, Any]) -> str | None:
    """Extract a stable tool name from a tool definition."""

    name = tool_definition.get("name")
    if isinstance(name, str) and name:
        return name
    function_definition = tool_definition.get("function")
    if isinstance(function_definition, dict):
        function_name = function_definition.get("name")
        if isinstance(function_name, str) and function_name:
            return function_name
    tool_type = tool_definition.get("type")
    if isinstance(tool_type, str) and tool_type:
        return tool_type
    return None
