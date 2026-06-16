"""Provider-owned memory tool formatting and parsing helpers."""

from __future__ import annotations

import json
from typing import Any


def memory_tool_definitions(
    *,
    provider: str,
    memory_tools: list[dict[str, Any]],
    inject_tools: bool,
    use_native_tool: bool,
    native_tool_type: str,
    native_tool_name: str,
) -> list[dict[str, Any]]:
    """Return provider-native memory tool definitions."""
    if not inject_tools:
        return []

    if use_native_tool and provider == "anthropic":
        return [{"type": native_tool_type, "name": native_tool_name}]

    out: list[dict[str, Any]] = []
    for memory_tool in memory_tools:
        tool_name = memory_tool["function"]["name"]
        if provider == "anthropic":
            out.append(
                {
                    "name": tool_name,
                    "description": memory_tool["function"]["description"],
                    "input_schema": memory_tool["function"]["parameters"],
                }
            )
        else:
            out.append(dict(memory_tool))
    return out


def inject_memory_tool_definitions(
    *,
    tools: list[dict[str, Any]],
    provider: str,
    memory_tools: list[dict[str, Any]],
    existing_names: set[str],
) -> tuple[list[dict[str, Any]], bool]:
    """Append missing provider-native memory tool definitions."""
    was_injected = False
    for memory_tool in memory_tools:
        tool_name = memory_tool["function"]["name"]
        if tool_name in existing_names:
            continue

        if provider == "anthropic":
            tools.append(
                {
                    "name": tool_name,
                    "description": memory_tool["function"]["description"],
                    "input_schema": memory_tool["function"]["parameters"],
                }
            )
        else:
            tools.append(memory_tool)

        was_injected = True

    return tools, was_injected


def extract_memory_tool_calls(response: dict[str, Any], provider: str) -> list[dict[str, Any]]:
    """Extract provider-native memory tool calls from a response payload."""
    if provider == "anthropic":
        content = response.get("content", [])
        if isinstance(content, list):
            return [block for block in content if block.get("type") == "tool_use"]
        return []

    if provider == "openai":
        choices = response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            tc_list = list(message.get("tool_calls", []) or [])
            if tc_list:
                return tc_list

        output = response.get("output", [])
        if isinstance(output, list):
            return [
                item
                for item in output
                if isinstance(item, dict) and item.get("type") == "function_call"
            ]

    return []


def memory_tool_name(tool_call: dict[str, Any], provider: str) -> str:
    """Return the provider-native memory tool-call name."""
    if provider == "anthropic":
        return str(tool_call.get("name", ""))
    if provider == "openai":
        return str(tool_call.get("name") or tool_call.get("function", {}).get("name", ""))
    return str(tool_call.get("name", "") or tool_call.get("function", {}).get("name", ""))


def memory_tool_id(tool_call: dict[str, Any], provider: str) -> str:
    """Return the provider-native memory tool-call id."""
    if provider == "openai":
        return str(tool_call.get("id") or tool_call.get("call_id", ""))
    return str(tool_call.get("id", ""))


def memory_tool_input(tool_call: dict[str, Any], provider: str) -> dict[str, Any]:
    """Return parsed provider-native memory tool input."""
    if provider == "anthropic":
        result = tool_call.get("input", {})
        return dict(result) if isinstance(result, dict) else {}

    args_str = tool_call.get("arguments") or tool_call.get("function", {}).get("arguments") or "{}"
    try:
        parsed = json.loads(args_str)
        return dict(parsed) if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def memory_tool_result(tool_id: str, content: str, provider: str) -> dict[str, Any]:
    """Return provider-native memory tool result data."""
    if provider == "anthropic":
        return {
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": content,
        }
    return {
        "role": "tool",
        "tool_call_id": tool_id,
        "content": content,
    }


def append_memory_context_tail(
    messages: list[dict[str, Any]],
    context_text: str,
    *,
    provider: str,
    frozen_message_count: int,
) -> tuple[list[dict[str, Any]], int]:
    """Append memory context to the provider-native latest user tail."""
    if provider == "anthropic":
        from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin

        new_messages = AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
            messages,
            context_text,
            frozen_message_count=frozen_message_count,
        )
        if new_messages is messages:
            return messages, 0
        return new_messages, len(context_text)

    if provider == "openai":
        from headroom.proxy.helpers import append_text_to_latest_user_chat_message

        return append_text_to_latest_user_chat_message(messages, context_text)

    raise ValueError(f"Unknown provider {provider!r}; expected 'anthropic' or 'openai'")
