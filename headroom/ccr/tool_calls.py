"""Provider-shaped CCR tool-call extraction and classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tool_injection import CCR_TOOL_NAME, parse_tool_call


@dataclass
class CCRToolCall:
    """Represents a detected CCR retrieval tool call."""

    tool_call_id: str
    hash_key: str


def extract_tool_calls(response: dict[str, Any], provider: str) -> list[dict[str, Any]]:
    """Extract provider-native tool-call objects from a response payload."""
    if provider == "anthropic":
        content = response.get("content", [])
        if isinstance(content, list):
            return [
                block
                for block in content
                if isinstance(block, dict) and block.get("type") == "tool_use"
            ]
        return []

    if provider == "openai":
        choices = response.get("choices", [])
        if not isinstance(choices, list) or not choices:
            return []
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            return []
        message = first_choice.get("message", {})
        if not isinstance(message, dict):
            return []
        tool_calls = message.get("tool_calls", [])
        return list(tool_calls) if isinstance(tool_calls, list) else []

    if provider == "google":
        candidates = response.get("candidates", [])
        if not isinstance(candidates, list) or not candidates:
            return []
        first_candidate = candidates[0]
        if not isinstance(first_candidate, dict):
            return []
        content = first_candidate.get("content", {})
        if not isinstance(content, dict):
            return []
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            return []
        return [part for part in parts if isinstance(part, dict) and "functionCall" in part]

    if provider == "openai_responses":
        output = response.get("output", [])
        if isinstance(output, list):
            return [
                item
                for item in output
                if isinstance(item, dict) and item.get("type") == "function_call"
            ]
        return []

    return []


def is_ccr_tool_call(tool_call: dict[str, Any]) -> bool:
    """Return true when a provider-native tool call names the CCR retrieval tool."""
    return (
        tool_call.get("name") == CCR_TOOL_NAME
        or (tool_call.get("function") or {}).get("name") == CCR_TOOL_NAME
        or (tool_call.get("functionCall") or {}).get("name") == CCR_TOOL_NAME
    )


def has_ccr_tool_calls(response: dict[str, Any], provider: str) -> bool:
    """Return true when ``response`` contains at least one CCR tool call."""
    return any(is_ccr_tool_call(tool_call) for tool_call in extract_tool_calls(response, provider))


def tool_call_id_for_provider(tool_call: dict[str, Any], provider: str) -> str:
    """Return the provider-specific identifier used by the matching tool result."""
    if provider == "google":
        function_call = tool_call.get("functionCall", {})
        if isinstance(function_call, dict):
            name = function_call.get("name", CCR_TOOL_NAME)
            return str(name)
        return CCR_TOOL_NAME
    if provider == "openai_responses":
        call_id = tool_call.get("call_id", tool_call.get("id", ""))
        return str(call_id)
    return str(tool_call.get("id", ""))


def parse_ccr_tool_calls(
    response: dict[str, Any],
    provider: str,
) -> tuple[list[CCRToolCall], list[dict[str, Any]]]:
    """Split provider-native tool calls into CCR retrievals and other tools."""
    ccr_calls: list[CCRToolCall] = []
    other_calls: list[dict[str, Any]] = []

    for tool_call in extract_tool_calls(response, provider):
        hash_key = parse_tool_call(tool_call, provider)
        if hash_key is None:
            other_calls.append(tool_call)
            continue

        ccr_calls.append(
            CCRToolCall(
                tool_call_id=tool_call_id_for_provider(tool_call, provider),
                hash_key=hash_key,
            )
        )

    return ccr_calls, other_calls
