"""Pure structural turn classification for output shaping."""

from __future__ import annotations

from enum import Enum
from typing import Any


class TurnKind(Enum):
    """Structural classification of the latest conversation turn."""

    NEW_USER_ASK = "new_user_ask"
    MECHANICAL_CONTINUATION = "mechanical_continuation"
    ERROR_CONTINUATION = "error_continuation"
    UNKNOWN = "unknown"


_OPENAI_RESPONSES_OUTPUT_ITEM_TYPES = frozenset(
    {
        "custom_tool_call_output",
        "function_call_output",
        "local_shell_call_output",
        "apply_patch_call_output",
    }
)


def classify_turn(messages: list[dict[str, Any]]) -> TurnKind:
    """Classify the latest Anthropic-style turn from message structure only."""
    if not messages:
        return TurnKind.UNKNOWN
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return TurnKind.UNKNOWN

    content = last.get("content")
    if isinstance(content, str):
        return TurnKind.NEW_USER_ASK if content.strip() else TurnKind.UNKNOWN
    if not isinstance(content, list) or not content:
        return TurnKind.UNKNOWN

    saw_tool_result = False
    saw_error = False
    for block in content:
        if not isinstance(block, dict):
            return TurnKind.UNKNOWN
        btype = block.get("type")
        if btype == "tool_result":
            saw_tool_result = True
            if block.get("is_error") is True:
                saw_error = True
        elif btype == "text":
            return TurnKind.NEW_USER_ASK
        elif btype in ("image", "document"):
            return TurnKind.NEW_USER_ASK

    if saw_error:
        return TurnKind.ERROR_CONTINUATION
    if saw_tool_result:
        return TurnKind.MECHANICAL_CONTINUATION
    return TurnKind.UNKNOWN


def _responses_part_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        texts: list[str] = []
        for part in value:
            if isinstance(part, str):
                texts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                texts.append(part["text"])
        return "\n".join(text for text in texts if text)
    return ""


def _responses_user_signal(item: dict[str, Any]) -> bool:
    item_type = item.get("type")
    role = item.get("role")
    if role == "user":
        content = item.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in {
                    "input_file",
                    "input_image",
                }:
                    return True
        text = _responses_part_text(content)
        return bool(text.strip())
    if item_type == "input_text":
        text = _responses_part_text(item.get("text"))
        return bool(text.strip())
    if item_type == "input_image":
        return True
    return False


def classify_openai_responses_input(input_data: Any) -> TurnKind:
    """Classify OpenAI Responses ``input`` without content heuristics."""
    if isinstance(input_data, str):
        return TurnKind.NEW_USER_ASK if input_data.strip() else TurnKind.UNKNOWN
    if not isinstance(input_data, list) or not input_data:
        return TurnKind.UNKNOWN

    saw_tool_output = False
    saw_unknown = False
    for item in input_data:
        if not isinstance(item, dict):
            saw_unknown = True
            continue
        item_type = item.get("type")
        if item_type in _OPENAI_RESPONSES_OUTPUT_ITEM_TYPES:
            saw_tool_output = True
            continue
        if _responses_user_signal(item):
            return TurnKind.NEW_USER_ASK
        if item_type in {"message", "function_call", "reasoning"}:
            continue
        saw_unknown = True

    if saw_tool_output and not saw_unknown:
        return TurnKind.MECHANICAL_CONTINUATION
    return TurnKind.UNKNOWN
