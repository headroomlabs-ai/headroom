"""Edge 1 — intent query. No model; ~2 days of work per the protocol.

Extracts the newest assistant reasoning text plus its tool-call arguments as
the relevance query every other edge scores against. Falls back to the
tool-call arguments alone, then to the last user text message, per the
protocol's own risk mitigation for turns with no visible reasoning (tool call
only, or redacted thinking).

Read-only over `messages` — this edge never mutates anything, so it can't be
the thing that trips the signed-thinking-block passthrough in
`body_forwarding.py`. It also never returns the *content* of a thinking or
redacted_thinking block: only `text` and `tool_use` blocks are read, which is
the structural enforcement for cache-safety rule #3 (never touch thinking).
"""

from __future__ import annotations

from typing import Any

_ARG_KEYS_OF_INTEREST = ("command", "pattern", "path", "file_path", "url", "query")


def _tool_use_query_fragment(block: dict[str, Any]) -> str:
    tool_input = block.get("input")
    if not isinstance(tool_input, dict):
        return ""
    parts = [str(tool_input[key]) for key in _ARG_KEYS_OF_INTEREST if tool_input.get(key)]
    if parts:
        return " ".join(parts)
    # No recognized key — fall back to the whole input, still useful signal.
    return " ".join(str(v) for v in tool_input.values() if isinstance(v, (str, int, float)))


def _text_from_content(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content] if content.strip() else []
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif block_type == "tool_use":
            fragment = _tool_use_query_fragment(block)
            if fragment:
                texts.append(fragment)
        # Deliberately no branch for "thinking" / "redacted_thinking": those
        # blocks are never read, per cache-safety rule #3.
    return texts


def _last_message_text(messages: list[dict[str, Any]], role: str) -> list[str]:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != role:
            continue
        texts = _text_from_content(message.get("content"))
        if texts:
            return texts
    return []


def extract_intent_query(messages: list[dict[str, Any]]) -> str:
    """Newest assistant text + tool-call args; else last user text; else ''."""
    assistant_texts = _last_message_text(messages, "assistant")
    if assistant_texts:
        return " ".join(assistant_texts).strip()
    user_texts = _last_message_text(messages, "user")
    return " ".join(user_texts).strip()
