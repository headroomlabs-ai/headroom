"""Shared, read-mostly helpers over Anthropic-shape message lists.

Every function here is scoped to `tool_use` (assistant) and `tool_result`
(user) blocks only — never `thinking` / `redacted_thinking` — which is the
structural half of cache-safety rule #3 (never touch thinking blocks).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def collect_tool_use_index(messages: list[dict[str, Any]]) -> dict[str, tuple[str, Any]]:
    """tool_call_id -> (tool_name, tool_input), scanning assistant `tool_use` blocks only."""
    index: dict[str, tuple[str, Any]] = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_use_id = block.get("id")
                if isinstance(tool_use_id, str):
                    index[tool_use_id] = (block.get("name", ""), block.get("input"))
    return index


def get_block_text(block: dict[str, Any]) -> str | None:
    content = block.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [c.get("text") for c in content if isinstance(c, dict) and c.get("type") == "text"]
        texts = [t for t in texts if isinstance(t, str)]
        if texts:
            return "\n".join(texts)
    return None


def set_block_text(block: dict[str, Any], new_text: str) -> None:
    content = block.get("content")
    if isinstance(content, str):
        block["content"] = new_text
        return
    if isinstance(content, list):
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                c["text"] = new_text
                return
    block["content"] = new_text


def iter_tool_result_blocks(
    messages: list[dict[str, Any]],
) -> Iterator[tuple[int, int, dict[str, Any], str | None]]:
    """Yields (message_index, block_index, block, tool_use_id) for every
    `tool_result` block in a `user`-role message. Read-only iteration —
    callers mutate `block` in place via `set_block_text` if they choose to.
    """
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                yield message_index, block_index, block, block.get("tool_use_id")
