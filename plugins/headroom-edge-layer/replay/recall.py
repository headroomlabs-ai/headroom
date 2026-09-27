"""Next-step recall: does a compressed tool result still carry what the
agent used from it in the next few turns?

For each compressed tool result, collect what the agent used from it in the
next N turns — file paths, line numbers, identifiers, error strings in later
tool arguments and assistant text — and report what share of those items
are still substrings of the compressed version. Reported per bucket, not as
one global number, per the protocol.
"""

from __future__ import annotations

import json
import re
from typing import Any

_PATH_RE = re.compile(r"\b[\w./-]+\.[A-Za-z]{1,8}\b")
_LINE_REF_RE = re.compile(r":\d+:")
_ERROR_RE = re.compile(r"\b\w*(?:Error|Exception|Traceback)\w*\b")
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{4,}\b")


def _texts_from_message(message: dict[str, Any]) -> list[str]:
    content = message.get("content")
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            texts.append(block["text"])
        elif block_type == "tool_use" and isinstance(block.get("input"), dict):
            texts.append(json.dumps(block["input"]))
        elif block_type == "tool_result":
            inner = block.get("content")
            if isinstance(inner, str):
                texts.append(inner)
            elif isinstance(inner, list):
                texts.extend(
                    c.get("text", "") for c in inner if isinstance(c, dict) and c.get("type") == "text"
                )
    return texts


def extract_referenced_items(messages: list[dict[str, Any]], *, max_items: int = 200) -> set[str]:
    """Distinct paths / line refs / error tokens / identifiers mentioned
    across `messages`, capped so a pathological turn can't blow up scoring.
    """
    items: set[str] = set()
    for message in messages:
        for text in _texts_from_message(message):
            items.update(_PATH_RE.findall(text))
            items.update(_ERROR_RE.findall(text))
            # Identifiers are the noisiest signal (lots of false positives),
            # so cap how many any single text block contributes.
            items.update(_IDENTIFIER_RE.findall(text)[:20])
            if len(items) >= max_items:
                return set(list(items)[:max_items])
    return items


def next_step_recall(compressed_text: str, next_messages: list[dict[str, Any]]) -> float | None:
    """Share of items referenced in `next_messages` still present verbatim in
    `compressed_text`. Returns None (not 0.0 or 1.0) when there's nothing to
    check — a "no signal" result must not be mistaken for perfect recall.
    """
    referenced = extract_referenced_items(next_messages)
    if not referenced:
        return None
    present = sum(1 for item in referenced if item in compressed_text)
    return present / len(referenced)
