"""Shared tool-schema compaction for Headroom proxy handlers.

Strips JSON Schema annotation keys ($schema, title, examples, etc.)
and normalises description whitespace to reduce the token cost of
tool definitions without changing their semantics.

Both the OpenAI and Anthropic handlers call the same compaction
logic from this module.
"""

from __future__ import annotations

import copy
import json
from typing import Any

# Keys that are JSON Schema annotations, not constraints.
# Removing them does not change the set of valid inputs.
TOOL_SCHEMA_DROP_KEYS: frozenset[str] = frozenset({
    "$id",
    "$schema",
    "$comment",
    "deprecated",
    "examples",
    "example",
    "markdownDescription",
    "readOnly",
    "title",
    "writeOnly",
})


def _json_byte_len(value: Any) -> int:
    """Byte length of compact JSON serialisation (for size comparisons)."""
    return len(json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":")))


def compact_tool_schema_value(
    value: Any,
    _parent_key: str | None = None,
) -> Any:
    """Recursively compact a tool-schema structure.

    - Drops annotation keys (``TOOL_SCHEMA_DROP_KEYS``) unless they appear
      as property *names* inside a ``properties`` object (e.g. a field
      literally named ``"title"`` must survive).
    - Normalises ``description`` strings by collapsing whitespace.
    """
    if isinstance(value, list):
        return [compact_tool_schema_value(item, _parent_key) for item in value]

    if not isinstance(value, dict):
        return value

    compacted: dict[str, Any] = {}
    for key, child in value.items():
        # Don't drop keys that are property *names* inside a JSON Schema
        # `properties` object — only drop them when they are schema annotations.
        if _parent_key != "properties" and key in TOOL_SCHEMA_DROP_KEYS:
            continue

        if key == "description" and isinstance(child, str):
            compacted[key] = " ".join(child.split())
            continue

        compacted[key] = compact_tool_schema_value(child, key)

    return compacted


def compact_tools(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], bool, int, int]:
    """Compact the ``tools`` array in *payload*.

    Returns ``(updated_payload, modified, before_bytes, after_bytes)``.
    If compaction did not reduce size, the original payload is returned
    unchanged and *modified* is ``False``.
    """
    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return payload, False, 0, 0

    compacted_tools = compact_tool_schema_value(tools)
    before = _json_byte_len(tools)
    after = _json_byte_len(compacted_tools)
    if after >= before:
        return payload, False, before, after

    updated = copy.deepcopy(payload)
    updated["tools"] = compacted_tools
    return updated, True, before, after
