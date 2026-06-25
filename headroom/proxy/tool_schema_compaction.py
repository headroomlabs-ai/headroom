"""Shared tool-schema compaction for Headroom proxy handlers.

Strips JSON Schema annotation keys ($schema, title, examples, etc.)
and normalises description whitespace to reduce the token cost of
tool definitions without changing their semantics.

Both the OpenAI and Anthropic handlers call the same compaction
logic from this module.

**Layer 2 — Tool Description Compaction**

Truncates tool and parameter ``description`` strings to a configurable
maximum length, preserving the first complete sentence so that the model
can still select the right tool.  Opt-in via
``HEADROOM_TOOL_DESC_MAX_CHARS`` (default ``0`` = disabled).
"""

from __future__ import annotations

import copy
import json
import os
import re
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

# ---------------------------------------------------------------------------
# Env-var helpers
# ---------------------------------------------------------------------------

_TOOL_DESC_MAX_CHARS: int | None = None


def tool_desc_max_chars() -> int:
    """Return the configured max description length (cached per-process).

    ``HEADROOM_TOOL_DESC_MAX_CHARS=0`` (default) disables truncation.
    """
    global _TOOL_DESC_MAX_CHARS
    if _TOOL_DESC_MAX_CHARS is None:
        try:
            _TOOL_DESC_MAX_CHARS = int(
                os.environ.get("HEADROOM_TOOL_DESC_MAX_CHARS", "0")
            )
        except ValueError:
            _TOOL_DESC_MAX_CHARS = 0
    return _TOOL_DESC_MAX_CHARS


# ---------------------------------------------------------------------------
# Layer 1: annotation-key compaction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Layer 2: description truncation
# ---------------------------------------------------------------------------

_FIRST_SENTENCE_RE = re.compile(r"^(.*?[.!?])(?:\s|$)", re.DOTALL)


def _truncate_description(desc: str, max_chars: int) -> str:
    """Truncate *desc* to *max_chars*, preserving the first complete sentence.

    Strategy:
    - *max_chars* ≤ 0: return *desc* unchanged (feature disabled).
    - Short descriptions (≤ *max_chars*) pass through unchanged.
    - Normalise whitespace before any truncation.
    - If the first sentence fits in *max_chars*, keep it and optionally
      append the second sentence when the combined length ≤ 1.5× *max_chars*.
    - If the first sentence alone exceeds *max_chars*, hard-truncate
      and append ``…``.
    """
    if max_chars <= 0:
        return desc

    # Normalise whitespace first (mirrors Layer 1 behaviour).
    desc = " ".join(desc.split())

    if len(desc) <= max_chars:
        return desc

    m = _FIRST_SENTENCE_RE.match(desc)
    if m and len(m.group(1)) <= max_chars:
        first = m.group(1)
        rest = desc[len(first):].strip()
        if rest:
            m2 = _FIRST_SENTENCE_RE.match(rest)
            if m2 and len(first) + 1 + len(m2.group(1)) <= int(max_chars * 1.5):
                return f"{first} {m2.group(1)}"
        return first

    # First sentence too long → hard truncation.
    return desc[:max_chars].rstrip() + "…"


def _truncate_descriptions_in_schema(
    value: Any,
    max_chars: int,
) -> Any:
    """Recursively truncate ``description`` fields in a tool-schema structure."""
    if isinstance(value, list):
        return [_truncate_descriptions_in_schema(item, max_chars) for item in value]

    if not isinstance(value, dict):
        return value

    compacted: dict[str, Any] = {}
    for key, child in value.items():
        if key == "description" and isinstance(child, str):
            compacted[key] = _truncate_description(child, max_chars)
        else:
            compacted[key] = _truncate_descriptions_in_schema(child, max_chars)

    return compacted


def compact_tool_descriptions(
    payload: dict[str, Any],
    max_chars: int = 0,
) -> tuple[dict[str, Any], bool, int, int]:
    """Truncate tool descriptions in *payload* to *max_chars*.

    Returns ``(updated_payload, modified, before_bytes, after_bytes)``.
    If *max_chars* is 0 (default) or compaction doesn't reduce size,
    the original payload is returned unchanged.
    """
    if max_chars <= 0:
        return payload, False, 0, 0

    tools = payload.get("tools")
    if not isinstance(tools, list) or not tools:
        return payload, False, 0, 0

    compacted_tools = _truncate_descriptions_in_schema(tools, max_chars)
    before = _json_byte_len(tools)
    after = _json_byte_len(compacted_tools)
    if after >= before:
        return payload, False, before, after

    updated = copy.deepcopy(payload)
    updated["tools"] = compacted_tools
    return updated, True, before, after
