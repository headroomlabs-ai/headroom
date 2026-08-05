"""Canonical byte serialization for sticky memory tool definitions."""

from __future__ import annotations

import json
from typing import Any


def serialize_tool_definition_canonical(tool_definition: dict[str, Any]) -> bytes:
    """Serialize a tool definition to deterministic compact UTF-8 JSON bytes."""

    return json.dumps(
        tool_definition,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
