"""Policy helpers for replaying memory-tool golden definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast


@dataclass(frozen=True)
class MemoryToolDefinitionReplay:
    """Memory tool definition selected for sticky replay."""

    tool_name: str
    tool_definition: dict[str, Any]
    canonical_bytes: bytes


def serialize_memory_tool_definition_canonical(tool_definition: dict[str, Any]) -> bytes:
    """Return stable canonical bytes for a memory tool definition."""

    return json.dumps(
        tool_definition,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def replay_golden_memory_tool_definition(
    *,
    tool_name: str,
    golden_tool_bytes: bytes,
) -> MemoryToolDefinitionReplay:
    """Decode a stored memory tool definition and preserve its original bytes."""

    tool_definition = json.loads(golden_tool_bytes.decode("utf-8"))
    return MemoryToolDefinitionReplay(
        tool_name=tool_name,
        tool_definition=cast(dict[str, Any], tool_definition),
        canonical_bytes=golden_tool_bytes,
    )
