"""Policy helpers for replaying CCR golden tool definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, cast

from headroom.ccr.tool_injection import create_ccr_tool_definition


@dataclass(frozen=True)
class CcrToolDefinitionReplay:
    """CCR tool definition selected for sticky replay or fresh injection."""

    tool_definition: dict[str, Any]
    canonical_bytes: bytes
    used_golden_bytes: bool


def serialize_ccr_tool_definition_canonical(tool_definition: dict[str, Any]) -> bytes:
    """Return stable canonical bytes for a CCR tool definition."""

    return json.dumps(
        tool_definition,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def replay_golden_ccr_tool_definition(golden_tool_bytes: bytes) -> CcrToolDefinitionReplay:
    """Decode a stored CCR tool definition and preserve its original bytes."""

    tool_definition = json.loads(golden_tool_bytes.decode("utf-8"))
    return CcrToolDefinitionReplay(
        tool_definition=cast(dict[str, Any], tool_definition),
        canonical_bytes=golden_tool_bytes,
        used_golden_bytes=True,
    )


def create_fresh_ccr_tool_definition(
    provider: Literal["anthropic", "openai", "google"],
) -> CcrToolDefinitionReplay:
    """Create and canonicalize a fresh CCR tool definition for ``provider``."""

    tool_definition = create_ccr_tool_definition(provider)
    return CcrToolDefinitionReplay(
        tool_definition=tool_definition,
        canonical_bytes=serialize_ccr_tool_definition_canonical(tool_definition),
        used_golden_bytes=False,
    )
