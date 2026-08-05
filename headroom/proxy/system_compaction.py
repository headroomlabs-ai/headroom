"""System-prompt compaction for Headroom proxy handlers.

Compresses the ``system`` field in Anthropic Messages API requests
using the existing ContentRouter (CCR), reducing the token cost of
static context blocks (CLAUDE.md, rules, hooks, MCP instructions)
without removing them entirely.

Opt-in via ``HEADROOM_SYSTEM_COMPACT=1`` (default disabled).
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

# Minimum length (chars) for a system block to be eligible for compression.
# Short instruction blocks (< 500 chars) are almost always critical and
# should not be lossily compressed.
_SYSTEM_COMPACT_MIN_CHARS = 500


def system_compact_enabled() -> bool:
    """Return whether system-prompt compaction is enabled via env var."""
    return os.environ.get("HEADROOM_SYSTEM_COMPACT", "").strip() in ("1", "true")


def system_compact_min_chars() -> int:
    """Return the minimum block length for compression (env-configurable)."""
    try:
        return int(
            os.environ.get("HEADROOM_SYSTEM_COMPACT_MIN_CHARS", str(_SYSTEM_COMPACT_MIN_CHARS))
        )
    except ValueError:
        return _SYSTEM_COMPACT_MIN_CHARS


def _json_byte_len(value: Any) -> int:
    """Byte length of compact JSON serialisation (for size comparisons)."""
    return len(json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":")))


def _compact_system_blocks(
    blocks: list[dict[str, Any]],
    router: Any,
    model: str,
    request_id: str,
    min_chars: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Compress eligible text blocks in *blocks* using *router*.

    Returns ``(updated_blocks, modified)``.  Blocks shorter than
    *min_chars* are left untouched.  ``cache_control`` and other
    non-text fields are preserved unchanged.
    """
    modified = False
    updated: list[dict[str, Any]] = []

    for block in blocks:
        if not isinstance(block, dict) or block.get("type") != "text":
            updated.append(block)
            continue

        text = block.get("text", "")
        if not isinstance(text, str) or len(text) < min_chars:
            updated.append(block)
            continue

        # Attempt CCR compression.
        try:
            result = router.compress(text, context="")
            compressed_text = result.compressed if hasattr(result, "compressed") else str(result)
        except Exception:
            # CCR failure → leave block unchanged.
            updated.append(block)
            continue

        if not isinstance(compressed_text, str) or len(compressed_text) >= len(text):
            # Compression didn't help → leave unchanged.
            updated.append(block)
            continue

        new_block: dict[str, Any] = {"type": "text", "text": compressed_text}
        # Preserve any other fields (cache_control, etc.)
        for k, v in block.items():
            if k not in ("type", "text"):
                new_block[k] = v
        updated.append(new_block)
        modified = True

    return updated, modified


def compact_system_prompt(
    payload: dict[str, Any],
    router: Any,
    model: str,
    request_id: str,
) -> tuple[dict[str, Any], bool, int, int]:
    """Compress the ``system`` field in *payload* using CCR.

    Returns ``(updated_payload, modified, before_bytes, after_bytes)``.
    If compaction doesn't reduce size, the original payload is returned
    unchanged and *modified* is ``False``.
    """
    system = payload.get("system")
    if system is None:
        return payload, False, 0, 0

    min_chars = system_compact_min_chars()

    # Anthropic system field can be a string or a list of content blocks.
    if isinstance(system, str):
        if len(system) < min_chars:
            return payload, False, 0, 0
        blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
    elif isinstance(system, list):
        blocks = system
    else:
        return payload, False, 0, 0

    before = _json_byte_len(blocks)

    updated_blocks, modified = _compact_system_blocks(
        blocks,
        router,
        model,
        request_id,
        min_chars,
    )

    after = _json_byte_len(updated_blocks)
    if not modified or after >= before:
        return payload, False, before, after

    updated = copy.deepcopy(payload)
    # Restore in the original format.
    if isinstance(system, str):
        # Single-string system → reassemble from compressed blocks.
        texts = [b.get("text", "") for b in updated_blocks if b.get("type") == "text"]
        updated["system"] = "\n".join(texts)
    else:
        updated["system"] = updated_blocks

    return updated, True, before, after
