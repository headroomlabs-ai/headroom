"""Transforms must never rewrite bytes the provider has already hashed.

A ``cache_control`` marker means the provider is billing everything up to and
including that point at the cache-read rate (0.1x on Anthropic). Rewriting any
of those bytes busts the entry and re-bills the whole prefix at the write rate
(1.25x) -- a 12.5x swing that dwarfs anything the transform saves.

Preserving the ``cache_control`` *field* does not preserve the *entry*: the
provider's key is the content, not the marker.
"""

from __future__ import annotations

import json
from typing import Any

from headroom.proxy.system_compaction import _compact_system_blocks
from headroom.proxy.tool_schema_compaction import compact_tool_descriptions, compact_tools


def _text(chars: int, seed: str = "x") -> str:
    return (seed * 10 + " context line. ") * (chars // 25 + 1)


class _StubRouter:
    """Router whose compression is obvious and always shrinks."""

    def compress(self, text: str, context: str = "") -> Any:  # noqa: ARG002
        class _R:
            compressed = "SHRUNK"

        return _R()


# --------------------------------------------------------------------------
# tool description compaction
# --------------------------------------------------------------------------


def test_tool_desc_compaction_runs_even_when_a_tool_is_marked() -> None:
    """A marker does NOT freeze the tools, and skipping was the expensive choice.

    The provider never saw the client's bytes -- it only ever sees ours -- so
    there is nothing to preserve. Compaction is deterministic, so every turn
    sends the same compacted bytes and the cache hits. Measured against the real
    API with 30 pinned tools over 6 turns, always-compacted billed 15,270 in
    prefix tokens against 21,938 for always-raw: skipping made the cached prefix
    30% larger for the whole session to avoid a bust that never happened.
    """
    payload = {
        "tools": [
            {"name": "a", "description": _text(4000), "input_schema": {}},
            {
                "name": "b",
                "description": _text(4000, "y"),
                "input_schema": {},
                "cache_control": {"type": "ephemeral"},
            },
        ]
    }

    out, modified, before_bytes, after_bytes = compact_tool_descriptions(payload, max_chars=50)

    assert modified is True
    assert after_bytes < before_bytes
    # The marker itself is carried through untouched -- we shrink the schema,
    # we do not take over the client's cache management.
    assert out["tools"][1]["cache_control"] == {"type": "ephemeral"}


def test_compaction_is_byte_stable_across_calls() -> None:
    """The property the whole argument rests on.

    Cache-safety here is not "we avoided the bytes", it is "we produce the same
    bytes every time". If this ever stops holding, the removed guard has to come
    back -- so it is asserted rather than assumed.
    """

    def _payload() -> dict[str, Any]:
        return {
            "tools": [
                {
                    "name": f"t{i}",
                    "description": "  Lots   of    whitespace.  " * 20,
                    "input_schema": {
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "title": "Args",
                        "examples": [{"a": 1}],
                        "type": "object",
                        "properties": {"a": {"type": "string", "description": "  x  y  "}},
                    },
                }
                for i in range(5)
            ]
        }

    first, _, _, _ = compact_tools(_payload())
    second, _, _, _ = compact_tools(_payload())

    assert json.dumps(first["tools"], sort_keys=True) == json.dumps(second["tools"], sort_keys=True)


def test_a_marked_tools_array_is_compacted_by_layer_one_too() -> None:
    """Layer 1 never had a guard, and now it deliberately never will."""
    payload = {
        "tools": [
            {
                "name": "a",
                "description": "d",
                "input_schema": {"type": "object", "title": "T", "properties": {}},
                "cache_control": {"type": "ephemeral"},
            }
        ]
    }

    out, modified, _, _ = compact_tools(payload)

    assert modified is True
    assert "title" not in out["tools"][0]["input_schema"]


def test_tool_desc_compaction_still_runs_when_nothing_is_marked() -> None:
    """The common case -- no markers -- must keep working."""
    payload = {
        "tools": [
            {"name": "a", "description": _text(4000), "input_schema": {}},
            {"name": "b", "description": _text(4000, "y"), "input_schema": {}},
        ]
    }

    out, modified, before_bytes, after_bytes = compact_tool_descriptions(payload, max_chars=50)

    assert modified is True
    assert after_bytes < before_bytes
    assert all(len(t["description"]) <= 80 for t in out["tools"])


# --------------------------------------------------------------------------
# system prompt compaction
# --------------------------------------------------------------------------


def test_system_compaction_leaves_blocks_at_or_before_the_marker() -> None:
    """Blocks inside the cached span keep their exact bytes."""
    blocks = [
        {"type": "text", "text": _text(2000, "a")},
        {"type": "text", "text": _text(2000, "b"), "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": _text(2000, "c")},
    ]
    original = [dict(b) for b in blocks]

    out, modified = _compact_system_blocks(blocks, _StubRouter(), "m", "rid", 500)

    # frozen: identical bytes, marker intact
    assert out[0] == original[0]
    assert out[1] == original[1]
    assert out[1]["cache_control"] == {"type": "ephemeral"}
    # after the breakpoint: free to compact
    assert out[2]["text"] == "SHRUNK"
    assert modified is True


def test_system_compaction_unrestricted_without_markers() -> None:
    blocks = [
        {"type": "text", "text": _text(2000, "a")},
        {"type": "text", "text": _text(2000, "b")},
    ]

    out, modified = _compact_system_blocks(blocks, _StubRouter(), "m", "rid", 500)

    assert modified is True
    assert [b["text"] for b in out] == ["SHRUNK", "SHRUNK"]


def test_system_compaction_noop_when_marker_is_last() -> None:
    """Marker on the final block freezes everything."""
    blocks = [
        {"type": "text", "text": _text(2000, "a")},
        {"type": "text", "text": _text(2000, "b"), "cache_control": {"type": "ephemeral"}},
    ]
    original = [dict(b) for b in blocks]

    out, modified = _compact_system_blocks(blocks, _StubRouter(), "m", "rid", 500)

    assert modified is False
    assert out == original
