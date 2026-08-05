from __future__ import annotations

from headroom.proxy.body_forwarding import serialize_body_canonical
from headroom.proxy.tool_definition_serialization import serialize_tool_definition_canonical


def test_serialize_tool_definition_canonical_uses_compact_separators() -> None:
    tool = {"name": "memory_save", "input_schema": {"type": "object"}}

    assert serialize_tool_definition_canonical(tool) == (
        b'{"name":"memory_save","input_schema":{"type":"object"}}'
    )


def test_serialize_tool_definition_canonical_preserves_unicode() -> None:
    tool = {"name": "memory_save", "description": "remember cafe notes"}
    tool["description"] = "remember caf\u00e9 notes"

    assert b"caf\xc3\xa9" in serialize_tool_definition_canonical(tool)


def test_serialize_tool_definition_canonical_preserves_insertion_order() -> None:
    first = {"name": "memory_save", "description": "desc"}
    second = {"description": "desc", "name": "memory_save"}

    assert serialize_tool_definition_canonical(first) != serialize_tool_definition_canonical(second)


def test_serialize_tool_definition_canonical_matches_body_canonicalizer() -> None:
    tool = {"name": "memory_save", "input_schema": {"type": "object"}}

    assert serialize_tool_definition_canonical(tool) == serialize_body_canonical(tool)
