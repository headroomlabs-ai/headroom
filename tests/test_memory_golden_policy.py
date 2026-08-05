from __future__ import annotations

import pytest

from headroom.proxy.helpers import serialize_tool_definition_canonical
from headroom.proxy.memory_golden_policy import (
    replay_golden_memory_tool_definition,
    serialize_memory_tool_definition_canonical,
)


def test_replays_golden_memory_tool_without_reserializing() -> None:
    golden = b'{ "name" : "memory_save" , "description" : "client bytes" }'

    replay = replay_golden_memory_tool_definition(
        tool_name="memory_save",
        golden_tool_bytes=golden,
    )

    assert replay.tool_name == "memory_save"
    assert replay.tool_definition["name"] == "memory_save"
    assert replay.canonical_bytes == golden


def test_rejects_invalid_golden_json() -> None:
    with pytest.raises(ValueError):
        replay_golden_memory_tool_definition(
            tool_name="memory_save",
            golden_tool_bytes=b"not-json",
        )


def test_rejects_non_utf8_golden_bytes() -> None:
    with pytest.raises(UnicodeDecodeError):
        replay_golden_memory_tool_definition(
            tool_name="memory_save",
            golden_tool_bytes=b"\x80\x81",
        )


def test_memory_canonical_serializer_matches_existing_helper() -> None:
    tool_definition = {
        "name": "memory_search",
        "description": "Find memory",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    }

    assert serialize_memory_tool_definition_canonical(
        tool_definition
    ) == serialize_tool_definition_canonical(tool_definition)
