from __future__ import annotations

import pytest

from headroom.ccr.tool_injection import CCR_TOOL_NAME, create_ccr_tool_definition
from headroom.proxy.ccr_golden_policy import (
    create_fresh_ccr_tool_definition,
    replay_golden_ccr_tool_definition,
    serialize_ccr_tool_definition_canonical,
)


def test_replays_golden_definition_without_reserializing() -> None:
    golden = b'{ "name" : "headroom_retrieve" , "description" : "client bytes" }'

    replay = replay_golden_ccr_tool_definition(golden)

    assert replay.tool_definition["name"] == CCR_TOOL_NAME
    assert replay.canonical_bytes == golden
    assert replay.used_golden_bytes is True


def test_rejects_invalid_golden_json() -> None:
    with pytest.raises(ValueError):
        replay_golden_ccr_tool_definition(b"not-json")


def test_rejects_non_utf8_golden_bytes() -> None:
    with pytest.raises(UnicodeDecodeError):
        replay_golden_ccr_tool_definition(b"\x80\x81")


def test_fresh_definition_uses_canonical_bytes() -> None:
    replay = create_fresh_ccr_tool_definition("anthropic")

    assert replay.tool_definition == create_ccr_tool_definition("anthropic")
    assert replay.canonical_bytes == serialize_ccr_tool_definition_canonical(replay.tool_definition)
    assert replay.used_golden_bytes is False
