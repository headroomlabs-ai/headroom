"""Tests for normalizing client-specific tool names."""

from __future__ import annotations

import pytest

from headroom.config import DEFAULT_EXCLUDE_TOOLS, is_tool_excluded


@pytest.mark.parametrize(
    "name",
    [
        "headroom_retrieve",
        "mcp__headroom__headroom_retrieve",
        "mcp_headroom_headroom_retrieve",
        "headroom_headroom_retrieve",
    ],
)
def test_retrieve_tool_aliases_are_excluded(name: str) -> None:
    assert is_tool_excluded(name, DEFAULT_EXCLUDE_TOOLS)


def test_unrelated_underscored_tool_names_are_not_excluded() -> None:
    assert not is_tool_excluded("my_headroom_retrieve", DEFAULT_EXCLUDE_TOOLS)
    assert not is_tool_excluded("other_Read", DEFAULT_EXCLUDE_TOOLS)
