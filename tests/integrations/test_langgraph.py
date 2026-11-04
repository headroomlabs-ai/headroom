"""Regression tests for qualified CCR retrieval tool names in LangGraph."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("headroom._core")

try:
    from langchain_core.messages import AIMessage, ToolMessage
except ImportError:
    pytest.skip("LangChain not installed", allow_module_level=True)

from headroom.integrations.langchain.langgraph import compress_tool_messages


def _large_output() -> str:
    return json.dumps([{"id": i, "name": f"item_{i}", "value": "x" * 30} for i in range(200)])


def _messages(tool_name: str) -> list:
    return [
        AIMessage(content="", tool_calls=[{"id": "call_1", "name": tool_name, "args": {}}]),
        ToolMessage(content=_large_output(), tool_call_id="call_1"),
    ]


@pytest.mark.parametrize(
    "tool_name",
    ["mcp__Headroom__headroom_retrieve", "mcp_Headroom_headroom_retrieve"],
)
def test_qualified_ccr_retrieval_message_is_preserved(tool_name: str) -> None:
    messages = _messages(tool_name)
    original = messages[1].content

    result = compress_tool_messages(messages)

    assert result.messages[1].content == original
    assert result.metrics[0].skip_reason == "tool_excluded"


def test_incomplete_tool_calls_do_not_hide_later_qualified_name() -> None:
    messages = [
        AIMessage(
            content="",
            tool_calls=[
                {"id": None, "name": "incomplete", "args": {}},
                {"id": "ignored", "name": "", "args": {}},
                {
                    "id": "call_1",
                    "name": "mcp__Headroom__headroom_retrieve",
                    "args": {},
                },
            ],
        ),
        ToolMessage(content=_large_output(), tool_call_id="call_1"),
    ]
    original = messages[1].content

    result = compress_tool_messages(messages)

    assert result.messages[1].content == original
    assert result.metrics[0].skip_reason == "tool_excluded"


def test_near_match_ccr_tool_name_is_not_excluded() -> None:
    messages = _messages("mcp__Headroom__headroom_retrieve_extra")
    original = messages[1].content

    result = compress_tool_messages(messages)

    assert result.metrics[0].skip_reason != "tool_excluded"
    assert result.messages[1].content != original


@pytest.mark.parametrize(
    "tool_name",
    ["mcp__Headroom__headroom_retrieve", "mcp_Headroom_headroom_retrieve"],
)
def test_qualified_name_on_the_tool_message_is_enough(tool_name: str) -> None:
    """`ToolNode` populates `ToolMessage.name`, so the id index is only a fallback."""
    messages = [ToolMessage(content=_large_output(), tool_call_id="call_1", name=tool_name)]
    original = messages[0].content

    result = compress_tool_messages(messages)

    assert result.messages[0].content == original
    assert result.metrics[0].skip_reason == "tool_excluded"
