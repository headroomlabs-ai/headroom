"""The gateway body never carries more cache_control blocks than Anthropic accepts.

A LiteLLM benchmark run saw ``A maximum of 4 blocks with cache_control may be
provided. Found 5.`` on a turn whose only Headroom work was tool-result
compression and tool schema compaction. The client (Claude Code) never sends
more than four, so the fifth was added on the way through. Whatever the path,
the last stop before the body leaves must hold the budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from headroom.proxy.helpers import count_cache_breakpoints, enforce_cache_breakpoint_budget
from headroom.proxy.turn_hooks import register_turn_hook
from tests.gateway.conftest import compress

CC = {"type": "ephemeral"}


def _system(markers: int) -> list[dict[str, Any]]:
    return [
        {
            "type": "text",
            "text": f"system part {i}",
            **({"cache_control": CC} if i < markers else {}),
        }
        for i in range(2)
    ]


def _messages(marked: set[int], n: int = 6) -> list[dict[str, Any]]:
    out = []
    for i in range(n):
        block = {"type": "text", "text": f"message {i}"}
        if i in marked:
            block["cache_control"] = CC
        out.append({"role": "user" if i % 2 == 0 else "assistant", "content": [block]})
    return out


def test_under_budget_is_returned_by_identity() -> None:
    system, messages = _system(2), _messages({4, 5})
    s, m, t, stats = enforce_cache_breakpoint_budget(system, messages, None)
    assert (s, m, t) == (system, messages, None)
    assert s is system and m is messages
    assert stats["repaired"] is False


def test_extra_message_marker_goes_back_to_client_positions() -> None:
    client = _messages({4, 5})
    outbound = _messages({2, 4, 5})  # one marker a replay or transform carried in
    s, m, _, stats = enforce_cache_breakpoint_budget(
        _system(2), outbound, None, client_messages=client
    )
    assert stats["repaired"] is True
    assert count_cache_breakpoints(s, m, None)["total"] == 4
    marked = [i for i, msg in enumerate(m) if "cache_control" in msg["content"][0]]
    assert marked == [4, 5]


def test_without_client_positions_the_oldest_message_markers_go_first() -> None:
    outbound = _messages({1, 3, 5})
    s, m, _, stats = enforce_cache_breakpoint_budget(_system(2), outbound, None)
    assert count_cache_breakpoints(s, m, None)["total"] == 4
    marked = [i for i, msg in enumerate(m) if "cache_control" in msg["content"][0]]
    # The newest marker is the write anchor for the growing tail: it stays.
    assert marked == [3, 5]
    assert all("cache_control" in b for b in s)
    # Copy-on-write: the caller's objects are untouched.
    assert "cache_control" in outbound[1]["content"][0]


@dataclass
class _MarkerLeakHook:
    """Stands in for any stage that leaves an extra marker on an old message."""

    name: str = "test_marker_leak"
    savings_source: str = "test_marker_leak"
    stream_safe: bool = True

    def on_request(self, ctx: Any) -> None:
        messages = [dict(m) for m in ctx.messages]
        first = messages[0]
        messages[0] = {**first, "content": [{**first["content"][0], "cache_control": CC}]}
        ctx.messages = messages

    async def on_response(self, ctx: Any, response: Any, call_model: Any) -> None:
        return None


def test_gateway_body_is_held_to_four_markers(headroom_client) -> None:
    register_turn_hook(_MarkerLeakHook())
    body = {
        "model": "claude-sonnet-4-5",
        "system": _system(2),
        "messages": _messages({4, 5}),
        "gateway": {"can_redrive": False, "can_relay_response": False},
    }
    resp = compress(headroom_client, body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    out = data["body"]
    assert (
        count_cache_breakpoints(out.get("system"), out["messages"], out.get("tools"))["total"] == 4
    )
    assert data["messages"] == out["messages"]
    assert "cache_breakpoint_budget" in data["transforms_applied"]


def test_openai_chat_replayed_markers_capped_oldest_first() -> None:
    # Issue #4471: a replayed frozen prefix keeps an old message-level marker
    # and the message count differs from the client's, so the mirror step
    # bails; the budget guard must still cap OpenAI chat shapes at four.
    client = [
        {"role": "system", "content": "s1", "cache_control": CC},
        {"role": "system", "content": "s2", "cache_control": CC},
        {"role": "user", "content": "read f1"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "r", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "a", "content": "f1 body", "cache_control": CC},
    ]
    tools = [{"type": "function", "function": {"name": "r"}, "cache_control": CC}]
    replayed = [dict(m) for m in client]
    replayed[3] = {
        **replayed[3],
        "tool_calls": [{**replayed[3]["tool_calls"][0], "cache_control": CC}],
    }
    replayed.insert(2, {"role": "user", "content": "earlier", "cache_control": CC})

    assert count_cache_breakpoints(None, replayed, tools)["total"] == 6
    _, messages, tools_out, stats = enforce_cache_breakpoint_budget(
        None, replayed, tools, client_messages=None
    )
    assert stats["repaired"]
    assert count_cache_breakpoints(None, messages, tools_out)["total"] <= 4
    assert "cache_control" in messages[-1]


def test_openai_chat_budget_drops_stale_tool_marker_keeps_systems() -> None:
    # Issue #4471: on the OpenAI chat path client_messages=None so
    # normalize_message_cache_control's block-level legacy fallback cannot add
    # a marker to a list-content user message and then cost a real system
    # marker. Only the oldest-first drop runs, so stale replayed markers go
    # first while system markers and the latest tool anchor are kept.
    messages = [
        # older replayed tool result carrying a marker: oldest, dropped first
        {"role": "tool", "tool_call_id": "t1", "content": "stale", "cache_control": CC},
        {"role": "system", "content": "system one", "cache_control": CC},
        {"role": "system", "content": "system two", "cache_control": CC},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        # latest tool result carrying a marker: kept
        {"role": "tool", "tool_call_id": "t2", "content": "fresh", "cache_control": CC},
    ]
    tools = [{"type": "function", "function": {"name": "r"}, "cache_control": CC}]

    assert count_cache_breakpoints(None, messages, tools)["total"] == 5
    _, out, tools_out, stats = enforce_cache_breakpoint_budget(
        None, messages, tools, client_messages=None
    )
    assert stats["repaired"]
    assert count_cache_breakpoints(None, out, tools_out)["total"] <= 4
    assert "cache_control" in out[1]
    assert "cache_control" in out[2]
    assert "cache_control" in out[-1]
    assert "cache_control" not in out[0]
    assert out[3] == {"role": "user", "content": [{"type": "text", "text": "hi"}]}


def test_count_includes_tool_calls_markers() -> None:
    msg = {"role": "assistant", "tool_calls": [{"id": "a", "cache_control": CC}]}
    assert count_cache_breakpoints(None, [msg], None)["total"] == 1
