"""/v1/compress nets tool definitions a turn hook adds against its saving.

A hook that folds messages but adds a tool (skill search's search tool) sends
those tool tokens too; ``tokens_after`` must include them, as on the chat path.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from headroom.proxy.savings_attribution import from_tags
from headroom.proxy.turn_hooks import register_turn_hook
from tests.gateway.conftest import compress

_LONG = "line of tool output that a hook can fold away entirely " * 60
_ADDED_TOOL = {
    "name": "search_skills",
    "description": "Find a skill by what it does. " * 40,
    "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}},
}


class _FoldAndAddTool:
    name = savings_source = "skill_search"
    stream_safe = True

    def __init__(self, add_tool: bool) -> None:
        self.add_tool = add_tool

    def on_request(self, ctx: Any) -> None:
        ctx.messages = [{**m, "content": "folded"} for m in ctx.messages]
        if self.add_tool:
            ctx.tools = [*(ctx.tools or []), dict(_ADDED_TOOL)]


def _run(make_headroom_client: Any, *, add_tool: bool) -> dict[str, Any]:
    register_turn_hook(_FoldAndAddTool(add_tool))
    client = make_headroom_client(optimize=False)
    body = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": _LONG}],
        "tools": [{"name": "noop", "description": "x", "input_schema": {"type": "object"}}],
        "gateway": {"can_redrive": False, "can_relay_response": False},
    }
    resp = compress(client, body)
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_tools_a_hook_adds_count_as_sent(make_headroom_client) -> None:
    from headroom.proxy.turn_hooks import clear_turn_hooks

    plain = _run(make_headroom_client, add_tool=False)
    clear_turn_hooks()
    grown = _run(make_headroom_client, add_tool=True)
    assert plain["tokens_saved"] > 0
    added = grown["tokens_after"] - plain["tokens_after"]
    assert added > 100  # the search tool's definition
    assert grown["tokens_saved"] == plain["tokens_saved"] - added
    assert grown["tokens_before"] == plain["tokens_before"]


class _AddTextRemoveTools:
    name = savings_source = "rewriter"
    stream_safe = True

    def __init__(self, repeat: int) -> None:
        self.repeat = repeat

    def on_request(self, ctx: Any) -> None:
        ctx.messages = [*ctx.messages, {"role": "user", "content": "steer. " * self.repeat}]
        ctx.tools = []


@pytest.mark.parametrize("session", [False, True], ids=["stateless", "session"])
@pytest.mark.parametrize("repeat", [5, 2_000], ids=["net-positive", "net-negative"])
def test_message_growth_offsets_tools_a_hook_removed(
    make_headroom_client, outcome_spy, repeat, session
) -> None:
    """The opposite signed case on /v1/compress: a hook appends message text
    and drops the tools. The headline is removed tools minus added text,
    clamped once, not the full tool removal."""
    from headroom.proxy.tool_schema_savings_policy import headline_tokens_saved
    from headroom.tokenizers import get_tokenizer

    register_turn_hook(_AddTextRemoveTools(repeat))
    client = make_headroom_client(optimize=False)
    outcomes = outcome_spy(client)
    before = [{"role": "user", "content": "hi"}]
    body = {
        "model": "claude-sonnet-4-5",
        "messages": before,
        "tools": [dict(_ADDED_TOOL)],
        "gateway": {"can_redrive": False, "can_relay_response": False},
    }
    if session:
        # Session turns recount the final (post-hook) messages: the added
        # text must not be counted a second time on top of that.
        body["config"] = {"session_id": f"growth-{repeat}"}
    resp = compress(client, body)
    assert resp.status_code == 200, resp.text
    tok = get_tokenizer("claude-sonnet-4-5")
    growth = tok.count_messages(
        [*before, {"role": "user", "content": "steer. " * repeat}]
    ) - tok.count_messages(before)
    removed = tok.count_text(json.dumps([_ADDED_TOOL], default=str))
    net = removed - growth
    assert (net > 0) == (repeat == 5)
    outcome = outcomes[-1]
    headline = headline_tokens_saved(outcome.tokens_saved, outcome.tags)
    # The gateway compacts tool schemas before hooks run, so compare with the
    # hook's own measured deltas (the runner's ledger), not the raw schema.
    entries = from_tags(outcome.tags)
    if net > 0:
        (entry,) = entries
        hook_net = entry["details"]["tool_tokens_saved"] + entry["details"]["message_tokens_saved"]
        assert entry["details"]["message_tokens_saved"] < 0
        assert headline == hook_net == entry["tokens"]
    else:
        assert entries == []
        assert headline == 0
