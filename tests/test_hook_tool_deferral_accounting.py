"""A turn hook's measured tool delta ignores the ``defer_loading`` flag.

Deferral is booked through ``tool_search_deferred_tokens``. Measuring the raw
tool array would see only the flag text appear or disappear, so un-deferring a
tool (tool search's hot-tools hook) was booked as a realized saving although
it makes the request larger.
"""

from __future__ import annotations

import json

import pytest

from headroom.proxy.savings_attribution import from_tags, record_savings
from headroom.proxy.tool_schema_savings_policy import without_deferral_flags
from headroom.proxy.turn_hooks import (
    TurnContext,
    clear_turn_hooks,
    register_turn_hook,
    run_request_hooks,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_turn_hooks()
    yield
    clear_turn_hooks()


def _tool(name: str, size: int = 400) -> dict:
    return {"name": name, "description": "d" * size, "input_schema": {"type": "object"}}


def _count(value: object) -> int:
    # Stand-in tokenizer: one token per 4 characters of the serialised array.
    return len(json.dumps(value, default=str)) // 4 if value else 0


class _UndeferHook:
    """Shaped like headroom-tool-search's HotToolsHook: drops the flag in place."""

    name = savings_source = "tool_search"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        for tool in ctx.tools:
            tool.pop("defer_loading", None)


class _DeferHook:
    """Shaped like NativeDeferralHook: flags tools, books through the tag."""

    name = savings_source = "tool_search"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        ctx.tools = [{**t, "defer_loading": True} for t in ctx.tools]


class _RemoveHook:
    """Shaped like a router: really removes a tool from the array."""

    name = savings_source = "tool_router"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        ctx.tools = ctx.tools[:1]


def _run(hook: object, tools: list) -> list[dict]:
    register_turn_hook(hook)
    ctx = TurnContext(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        count_messages=_count,
        count_tools=_count,
    )
    run_request_hooks(ctx)
    return from_tags(ctx.tags)


def test_undeferring_a_tool_is_not_booked_as_a_saving() -> None:
    tools = [{**_tool("a"), "defer_loading": True}, {**_tool("b"), "defer_loading": True}]
    assert _run(_UndeferHook(), tools) == []


def test_deferring_is_left_to_the_deferral_tag() -> None:
    # Only the flag changes, so the measured delta is zero and the hook's own
    # tool_search_deferred_tokens tag stays the single place deferral is booked.
    assert _run(_DeferHook(), [_tool("a"), _tool("b")]) == []


def test_a_real_removal_is_still_measured() -> None:
    (entry,) = _run(_RemoveHook(), [_tool("a"), _tool("b")])
    assert entry["source"] == "tool_router"
    assert entry["tokens"] == _count([_tool("a"), _tool("b")]) - _count([_tool("a")])


def test_without_deferral_flags_never_mutates_and_passes_non_lists() -> None:
    tools = [{**_tool("a"), "defer_loading": True}, "not-a-dict"]
    out = without_deferral_flags(tools)
    assert "defer_loading" in tools[0]
    assert out[0] == _tool("a")
    assert out[1] == "not-a-dict"
    assert without_deferral_flags(None) is None


# ── deferral credit is capped at what is still deferred after hooks ──────────


class _BookDeferral:
    """Shaped like the native deferral hook: flags tools and books them."""

    name = savings_source = "tool_search"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        ctx.tools = [{**t, "defer_loading": True} for t in ctx.tools]
        ctx.tags["tool_search_deferred_tokens"] = _count(ctx.tools)
        ctx.tags["tool_search_deferred_tools"] = len(ctx.tools)
        record_savings(ctx.tags, "tool_search", tokens=_count(ctx.tools), estimated=True)


class _UndeferFirstGrowSecond:
    name = savings_source = "other"
    stream_safe = True

    def __init__(self, grow: int = 0) -> None:
        self.grow = grow

    def on_request(self, ctx: TurnContext) -> None:
        ctx.tools[0].pop("defer_loading", None)
        ctx.tools[1]["description"] += "g" * self.grow


def _booked(*hooks: object, messages: list[dict] | None = None) -> tuple[dict, list]:
    for hook in hooks:
        register_turn_hook(hook)
    tags: dict = {}
    ctx = TurnContext(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=messages or [{"role": "user", "content": "hi"}],
        tools=[_tool("a"), _tool("b")],
        count_messages=_count,
        count_tools=_count,
        tags=tags,
    )
    run_request_hooks(ctx)
    return tags, [e for e in from_tags(tags) if e["source"] == "tool_search"]


def test_undeferring_a_booked_tool_lowers_the_headline_and_the_ledger() -> None:
    tags, (entry,) = _booked(_BookDeferral(), _UndeferFirstGrowSecond())
    expected = _count([{**_tool("b"), "defer_loading": True}])
    assert tags["tool_search_deferred_tokens"] == expected
    assert tags["tool_search_deferred_tools"] == 1
    assert entry["tokens"] == expected  # /stats by-source agrees with the headline


def test_growing_another_deferred_schema_cannot_keep_the_credit() -> None:
    # b grows by far more than a's share: the final deferred set outweighs the
    # booking, yet a is billed now, so its share must still go.
    tags, (entry,) = _booked(_BookDeferral(), _UndeferFirstGrowSecond(grow=4_000))
    expected = _count([{**_tool("b"), "defer_loading": True}])
    assert tags["tool_search_deferred_tokens"] == expected
    assert entry["tokens"] == expected


def test_reconcile_never_raises_the_credit_or_touches_unbooked_turns() -> None:
    tags, (entry,) = _booked(_BookDeferral())
    both = _count([{**_tool("a"), "defer_loading": True}, {**_tool("b"), "defer_loading": True}])
    assert tags["tool_search_deferred_tokens"] == both == entry["tokens"]
    clear_turn_hooks()
    untouched, _ = _booked(_UndeferFirstGrowSecond())
    assert "tool_search_deferred_tokens" not in untouched


def test_anthropic_headline_drops_the_credit_of_a_tool_a_hook_undefers(monkeypatch) -> None:
    """End to end: the built-in deferral books 14 MCP tools, then a hook
    (shaped like tool search's hot tools) un-defers one. The headline credit
    must cover only the 13 still deferred."""
    import httpx
    import respx

    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.loopback_guard import require_loopback
    from headroom.proxy.server import ProxyConfig, create_app
    from headroom.tokenizers import get_tokenizer

    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", "1")

    class _UndeferOne:
        name = savings_source = "tool_search"
        stream_safe = True

        def on_request(self, ctx: TurnContext) -> None:
            for tool in ctx.tools:
                if isinstance(tool, dict) and tool.get("name") == "mcp__srv__tool_0":
                    tool.pop("defer_loading", None)

    register_turn_hook(_UndeferOne())
    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
        )
    )
    app.dependency_overrides[require_loopback] = lambda: None
    outcomes: list = []

    async def _spy(_self, outcome, *a, **kw):  # noqa: ANN001, ANN002, ANN003, ANN202
        outcomes.append(outcome)

    monkeypatch.setattr(type(app.state.proxy), "_record_request_outcome", _spy, raising=True)
    tools = [
        {
            "name": f"mcp__srv__tool_{i}",
            "description": "Look things up. " * 30,
            "input_schema": {"type": "object"},
        }
        for i in range(14)
    ]
    with respx.mock:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "a",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5",
                    "content": [{"type": "text", "text": "ok"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 100, "output_tokens": 1},
                },
            )
        )
        with TestClient(app) as client:
            result = client.post(
                "/v1/messages",
                json={
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hi"}],
                    "tools": tools,
                },
                headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
            )
    assert result.status_code == 200
    tags = outcomes[-1].tags
    assert tags.get("tool_search_mode") == "headroom"
    still = [{**t, "defer_loading": True} for t in tools[1:]]
    tok = get_tokenizer("claude-sonnet-4-5")
    assert tags["tool_search_deferred_tools"] == 13
    assert tags["tool_search_deferred_tokens"] == tok.count_text(json.dumps(still, default=str))
    (entry,) = [e for e in from_tags(tags) if e["source"] == "tool_search"]
    assert entry["tokens"] == tags["tool_search_deferred_tokens"]


class _ShrinkSecondUndeferFirst:
    """A tool_search-sourced hook: shortens b (still deferred), un-defers a."""

    name = savings_source = "tool_search"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        ctx.tools[0].pop("defer_loading", None)
        ctx.tools[1]["description"] = "short"


def test_shortening_an_unloaded_deferred_tool_keeps_its_booked_credit() -> None:
    # b is deferred and unloaded on both sides of the hook: it is not billed, so
    # the hook books nothing for shortening it and the deferral keeps crediting
    # b's booked schema -- the tokens a client without the proxy would pay.
    tags, entries = _booked(_BookDeferral(), _ShrinkSecondUndeferFirst())
    (deferral_entry,) = entries
    assert deferral_entry.get("estimated")
    booked_b = {**_tool("b"), "defer_loading": True}
    assert tags["tool_search_deferred_tokens"] == _count([booked_b])
    assert tags["tool_search_deferred_tools"] == 1
    assert deferral_entry["tokens"] == tags["tool_search_deferred_tokens"]


def test_a_shortened_loaded_deferred_schema_is_credited_once() -> None:
    # Once a tool_reference loads b it is billed: the shrink is the hook's own
    # saving and the deferral credits b at its final (smaller) size, so those
    # tokens are not counted twice.
    tags, entries = _booked(
        _BookDeferral(), _ShrinkSecondUndeferFirst(), messages=_history_loading("b")
    )
    measured = [e for e in entries if not e.get("estimated")]
    (hook_entry,) = measured
    shrunk = {**_tool("b"), "description": "short"}
    assert hook_entry["tokens"] == _count([_tool("a"), _tool("b")]) - _count([_tool("a"), shrunk])
    assert tags["tool_search_deferred_tokens"] == _count([{**shrunk, "defer_loading": True}])


# ── a deferred tool is billed only once a tool_reference loads it ────────────


def _history_loading(name: str, *, server_side: bool = False) -> list[dict]:
    """A transcript in which a tool search returned ``name``."""
    reference = {"type": "tool_reference", "tool_name": name}
    if server_side:
        result = {
            "type": "tool_search_tool_result",
            "tool_use_id": "srv_1",
            "content": {"type": "tool_search_tool_search_result", "tool_references": [reference]},
        }
        return [
            {"role": "user", "content": "find it"},
            {"role": "assistant", "content": [result]},
            {"role": "user", "content": "go on"},
        ]
    # Client-side search (VS Code Copilot Chat): a plain tool_use/tool_result.
    return [
        {"role": "user", "content": "find it"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "tool_search", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": [reference]}],
        },
    ]


class _CondenseAll:
    """Shaped like a description-trim hook: shortens every tool in place."""

    name = savings_source = "tool_desc_trim"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        for tool in ctx.tools:
            if "type" not in tool:  # a server tool has no description
                tool["description"] = "short"


def _client_deferred_run(messages: list[dict]) -> list[dict]:
    """Copilot's shape: one resident tool, two the client itself deferred."""
    tools = [
        {"name": "tool_search_tool_regex", "type": "tool_search_tool_regex_20251119"},
        _tool("resident"),
        {**_tool("loaded"), "defer_loading": True},
        {**_tool("unloaded"), "defer_loading": True},
    ]
    register_turn_hook(_CondenseAll())
    ctx = TurnContext(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=messages,
        tools=tools,
        count_messages=_count,
        count_tools=_count,
    )
    run_request_hooks(ctx)
    return from_tags(ctx.tags)


def _condensed_delta(*names: str) -> int:
    full = [_tool(n) for n in names]
    short = [{**_tool(n), "description": "short"} for n in names]
    return _count(full) - _count(short)


def test_condensing_an_unloaded_deferred_tool_books_nothing() -> None:
    (entry,) = _client_deferred_run([{"role": "user", "content": "hi"}])
    # Only the resident tool's trim is on the bill.
    assert entry["tokens"] == _condensed_delta("resident")


@pytest.mark.parametrize("server_side", [False, True])
def test_condensing_a_loaded_deferred_tool_books_its_delta(server_side: bool) -> None:
    (entry,) = _client_deferred_run(_history_loading("loaded", server_side=server_side))
    # The stand-in tokenizer floors, so tool order moves the total by a token.
    assert entry["tokens"] == pytest.approx(_condensed_delta("resident", "loaded"), abs=1)


def test_deferred_tools_still_count_on_other_providers() -> None:
    # OpenAI tool search loads tools outside an Anthropic history, which this
    # accounting cannot read: every deferred tool is measured as before.
    register_turn_hook(_CondenseAll())
    tools = [{**_tool("a"), "defer_loading": True}]
    ctx = TurnContext(
        provider="openai",
        model="gpt-5",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        count_messages=_count,
        count_tools=_count,
    )
    run_request_hooks(ctx)
    (entry,) = from_tags(ctx.tags)
    assert entry["tokens"] == _condensed_delta("a")


class _DropDeferred:
    name = savings_source = "tool_router"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        ctx.tools = [t for t in ctx.tools if not t.get("defer_loading")]


def test_removing_an_unloaded_deferred_tool_books_nothing() -> None:
    register_turn_hook(_DropDeferred())
    ctx = TurnContext(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        tools=[_tool("a"), {**_tool("b"), "defer_loading": True}],
        count_messages=_count,
        count_tools=_count,
    )
    run_request_hooks(ctx)
    assert from_tags(ctx.tags) == []


class _ShrinkBAndDropItsReference:
    """Trims b while b is loaded, then folds away the message that loaded it."""

    name = savings_source = "fold"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        ctx.tools[1]["description"] = "short"
        ctx.messages = [{"role": "user", "content": "folded"}]


def test_a_tool_billed_when_booked_is_not_credited_twice_after_a_fold() -> None:
    # b was loaded (billed) when booked, so the hook's trim is a real saving;
    # the fold then unloads b, but its credit must not go back to the full
    # booked schema on top of the trim.
    tags, entries = _booked(
        _BookDeferral(), _ShrinkBAndDropItsReference(), messages=_history_loading("b")
    )
    shrunk = {**_tool("b"), "description": "short", "defer_loading": True}
    assert tags["tool_search_deferred_tokens"] == _count(
        [{**_tool("a"), "defer_loading": True}, shrunk]
    )


def test_the_gateway_turn_books_only_loaded_deferred_tools() -> None:
    from headroom.proxy import gateway_turn as gt
    from headroom.proxy.gateway_turn import GatewayCapabilities

    register_turn_hook(_CondenseAll())
    tools = [
        _tool("resident"),
        {**_tool("loaded"), "defer_loading": True},
        {**_tool("unloaded"), "defer_loading": True},
    ]
    tags: dict = {}
    tr = gt.RequestTransformer(
        provider="anthropic",
        model_name="claude-sonnet-4-5",
        tools=tools,
        config=None,
        tags=tags,
        caps=GatewayCapabilities(can_redrive=False),
    )
    tr.run(_history_loading("loaded"))
    from headroom.tokenizers import get_tokenizer

    tok = get_tokenizer("claude-sonnet-4-5")

    def count(value: list) -> int:
        return tok.count_text(json.dumps(value, default=str))

    full = [_tool("resident"), _tool("loaded")]
    short = [{**t, "description": "short"} for t in full]
    assert tags["turn_hook_tools_saved_tokens"] == count(full) - count(short)
