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


def _booked(*hooks: object) -> tuple[dict, list]:
    for hook in hooks:
        register_turn_hook(hook)
    tags: dict = {}
    ctx = TurnContext(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
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


def test_release_debits_the_deferral_entry_not_a_hooks_own_saving() -> None:
    tags, entries = _booked(_BookDeferral(), _ShrinkSecondUndeferFirst())
    deferral = [e for e in entries if e.get("estimated")]
    measured = [e for e in entries if not e.get("estimated")]
    (deferral_entry,) = deferral
    (hook_entry,) = measured
    shrunk = {**_tool("b"), "description": "short"}
    # The hook keeps its own measured saving (b's shrink, flags ignored)...
    assert hook_entry["tokens"] == _count([_tool("a"), _tool("b")]) - _count([_tool("a"), shrunk])
    # ...and only the deferral entry is debited, down to what is still deferred.
    assert deferral_entry["tokens"] == tags["tool_search_deferred_tokens"]


def test_a_shortened_deferred_schema_is_credited_once() -> None:
    # b's shrink is the hook's saving; the deferral credits b at its final
    # (smaller) size, so those tokens are not counted twice.
    tags, _ = _booked(_BookDeferral(), _ShrinkSecondUndeferFirst())
    shrunk = {**_tool("b"), "description": "short", "defer_loading": True}
    assert tags["tool_search_deferred_tokens"] == _count([shrunk])
    assert tags["tool_search_deferred_tools"] == 1
