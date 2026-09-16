"""Turn-hook registry + runners (headroom/proxy/turn_hooks.py).

The hook surface is opt-in: with nothing registered the runners must be exact
no-ops (the property the proxy relies on to stay byte-identical for everyone who
has no extension installed). These tests pin that, plus request-mutation,
response-replacement, the re-drive (``call_model``) loop, and the
never-raise guarantee.
"""

from __future__ import annotations

import pytest

from headroom.proxy.turn_hooks import (
    TurnContext,
    clear_turn_hooks,
    register_turn_hook,
    registered_turn_hooks,
    run_request_hooks,
    run_response_hooks,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_turn_hooks()
    yield
    clear_turn_hooks()


def _ctx(**kw):
    base = {"provider": "anthropic", "model": "claude-x", "messages": [], "tools": None}
    base.update(kw)
    return TurnContext(**base)


async def _noop_call_model(_messages):  # pragma: no cover - never invoked in no-op tests
    raise AssertionError("call_model must not be invoked when no hook re-drives")


# --- inert-when-empty (the load-bearing guarantee) ---------------------------


def test_request_runner_inert_when_empty():
    assert registered_turn_hooks() == []
    ctx = _ctx(tools=[{"name": "a"}])
    before = ctx.tools
    run_request_hooks(ctx)  # must not raise, must not touch ctx
    assert ctx.tools is before


@pytest.mark.asyncio
async def test_response_runner_returns_input_unchanged_when_empty():
    resp = {"id": "orig", "content": []}
    out = await run_response_hooks(_ctx(), resp, _noop_call_model)
    assert out is resp  # same object, untouched


# --- on_request mutation -----------------------------------------------------


def test_stream_safe_filter_runs_only_optted_in_hooks_on_stream():
    """On a streamed turn only ``stream_safe`` hooks' on_request runs (fold-only,
    no re-drive); buffered runs all. A hook that may re-drive stays buffered-only."""
    ran: list[str] = []

    class Fold:
        name = "fold"
        stream_safe = True  # opts in — safe on streaming

        def on_request(self, ctx: TurnContext) -> None:
            ran.append("fold")

    class Redrive:  # no stream_safe attr → buffered-only (default)
        name = "redrive"

        def on_request(self, ctx: TurnContext) -> None:
            ran.append("redrive")

    register_turn_hook(Fold())
    register_turn_hook(Redrive())

    ran.clear()
    run_request_hooks(_ctx(), stream_safe_only=True)  # streaming turn
    assert ran == ["fold"]

    ran.clear()
    run_request_hooks(_ctx())  # buffered turn (default)
    assert ran == ["fold", "redrive"]


def test_on_request_may_mutate_ctx():
    class Shrink:
        name = "shrink"

        def on_request(self, ctx: TurnContext) -> None:
            ctx.tools = [t for t in (ctx.tools or []) if t["name"] != "drop_me"]

    register_turn_hook(Shrink())
    ctx = _ctx(tools=[{"name": "keep"}, {"name": "drop_me"}])
    run_request_hooks(ctx)
    assert ctx.tools == [{"name": "keep"}]


def test_request_runner_attributes_savings_with_handler_counters():
    class Shrink:
        name = "internal-hook-name"
        savings_source = "tool_search"

        def on_request(self, ctx: TurnContext) -> None:
            ctx.tools = (ctx.tools or [])[:1]
            ctx.messages[0]["content"] = "short"

    register_turn_hook(Shrink())
    tags = {}
    ctx = _ctx(
        messages=[{"role": "user", "content": "a much longer value"}],
        tools=[{"name": "keep"}, {"name": "drop"}],
        tags=tags,
        count_messages=lambda messages: len(messages[0]["content"]),
        count_tools=lambda tools: len(tools or []),
    )

    run_request_hooks(ctx)

    from headroom.proxy.savings_attribution import from_tags

    assert from_tags(tags) == [
        {
            "source": "tool_search",
            "realized": True,
            "estimated": False,
            "tokens": 15,
            "usd": 0.0,
            "details": {"message_tokens_saved": 14, "tool_tokens_saved": 1},
        }
    ]


# --- on_response replacement + re-drive loop ---------------------------------


@pytest.mark.asyncio
async def test_on_response_can_replace_via_call_model():
    calls: list[list] = []

    async def call_model(messages):
        calls.append(messages)
        return {"id": "resolved", "content": [{"type": "text", "text": "done"}]}

    class ResolveOnce:
        name = "resolve"

        async def on_response(self, ctx, response, call_model):
            if response.get("id") == "needs-work":
                return await call_model(ctx.messages + [{"role": "user", "content": "go"}])
            return None

    register_turn_hook(ResolveOnce())
    out = await run_response_hooks(
        _ctx(messages=[{"role": "user", "content": "hi"}]), {"id": "needs-work"}, call_model
    )
    assert out["id"] == "resolved"
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_on_response_none_leaves_response_unchanged():
    class Observer:
        name = "observe"

        async def on_response(self, ctx, response, call_model):
            return None  # observe only

    register_turn_hook(Observer())
    resp = {"id": "orig"}
    out = await run_response_hooks(_ctx(), resp, _noop_call_model)
    assert out is resp


@pytest.mark.asyncio
async def test_replacements_chain_across_hooks():
    class First:
        name = "first"

        async def on_response(self, ctx, response, call_model):
            return {"id": "after-first", "seen": response["id"]}

    class Second:
        name = "second"

        async def on_response(self, ctx, response, call_model):
            return {"id": "after-second", "seen": response["id"]}

    register_turn_hook(First())
    register_turn_hook(Second())
    out = await run_response_hooks(_ctx(), {"id": "orig"}, _noop_call_model)
    assert out == {"id": "after-second", "seen": "after-first"}  # Second saw First's output


# --- a failing hook must never break the proxy -------------------------------


def test_failing_on_request_is_swallowed():
    class Boom:
        name = "boom"

        def on_request(self, ctx: TurnContext) -> None:
            raise RuntimeError("kaboom")

    register_turn_hook(Boom())
    run_request_hooks(_ctx())  # must not raise


@pytest.mark.asyncio
async def test_failing_on_response_is_skipped_and_original_survives():
    class Boom:
        name = "boom"

        async def on_response(self, ctx, response, call_model):
            raise RuntimeError("kaboom")

    class Good:
        name = "good"

        async def on_response(self, ctx, response, call_model):
            return {"id": "recovered"}

    register_turn_hook(Boom())
    register_turn_hook(Good())
    out = await run_response_hooks(_ctx(), {"id": "orig"}, _noop_call_model)
    assert out == {"id": "recovered"}  # Boom skipped, Good still ran


# --- hooks with only one method defined --------------------------------------


@pytest.mark.asyncio
async def test_hook_without_on_response_is_skipped():
    class OnlyRequest:
        name = "only-request"

        def on_request(self, ctx: TurnContext) -> None:
            pass

    register_turn_hook(OnlyRequest())
    resp = {"id": "orig"}
    out = await run_response_hooks(_ctx(), resp, _noop_call_model)
    assert out is resp
