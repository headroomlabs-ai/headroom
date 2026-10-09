"""A turn hook that folds messages but ADDS a tool saved the difference.

headroom-skill-search rewrites the skill list in the messages and appends a
``search_skills`` tool. The runner used to clamp the message and tool halves
separately, so the added tool never reduced the booked saving, and the
handlers left it out of the forwarded count, so the headline overstated too.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from headroom.proxy.savings_attribution import from_tags
from headroom.proxy.turn_hooks import (
    TurnContext,
    clear_turn_hooks,
    register_turn_hook,
    run_request_hooks,
)

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.loopback_guard import require_loopback  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402
from headroom.tokenizers import get_tokenizer  # noqa: E402

SEARCH_TOOL = {
    "name": "search_skills",
    "description": "Search the skill catalogue by keyword. " * 10,
    "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
}


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_turn_hooks()
    yield
    clear_turn_hooks()


class _FoldAndAddTool:
    """Shaped like SkillSearchHook: shrink a message, append a search tool."""

    name = savings_source = "skill_search"
    stream_safe = True

    def __init__(self, folded_text: str) -> None:
        self.folded_text = folded_text

    def on_request(self, ctx: TurnContext) -> None:
        ctx.messages = [{"role": "user", "content": self.folded_text}]
        ctx.tools = [*(ctx.tools or []), SEARCH_TOOL]


def _count(value: Any) -> int:
    return len(json.dumps(value, default=str)) // 4 if value else 0


def _run(hook: Any, text: str) -> TurnContext:
    register_turn_hook(hook)
    ctx = TurnContext(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": text}],
        tools=[],
        count_messages=_count,
        count_tools=_count,
    )
    run_request_hooks(ctx)
    return ctx


def test_ledger_entry_is_net_of_the_added_tool() -> None:
    long_text = "skill line\n" * 400
    ctx = _run(_FoldAndAddTool("short"), long_text)
    (entry,) = from_tags(ctx.tags)
    folded = _count([{"role": "user", "content": long_text}]) - _count(
        [{"role": "user", "content": "short"}]
    )
    added = _count([SEARCH_TOOL])
    assert entry["tokens"] == folded - added
    assert entry["details"]["tool_tokens_saved"] == -added


def test_no_entry_when_the_added_tool_outweighs_the_fold() -> None:
    ctx = _run(_FoldAndAddTool("short"), "a bit longer than short")
    assert from_tags(ctx.tags) == []


@respx.mock
def test_anthropic_headline_nets_a_tool_the_hook_added(monkeypatch) -> None:
    long_text = "skill line number one\n" * 300
    register_turn_hook(_FoldAndAddTool("short"))
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
    outcomes: list[Any] = []

    async def _spy(_self, outcome, *a, **kw):  # noqa: ANN001, ANN002, ANN003, ANN202
        outcomes.append(outcome)

    monkeypatch.setattr(type(app.state.proxy), "_record_request_outcome", _spy, raising=True)
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
                "messages": [{"role": "user", "content": long_text}],
            },
            headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
        )
    assert result.status_code == 200
    tok = get_tokenizer("claude-sonnet-4-5")
    folded = tok.count_messages([{"role": "user", "content": long_text}]) - tok.count_messages(
        [{"role": "user", "content": "short"}]
    )
    added = tok.count_text(json.dumps([SEARCH_TOOL], default=str))
    assert folded > added > 0
    assert outcomes[-1].tokens_saved == folded - added


class _FoldAndAddFunction(_FoldAndAddTool):
    def on_request(self, ctx: TurnContext) -> None:
        ctx.messages = [{"role": "user", "content": self.folded_text}]
        ctx.tools = [*(ctx.tools or []), {"type": "function", "function": SEARCH_TOOL}]


@respx.mock
def test_openai_chat_headline_nets_a_tool_the_hook_added(monkeypatch) -> None:
    long_text = "skill line number one\n" * 300
    register_turn_hook(_FoldAndAddFunction("short"))
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
    outcomes: list[Any] = []

    async def _spy(_self, outcome, *a, **kw):  # noqa: ANN001, ANN002, ANN003, ANN202
        outcomes.append(outcome)

    monkeypatch.setattr(type(app.state.proxy), "_record_request_outcome", _spy, raising=True)
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "a",
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 1},
            },
        )
    )
    with TestClient(app) as client:
        result = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": long_text}]},
            headers={"authorization": "Bearer sk-test"},
        )
    assert result.status_code == 200
    tok = get_tokenizer("gpt-4o")
    folded = tok.count_messages([{"role": "user", "content": long_text}]) - tok.count_messages(
        [{"role": "user", "content": "short"}]
    )
    added = tok.count_text(json.dumps([{"type": "function", "function": SEARCH_TOOL}], default=str))
    assert folded > added > 0
    assert outcomes[-1].tokens_saved == folded - added


# ── a later hook's loss offsets an earlier hook's gain ───────────────────────


class _Fold:
    name = savings_source = "fold"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        ctx.messages = [{"role": "user", "content": "x"}]


class _AddBigTool:
    name = savings_source = "adder"
    stream_safe = True

    def __init__(self, size: int) -> None:
        self.size = size

    def on_request(self, ctx: TurnContext) -> None:
        ctx.tools = [*(ctx.tools or []), {"name": "t", "description": "d" * self.size}]


def _sequence(*hooks: object) -> tuple[list[dict], int]:
    for hook in hooks:
        register_turn_hook(hook)
    count = lambda v: len(json.dumps(v, default=str)) // 4 if v else 0  # noqa: E731
    messages = [{"role": "user", "content": "m" * 800}]
    ctx = TurnContext(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=messages,
        tools=[],
        count_messages=count,
        count_tools=count,
    )
    before = count(messages)
    run_request_hooks(ctx)
    net = before - count(ctx.messages) - count(ctx.tools)
    return from_tags(ctx.tags), net


def test_a_later_hook_that_grows_more_than_was_saved_cancels_the_credit() -> None:
    entries, net = _sequence(_Fold(), _AddBigTool(2_000))
    assert net < 0
    assert entries == []


def test_a_later_partial_loss_caps_the_earlier_credit_at_the_net() -> None:
    entries, net = _sequence(_Fold(), _AddBigTool(200))
    assert net > 0
    assert [(e["source"], e["tokens"]) for e in entries] == [("fold", net)]
    (entry,) = entries
    details = entry["details"]
    # The breakdown sums to the credited amount; the raw deltas are kept.
    assert details["message_tokens_saved"] + details["tool_tokens_saved"] == net
    assert details["uncapped_message_tokens_saved"] > net
    assert details["uncapped_tool_tokens_saved"] == 0


# ── message growth offsets tool removal in the handler headline ──────────────

BIG_TOOL = {
    "name": "big_tool",
    "description": "Does many things in many ways. " * 40,
    "input_schema": {"type": "object"},
}


class _AddTextRemoveTools:
    """The opposite signed case: append message text, drop every tool."""

    name = savings_source = "rewriter"
    stream_safe = True

    def __init__(self, added_text: str) -> None:
        self.added_text = added_text

    def on_request(self, ctx: TurnContext) -> None:
        ctx.messages = [*ctx.messages, {"role": "user", "content": self.added_text}]
        ctx.tools = []


def _app_with_spy(monkeypatch) -> tuple[Any, list[Any]]:
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
    outcomes: list[Any] = []

    async def _spy(_self, outcome, *a, **kw):  # noqa: ANN001, ANN002, ANN003, ANN202
        outcomes.append(outcome)

    monkeypatch.setattr(type(app.state.proxy), "_record_request_outcome", _spy, raising=True)
    return app, outcomes


def _expected(model: str, before: list, added_text: str, tools: list) -> int:
    tok = get_tokenizer(model)
    grown = tok.count_messages([*before, {"role": "user", "content": added_text}])
    message_growth = grown - tok.count_messages(before)
    removed = tok.count_text(json.dumps(tools, default=str))
    return removed - message_growth


@pytest.mark.parametrize("repeat", [5, 400], ids=["net-positive", "net-negative"])
@respx.mock
def test_anthropic_headline_nets_added_text_against_removed_tools(monkeypatch, repeat) -> None:
    from headroom.proxy.tool_schema_savings_policy import headline_tokens_saved

    added = "extra steering text. " * repeat
    register_turn_hook(_AddTextRemoveTools(added))
    app, outcomes = _app_with_spy(monkeypatch)
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
    before = [{"role": "user", "content": "hi"}]
    with TestClient(app) as client:
        result = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 16,
                "messages": before,
                "tools": [BIG_TOOL],
            },
            headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
        )
    assert result.status_code == 200
    net = _expected("claude-sonnet-4-5", before, added, [BIG_TOOL])
    assert (net > 0) == (repeat == 5)
    outcome = outcomes[-1]
    assert headline_tokens_saved(outcome.tokens_saved, outcome.tags) == max(0, net)


@pytest.mark.parametrize("repeat", [5, 400], ids=["net-positive", "net-negative"])
@respx.mock
def test_openai_chat_headline_nets_added_text_against_removed_tools(monkeypatch, repeat) -> None:
    from headroom.proxy.tool_schema_savings_policy import headline_tokens_saved

    added = "extra steering text. " * repeat
    register_turn_hook(_AddTextRemoveTools(added))
    app, outcomes = _app_with_spy(monkeypatch)
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "a",
                "choices": [
                    {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 1},
            },
        )
    )
    tools = [{"type": "function", "function": BIG_TOOL}]
    before = [{"role": "user", "content": "hi"}]
    with TestClient(app) as client:
        result = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": before, "tools": tools},
            headers={"authorization": "Bearer sk-test"},
        )
    assert result.status_code == 200
    net = _expected("gpt-4o", before, added, tools)
    assert (net > 0) == (repeat == 5)
    outcome = outcomes[-1]
    assert headline_tokens_saved(outcome.tokens_saved, outcome.tags) == max(0, net)
