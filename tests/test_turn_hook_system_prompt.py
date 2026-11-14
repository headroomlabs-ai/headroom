"""Turn hooks can read and rewrite Anthropic's top-level system prompt.

The system prompt is part of every billed request, but it lived outside
``TurnContext``, so no hook could shrink it, and any change a hook made to it
went unbooked.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from headroom.proxy.savings_attribution import from_tags
from headroom.proxy.tool_schema_savings_policy import TURN_HOOK_SYSTEM_SAVED_TAG
from headroom.proxy.turn_hooks import (
    TurnContext,
    clear_turn_hooks,
    register_turn_hook,
    run_request_hooks,
)

LONG = "Follow the house style guide in every answer. " * 40


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch):
    monkeypatch.setenv("HEADROOM_TOOL_SEARCH", "0")
    clear_turn_hooks()
    yield
    clear_turn_hooks()


def _count(messages: list[dict]) -> int:
    return len(json.dumps(messages, default=str)) // 4


class _ShortenSystem:
    """Copies the blocks and shortens the long text one (cache_control kept)."""

    name = savings_source = "system_trim"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        ctx.system = [
            {**block, "text": "Use the house style."} if block.get("text") == LONG else block
            for block in ctx.system
        ]


class _GrowSystem:
    name = savings_source = "system_grow"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        ctx.system = ctx.system + " Also: " + LONG


class _DropTool:
    name = savings_source = "tool_router"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        ctx.tools = ctx.tools[:1]


def _system_blocks() -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": "You are a helpful assistant."},
        {"type": "text", "text": LONG, "cache_control": {"type": "ephemeral"}},
    ]


def test_a_hook_shrinking_the_system_prompt_is_booked() -> None:
    register_turn_hook(_ShortenSystem())
    system = _system_blocks()
    ctx = TurnContext(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        count_messages=_count,
        system=system,
    )
    run_request_hooks(ctx)
    (entry,) = from_tags(ctx.tags)
    expected = _count([{"role": "system", "content": system}]) - _count(
        [{"role": "system", "content": ctx.system}]
    )
    assert expected > 0
    assert entry["source"] == "system_trim"
    assert entry["tokens"] == expected
    assert entry["details"]["system_tokens_saved"] == expected


def test_a_hook_growing_the_system_prompt_offsets_other_savings() -> None:
    register_turn_hook(_DropTool())
    register_turn_hook(_GrowSystem())
    tools = [{"name": "a"}, {"name": "b", "description": "d" * 4_000}]
    ctx = TurnContext(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
        count_messages=_count,
        count_tools=lambda value: len(json.dumps(value)) // 4 if value else 0,
        system="Be brief.",
    )
    tool_saving = len(json.dumps(tools)) // 4 - len(json.dumps(tools[:1])) // 4
    run_request_hooks(ctx)
    growth = _count([{"role": "system", "content": ctx.system}]) - _count(
        [{"role": "system", "content": "Be brief."}]
    )
    (entry,) = from_tags(ctx.tags)
    assert entry["tokens"] == tool_saving - growth


def test_hooks_that_ignore_the_system_prompt_book_as_before() -> None:
    register_turn_hook(_DropTool())
    ctx = TurnContext(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "a"}, {"name": "b"}],
        count_messages=_count,
        count_tools=lambda value: len(json.dumps(value)) // 4 if value else 0,
        system=_system_blocks(),
    )
    run_request_hooks(ctx)
    (entry,) = from_tags(ctx.tags)
    assert "system_tokens_saved" not in entry["details"]
    without_system = TurnContext(
        provider="anthropic",
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "a"}, {"name": "b"}],
        count_messages=_count,
        count_tools=lambda value: len(json.dumps(value)) // 4 if value else 0,
    )
    run_request_hooks(without_system)
    (baseline,) = from_tags(without_system.tags)
    assert entry["tokens"] == baseline["tokens"]


def _send(monkeypatch, body: dict) -> tuple[Any, dict]:
    """POST ``body`` through /v1/messages; return the outcome and the upstream body."""
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.loopback_guard import require_loopback
    from headroom.proxy.server import ProxyConfig, create_app

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
    with respx.mock:
        route = respx.post("https://api.anthropic.com/v1/messages").mock(
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
                    **body,
                },
                headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
            )
    assert result.status_code == 200, result.text
    return outcomes[-1], json.loads(route.calls.last.request.content)


def _system_count(system: Any) -> int:
    from headroom.tokenizers import get_tokenizer

    count = get_tokenizer("claude-sonnet-4-5").count_messages
    return count([{"role": "system", "content": system}]) if system else 0


def test_anthropic_forwards_and_books_a_hook_rewritten_system_block(monkeypatch) -> None:
    register_turn_hook(_ShortenSystem())
    outcome, upstream = _send(monkeypatch, {"system": _system_blocks()})
    assert upstream["system"] == [
        {"type": "text", "text": "You are a helpful assistant."},
        {"type": "text", "text": "Use the house style.", "cache_control": {"type": "ephemeral"}},
    ]
    expected = _system_count(_system_blocks()) - _system_count(upstream["system"])
    assert outcome.tags[TURN_HOOK_SYSTEM_SAVED_TAG] == expected > 0
    assert "turn_hook:system" in outcome.transforms_applied


class _EditInPlace:
    name = savings_source = "system_trim"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        ctx.system[1]["text"] = "short"


def test_an_in_place_system_edit_is_forwarded_and_booked(monkeypatch) -> None:
    register_turn_hook(_EditInPlace())
    outcome, upstream = _send(monkeypatch, {"system": _system_blocks()})
    assert upstream["system"][1] == {
        "type": "text",
        "text": "short",
        "cache_control": {"type": "ephemeral"},
    }
    expected = _system_count(_system_blocks()) - _system_count(upstream["system"])
    assert outcome.tags[TURN_HOOK_SYSTEM_SAVED_TAG] == expected > 0


class _DropSystem:
    name = savings_source = "system_trim"
    stream_safe = True

    def on_request(self, ctx: TurnContext) -> None:
        ctx.system = None


def test_a_hook_removing_the_system_prompt_drops_it_upstream(monkeypatch) -> None:
    register_turn_hook(_DropSystem())
    outcome, upstream = _send(monkeypatch, {"system": _system_blocks()})
    assert "system" not in upstream
    assert outcome.tags[TURN_HOOK_SYSTEM_SAVED_TAG] == _system_count(_system_blocks())


def test_steering_is_not_part_of_a_hooks_system_delta(monkeypatch) -> None:
    from headroom.proxy.output_verbosity_policy import STEERING_SENTINEL
    from headroom.tokenizers import get_tokenizer

    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "1")
    monkeypatch.setenv("HEADROOM_OUTPUT_HOLDOUT", "0")
    register_turn_hook(_ShortenSystem())
    outcome, upstream = _send(monkeypatch, {"system": _system_blocks()})
    steering = upstream["system"][-1]["text"]
    assert steering.startswith(STEERING_SENTINEL)
    hook_delta = _system_count(_system_blocks()) - _system_count(upstream["system"][:-1])
    steer_tokens = get_tokenizer("claude-sonnet-4-5").count_text(steering)
    assert outcome.tags[TURN_HOOK_SYSTEM_SAVED_TAG] == hook_delta - steer_tokens > 0
