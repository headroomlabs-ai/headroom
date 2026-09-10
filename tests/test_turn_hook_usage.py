"""A turn hook's re-drives are billed calls and must reach token accounting.

A hook that resolves an injected tool call re-drives the model. Both OpenAI
handlers read usage from exactly ONE response — the original, or whichever the
hook returned in its place, because the handler swaps `response` for it. Every
other upstream call on that turn is spend nothing else records.

Getting that wrong is not a rounding error for a token-saving feature: it lets
the feature hide its own overhead behind the saving it claims. The first version
of this recorded only the re-drives and added them unconditionally, so a single
re-drive billed `B + B` and dropped the original `A` entirely. The handler tests
at the bottom are what catch that class of mistake; the unit tests above them
cannot, because the bug lives in how the accumulator composes with the response
swap rather than in the accumulator itself.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from headroom.proxy.handlers.anthropic import _AnthropicTurnHookUsage
from headroom.proxy.handlers.openai import (
    CHAT_USAGE_KEYS,
    RESPONSES_USAGE_KEYS,
    TurnHookUsage,
)

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.loopback_guard import require_loopback  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402
from headroom.proxy.turn_hooks import clear_turn_hooks, register_turn_hook  # noqa: E402

# --- unit: the accumulator -----------------------------------------------


def _chat(prompt: int, completion: int, cached: int = 0) -> dict[str, Any]:
    return {
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        }
    }


def test_no_redrive_adds_nothing() -> None:
    """The common path: one upstream call, which the usage block reads itself."""
    u = TurnHookUsage()
    original = _chat(100, 10)
    u.record(original, **CHAT_USAGE_KEYS)
    u.settle(original)
    assert u.extra_calls == 0
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens) == (0, 0, 0)


def test_one_redrive_leaves_the_original_to_add() -> None:
    """A + B billed; the block will read B; so A is the delta."""
    u = TurnHookUsage()
    a, b = _chat(100, 10, 60), _chat(150, 20, 90)
    u.record(a, **CHAT_USAGE_KEYS)
    u.record(b, **CHAT_USAGE_KEYS)
    u.settle(b)
    assert u.extra_calls == 1
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens) == (100, 10, 60)


def test_two_redrives_leave_the_original_and_the_middle() -> None:
    u = TurnHookUsage()
    a, b, c = _chat(100, 10), _chat(150, 20), _chat(200, 30)
    for r in (a, b, c):
        u.record(r, **CHAT_USAGE_KEYS)
    u.settle(c)
    assert u.extra_calls == 2
    assert (u.input_tokens, u.output_tokens) == (250, 30)


def test_hook_that_keeps_the_original_still_pays_for_the_redrive() -> None:
    """Re-drove, then returned the original anyway. B was still billed."""
    u = TurnHookUsage()
    a, b = _chat(100, 10), _chat(150, 20)
    u.record(a, **CHAT_USAGE_KEYS)
    u.record(b, **CHAT_USAGE_KEYS)
    u.settle(a)
    assert u.extra_calls == 1
    assert (u.input_tokens, u.output_tokens) == (150, 20)


def test_synthesised_response_matches_nothing_and_over_counts() -> None:
    """Nothing is subtracted when the hook invents a response. Over-counting is
    the safe direction for a bill; under-counting is the bug this file exists
    for."""
    u = TurnHookUsage()
    a, b = _chat(100, 10), _chat(150, 20)
    u.record(a, **CHAT_USAGE_KEYS)
    u.record(b, **CHAT_USAGE_KEYS)
    u.settle({"usage": {"prompt_tokens": 999}})
    assert u.extra_calls == 2
    assert u.input_tokens == 250


def test_responses_shape_uses_its_own_key_names() -> None:
    u = TurnHookUsage()
    a = {
        "usage": {
            "input_tokens": 400,
            "output_tokens": 40,
            "input_tokens_details": {"cached_tokens": 300},
        }
    }
    b = {"usage": {"input_tokens": 500, "output_tokens": 50}}
    u.record(a, **RESPONSES_USAGE_KEYS)
    u.record(b, **RESPONSES_USAGE_KEYS)
    u.settle(b)
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens) == (400, 40, 300)

    # Chat keys must not read a Responses payload: a silent 0 looks exactly like
    # "the hook cost nothing".
    v = TurnHookUsage()
    v.record(a, **CHAT_USAGE_KEYS)
    v.record(b, **CHAT_USAGE_KEYS)
    v.settle(b)
    assert v.input_tokens == 0
    assert v.extra_calls == 1, "the call still happened even if its shape was unreadable"


def test_never_raises_on_a_shape_it_does_not_recognise() -> None:
    """A hook must not be able to 500 a request by returning something odd."""
    u = TurnHookUsage()
    for payload in (
        None,
        {},
        [],
        "not a dict",
        {"usage": None},
        {"usage": "nope"},
        {"usage": {"prompt_tokens": None, "completion_tokens": "x"}},
        {"usage": {"prompt_tokens": -5, "prompt_tokens_details": "nope"}},
    ):
        u.record(payload, **CHAT_USAGE_KEYS)
    u.settle(object())
    assert u.extra_calls == 8
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens) == (0, 0, 0)


def test_anthropic_accumulator_includes_disjoint_cache_buckets() -> None:
    usage = _AnthropicTurnHookUsage()
    first = {
        "usage": {
            "input_tokens": 100,
            "output_tokens": 10,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 25,
            "cache_creation": {
                "ephemeral_5m_input_tokens": 20,
                "ephemeral_1h_input_tokens": 5,
            },
        }
    }
    final = {
        "usage": {
            "input_tokens": 150,
            "output_tokens": 20,
            "cache_read_input_tokens": 70,
            "cache_creation_input_tokens": 30,
        }
    }
    usage.record(first)
    usage.record(final)
    usage.settle(final)
    assert usage.input_tokens == 175
    assert usage.output_tokens == 10
    assert usage.cache_read_tokens == 50
    assert usage.cache_write_tokens == 25
    assert usage.cache_write_5m_tokens == 20
    assert usage.cache_write_1h_tokens == 5


# --- handler level: what the unit tests above structurally cannot see -----


class _RedriveOnce:
    """Minimal hook: re-drive the model exactly once, return the new response."""

    name = "test_redrive"
    stream_safe = False

    def __init__(self) -> None:
        self.calls = 0

    def on_request(self, ctx: Any) -> None:  # pragma: no cover - nothing to do
        return None

    async def on_response(self, ctx: Any, response: Any, call_model: Any) -> Any:
        if self.calls:
            return None
        self.calls += 1
        return await call_model(ctx.messages)


@pytest.fixture
def _no_hooks():
    clear_turn_hooks()
    yield
    clear_turn_hooks()


def _app_and_outcomes(monkeypatch):
    """App with a spy on the outcome record, which is where the billed token
    counts land (`provider_input_tokens` / `output_tokens`)."""
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
    proxy = app.state.proxy

    # Patched on the type, so the bound-call self arrives as the first argument.
    async def _spy(_self, outcome, *a, **kw):
        outcomes.append(outcome)

    monkeypatch.setattr(type(proxy), "_record_request_outcome", _spy, raising=True)
    return app, outcomes


@respx.mock
def test_chat_bills_the_original_plus_the_redrive(monkeypatch, _no_hooks) -> None:
    """A=100/10, B=150/20 -> 250 in / 30 out.

    The bug this pins reported 300/40 (B twice, A dropped).
    """
    register_turn_hook(_RedriveOnce())
    app, outcomes = _app_and_outcomes(monkeypatch)

    bodies = [
        {
            "id": "a",
            "choices": [
                {"message": {"role": "assistant", "content": "A"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10},
        },
        {
            "id": "b",
            "choices": [
                {"message": {"role": "assistant", "content": "B"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 150, "completion_tokens": 20},
        },
    ]
    sent = iter(bodies)
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=lambda request: httpx.Response(200, json=next(sent))
    )

    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            headers={"authorization": "Bearer sk-test"},
        )
    assert r.status_code == 200
    assert json.loads(r.content)["id"] == "b", "the hook's response is what the client gets"
    assert outcomes, "an outcome must be recorded"
    o = outcomes[-1]
    assert o.provider_input_tokens == 250, f"want A+B=250, got {o.provider_input_tokens}"
    assert o.output_tokens == 30, f"want A+B=30, got {o.output_tokens}"


@respx.mock
def test_responses_bills_the_original_plus_the_redrive(monkeypatch, _no_hooks) -> None:
    """Same arithmetic on /v1/responses, whose usage keys differ."""
    register_turn_hook(_RedriveOnce())
    app, outcomes = _app_and_outcomes(monkeypatch)

    bodies = [
        {
            "id": "a",
            "output": [{"type": "message", "role": "assistant", "content": []}],
            "usage": {"input_tokens": 400, "output_tokens": 40},
        },
        {
            "id": "b",
            "output": [{"type": "message", "role": "assistant", "content": []}],
            "usage": {"input_tokens": 500, "output_tokens": 50},
        },
    ]
    sent = iter(bodies)
    respx.post("https://api.openai.com/v1/responses").mock(
        side_effect=lambda request: httpx.Response(200, json=next(sent))
    )

    with TestClient(app) as client:
        r = client.post(
            "/v1/responses",
            json={
                "model": "gpt-4o",
                "input": [{"type": "message", "role": "user", "content": []}],
                "stream": False,
            },
            headers={"authorization": "Bearer sk-test"},
        )
    assert r.status_code == 200
    assert outcomes, "an outcome must be recorded"
    o = outcomes[-1]
    assert o.provider_input_tokens == 900, f"want A+B=900, got {o.provider_input_tokens}"
    assert o.output_tokens == 90, f"want A+B=90, got {o.output_tokens}"


@respx.mock
def test_no_hook_registered_bills_exactly_the_one_call(monkeypatch, _no_hooks) -> None:
    """The regression guard in the other direction: with no hook, accounting must
    be untouched — this whole mechanism has to be inert on a stock proxy."""
    app, outcomes = _app_and_outcomes(monkeypatch)
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "a",
                "choices": [
                    {"message": {"role": "assistant", "content": "A"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )
    )

    with TestClient(app) as client:
        r = client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            headers={"authorization": "Bearer sk-test"},
        )
    assert r.status_code == 200
    o = outcomes[-1]
    assert o.provider_input_tokens == 100
    assert o.output_tokens == 10


@respx.mock
def test_anthropic_bills_original_plus_hook_redrive(monkeypatch, _no_hooks) -> None:
    """Anthropic A=175 total input, B=250 -> 425; outputs 10+20."""
    register_turn_hook(_RedriveOnce())
    app, outcomes = _app_and_outcomes(monkeypatch)

    def response(ident: str, input_tokens: int, output_tokens: int, read: int, write: int):
        return {
            "id": ident,
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": ident}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_read_input_tokens": read,
                "cache_creation_input_tokens": write,
            },
        }

    sent = iter(
        [
            response("a", 100, 10, 50, 25),
            response("b", 150, 20, 70, 30),
        ]
    )
    respx.post("https://api.anthropic.com/v1/messages").mock(
        side_effect=lambda request: httpx.Response(200, json=next(sent))
    )

    with TestClient(app) as client:
        result = client.post(
            "/v1/messages",
            json={
                "model": "claude-sonnet-4-5",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hi"}],
            },
            headers={
                "x-api-key": "sk-ant-test",
                "anthropic-version": "2023-06-01",
            },
        )

    assert result.status_code == 200
    outcome = outcomes[-1]
    assert outcome.provider_input_tokens == 425
    assert outcome.output_tokens == 30
    assert outcome.cache_read_tokens == 120
    assert outcome.cache_write_tokens == 55
    assert outcome.uncached_input_tokens == 250
