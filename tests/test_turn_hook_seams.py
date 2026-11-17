"""The two turn-hook seams the gateway-path extensions rely on.

* ``priority`` — hooks run lowest-first, registration order for ties, default
  ``DEFAULT_HOOK_PRIORITY`` when absent or unparseable. Entry-point discovery
  order is not something an operator controls, so a hook whose gate depends on
  another hook's output (tool deferral after a model router) must be able to
  say so.
* ``TurnContext.provider_headers`` + ``merge_provider_headers`` — a hook may
  ask for a provider header; the handler honours only allow-listed names and
  merges ``anthropic-beta`` behind the client's own tokens.
"""

from __future__ import annotations

from typing import Any

import pytest

from headroom.proxy.turn_hooks import (
    DEFAULT_HOOK_PRIORITY,
    PROVIDER_HEADER_ALLOWLIST,
    TurnContext,
    clear_turn_hooks,
    merge_provider_headers,
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


class _Hook:
    stream_safe = True

    def __init__(self, name: str, priority: Any = None, log: list[str] | None = None) -> None:
        self.name = name
        if priority is not None:
            self.priority = priority
        self._log = log if log is not None else []

    def on_request(self, ctx: TurnContext) -> None:
        self._log.append(f"req:{self.name}")

    async def on_response(self, ctx: TurnContext, response: dict, call_model: Any) -> None:
        self._log.append(f"resp:{self.name}")
        return None


def _ctx() -> TurnContext:
    return TurnContext(provider="anthropic", model="claude-sonnet-4-5", messages=[], tools=[])


# --------------------------------------------------------------------------- #
# priority                                                                     #
# --------------------------------------------------------------------------- #


def test_registered_hooks_sort_by_priority_with_stable_ties() -> None:
    register_turn_hook(_Hook("late", 200))
    register_turn_hook(_Hook("default-a"))
    register_turn_hook(_Hook("router", 10))
    register_turn_hook(_Hook("default-b"))
    register_turn_hook(_Hook("bad", "not-a-number"))
    assert [h.name for h in registered_turn_hooks()] == [
        "router",
        "default-a",
        "default-b",
        "bad",
        "late",
    ]


def test_default_priority_is_one_hundred() -> None:
    assert DEFAULT_HOOK_PRIORITY == 100


@pytest.mark.asyncio
async def test_request_and_response_hooks_run_in_priority_order() -> None:
    log: list[str] = []
    register_turn_hook(_Hook("shaping", 200, log))
    register_turn_hook(_Hook("fold", 50, log))
    register_turn_hook(_Hook("router", 10, log))
    ctx = _ctx()
    run_request_hooks(ctx)
    assert log == ["req:router", "req:fold", "req:shaping"]
    log.clear()

    async def _call_model(messages: list) -> dict:
        return {}

    await run_response_hooks(ctx, {}, _call_model)
    assert log == ["resp:router", "resp:fold", "resp:shaping"]


def test_router_before_gate_changes_what_the_gate_sees() -> None:
    """The reason the seam exists: a priority-10 router rewrites ``ctx.model``
    and a priority-200 gate registered EARLIER still sees the routed id."""
    seen: list[str] = []

    class Gate:
        name = "gate"
        stream_safe = True
        priority = 200

        def on_request(self, ctx: TurnContext) -> None:
            seen.append(ctx.model)

    class Router:
        name = "router"
        stream_safe = True
        priority = 10

        def on_request(self, ctx: TurnContext) -> None:
            ctx.model = "anthropic.claude-sonnet-4-5-v1:0"

    register_turn_hook(Gate())
    register_turn_hook(Router())
    run_request_hooks(_ctx())
    assert seen == ["anthropic.claude-sonnet-4-5-v1:0"]


# --------------------------------------------------------------------------- #
# provider_headers                                                             #
# --------------------------------------------------------------------------- #


def test_context_has_its_own_empty_provider_headers() -> None:
    a, b = _ctx(), _ctx()
    assert a.provider_headers == {} and b.provider_headers == {}
    a.provider_headers["anthropic-beta"] = "x"
    assert b.provider_headers == {}


def test_allowlist_is_anthropic_beta_only_today() -> None:
    assert PROVIDER_HEADER_ALLOWLIST == frozenset({"anthropic-beta"})


@pytest.mark.parametrize("requested", [None, {}, {"anthropic-beta": ""}, {"anthropic-beta": "  "}])
def test_nothing_requested_merges_to_nothing(requested: Any) -> None:
    assert merge_provider_headers({"anthropic-beta": "client-1"}, requested) == {}


def test_disallowed_names_are_dropped() -> None:
    out = merge_provider_headers(
        None,
        {"Authorization": "Bearer x", "x-api-key": "k", "anthropic-beta": "tok-a"},
    )
    assert out == {"anthropic-beta": "tok-a"}


def test_beta_merges_client_first_and_deduplicates() -> None:
    out = merge_provider_headers(
        {"Anthropic-Beta": "client-1,tok-a"},
        {"ANTHROPIC-BETA": "tok-a, tok-b ,tok-a"},
    )
    assert out == {"anthropic-beta": "client-1,tok-a,tok-b"}


def test_beta_without_client_value_is_just_the_hooks() -> None:
    assert merge_provider_headers({}, {"anthropic-beta": "tok-a"}) == {"anthropic-beta": "tok-a"}
    assert merge_provider_headers({"anthropic-beta": ""}, {"anthropic-beta": "tok-a"}) == {
        "anthropic-beta": "tok-a"
    }


def test_non_string_values_are_ignored() -> None:
    assert merge_provider_headers({}, {"anthropic-beta": ["tok-a"]}) == {}  # type: ignore[dict-item]
