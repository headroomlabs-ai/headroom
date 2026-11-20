"""WS Responses: a turn hook's system-prompt cut reaches the headline.

``_compress_openai_responses_payload`` books a system-only reduction on
``turn_hook_system_saved_tokens``, outside the message-side ``tokens_saved`` it
returns. The WebSocket paths must carry that tag to the outcome that claims the
frame's ledger rows, under the same gates, so headline and by-source agree.
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any
from unittest.mock import patch

import pytest

from headroom.proxy.savings_attribution import from_tags
from headroom.proxy.tool_schema_savings_policy import (
    TURN_HOOK_SYSTEM_SAVED_TAG,
    headline_tokens_saved,
)
from headroom.proxy.turn_hooks import clear_turn_hooks, count_system_tokens, register_turn_hook
from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
from tests.test_openai_codex_ws_lifecycle import (
    _DummyOpenAIHandler,
    _FakeUpstream,
    _FakeWebSocket,
    _make_fake_websockets_module,
)
from tests.test_openai_responses_context_compaction import _HandlerHarness

LONG = "a long system prompt " * 40


class _GatedUpstream(_FakeUpstream):
    """Streams turn ``k``'s events only after the ``k``-th frame is sent."""

    def __init__(self, turns: list[list[str]]) -> None:
        super().__init__([])
        self._turns = turns
        self.on_done: Any = lambda: None

    async def _iter(self):
        for k, events in enumerate(self._turns, start=1):
            while len(self.sent) < k:
                await asyncio.sleep(0.001)
            for event in events:
                yield event
        self.on_done()


class _System:
    """Rewrites the developer prompt; the core books its by-source row."""

    name = "system"

    def __init__(self, new: str) -> None:
        self.new = new

    def on_request(self, ctx: Any) -> None:
        ctx.system = self.new


def _frame() -> str:
    return json.dumps(
        {
            "type": "response.create",
            "response": {
                "model": "gpt-5.5",
                "input": [
                    {"role": "developer", "content": LONG},
                    {"role": "user", "content": "hi"},
                ],
            },
        }
    )


def _usage(event: str, rid: str, input_tokens: int) -> str:
    return json.dumps(
        {
            "type": event,
            "response": {"id": rid, "usage": {"input_tokens": input_tokens, "output_tokens": 1}},
        }
    )


def _turn(rid: str, input_tokens: int, event: str = "response.completed") -> list[str]:
    return [
        json.dumps({"type": "response.created", "response": {"id": rid}}),
        _usage(event, rid, input_tokens),
    ]


async def _run(
    frames: list[str],
    upstream: _GatedUpstream | None,
    new_system: str,
    *,
    fallback_input: int = 0,
) -> list[Any]:
    harness = _HandlerHarness(ContentRouter(ContentRouterConfig()))
    handler = _DummyOpenAIHandler()
    handler.config.optimize = True
    handler._compress_openai_responses_payload = harness._compress_openai_responses_payload  # type: ignore[method-assign]
    outcomes: list[Any] = []

    async def _record(outcome: Any) -> None:
        outcomes.append(outcome)

    handler._record_request_outcome = _record  # type: ignore[method-assign]
    if upstream is None:

        async def _fallback(*_args: Any) -> tuple[int, int, int, int, int, bool]:
            return fallback_input, 1, 0, 0, fallback_input, False

        handler._ws_http_fallback = _fallback  # type: ignore[method-assign]
        client = _FakeWebSocket(frames=frames)
        websockets = _make_fake_websockets_module(None, connect_error=OSError("refused"))
    else:
        # The client stays connected until the upstream has streamed every turn.
        client = _FakeWebSocket(frames=frames, hold_after_initial=True)
        upstream.on_done = client.trigger_disconnect
        websockets = _make_fake_websockets_module(upstream)
    clear_turn_hooks()
    register_turn_hook(_System(new_system))
    try:
        with patch.dict(sys.modules, {"websockets": websockets}):
            await asyncio.wait_for(handler.handle_openai_responses_ws(client), timeout=5)
    finally:
        clear_turn_hooks()
    return outcomes


def _system_cut() -> int:
    """What one frame's hook removes, as the core counts it."""
    harness = _HandlerHarness(ContentRouter(ContentRouterConfig()))
    counter = harness.openai_provider.get_token_counter("gpt-5.5")
    count = lambda value: counter.count_text(json.dumps(value))  # noqa: E731
    return count_system_tokens(count, LONG) - count_system_tokens(count, "short")


def _headline(outcome: Any) -> int:
    return headline_tokens_saved(outcome.tokens_saved, outcome.tags)


def _by_source(outcome: Any) -> int:
    return sum(int(row.get("tokens") or 0) for row in from_tags(outcome.tags))


@pytest.mark.asyncio
async def test_a_system_cut_on_the_first_and_a_later_frame_reaches_each_turns_headline() -> None:
    cut = _system_cut()
    assert cut > 0
    upstream = _GatedUpstream([_turn("r_1", 100), _turn("r_2", 160)])
    outcomes = await _run([_frame(), _frame()], upstream, "short")

    turns = [o for o in outcomes if o.provider_input_tokens]
    assert [o.provider_input_tokens for o in turns] == [100, 160]
    assert [_headline(o) for o in turns] == [cut, cut]
    assert [_by_source(o) for o in turns] == [cut, cut]
    assert sum(int(o.tags.get(TURN_HOOK_SYSTEM_SAVED_TAG, 0) or 0) for o in outcomes) == 2 * cut


@pytest.mark.asyncio
async def test_an_unclaimed_system_cut_rides_with_the_session_residual() -> None:
    cut = _system_cut()
    # The upstream WebSocket refuses, so the HTTP fallback serves the turn: its
    # usage reaches the session totals with no per-turn record to claim it.
    outcomes = await _run([_frame()], None, "short", fallback_input=100)

    [residual] = [o for o in outcomes if o.provider_input_tokens]
    assert residual.tags.get("transport") == "websocket"
    assert _headline(residual) == cut
    assert _by_source(residual) == cut


@pytest.mark.asyncio
async def test_a_system_cut_with_no_usage_is_not_booked() -> None:
    upstream = _GatedUpstream([[json.dumps({"type": "response.created", "response": {}})]])
    outcomes = await _run([_frame()], upstream, "short")

    assert all(_headline(o) == 0 for o in outcomes)
    assert all(_by_source(o) == 0 for o in outcomes)


@pytest.mark.asyncio
async def test_a_growing_system_prompt_books_no_saving() -> None:
    upstream = _GatedUpstream([_turn("r_1", 100), _turn("r_2", 160)])
    outcomes = await _run([_frame(), _frame()], upstream, LONG + " more words" * 50)

    assert all(_headline(o) == 0 for o in outcomes)
    assert not any(TURN_HOOK_SYSTEM_SAVED_TAG in (o.tags or {}) for o in outcomes)
