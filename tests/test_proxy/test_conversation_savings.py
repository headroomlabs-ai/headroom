"""Compression savings are counted once per conversation, not once per turn.

OpenAI's ``/v1/responses`` re-sends the whole transcript every turn and the
router recompresses all of it, so a turn's ``tokens_saved`` is the running
total for the conversation. Summing that across turns counts every removed
token again on every remaining turn. Anthropic's frozen cached prefix means
its ``tokens_saved`` is already novel-only, which is why the two providers'
lifetime totals were never comparable.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from headroom.proxy.conversation_savings import (
    ConversationSavings,
    get_conversation_savings,
    reset_conversation_savings,
    savings_conversation_key,
)
from headroom.proxy.outcome import RequestOutcome, emit_request_outcome


@pytest.fixture(autouse=True)
def _clean_ledger() -> Any:
    reset_conversation_savings()
    yield
    reset_conversation_savings()


# ── The ledger ─────────────────────────────────────────────────────────


def test_first_turn_of_a_conversation_is_entirely_novel() -> None:
    ledger = ConversationSavings()
    assert ledger.novel("conv-a", 76_020) == 76_020


def test_later_turns_count_only_what_the_total_grew_by() -> None:
    """The measured shape: a turn removes 76k from a transcript whose earlier
    turns already accounted for 62k of it."""
    ledger = ConversationSavings()
    ledger.novel("conv-a", 62_806)
    assert ledger.novel("conv-a", 76_020) == 13_214


def test_a_long_conversation_counts_each_removed_token_once() -> None:
    """The regression itself. Twenty turns of a conversation that removed
    10k tokens on turn one and nothing new after: the old accounting booked
    200k, which is also what made the savings rate exceed its own denominator.
    """
    ledger = ConversationSavings()
    total = sum(ledger.novel("conv-a", 10_000) for _ in range(20))
    assert total == 10_000


def test_conversations_do_not_leak_into_each_other() -> None:
    ledger = ConversationSavings()
    assert ledger.novel("conv-a", 5_000) == 5_000
    assert ledger.novel("conv-b", 8_000) == 8_000
    assert ledger.novel("conv-a", 6_000) == 1_000


def test_a_shrinking_total_is_a_compaction_not_a_negative_saving() -> None:
    """A compaction (or a client dropping history) shrinks the transcript.
    Nothing was removed for the first time, and the next turn must count from
    the new lower base rather than waiting out the old high-water mark."""
    ledger = ConversationSavings()
    ledger.novel("conv-a", 50_000)
    assert ledger.novel("conv-a", 9_000) == 0
    assert ledger.novel("conv-a", 11_000) == 2_000


def test_missing_identity_declines_to_answer() -> None:
    """None means "this path does not distinguish" so the funnel falls back to
    ``tokens_saved`` -- which is already novel-only on every provider that
    freezes its cached prefix. Attributing zero here would silently delete
    real savings."""
    ledger = ConversationSavings()
    assert ledger.novel(None, 5_000) is None
    assert ledger.novel("", 5_000) is None
    assert ledger.novel("conv-a", None) is None


def test_the_oldest_conversation_is_forgotten_first() -> None:
    """Bounded memory. A forgotten conversation re-counts its transcript once,
    which is a one-off overcount, not an unbounded leak."""
    ledger = ConversationSavings(max_conversations=2)
    ledger.novel("conv-a", 1_000)
    ledger.novel("conv-b", 1_000)
    ledger.novel("conv-c", 1_000)  # evicts conv-a, the oldest
    # conv-a is gone, so its transcript is re-counted from zero once. Re-
    # admitting it evicts the next-oldest (conv-b); conv-c stays warm.
    assert ledger.novel("conv-a", 1_000) == 1_000
    assert ledger.novel("conv-c", 1_500) == 500


def test_touching_a_conversation_keeps_it_warm() -> None:
    ledger = ConversationSavings(max_conversations=2)
    ledger.novel("conv-a", 1_000)
    ledger.novel("conv-b", 1_000)
    ledger.novel("conv-a", 2_000)  # conv-a is now the most recent
    ledger.novel("conv-c", 1_000)  # evicts conv-b, not conv-a
    assert ledger.novel("conv-a", 3_000) == 1_000


# ── The funnel ─────────────────────────────────────────────────────────


class _Handler:
    """Minimal stand-in for the parts of HeadroomProxy the funnel touches."""

    def __init__(self) -> None:
        self.metrics = MagicMock()
        self.metrics.record_request = AsyncMock()
        self.cost_tracker = MagicMock()
        self.logger = None


def _responses_outcome(**overrides: Any) -> RequestOutcome:
    defaults: dict[str, Any] = {
        "request_id": "req-1",
        "provider": "openai",
        "model": "gpt-6-astra",
        "original_tokens": 156_260,
        "optimized_tokens": 80_240,
        "output_tokens": 177,
        "tokens_saved": 76_020,
        "attempted_input_tokens": 90_000,
        "cache_read_tokens": 66_304,
        "conversation_key": "conv-a",
        "conversation_tokens_saved": 76_020,
    }
    defaults.update(overrides)
    return RequestOutcome(**defaults)


@pytest.mark.asyncio
async def test_running_totals_see_the_novel_figure() -> None:
    """Two turns of one conversation whose running total grew by 13,214. The
    savings tracker and the cost tracker must see the growth, not the total
    twice -- that is the inflation the whole change exists to remove."""
    handler = _Handler()
    await emit_request_outcome(
        handler, _responses_outcome(tokens_saved=62_806, conversation_tokens_saved=62_806)
    )
    await emit_request_outcome(handler, _responses_outcome())

    second_call = handler.metrics.record_request.await_args_list[1]
    assert second_call.kwargs["tokens_saved"] == 13_214
    assert handler.cost_tracker.record_tokens.call_args_list[1].args[1] == 13_214


@pytest.mark.asyncio
async def test_the_request_log_keeps_the_wire_truth() -> None:
    """Those 76,020 tokens really did leave this request's payload. Per
    request surfaces describe the wire; only running totals de-duplicate."""
    handler = _Handler()
    handler.logger = MagicMock()
    await emit_request_outcome(handler, _responses_outcome(conversation_tokens_saved=62_806))
    await emit_request_outcome(handler, _responses_outcome())

    logged = handler.logger.log.call_args_list[1].args[0]
    assert logged.tokens_saved == 76_020
    assert logged.input_tokens_original == 156_260
    assert logged.input_tokens_optimized == 80_240


@pytest.mark.asyncio
async def test_paths_without_conversation_identity_are_unchanged() -> None:
    """Anthropic and every other handler leave the pair unset. Their
    ``tokens_saved`` is already novel-only, so it must reach the trackers
    untouched however many turns go by."""
    handler = _Handler()
    for _ in range(3):
        await emit_request_outcome(
            handler,
            RequestOutcome(
                request_id="req-1",
                provider="anthropic",
                model="claude-opus-5",
                original_tokens=114_024,
                optimized_tokens=113_566,
                output_tokens=125,
                tokens_saved=458,
                attempted_input_tokens=1_700,
            ),
        )
    assert [
        call.kwargs["tokens_saved"] for call in handler.metrics.record_request.await_args_list
    ] == [458, 458, 458]


@pytest.mark.asyncio
async def test_a_websocket_residual_does_not_re_book_its_session() -> None:
    """The session-end residual carries the same running total the last
    per-turn record already accounted for, so its novel contribution is zero.
    When no per-turn record fired, it carries the whole figure instead."""
    handler = _Handler()
    await emit_request_outcome(handler, _responses_outcome(conversation_tokens_saved=40_000))
    await emit_request_outcome(
        handler,
        _responses_outcome(tokens_saved=0, conversation_tokens_saved=40_000),
    )
    assert handler.metrics.record_request.await_args_list[1].kwargs["tokens_saved"] == 0


@pytest.mark.asyncio
async def test_the_ledger_is_shared_across_requests() -> None:
    """One conversation spans many requests and, on the HTTP path, many
    handler invocations. A per-request ledger would never see turn N-1."""
    get_conversation_savings().novel("conv-a", 70_000)
    handler = _Handler()
    await emit_request_outcome(handler, _responses_outcome())
    assert handler.metrics.record_request.await_args_list[0].kwargs["tokens_saved"] == 6_020


def test_the_streaming_constructor_carries_the_pair() -> None:
    """Codex streams ``/v1/responses``, so the streaming finalizer is the path
    that actually books most Codex savings. Dropping the pair there would fix
    the accounting only for non-streaming turns."""
    outcome = RequestOutcome.from_stream(
        body={"model": "gpt-6-astra", "input": []},
        provider="openai",
        model="gpt-6-astra",
        request_id="req-1",
        original_tokens=156_260,
        optimized_tokens=80_240,
        output_tokens=177,
        tokens_saved=76_020,
        transforms_applied=[],
        total_latency_ms=1.0,
        overhead_ms=1.0,
        tags={},
        client="codex",
        conversation_key="conv-a",
        conversation_tokens_saved=76_020,
    )
    assert outcome.conversation_key == "conv-a"
    assert outcome.conversation_tokens_saved == 76_020


def test_the_streaming_constructor_defaults_the_pair_off() -> None:
    """Every other streaming caller (Anthropic SSE, Bedrock, OpenAI chat via
    backend) must keep reporting novel-only savings unchanged."""
    outcome = RequestOutcome.from_stream(
        body={"model": "claude-opus-5", "messages": []},
        provider="anthropic",
        model="claude-opus-5",
        request_id="req-1",
        original_tokens=114_024,
        optimized_tokens=113_566,
        output_tokens=125,
        tokens_saved=458,
        transforms_applied=[],
        total_latency_ms=1.0,
        overhead_ms=1.0,
        tags={},
        client="claude-code",
    )
    assert outcome.conversation_key is None
    assert outcome.conversation_tokens_saved is None


# --- identity: real Responses bodies, not an injected key -------------------

_INSTRUCTIONS = "You are Codex, a coding agent running in the user's terminal. " * 20


def _responses_body(text: str, **extra: Any) -> dict[str, Any]:
    return {
        "model": "gpt-6-astra",
        "instructions": _INSTRUCTIONS,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": text}]}],
        "store": False,
        **extra,
    }


def test_shared_instructions_with_distinct_input_are_not_one_conversation() -> None:
    """Two ordinary payloads, same model and instructions, different user
    input. The holdout key merges them (its Responses fallback is the
    instructions prefix); the savings key must not, or independent sessions
    suppress each other's savings through the process-wide ledger."""
    from headroom.proxy.output_savings_policy import conversation_key_from_body

    a_body = _responses_body("fix the failing test")
    b_body = _responses_body("write the release notes")
    assert conversation_key_from_body(a_body) == conversation_key_from_body(b_body)
    a, b = savings_conversation_key(a_body), savings_conversation_key(b_body)
    assert a is None and b is None
    ledger = ConversationSavings()
    assert ledger.novel(a, 100) is None and ledger.novel(b, 100) is None


def test_prompt_cache_key_is_cache_routing_not_identity() -> None:
    """OpenAI documents one prompt_cache_key shared across a user's sessions
    and forks, and across independent single-turn requests. Two conversations
    under one key must not share a running total, so the key alone yields no
    identity; a real session id beside it does, and wins."""
    a = _responses_body("fix the failing test", prompt_cache_key="shared-support-prefix")
    b = _responses_body("write the release notes", prompt_cache_key="shared-support-prefix")
    assert savings_conversation_key(a) is None and savings_conversation_key(b) is None
    a_key = savings_conversation_key(a, session_id="session-0")
    b_key = savings_conversation_key(b, session_id="session-1")
    assert a_key is not None and b_key is not None and a_key != b_key
    assert savings_conversation_key(a, session_id="session-0") == a_key


def test_explicit_ids_and_caller_vouched_sessions() -> None:
    assert savings_conversation_key(_responses_body("x", metadata={"conversation_id": "c-1"}))
    assert savings_conversation_key(_responses_body("x", thread_id="t-1"))
    ws_a = savings_conversation_key(_responses_body("x"), session_id="ws:a")
    ws_b = savings_conversation_key(_responses_body("x"), session_id="ws:b")
    assert ws_a is not None and ws_b is not None and ws_a != ws_b
    frame = {"type": "response.create", "response": _responses_body("x", prompt_cache_key="c")}
    assert savings_conversation_key(frame) == savings_conversation_key(
        _responses_body("x", prompt_cache_key="c")
    )


def test_server_side_state_means_incremental_input() -> None:
    """With ``previous_response_id`` or a ``conversation`` the provider holds
    the history and each payload is a fresh increment: its ``tokens_saved``
    is per-request already, even under an explicit conversation id."""
    assert (
        savings_conversation_key(
            _responses_body("more", prompt_cache_key="conv-1", previous_response_id="resp_1")
        )
        is None
    )
    assert savings_conversation_key(_responses_body("more", conversation="conv_abc")) is None
    assert (
        savings_conversation_key(_responses_body("more", conversation={"id": "conv_abc"})) is None
    )
    assert savings_conversation_key({"model": "gpt-6-astra", "messages": []}) is None
