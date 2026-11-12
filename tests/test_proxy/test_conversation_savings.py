"""Compression savings are booked per request, the same way on every path.

OpenAI's ``/v1/responses`` re-sends the whole transcript every turn and the
router recompresses all of it, so a turn's ``tokens_saved`` includes the
removals of earlier turns. Anthropic does the same: the frozen cached prefix is
re-sent with its removals, and a turn that removes nothing new still books
them. The running totals take each request's figure on both paths, so the two
providers' savings are counted the same way.

The conversation keys still identify a Responses conversation, for pipeline
extensions and for the per-request savings record's previous-request lookup.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from headroom.proxy.conversation_savings import savings_conversation_key
from headroom.proxy.outcome import RequestOutcome, emit_request_outcome

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
    }
    defaults.update(overrides)
    return RequestOutcome(**defaults)


def _anthropic_outcome() -> RequestOutcome:
    return RequestOutcome(
        request_id="req-1",
        provider="anthropic",
        model="claude-opus-5",
        original_tokens=114_024,
        optimized_tokens=113_566,
        output_tokens=125,
        tokens_saved=458,
        attempted_input_tokens=1_700,
    )


def _booked(handler: _Handler) -> list[int]:
    return [call.kwargs["tokens_saved"] for call in handler.metrics.record_request.await_args_list]


@pytest.mark.asyncio
async def test_running_totals_book_each_responses_request() -> None:
    """Two turns of one conversation: 62,806 removed from the first request,
    76,020 from the second, which re-sends the first. Both trackers book both
    figures, as they do for Anthropic."""
    handler = _Handler()
    await emit_request_outcome(handler, _responses_outcome(tokens_saved=62_806))
    await emit_request_outcome(handler, _responses_outcome())

    assert _booked(handler) == [62_806, 76_020]
    assert [c.args[1] for c in handler.cost_tracker.record_tokens.call_args_list] == [
        62_806,
        76_020,
    ]


@pytest.mark.asyncio
async def test_a_re_sent_removal_is_booked_the_same_on_both_providers() -> None:
    """Three turns that remove nothing new: each request still carries the
    earlier removals, and both providers book them on every turn."""
    handler = _Handler()
    for _ in range(3):
        await emit_request_outcome(handler, _anthropic_outcome())
        await emit_request_outcome(handler, _responses_outcome(tokens_saved=10_000))
    assert _booked(handler) == [458, 10_000] * 3


@pytest.mark.asyncio
async def test_the_request_log_keeps_the_wire_truth() -> None:
    handler = _Handler()
    handler.logger = MagicMock()
    await emit_request_outcome(handler, _responses_outcome())

    logged = handler.logger.log.call_args_list[0].args[0]
    assert logged.tokens_saved == 76_020
    assert logged.input_tokens_original == 156_260
    assert logged.input_tokens_optimized == 80_240


def test_the_streaming_constructor_carries_the_conversation_key() -> None:
    """Codex streams ``/v1/responses``, so the streaming finalizer carries the
    key extensions attribute by; every other streaming caller leaves it off."""
    common: dict[str, Any] = {
        "request_id": "req-1",
        "original_tokens": 156_260,
        "optimized_tokens": 80_240,
        "output_tokens": 177,
        "tokens_saved": 76_020,
        "transforms_applied": [],
        "total_latency_ms": 1.0,
        "overhead_ms": 1.0,
        "tags": {},
    }
    codex = RequestOutcome.from_stream(
        body={"model": "gpt-6-astra", "input": []},
        provider="openai",
        model="gpt-6-astra",
        client="codex",
        conversation_key="conv-a",
        **common,
    )
    claude = RequestOutcome.from_stream(
        body={"model": "claude-opus-5", "messages": []},
        provider="anthropic",
        model="claude-opus-5",
        client="claude-code",
        **common,
    )
    assert codex.conversation_key == "conv-a"
    assert claude.conversation_key is None


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
    would share one previous request in the savings record."""
    from headroom.proxy.output_savings_policy import conversation_key_from_body

    a_body = _responses_body("fix the failing test")
    b_body = _responses_body("write the release notes")
    assert conversation_key_from_body(a_body) == conversation_key_from_body(b_body)
    assert savings_conversation_key(a_body) is None
    assert savings_conversation_key(b_body) is None


def test_prompt_cache_key_is_cache_routing_not_identity() -> None:
    """OpenAI documents one prompt_cache_key shared across a user's sessions
    and forks, and across independent single-turn requests. Two conversations
    under one key must not share a previous request, so the key alone yields no
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


def _codex_body(text: str, thread_id: str, *, session_id: str = "root-thread") -> dict[str, Any]:
    """Codex's shape: both ids flat in ``client_metadata``, the session id
    being the root thread's id on every request of the session."""
    return _responses_body(text, client_metadata={"session_id": session_id, "thread_id": thread_id})


def test_a_thread_id_wins_over_the_session_beside_it() -> None:
    root = savings_conversation_key(_codex_body("fix it", "root-thread"))
    guardian = savings_conversation_key(_codex_body("review this", "guardian-thread"))
    subagent = savings_conversation_key(_codex_body("explore", "subagent-thread"))
    assert None not in (root, guardian, subagent)
    assert len({root, guardian, subagent}) == 3
    top = _responses_body("x", session_id="s-1", thread_id="t-1")
    assert savings_conversation_key(top) == savings_conversation_key(
        _responses_body("y", thread_id="t-1")
    )


def test_server_side_state_means_incremental_input() -> None:
    """With ``previous_response_id`` or a ``conversation`` the provider holds
    the history and each payload is a fresh increment, not a transcript to
    compare with the previous request, even under an explicit conversation id."""
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


@pytest.mark.asyncio
async def test_pipeline_extensions_read_the_booked_figure_from_the_snapshot() -> None:
    """An extension sees the figure the totals book, so it never has to parse
    PERF lines. ``novel_tokens_saved`` equals it on every path."""
    from headroom.pipeline import PipelineExtensionManager, PipelineStage

    seen: list[Any] = []

    class _Recorder:
        def on_pipeline_event(self, event: Any) -> None:
            if event.stage is PipelineStage.OUTCOME_OBSERVED:
                seen.append(event.outcome)

    handler = _Handler()
    handler.pipeline_extensions = PipelineExtensionManager(extensions=[_Recorder()], discover=False)
    await emit_request_outcome(handler, _responses_outcome(tokens_saved=62_806))
    await emit_request_outcome(handler, _responses_outcome())

    assert [(s.tokens_saved, s.novel_tokens_saved) for s in seen] == [
        (62_806, 62_806),
        (76_020, 76_020),
    ]


@pytest.mark.asyncio
async def test_the_snapshot_says_whose_request_it_was() -> None:
    """Project, client, conversation and the pre-compression size reach the
    extension, so it can attribute what it measures without patching core."""
    from headroom.pipeline import PipelineExtensionManager, PipelineStage

    seen: list[Any] = []

    class _Recorder:
        def on_pipeline_event(self, event: Any) -> None:
            if event.stage is PipelineStage.OUTCOME_OBSERVED:
                seen.append(event.outcome)

    handler = _Handler()
    handler.pipeline_extensions = PipelineExtensionManager(extensions=[_Recorder()], discover=False)
    await emit_request_outcome(handler, _responses_outcome(project="web", client="codex"))
    await emit_request_outcome(handler, _responses_outcome(request_id="req-2"))

    assert dict(seen[0].attribution) == {"project": "web", "client": "codex"}
    assert seen[0].conversation_key == "conv-a"
    assert seen[0].original_tokens == 156_260
    assert dict(seen[1].attribution) == {}
