"""Compression savings are counted once per conversation, not once per turn.

OpenAI's ``/v1/responses`` re-sends the whole transcript every turn and the
router recompresses all of it, so a turn's ``tokens_saved`` is the running
total for the conversation. Summing that across turns counts every removed
token again on every remaining turn. Anthropic's ``/v1/messages`` replays its
frozen prefix in the compressed form forwarded last turn, so its
``tokens_saved`` is a running total too (#5069).
"""

from __future__ import annotations

from types import SimpleNamespace
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
    """Handlers that leave the pair unset report novel-only ``tokens_saved``,
    so it must reach the trackers untouched however many turns go by."""
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
    """A streaming caller that passes no pair (OpenAI chat via backend) must
    keep reporting its savings unchanged."""
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


# --- Anthropic /v1/messages (#5069) -------------------------------------------


def _anthropic_outcome(tokens_saved: int, **overrides: Any) -> RequestOutcome:
    defaults: dict[str, Any] = {
        "request_id": "req-1",
        "provider": "anthropic",
        "model": "claude-opus-5",
        "original_tokens": 40_000 + tokens_saved,
        "optimized_tokens": 40_000,
        "output_tokens": 125,
        "tokens_saved": tokens_saved,
        "attempted_input_tokens": 40_000 + tokens_saved,
        "conversation_key": "anthropic:lineage-a",
        "conversation_tokens_saved": tokens_saved,
    }
    defaults.update(overrides)
    return RequestOutcome(**defaults)


@pytest.mark.asyncio
async def test_anthropic_noop_turns_do_not_re_book_the_frozen_prefix(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The shape reported in #5069: consecutive noop turns of one Claude Code
    session all logged ``tok_saved=856``, because the replayed prefix keeps the
    first turn's compression. The running totals must book it once, and the
    PERF line must say so with ``tok_novel``."""
    handler = _Handler()
    with caplog.at_level("INFO", logger="headroom.proxy"):
        for _ in range(4):
            await emit_request_outcome(handler, _anthropic_outcome(856))
        await emit_request_outcome(handler, _anthropic_outcome(2_342))

    booked = [
        call.kwargs["tokens_saved"] for call in handler.metrics.record_request.await_args_list
    ]
    assert booked == [856, 0, 0, 0, 1_486]
    perf = [r.getMessage() for r in caplog.records if " PERF " in r.getMessage()]
    assert len(perf) == 5
    assert "tok_novel" not in perf[0]  # equal to tok_saved: not repeated
    assert "tok_saved=856 tok_novel=0 " in perf[1]
    assert "tok_saved=2342 tok_novel=1486 " in perf[4]


def _anthropic_app(monkeypatch: pytest.MonkeyPatch, outcomes: list[Any]) -> Any:
    pytest.importorskip("fastapi")
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

    async def _spy(_self: Any, outcome: Any, *a: Any, **kw: Any) -> None:
        outcomes.append(outcome)

    monkeypatch.setattr(type(app.state.proxy), "_record_request_outcome", _spy, raising=True)
    return app


def test_anthropic_handler_keys_savings_by_conversation_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The handler must hand the funnel the pair: one key per conversation,
    stable across its turns, and the request's ``tokens_saved`` as the running
    total. Without it the funnel falls back to booking ``tokens_saved`` every
    turn, which is the #5069 inflation."""
    httpx = pytest.importorskip("httpx")
    respx = pytest.importorskip("respx")
    from fastapi.testclient import TestClient

    outcomes: list[Any] = []
    app = _anthropic_app(monkeypatch, outcomes)
    reply = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 100, "output_tokens": 1},
    }
    headers = {"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"}
    turn_one = [{"role": "user", "content": "first question in conversation A"}]
    turn_two = [
        *turn_one,
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "follow-up in conversation A"},
    ]
    other = [{"role": "user", "content": "an unrelated conversation B"}]

    with respx.mock:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=reply)
        )
        with TestClient(app) as client:
            for messages in (turn_one, turn_two, other):
                result = client.post(
                    "/v1/messages",
                    json={"model": "claude-sonnet-4-5", "max_tokens": 16, "messages": messages},
                    headers=headers,
                )
                assert result.status_code == 200

    first, second, unrelated = outcomes
    assert first.conversation_key and first.conversation_key.startswith("anthropic:")
    assert second.conversation_key == first.conversation_key
    assert unrelated.conversation_key != first.conversation_key
    for outcome in outcomes:
        assert outcome.conversation_tokens_saved == outcome.tokens_saved


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


def test_a_side_thread_in_the_session_does_not_re_book_the_root() -> None:
    """Root turns at running totals 1000 then 1200, with the guardian's
    request (total 0) in between. Keyed on the shared session the guardian
    reset the total and the root booked 1000 + 0 + 1200; keyed per thread
    the root books 1000 + 200."""
    ledger = ConversationSavings()
    root = savings_conversation_key(_codex_body("fix it", "root-thread"))
    guardian = savings_conversation_key(_codex_body("review this", "guardian-thread"))
    booked = [ledger.novel(root, 1000), ledger.novel(guardian, 0), ledger.novel(root, 1200)]
    assert booked == [1000, 0, 200]


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


@pytest.mark.asyncio
async def test_pipeline_extensions_read_both_figures_from_the_snapshot() -> None:
    """An extension sees the wire figure and the novel share the totals book,
    so it never has to parse PERF lines to tell re-sent history from new."""
    from headroom.pipeline import PipelineExtensionManager, PipelineStage

    seen: list[Any] = []

    class _Recorder:
        def on_pipeline_event(self, event: Any) -> None:
            if event.stage is PipelineStage.OUTCOME_OBSERVED:
                seen.append(event.outcome)

    handler = _Handler()
    handler.pipeline_extensions = PipelineExtensionManager(extensions=[_Recorder()], discover=False)
    await emit_request_outcome(
        handler, _responses_outcome(tokens_saved=62_806, conversation_tokens_saved=62_806)
    )
    await emit_request_outcome(handler, _responses_outcome())

    assert [(s.tokens_saved, s.novel_tokens_saved) for s in seen] == [
        (62_806, 62_806),
        (76_020, 13_214),
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


_LOG_OUTPUT = "\n".join(
    f"2026-11-27T07:{i % 60:02d}:00 INFO worker-{i % 7} step {i} completed ok" for i in range(600)
)
_EXPANDED_OUTPUT = "\n".join(
    f"src/auth/middleware_{i}.py session token login handler" for i in range(40)
)
_EXPANDED_SAMPLE = "\n".join(_EXPANDED_OUTPUT.splitlines()[:4])
_EXPANSION_QUERY = "list the auth middleware files"


def _expansion_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, *, optimize: bool
) -> tuple[Any, int]:
    """Drive one real ``/v1/messages`` turn whose query proactively expands a
    tracked compression; return the outcome the handler hands the funnel and
    the expansion size it counted."""
    pytest.importorskip("fastapi")
    import httpx
    from fastapi.testclient import TestClient

    from headroom.cache.compression_store import (
        detach_compression_store,
        get_compression_store,
        reset_compression_store,
    )
    from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
    from headroom.proxy.server import ProxyConfig, create_app

    # A private in-memory CCR store. The default is the developer's SQLite file,
    # and reset_compression_store() deletes every row in the store it resets, so
    # detach any store an earlier test left behind instead of resetting it.
    monkeypatch.setenv("HEADROOM_CCR_BACKEND", "memory")
    detach_compression_store()
    app = create_app(
        ProxyConfig(
            optimize=optimize,
            mode="token",
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_handle_responses=False,
            ccr_context_tracking=True,
            ccr_proactive_expansion=True,
            image_optimize=False,
        )
    )
    outcomes: list[Any] = []

    async def _spy(_self: Any, outcome: Any, *a: Any, **kw: Any) -> None:
        outcomes.append(outcome)

    monkeypatch.setattr(type(app.state.proxy), "_record_request_outcome", _spy, raising=True)
    counted: list[int] = []
    real_count = AnthropicHandlerMixin._proactive_expansion_tokens.__func__

    def _counting(cls: Any, messages: Any, count_text: Any) -> int:
        n = real_count(cls, messages, count_text)
        counted.append(n)
        return n

    monkeypatch.setattr(
        AnthropicHandlerMixin, "_proactive_expansion_tokens", classmethod(_counting)
    )

    async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
        return httpx.Response(
            200,
            json={
                "id": "msg_expansion",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 20, "output_tokens": 1},
            },
        )

    def tool_turn(tid: str, command: str, content: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "assistant",
                "content": [
                    {"type": "tool_use", "id": tid, "name": "Bash", "input": {"command": command}}
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tid, "content": content}],
            },
        ]

    cwd = str(tmp_path)
    try:
        with TestClient(app) as client:
            proxy = client.app.state.proxy
            hash_key = get_compression_store().store(
                _EXPANDED_OUTPUT,
                _EXPANDED_SAMPLE,
                original_item_count=40,
                compressed_item_count=4,
                tool_name="Bash",
                query_context=_EXPANSION_QUERY,
            )
            workspace_key, _ = proxy._resolve_ccr_workspace(
                SimpleNamespace(headers={"x-headroom-cwd": cwd}), {}
            )
            proxy.ccr_context_tracker.track_compression(
                hash_key=hash_key,
                turn_number=1,
                tool_name="Bash",
                original_count=40,
                compressed_count=4,
                workspace_key=workspace_key,
                query_context=_EXPANSION_QUERY,
                sample_content=_EXPANDED_SAMPLE,
            )
            proxy._retry_request = _fake_retry

            messages: list[dict[str, Any]] = [{"role": "user", "content": "run the worker"}]
            messages += tool_turn("t0", "./worker", _LOG_OUTPUT)
            messages += tool_turn(
                "t1", "ls", f"[40 items compressed to 4. Retrieve more: hash={hash_key}]"
            )
            messages += [
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": _EXPANSION_QUERY},
            ]
            response = client.post(
                "/v1/messages",
                headers={
                    "x-api-key": "test-key",
                    "anthropic-version": "2023-06-01",
                    "x-headroom-cwd": cwd,
                },
                json={"model": "claude-sonnet-4-6", "max_tokens": 16, "messages": messages},
            )
            assert response.status_code == 200, response.text
    finally:
        reset_compression_store()
    (outcome,) = outcomes
    assert counted and counted[-1] > 0, "setup: the turn did not proactively expand"
    return outcome, counted[-1]


def test_proactive_expansion_is_added_back_to_the_anthropic_running_total(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Expansion text is sent on one turn only and the tracker strips it, so the
    replayed prefix keeps the removal. The handler must feed the ledger the
    wire saving with the expansion added back; otherwise expand-then-not
    re-books the same removal on the next turn (#5069 follow-up). Asserted on
    the outcome the real handler hands the funnel."""
    outcome, expansion = _expansion_outcome(monkeypatch, tmp_path, optimize=True)
    assert outcome.tokens_saved > 0, "setup: nothing was compressed"
    assert outcome.conversation_key and outcome.conversation_key.startswith("anthropic:")
    assert outcome.conversation_tokens_saved == outcome.tokens_saved + expansion


def test_proactive_expansion_alone_books_nothing_on_the_anthropic_ledger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A turn that only expands removes nothing. Adding the expansion back to
    the clamped per-request saving would book the expansion's own size as a
    removal; the running total has to come from the signed wire saving."""
    outcome, _ = _expansion_outcome(monkeypatch, tmp_path, optimize=False)
    assert outcome.tokens_saved == 0
    assert outcome.conversation_tokens_saved == 0
