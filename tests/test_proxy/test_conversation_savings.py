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

import json
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


@pytest.fixture
def _ten_and_one_per_million(monkeypatch: pytest.MonkeyPatch) -> Any:
    """``warm-model`` at $10/M uncached input and $1/M cache reads."""
    from types import SimpleNamespace

    from headroom.pricing.counterfactual import resolve_rates

    resolve_rates.cache_clear()
    monkeypatch.setattr(
        "headroom.pricing.counterfactual._litellm",
        lambda: SimpleNamespace(
            model_cost={
                "warm-model": {
                    "input_cost_per_token": 10e-6,
                    "cache_read_input_token_cost": 1e-6,
                    "litellm_provider": "openai",
                }
            }
        ),
    )
    monkeypatch.setattr(
        "headroom.pricing.litellm_pricing.resolve_litellm_model", lambda model: model
    )
    yield
    resolve_rates.cache_clear()


def _use_real_sinks(handler: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Any:
    """Give ``handler`` the production metrics, savings tracker, ledger and
    cost tracker, writing under ``tmp_path``. Returns the ledger path."""
    from headroom.proxy.cost import CostTracker
    from headroom.proxy.prometheus_metrics import PrometheusMetrics
    from headroom.proxy.savings_tracker import SavingsTracker

    ledger = tmp_path / "savings_events.jsonl"
    monkeypatch.setenv("HEADROOM_SAVINGS_EVENTS_PATH", str(ledger))
    handler.cost_tracker = CostTracker()
    handler.metrics = PrometheusMetrics(
        savings_tracker=SavingsTracker(path=str(tmp_path / "proxy_savings.json")),
        cost_tracker=handler.cost_tracker,
    )
    return ledger


def _assert_the_warm_replay_was_booked(handler: Any, ledger: Any) -> None:
    """Two turns removing 1,024 tokens each; turn two's removals all sat in
    the replayed prefix, read from cache (2,048 cached + 1,024 uncached)."""
    from headroom.savings_ledger import aggregate_savings

    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    events = [e for e in events if e.get("saved")]
    m = handler.metrics
    lifetime = m.savings_tracker.snapshot()["lifetime"]
    durable = m.savings_tracker.lifetime_response()
    cost = handler.cost_tracker.stats()

    # Token booking: both turns, every sink.
    assert m.tokens_saved_total == 2_048
    assert lifetime["tokens_saved"] == 2_048
    assert durable["tokens"]["saved"] == 2_048
    assert [e["saved"] for e in events] == [1_024, 1_024]
    assert cost["total_compression_tokens_saved"] == 2_048
    # Cache-aware valuation: turn two adds a cache read's worth.
    expected = 1_024 * 10e-6 + 1_024 * 1e-6
    assert lifetime["compression_savings_usd"] == pytest.approx(expected, abs=1e-6)
    assert durable["cost"]["compression_savings_usd"] == pytest.approx(expected, abs=1e-6)
    assert sum(e["cost_effective_usd"] for e in events) == pytest.approx(expected, abs=1e-6)
    assert cost["cache_aware_savings_usd"] == round(expected, 4) == 0.0113
    # New-input rate: turn two's removals were cache reads, not new input, so
    # only turn one's 1,024 count, against 3,072 + 1,024 new tokens.
    saved, new = m.new_input_saved_tokens_total, m.new_input_tokens_total
    assert (saved, new) == (1_024, 4_096)
    assert round(saved / (new + saved) * 100, 2) == 20.0  # /stats' formula
    assert aggregate_savings(ledger).lifetime["new_input_savings_percent"] == 20.0


@pytest.mark.asyncio
async def test_a_warm_replay_books_its_tokens_at_the_cache_read_price(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, _ten_and_one_per_million: Any
) -> None:
    """Two turns of one conversation, each removing 1,024 tokens. Turn two
    re-sends turn one, every removal sits in that replayed prefix, and the
    provider reports 2,048 cached plus 1,024 uncached tokens. At $10/M
    uncached and $1/M cache reads, every sink books 1,024 + 1,024 tokens and
    $0.01024 + $0.001024: the replayed removal is worth a cache read, not the
    live-zone rate that made the total $0.02048. The new-input rate is 20%,
    not the 33.33% that counting the replayed removal as new input gives."""
    from headroom.proxy.savings import ItemMeasure

    handler = _Handler()
    ledger = _use_real_sinks(handler, monkeypatch, tmp_path)
    turn = {
        "model": "warm-model",
        "original_tokens": 4_096,
        "optimized_tokens": 3_072,
        "output_tokens": 50,
        "tokens_saved": 1_024,
        "attempted_input_tokens": 4_096,
    }
    await emit_request_outcome(
        handler,
        _responses_outcome(
            **turn,
            cache_read_tokens=0,
            uncached_input_tokens=3_072,
            provider_input_tokens=3_072,
            savings_measure=ItemMeasure(
                sent_tokens=3_072,
                removed_old=0,
                removed_new=1_024,
                sent_shared_prefix=0,
                bust=0,
            ),
        ),
    )
    await emit_request_outcome(
        handler,
        _responses_outcome(
            **turn,
            request_id="req-2",
            cache_read_tokens=2_048,
            uncached_input_tokens=1_024,
            provider_input_tokens=3_072,
            savings_measure=ItemMeasure(
                sent_tokens=3_072,
                removed_old=1_024,
                removed_new=0,
                sent_shared_prefix=2_048,
                bust=0,
            ),
        ),
    )

    _assert_the_warm_replay_was_booked(handler, ledger)


@pytest.mark.asyncio
async def test_a_warm_replay_over_websocket_is_booked_the_same(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, _ten_and_one_per_million: Any
) -> None:
    """The same two turns through the WebSocket handler: turn two's
    ``response.create`` re-sends turn one's input, the compressor removes
    1,024 tokens from each frame, and the upstream reports the same usage.
    The handler measures each frame against the previous one, so the sinks
    book what the HTTP path books."""
    import asyncio
    import copy
    import sys
    from unittest.mock import patch

    from tests.test_openai_codex_ws_lifecycle import (
        _DummyOpenAIHandler,
        _FakeUpstream,
        _FakeWebSocket,
        _FakeWebSocketDisconnect,
        _make_fake_websockets_module,
    )

    big, small = "log line " * 1_500, "3,000 log lines, compressed"

    def message(role: str, text: str) -> dict[str, Any]:
        kind = "output_text" if role == "assistant" else "input_text"
        return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}

    turn_one = [message("user", big)]
    turn_two = [*turn_one, message("assistant", "Mostly timeouts."), message("user", "Which?")]

    def frame(items: list[dict[str, Any]]) -> str:
        return json.dumps(
            {"type": "response.create", "response": {"model": "warm-model", "input": items}}
        )

    def completed(n: int, cached: int, uncached: int) -> str:
        usage = {
            "input_tokens": cached + uncached,
            "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 0},
            "output_tokens": 50,
        }
        response = {"id": f"resp_{n}", "model": "warm-model", "usage": usage}
        return json.dumps({"type": "response.completed", "response": response})

    class _Client(_FakeWebSocket):
        """Like Codex: the next response.create follows the last completion."""

        received = 0

        async def receive_text(self) -> str:
            turn, self.received = self.received, self.received + 1
            while len(self.sent_text) < turn:
                await asyncio.sleep(0.001)
            if self._frames:
                return self._frames.pop(0)
            raise _FakeWebSocketDisconnect("client closed")

    class _Upstream(_FakeUpstream):
        """Completes each turn only once its frame was sent."""

        async def _iter(self):  # noqa: ANN202
            for n, event in enumerate(self._events, start=1):
                while len(self.sent) < n:
                    await asyncio.sleep(0.001)
                yield event
            await asyncio.Event().wait()

    def compress(payload: dict[str, Any], **_kwargs: Any) -> tuple[Any, ...]:
        out = copy.deepcopy(payload)
        for item in out["input"]:
            for part in item["content"]:
                if part["text"] == big:
                    part["text"] = small
        return out, True, 1_024, ["router:test"], "compressed", 1_000, 100, 4_096

    handler = _DummyOpenAIHandler()
    handler.config.optimize = True
    handler._compress_openai_responses_payload = compress  # type: ignore[method-assign]
    ledger = _use_real_sinks(handler, monkeypatch, tmp_path)
    upstream = _Upstream([completed(1, 0, 3_072), completed(2, 2_048, 1_024)])

    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        await asyncio.wait_for(
            handler.handle_openai_responses_ws(_Client(frames=[frame(turn_one), frame(turn_two)])),
            timeout=10.0,
        )

    sent = [json.loads(raw).get("response") or json.loads(raw) for raw in upstream.sent]
    assert [body["input"][0]["content"][0]["text"] for body in sent] == [small, small]
    _assert_the_warm_replay_was_booked(handler, ledger)


def test_a_cold_replay_prices_like_the_live_zone(_ten_and_one_per_million: Any) -> None:
    """With nothing read from cache (an expired prefix), a replayed removal
    is a miss, priced like any other removal."""
    from headroom.proxy.savings_tracker import estimate_request_savings_usd

    def compression_usd(prefix: int) -> float:
        priced = estimate_request_savings_usd(
            "warm-model",
            compression_tokens_saved=1_024,
            compression_prefix_tokens_saved=prefix,
            uncached_input_tokens=3_072,
            provider="openai",
        )
        return float(priced["compression"])

    assert compression_usd(1_024) == pytest.approx(compression_usd(0)) == pytest.approx(0.01024)


@pytest.mark.asyncio
async def test_a_cold_replay_stays_in_the_new_input_numerator(
    _ten_and_one_per_million: Any,
) -> None:
    """With nothing read from cache, the replayed prefix was billed as new
    input, so its removal counts on the new-input basis like any other."""
    from headroom.proxy.prometheus_metrics import PrometheusMetrics

    metrics = PrometheusMetrics(stateless=True)
    common: dict[str, Any] = {
        "provider": "openai",
        "model": "warm-model",
        "input_tokens": 3_072,
        "output_tokens": 50,
        "tokens_saved": 1_024,
        "latency_ms": 1.0,
        "prefix_tokens_saved": 1_024,
    }
    await metrics.record_request(**common, uncached_input_tokens=3_072)
    await metrics.record_request(**common, cache_read_tokens=512, uncached_input_tokens=2_560)

    # Cold: all 1,024 count. Half-warm: the 512 read from cache do not.
    assert metrics.new_input_saved_tokens_total == 1_024 + 512


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
