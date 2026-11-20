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
from dataclasses import dataclass
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


_BIG, _SMALL = "log line " * 1_500, "3,000 log lines, compressed"


def _message(role: str, text: str) -> dict[str, Any]:
    kind = "output_text" if role == "assistant" else "input_text"
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def _create(items: list[dict[str, Any]], **fields: Any) -> str:
    response = {"model": "warm-model", **fields, "input": items}
    return json.dumps({"type": "response.create", "response": response})


def _completed(n: int, cached: int, uncached: int) -> str:
    usage = {
        "input_tokens": cached + uncached,
        "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 0},
        "output_tokens": 50,
    }
    response = {"id": f"resp_{n}", "model": "warm-model", "usage": usage}
    return json.dumps({"type": "response.completed", "response": response})


async def _run_ws_session(
    handler: Any,
    frames: list[str],
    events: list[str | list[str]],
    *,
    pipelined: frozenset[int] = frozenset(),
) -> list[Any]:
    """Run one session through ``handle_openai_responses_ws``. Like Codex,
    each frame waits for the previous frame's events, except a frame whose
    index is in ``pipelined``, sent right after the one before it. The
    upstream answers frame n with ``events[n]`` (one event or a list), once
    it has received n frames. The compressor cuts every ``_BIG`` text to
    ``_SMALL``, 1,024 tokens each. Returns the bodies sent upstream."""
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

    assert isinstance(handler, _DummyOpenAIHandler)

    groups = [[e] if isinstance(e, str) else e for e in events]
    # Events the client has seen before it sends frame i (the last entry:
    # before it hangs up). A pipelined frame goes out with the one before it.
    waits = [sum(len(g) for g in groups[:i]) for i in range(len(groups) + 1)]
    for i in sorted(pipelined):
        waits[i] = waits[i - 1]
    # The upstream answers frame i once the client has sent every frame it
    # sends before seeing that answer, so pipelined frames are both in flight.
    needs = [sum(w <= waits[i] for w in waits[:-1]) for i in range(len(groups))]

    class _Client(_FakeWebSocket):
        received = 0

        async def receive_text(self) -> str:
            turn, self.received = self.received, self.received + 1
            while len(self.sent_text) < waits[min(turn, len(waits) - 1)]:
                await asyncio.sleep(0.001)
            if self._frames:
                return self._frames.pop(0)
            raise _FakeWebSocketDisconnect("client closed")

    class _Upstream(_FakeUpstream):
        async def _iter(self):  # noqa: ANN202
            for need, group in zip(needs, groups, strict=True):
                while len(self.sent) < need:
                    await asyncio.sleep(0.001)
                for event in group:
                    yield event
            await asyncio.Event().wait()

    def compress(payload: dict[str, Any], **_kwargs: Any) -> tuple[Any, ...]:
        out = copy.deepcopy(payload)
        parts = [
            part
            for item in out["input"]
            for part in item.get("content") or []
            if part.get("text") == _BIG
        ]
        for part in parts:
            part["text"] = _SMALL
        if not parts:
            return payload, False, 0, [], None, 0, 0, 0
        return out, True, 1_024 * len(parts), ["router:test"], "compressed", 1_000, 100, 4_096

    handler.config.optimize = True
    handler._compress_openai_responses_payload = compress
    upstream = _Upstream([e for g in groups for e in g])
    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        await asyncio.wait_for(
            handler.handle_openai_responses_ws(_Client(frames=frames)), timeout=10.0
        )
    return [json.loads(raw).get("response") or json.loads(raw) for raw in upstream.sent]


@pytest.mark.asyncio
async def test_a_warm_replay_over_websocket_is_booked_the_same(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, _ten_and_one_per_million: Any
) -> None:
    """The same two turns through the WebSocket handler: turn two's
    ``response.create`` re-sends turn one's input, the compressor removes
    1,024 tokens from each frame, and the upstream reports the same usage.
    The handler measures each frame against the previous one, so the sinks
    book what the HTTP path books."""
    from tests.test_openai_codex_ws_lifecycle import _DummyOpenAIHandler

    turn_one = [_message("user", _BIG)]
    turn_two = [*turn_one, _message("assistant", "Mostly timeouts."), _message("user", "Which?")]
    handler = _DummyOpenAIHandler()
    ledger = _use_real_sinks(handler, monkeypatch, tmp_path)

    sent = await _run_ws_session(
        handler,
        [_create(turn_one), _create(turn_two)],
        [_completed(1, 0, 3_072), _completed(2, 2_048, 1_024)],
    )

    assert [body["input"][0]["content"][0]["text"] for body in sent] == [_SMALL, _SMALL]
    _assert_the_warm_replay_was_booked(handler, ledger)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome_b",
    [
        {"type": "response.failed", "response": {"id": "resp_2", "status": "failed"}},
        {"type": "response.incomplete", "response": {"id": "resp_2", "status": "incomplete"}},
        {"type": "error", "error": {"type": "invalid_request_error", "code": "rejected"}},
    ],
    ids=["failed", "incomplete", "rejected"],
)
async def test_a_failed_websocket_turn_never_becomes_the_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    _ten_and_one_per_million: Any,
    outcome_b: dict[str, Any],
) -> None:
    """Transcript A completes. A different transcript B fails without usage
    (or ends incomplete, or is rejected with an ``error`` and no response).
    A ``previous_response_id`` turn on A completes. Then B plus a new item
    removes 1,024 tokens, with 2,048 cached and 1,024 uncached. A is still
    the last successful transcript and shares no input item with B, so none
    of the last turn's removals sat in a replayed prefix: it books at the
    uncached rate and counts as new input. B's removals, deferred to the
    incremental turn's record, count as new input too."""
    from tests.test_openai_codex_ws_lifecycle import _DummyOpenAIHandler

    transcript_a = [_message("user", "Why does the deploy fail?")]
    transcript_b = [_message("user", _BIG)]
    handler = _DummyOpenAIHandler()
    ledger = _use_real_sinks(handler, monkeypatch, tmp_path)

    await _run_ws_session(
        handler,
        [
            _create(transcript_a),
            _create(transcript_b),
            _create([_message("user", "Go on.")], previous_response_id="resp_1"),
            _create([*transcript_b, _message("user", "Which?")]),
        ],
        [
            _completed(1, 0, 1_000),
            json.dumps(outcome_b),
            _completed(3, 1_000, 100),
            _completed(4, 2_048, 1_024),
        ],
    )

    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    last = events[-1]
    assert (last["saved"], last.get("saved_prefix", 0)) == (1_024, 0)
    assert last["cost_effective_usd"] == pytest.approx(1_024 * 10e-6, abs=1e-6)
    m = handler.metrics
    assert m.tokens_saved_total == 2_048
    assert m.new_input_saved_tokens_total == 2_048


# Full transcripts for the ordering cases. Only A and D carry ``_BIG``, so a
# re-send of either removes 1,024 tokens; no two share an input item.
_TRANSCRIPTS = {
    "A": [_message("user", _BIG)],
    "B": [_message("user", "Summarize the deploy log.")],
    "C": [_message("user", "List the failing routes.")],
    "D": [_message("user", "Group the errors by route."), _message("user", _BIG)],
}
_REJECTED = json.dumps(
    {"type": "error", "error": {"type": "invalid_request_error", "message": "rejected"}}
)


def _created(n: int) -> str:
    return json.dumps({"type": "response.created", "response": {"id": f"resp_{n}"}})


def _ended(n: int, kind: str) -> str:
    return json.dumps({"type": f"response.{kind}", "response": {"id": f"resp_{n}"}})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("steps", "probe", "replayed"),
    [
        # The review's case: the standalone error retires B, C becomes the baseline.
        pytest.param("A B:rejected C", "A", 0, id="rejected-then-completed"),
        # A rejected re-send of A defers its 1,024 to C's record, never as prefix.
        pytest.param("A A:rejected C", "A", 0, id="deferred-rejection"),
        pytest.param("A! B!:rejected C!", "A", 0, id="rejected-then-completed-unnamed"),
        pytest.param("A B:rejected B:rejected C", "A", 0, id="two-rejections"),
        # An error after response.created fails that response; C still completes.
        pytest.param("A B:mid-error C", "A", 0, id="error-mid-response"),
        pytest.param("A B:mid-error-failed C", "A", 0, id="error-mid-response-then-failed"),
        # An incremental turn is not measured: A stays the baseline.
        pytest.param("A B:rejected A:incremental", "A", 1_024, id="rejected-then-incremental"),
        pytest.param("A C:no-usage", "A", 1_024, id="completed-without-usage"),
        # D is sent before C's response ends; each response settles its own frame.
        pytest.param("A C D:pipelined", "D", 1_024, id="pipelined"),
        # B is rejected while D's response runs, then D completes.
        pytest.param("A D:created B:rejected-while-D-runs", "D", 1_024, id="rejected-mid-flight"),
    ],
)
async def test_each_websocket_response_settles_its_own_frame(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    _ten_and_one_per_million: Any,
    steps: str,
    probe: str,
    replayed: int,
) -> None:
    """Orderings of full transcripts on one connection, then a probe that
    re-sends one transcript plus a new item: 1,024 removals, 2,048 cached +
    1,024 uncached. Its prefix share is 1,024 only if that transcript's own
    frame was the last to complete with usage. A step is ``<transcript>``
    (completes; ``!`` without ``response.created``) or ``<transcript>:<how>``.
    Whatever the ordering, the session books every frame's removals once."""
    from tests.test_openai_codex_ws_lifecycle import _DummyOpenAIHandler

    frames: list[str] = []
    events: list[str | list[str]] = []
    pipelined: set[int] = set()
    named = "!" not in steps  # the upstream sends response.created
    for n, step in enumerate(steps.split(), start=1):
        name, _, how = step.partition(":")
        items = _TRANSCRIPTS[name.rstrip("!")]
        done = ([_created(n)] if named else []) + [_completed(n, 0, 1_000)]
        if how == "incremental":
            frames.append(_create([_message("user", "Go on.")], previous_response_id="resp_1"))
        else:
            frames.append(_create(items))
        events.append(
            {
                "": done,
                "rejected": [_REJECTED],
                "mid-error": [_created(n), _REJECTED],
                "mid-error-failed": [_created(n), _REJECTED, _ended(n, "failed")],
                "incremental": done,
                "no-usage": [_created(n), _ended(n, "completed")],
                "pipelined": done,
                "created": [_created(n)],
                "rejected-while-D-runs": [_REJECTED, _completed(n - 1, 0, 1_000)],
            }[how]
        )
        if how in {"pipelined", "rejected-while-D-runs"}:
            pipelined.add(n - 1)
    frames.append(_create([*_TRANSCRIPTS[probe], _message("user", "Which?")]))
    n = len(frames)
    events.append(([_created(n)] if named else []) + [_completed(n, 2_048, 1_024)])
    handler = _DummyOpenAIHandler()
    ledger = _use_real_sinks(handler, monkeypatch, tmp_path)

    await _run_ws_session(handler, frames, events, pipelined=frozenset(pipelined))

    booked = [json.loads(line) for line in ledger.read_text().splitlines()]
    last = [event for event in booked if event.get("saved")][-1]
    assert (last["saved"], last.get("saved_prefix", 0)) == (1_024, replayed)
    rate = 1e-6 if replayed else 10e-6
    assert last["cost_effective_usd"] == pytest.approx(1_024 * rate, abs=1e-6)
    removed = 1_024 * sum(raw.count(_BIG) for raw in frames)
    assert sum(event.get("saved", 0) for event in booked) == removed
    assert handler.metrics.tokens_saved_total == removed


@pytest.mark.asyncio
@pytest.mark.parametrize("named", [True, False], ids=["created", "no-created"])
async def test_a_pipelined_replay_books_its_own_removals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    _ten_and_one_per_million: Any,
    named: bool,
) -> None:
    """The review's case: A completes, then a cold C and a replay of A go out
    back to back, before C's response. C removes nothing; the replay removes
    1,024 tokens that sat in A's prefix, read from cache. Each ledger row
    books its own frame: C books 0, the replay books 1,024 as prefix."""
    from tests.test_openai_codex_ws_lifecycle import _DummyOpenAIHandler

    handler = _DummyOpenAIHandler()
    ledger = _use_real_sinks(handler, monkeypatch, tmp_path)

    def done(n: int, cached: int, uncached: int) -> list[str]:
        return ([_created(n)] if named else []) + [_completed(n, cached, uncached)]

    await _run_ws_session(
        handler,
        [
            _create(_TRANSCRIPTS["A"]),
            _create(_TRANSCRIPTS["C"]),
            _create([*_TRANSCRIPTS["A"], _message("user", "Which?")]),
        ],
        [done(1, 0, 3_072), done(2, 0, 1_000), done(3, 2_048, 1_024)],
        pipelined=frozenset({2}),
    )

    rows = [
        (e.get("saved", 0), e.get("saved_prefix", 0), round(e.get("cost_effective_usd", 0), 6))
        for e in (json.loads(line) for line in ledger.read_text().splitlines())
    ]
    assert rows == [(1_024, 0, 0.01024), (0, 0, 0.0), (1_024, 1_024, 0.001024)]
    assert handler.metrics.tokens_saved_total == 2_048


@dataclass
class _Measure:
    """Stands in for an ItemMeasure: a name and a replayed share."""

    name: str
    removed_old: int = 10**6


@pytest.mark.parametrize(
    ("ops", "booked", "released", "unbooked"),
    [
        # Headroom's memory continuation (an unmeasured frame) is rejected.
        pytest.param(
            "sent:a/100 created:r1 ended:r1 sent:- errored sent:b/50 created:r2 ended:r2",
            "a:100 b:50",
            "",
            0,
            id="continuation-rejected",
        ),
        # A client frame is rejected after a continuation completed.
        pytest.param(
            "sent:a/100 created:r1 ended:r1 sent:- created:r2 ended:r2 sent:b/50 errored",
            "a:100 -:0",
            "b",
            50,
            id="rejected-after-continuation",
        ),
        # Disconnect with one frame bound to a response and one not.
        pytest.param(
            "sent:a/100 created:r1 sent:b/50", "", "ab", 150, id="disconnect-bound-and-unbound"
        ),
        # A completion nothing booked is released at close.
        pytest.param(
            "sent:a/100 created:r1 ended:r1/held", "", "a", 100, id="unbooked-completion-at-close"
        ),
        # An error naming its response fails that frame, not the newer one,
        # and its savings ride with the next completion.
        pytest.param(
            "sent:a/100 created:r1 sent:b/50 errored:r1 ended:r1/failed created:r2 ended:r2",
            "none b:150",
            "a",
            0,
            id="error-naming-its-response",
        ),
        # Once responses are named, an unknown id ends nothing.
        pytest.param(
            "sent:a/100 created:r1 ended:r9 ended:r1",
            "none a:100",
            "",
            0,
            id="unknown-id-once-named",
        ),
        # Fallback, no response.created: responses end frames in send order.
        pytest.param(
            "sent:a/100 sent:b/50 ended:x ended:y/failed",
            "a:100 none",
            "b",
            50,
            id="fallback-send-order",
        ),
        # A completion without usage defers its frame to the next completion.
        pytest.param(
            "sent:a/100 created:r1 ended:r1/no-usage sent:b/50 created:r2 ended:r2",
            "none b:150",
            "a",
            0,
            id="completed-without-usage",
        ),
        # The review's case: C and the replay D are in flight together.
        pytest.param(
            "sent:a/100 created:r1 ended:r1 sent:c/0 sent:d/80 created:r2 created:r3"
            " ended:r2 ended:r3",
            "a:100 c:0 d:80",
            "",
            0,
            id="pipelined",
        ),
    ],
)
def test_websocket_responses_pair_with_their_own_frames(
    ops: str, booked: str, released: str, unbooked: int
) -> None:
    """``_WsTurnMeasures`` on its own, then a disconnect: what each completion's
    record books (``measure:saved``, ``-`` for an unmeasured frame, ``none``
    for a response that did not complete with usage), which measures are
    released and can never become the baseline, and what is left for the
    session's residual record. Every frame's savings are booked exactly once."""
    from headroom.proxy.handlers.openai import _WsFrame, _WsTurnMeasures

    gone: list[Any] = []
    taken: list[str] = []
    sent = 0
    turns = _WsTurnMeasures(release=gone.append)
    for op in ops.split():
        verb, _, arg = op.partition(":")
        arg, _, how = arg.partition("/")
        if verb == "sent":
            saved = int(how or 0)
            sent += saved
            measure = None if arg == "-" else _Measure(arg)
            turns.sent(_WsFrame(measure, saved=saved))  # type: ignore[arg-type]
        elif verb == "created":
            turns.created(arg)
        elif verb == "ended":
            turns.ended(arg, completed=how in {"", "held"})
            if how != "held":
                booking = turns.take_completed()
                name = "none" if booking is None else getattr(booking.measure, "name", "-")
                taken.append(f"{name}:{booking.saved}" if booking else name)
        else:
            turns.errored(arg or None)
    left = turns.close()
    assert (" ".join(taken), "".join(m.name for m in gone), left.saved) == (
        booked,
        released,
        unbooked,
    )
    total = sum(int(t.rsplit(":", 1)[1]) for t in taken if ":" in t)
    assert total + left.saved == sent


def test_deferred_savings_never_count_as_a_replayed_prefix() -> None:
    """A failed frame's savings ride with the next completion, but the
    completed frame's replayed share stays capped at its own savings."""
    from headroom.proxy.handlers.openai import _WsFrame, _WsTurnMeasures

    turns = _WsTurnMeasures(release=lambda _measure: None)
    turns.sent(_WsFrame(_Measure("a"), saved=100))  # type: ignore[arg-type]
    turns.errored(None)
    turns.sent(_WsFrame(_Measure("b", removed_old=80), saved=50))  # type: ignore[arg-type]
    turns.created("r2")
    turns.ended("r2", completed=True)
    booking = turns.take_completed()
    assert booking is not None and booking.saved == 150
    assert booking.measure.removed_old == 50  # type: ignore[union-attr]


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
