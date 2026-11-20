"""CCR reactive-continuation overhead is folded into the billed cost totals,
cache-split per provider so cached tokens price at the right rate.

Regression coverage for the gap where `handle_response` returns only the final
continuation round, so the dropped rounds' real billed usage sat in /stats
`ccr_overhead` but was excluded from `cost_with_headroom` (the cost.py
"ATTRIBUTION ONLY" comment's premise that it was "already inside
cost_with_headroom" did not hold). The fix publishes the per-call dropped-round
usage, cache-split into (uncached_input, cache_read, cache_write, output), via a
ContextVar mirroring the proactive-expansion pattern; the outcome funnel adds
each bucket to the matching `cost_tracker.record_tokens` argument.

The fold is deliberately cost-only. `metrics.record_request` forwards
`cache_read_tokens` to SavingsTracker, which credits it as cache *savings*, so
folding there would let retrieval overhead RAISE the reported savings; the
output-shaper estimator reads `outcome.output_tokens` as this turn's completion
length; and the RequestLog row / PERF line describe the single round the client
received. Those surfaces stay on the final round — see the tests below that pin
each one.

All usage numbers below are SYNTHETIC fixture inputs chosen for arithmetic
clarity; assertions are on structural invariants (sums of dropped rounds,
cache-bucket placement), NOT on any production billing figure or percentage.

Context note: set/call/consume must run inside ONE asyncio.run. A ContextVar
set inside handle_response only propagates to callers in the SAME task/context
(the production flow: handle_response and emit_request_outcome run in the same
request coroutine). Production emit runs via asyncio.shield, which copies the
context, so the funnel's consume clears the copy rather than the caller's value —
`clear_pending_outcome_side_channels()` in `_record_request_outcome` is what
prevents a second outcome from the same context re-booking the same rounds. The
batch path (multiple handle_response calls, one emit) is a known follow-up: the
publish overwrites rather than sums, so earlier items go uncounted.
"""

from __future__ import annotations

import asyncio

from headroom.ccr.response_handler import CCRResponseHandler, CCRToolResult
from headroom.ccr.tool_injection import CCR_TOOL_NAME
from headroom.proxy.outcome import (
    RequestOutcome,
    clear_pending_outcome_side_channels,
    consume_pending_ccr_continuation_usage,
    emit_request_outcome,
    set_pending_ccr_continuation_usage,
)

# --- provider-shape parsing ----------------------------------------------


def test_extract_round_cost_usage_anthropic():
    f = CCRResponseHandler._extract_round_cost_usage
    got = f(
        {
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 50,
                "cache_read_input_tokens": 200,
                "cache_creation_input_tokens": 100,
            }
        },
        "anthropic",
    )
    # input_tokens is already uncached; cache fields map 1:1.
    assert got == (1000, 200, 100, 50)


def test_extract_round_cost_usage_openai_chat_splits_cached():
    f = CCRResponseHandler._extract_round_cost_usage
    got = f(
        {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "prompt_tokens_details": {"cached_tokens": 200},
            }
        },
        "openai",
    )
    # prompt_tokens is gross incl cached; cached lands in cache_read, not uncached.
    assert got == (800, 200, 0, 50)


def test_extract_round_cost_usage_openai_responses_shape():
    f = CCRResponseHandler._extract_round_cost_usage
    got = f(
        {
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 50,
                "input_tokens_details": {"cached_tokens": 200},
            }
        },
        "openai_responses",
    )
    assert got == (800, 200, 0, 50)


def test_extract_round_cost_usage_prefers_toplevel_cache_keys():
    f = CCRResponseHandler._extract_round_cost_usage
    got = f(
        {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "cache_read_input_tokens": 300,
                "prompt_tokens_details": {"cached_tokens": 999},
            }
        },
        "openai",
    )
    # Authoritative top-level Anthropic/Bedrock key wins over the details shape.
    assert got == (700, 300, 0, 50)


def test_extract_round_cost_usage_absent_usage():
    f = CCRResponseHandler._extract_round_cost_usage
    assert f({}, "anthropic") == (0, 0, 0, 0)
    assert f({"usage": None}, "openai") == (0, 0, 0, 0)
    assert f("not a dict", "anthropic") == (0, 0, 0, 0)


# --- handle_response publishes per-call dropped-round usage --------------


def _tool_use_response(rid, usage):
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": f"tu_{rid}",
                "name": CCR_TOOL_NAME,
                "input": {"hash": "a" * 24},
            }
        ],
        "stop_reason": "tool_use",
        "usage": usage,
    }


def _final_response(usage):
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": "final answer"}],
        "stop_reason": "end_turn",
        "usage": usage,
    }


def test_handle_response_publishes_per_call_dropped_round_usage(monkeypatch):
    """3 anthropic rounds (r1, r2 dropped; r3 final): the published 4-tuple
    equals the SUM of the two dropped rounds' cache-split usage only."""
    handler = CCRResponseHandler()
    monkeypatch.setattr(
        handler,
        "_execute_retrieval",
        lambda call: CCRToolResult(
            tool_call_id=call.tool_call_id,
            content="{}",
            success=True,
            items_retrieved=1,
            tokens_retrieved=10,
        ),
    )
    # Synthetic anthropic per-round usage (arbitrary, with cache fields).
    U1 = {
        "input_tokens": 100_000,
        "output_tokens": 500,
        "cache_read_input_tokens": 40_000,
        "cache_creation_input_tokens": 5_000,
    }
    U2 = {
        "input_tokens": 105_000,
        "output_tokens": 600,
        "cache_read_input_tokens": 42_000,
        "cache_creation_input_tokens": 6_000,
    }
    U3 = {
        "input_tokens": 110_000,
        "output_tokens": 800,
        "cache_read_input_tokens": 44_000,
        "cache_creation_input_tokens": 7_000,
    }

    call_count = [0]

    async def api_call(messages, tools):
        call_count[0] += 1
        if call_count[0] == 1:
            return _tool_use_response("r2", U2)
        return _final_response(U3)

    async def main():
        consume_pending_ccr_continuation_usage()
        final = await handler.handle_response(
            _tool_use_response("r1", U1),
            [{"role": "user", "content": "q"}],
            [],
            api_call,
            "anthropic",
        )
        return final, consume_pending_ccr_continuation_usage()

    final, published = asyncio.run(main())

    assert final["usage"]["input_tokens"] == U3["input_tokens"]  # final round returned
    assert published == (
        U1["input_tokens"] + U2["input_tokens"],
        U1["cache_read_input_tokens"] + U2["cache_read_input_tokens"],
        U1["cache_creation_input_tokens"] + U2["cache_creation_input_tokens"],
        U1["output_tokens"] + U2["output_tokens"],
    )


def test_no_continuation_publishes_nothing():
    async def main():
        consume_pending_ccr_continuation_usage()
        await CCRResponseHandler().handle_response(
            _final_response({"input_tokens": 50, "output_tokens": 20}),
            [{"role": "user", "content": "q"}],
            [],
            None,
            "anthropic",
        )
        return consume_pending_ccr_continuation_usage()

    assert asyncio.run(main()) is None


# --- outcome funnel fold -------------------------------------------------


class _Metrics:
    def __init__(self):
        self.requested = []

    async def record_failed(self, provider):
        pass

    async def record_request(self, **kwargs):
        self.requested.append(kwargs)


class _CostTracker:
    def __init__(self):
        self.recorded = []

    def record_tokens(self, model, tokens_saved, tokens_sent, **kwargs):
        self.recorded.append({"model": model, "tokens_sent": tokens_sent, **kwargs})


class _Handler:
    def __init__(self):
        self.metrics = _Metrics()
        self.cost_tracker = _CostTracker()


BASE_OUT, BASE_UIN, BASE_CR, BASE_CW = 800, 110_000, 30_000, 3_000
CT_UIN, CT_CR, CT_CW, CT_OUT = 205_000, 80_000, 11_000, 1_100  # dropped rounds (synthetic)


def _outcome(request_id="req-ccr-fold", **overrides):
    fields = {
        "request_id": request_id,
        "provider": "anthropic",
        "model": "m",
        "status_code": 200,
        "original_tokens": BASE_UIN,
        "optimized_tokens": BASE_UIN,
        "output_tokens": BASE_OUT,
        "tokens_saved": 0,
        "attempted_input_tokens": BASE_UIN,
        "uncached_input_tokens": BASE_UIN,
        "cache_read_tokens": BASE_CR,
        "cache_write_tokens": BASE_CW,
    }
    fields.update(overrides)
    return RequestOutcome(**fields)


def test_emit_request_outcome_folds_continuation_usage_into_cost_view():
    """The funnel adds each published bucket to the matching cost_tracker arg."""
    handler = _Handler()

    async def main():
        set_pending_ccr_continuation_usage((CT_UIN, CT_CR, CT_CW, CT_OUT))
        await emit_request_outcome(handler, _outcome())
        return consume_pending_ccr_continuation_usage()

    leftover = asyncio.run(main())

    rec = handler.cost_tracker.recorded[0]
    assert rec["uncached_tokens"] == BASE_UIN + CT_UIN
    assert rec["cache_read_tokens"] == BASE_CR + CT_CR
    assert rec["cache_write_tokens"] == BASE_CW + CT_CW
    assert rec["output_tokens"] == BASE_OUT + CT_OUT
    # tokens_sent is Headroom's own forwarded-bytes count, never a billed total.
    assert rec["tokens_sent"] == BASE_UIN
    # The TTL split is not invented for continuation rounds.
    assert rec["cache_write_5m_tokens"] == 0
    assert rec["cache_write_1h_tokens"] == 0
    assert leftover is None


def test_continuation_usage_does_not_inflate_the_savings_surface():
    """metrics.record_request feeds SavingsTracker, which credits cache_read as
    savings — CCR overhead must not raise the reported savings, so the fold stays
    off this call."""
    handler = _Handler()

    async def main():
        set_pending_ccr_continuation_usage((CT_UIN, CT_CR, CT_CW, CT_OUT))
        await emit_request_outcome(handler, _outcome())

    asyncio.run(main())

    rec = handler.metrics.requested[0]
    assert rec["cache_read_tokens"] == BASE_CR
    assert rec["cache_write_tokens"] == BASE_CW
    assert rec["uncached_input_tokens"] == BASE_UIN
    assert rec["output_tokens"] == BASE_OUT
    assert rec["input_tokens"] == BASE_UIN


def test_continuation_output_does_not_bias_output_shaper_estimator():
    """The shaper's counterfactual reads this turn's completion length; dropped
    continuation rounds are a different request's output."""
    from headroom.proxy import output_savings

    seen = []

    class _Recorder:
        def record_from_labels(self, labels, output_tokens):
            seen.append(output_tokens)

        def estimate_request_savings(self, labels, output_tokens):
            return 0

    original = output_savings.get_recorder
    output_savings.get_recorder = lambda: _Recorder()
    try:

        async def main():
            set_pending_ccr_continuation_usage((CT_UIN, CT_CR, CT_CW, CT_OUT))
            await emit_request_outcome(
                _Handler(),
                _outcome(transforms_applied=("output_shaper:arm=terse",)),
            )

        asyncio.run(main())
    finally:
        output_savings.get_recorder = original

    assert seen == [BASE_OUT]


def test_shielded_emit_does_not_leave_continuation_pending_for_the_next_outcome():
    """asyncio.shield copies the context, so the funnel's consume clears the copy.
    clear_pending_outcome_side_channels() is what keeps a second outcome from the
    same context from re-booking the same continuation rounds."""
    handler = _Handler()

    async def emit(outcome):
        # Mirrors HeadroomProxy._record_request_outcome.
        try:
            await asyncio.shield(emit_request_outcome(handler, outcome))
        finally:
            clear_pending_outcome_side_channels()

    async def main():
        set_pending_ccr_continuation_usage((CT_UIN, CT_CR, CT_CW, CT_OUT))
        await emit(_outcome("turn-1"))
        await emit(_outcome("turn-2"))

    asyncio.run(main())

    first, second = handler.cost_tracker.recorded
    assert first["uncached_tokens"] == BASE_UIN + CT_UIN
    assert second["uncached_tokens"] == BASE_UIN  # not booked twice
    assert second["cache_read_tokens"] == BASE_CR


def test_emit_request_outcome_5xx_does_not_fold_continuation():
    handler = _Handler()
    outcome = RequestOutcome(
        request_id="req-ccr-5xx",
        provider="anthropic",
        model="m",
        status_code=529,
        original_tokens=100,
        optimized_tokens=100,
        output_tokens=10,
        tokens_saved=0,
        attempted_input_tokens=100,
        uncached_input_tokens=100,
    )

    async def main():
        set_pending_ccr_continuation_usage((999_999, 999_999, 999_999, 999_999))
        await emit_request_outcome(handler, outcome)
        return consume_pending_ccr_continuation_usage()

    leftover = asyncio.run(main())
    assert handler.metrics.requested == []
    assert leftover is None


# --- end-to-end: handle_response -> emit in one task ---------------------


def test_end_to_end_handle_then_emit_folds_dropped_rounds(monkeypatch):
    """Full wiring in one task: handle_response publishes, emit consumes and folds
    into the cost view only — cost_tracker sees final + dropped rounds while the
    metrics/savings surface stays on the final round."""
    ccr = CCRResponseHandler()
    monkeypatch.setattr(
        ccr,
        "_execute_retrieval",
        lambda call: CCRToolResult(
            tool_call_id=call.tool_call_id,
            content="{}",
            success=True,
            items_retrieved=1,
            tokens_retrieved=10,
        ),
    )
    U1 = {
        "input_tokens": 100_000,
        "output_tokens": 500,
        "cache_read_input_tokens": 40_000,
        "cache_creation_input_tokens": 5_000,
    }
    U2 = {
        "input_tokens": 12_000,
        "output_tokens": 80,
        "cache_read_input_tokens": 1_000,
        "cache_creation_input_tokens": 0,
    }

    async def api_call(messages, tools):
        return _final_response(U2)

    async def main():
        consume_pending_ccr_continuation_usage()
        await ccr.handle_response(
            _tool_use_response("r1", U1),
            [{"role": "user", "content": "q"}],
            [],
            api_call,
            "anthropic",
        )
        # The handler would normally build the outcome from the final round.
        handler = _Handler()
        outcome = RequestOutcome(
            request_id="req-e2e",
            provider="anthropic",
            model="m",
            status_code=200,
            original_tokens=U2["input_tokens"],
            optimized_tokens=U2["input_tokens"],
            output_tokens=U2["output_tokens"],
            tokens_saved=0,
            attempted_input_tokens=U2["input_tokens"],
            uncached_input_tokens=U2["input_tokens"],
            cache_read_tokens=U2["cache_read_input_tokens"],
            cache_write_tokens=U2["cache_creation_input_tokens"],
        )
        await emit_request_outcome(handler, outcome)
        return handler.cost_tracker.recorded[0], handler.metrics.requested[0]

    cost_rec, metrics_rec = asyncio.run(main())
    # Cost view: final (U2) + dropped (U1) per bucket.
    assert cost_rec["uncached_tokens"] == U1["input_tokens"] + U2["input_tokens"]
    assert (
        cost_rec["cache_read_tokens"]
        == U1["cache_read_input_tokens"] + U2["cache_read_input_tokens"]
    )
    assert (
        cost_rec["cache_write_tokens"]
        == U1["cache_creation_input_tokens"] + U2["cache_creation_input_tokens"]
    )
    assert cost_rec["output_tokens"] == U1["output_tokens"] + U2["output_tokens"]
    # Savings / metrics view: final round only.
    assert metrics_rec["uncached_input_tokens"] == U2["input_tokens"]
    assert metrics_rec["cache_read_tokens"] == U2["cache_read_input_tokens"]
    assert metrics_rec["output_tokens"] == U2["output_tokens"]


# --- batch path: multiple CCR items accumulate, not overwrite ------------


def test_batch_two_ccr_items_both_reach_cost_view():
    """Two CCR-bearing batch items: both items' continuation overhead
    reaches cost_tracker.record_tokens, not just the last.

    Simulates BatchResultProcessor: multiple handle_response calls
    (each set_pending), then one emit_request_outcome (one consume).
    """
    handler = _Handler()
    handler.cost_tracker = _CostTracker()
    base_out, base_uin, base_cr, base_cw = 800, 110_000, 30_000, 3_000
    item1 = (205_000, 80_000, 11_000, 1_100)
    item2 = (100_000, 50_000, 5_000, 500)

    outcome = RequestOutcome(
        request_id="req-batch",
        provider="anthropic",
        model="m",
        status_code=200,
        original_tokens=base_uin,
        optimized_tokens=base_uin,
        output_tokens=base_out,
        tokens_saved=0,
        attempted_input_tokens=base_uin,
        uncached_input_tokens=base_uin,
        cache_read_tokens=base_cr,
        cache_write_tokens=base_cw,
    )

    async def main():
        consume_pending_ccr_continuation_usage()
        set_pending_ccr_continuation_usage(item1)
        set_pending_ccr_continuation_usage(item2)
        await emit_request_outcome(handler, outcome)
        return consume_pending_ccr_continuation_usage()

    leftover = asyncio.run(main())

    assert handler.cost_tracker.recorded, "cost_tracker.record_tokens was not called"
    ct = handler.cost_tracker.recorded[0]
    # Both items accumulated into cost_tracker, not just the last
    assert ct["uncached_tokens"] == base_uin + item1[0] + item2[0]
    assert ct["cache_read_tokens"] == base_cr + item1[1] + item2[1]
    assert ct["cache_write_tokens"] == base_cw + item1[2] + item2[2]
    assert ct["output_tokens"] == base_out + item1[3] + item2[3]
    # record_request stays on base outcome (cost-only fold)
    assert handler.metrics.requested[0]["uncached_input_tokens"] == base_uin
    # Consume cleared after emit
    assert leftover is None
