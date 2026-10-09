"""Unit tests for ``headroom.proxy.gateway_turn`` internals — no HTTP.

Three things have to hold or the two-half contract is unsafe:

* ``SuspendedHookRunner`` parks a hook coroutine between two HTTP requests
  and never leaks a task, whatever the hook does (re-drives, finishes at
  once, raises, or is cancelled mid-suspend);
* ``PendingTurnRegistry`` expires and evicts without losing a deferred
  outcome draft;
* ``normalize_usage`` reads every provider shape a gateway can relay and
  never reports a cache signal that was not there.
"""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest

from headroom.proxy import gateway_turn as gt
from headroom.proxy.gateway_turn import (
    GatewayCapabilities,
    GatewayRequestError,
    NormalizedUsage,
    PendingTurn,
    PendingTurnRegistry,
    SuspendedHookRunner,
    compact_gateway_tools,
    compute_obligations,
    normalize_usage,
    parse_gateway_block,
)
from headroom.proxy.outcome import RequestOutcome
from headroom.proxy.turn_hooks import TurnContext, clear_turn_hooks, register_turn_hook


@pytest.fixture(autouse=True)
def _no_hooks():
    clear_turn_hooks()
    yield
    clear_turn_hooks()


def _ctx(tools: Any = None) -> TurnContext:
    return TurnContext(
        provider="openai",
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=tools,
    )


# --------------------------------------------------------------------------- #
# SuspendedHookRunner                                                          #
# --------------------------------------------------------------------------- #


class _RedriveN:
    """Re-drive ``n`` times, then return the last provider response."""

    name = "redrive_n"
    stream_safe = False

    def __init__(self, n: int) -> None:
        self.n = n
        self.seen: list[dict[str, Any]] = []

    async def on_response(self, ctx: Any, response: Any, call_model: Any) -> Any:
        current = response
        for i in range(self.n):
            ctx.tools = [{"name": f"reloaded_{i}"}]
            current = await call_model(ctx.messages + [{"role": "assistant", "content": str(i)}])
            self.seen.append(current)
        return current


class _NoRedrive:
    name = "no_redrive"
    stream_safe = False

    async def on_response(self, ctx: Any, response: Any, call_model: Any) -> Any:
        return None


class _Replaces:
    name = "replaces"
    stream_safe = False

    async def on_response(self, ctx: Any, response: Any, call_model: Any) -> Any:
        return {"replaced": True, "from": response.get("id")}


class _Raises:
    name = "raises"
    stream_safe = False

    async def on_response(self, ctx: Any, response: Any, call_model: Any) -> Any:
        raise RuntimeError("boom")


async def _baseline() -> int:
    await asyncio.sleep(0)
    return len(asyncio.all_tasks())


def test_runner_start_suspends_then_resume_completes() -> None:
    async def scenario() -> None:
        base = await _baseline()
        hook = _RedriveN(1)
        register_turn_hook(hook)
        ctx = _ctx(tools=[{"name": "t"}])
        runner = SuspendedHookRunner(ctx)

        step = await runner.start({"id": "r1"})
        assert step.is_redrive
        assert step.messages is not None and step.messages[-1]["content"] == "0"
        assert step.tools == [{"name": "reloaded_0"}]
        assert runner.state == "awaiting_redrive"
        # The hook task is parked — exactly one extra task while suspended.
        assert len(asyncio.all_tasks()) == base + 1

        final = {"id": "r2"}
        step2 = await runner.resume(final)
        assert step2.kind == "done"
        # The hook returned the very object the gateway posted — nothing to
        # send back, the gateway forwards what it already holds.
        assert step2.response is None
        assert runner.state == "done"
        assert runner.rounds == 1
        assert hook.seen == [final]
        assert await _baseline() == base

    asyncio.run(scenario())


def test_runner_multiple_rounds() -> None:
    async def scenario() -> None:
        base = await _baseline()
        register_turn_hook(_RedriveN(3))
        runner = SuspendedHookRunner(_ctx())
        step = await runner.start({"id": "r0"})
        rounds = 0
        while step.is_redrive:
            rounds += 1
            step = await runner.resume({"id": f"r{rounds}"})
        assert rounds == 3
        assert step.kind == "done" and step.response is None
        assert await _baseline() == base

    asyncio.run(scenario())


def test_runner_cancel_mid_suspend_leaves_no_tasks() -> None:
    async def scenario() -> None:
        base = await _baseline()
        register_turn_hook(_RedriveN(2))
        runner = SuspendedHookRunner(_ctx())
        step = await runner.start({"id": "r0"})
        assert step.is_redrive
        assert len(asyncio.all_tasks()) == base + 1
        await runner.cancel()
        assert runner.state == "done"
        assert runner._task is not None and runner._task.done()
        assert await _baseline() == base
        # Resuming a cancelled runner is a programming error, not a hang.
        with pytest.raises(RuntimeError):
            await runner.resume({"id": "late"})

    asyncio.run(scenario())


def test_runner_hook_without_redrive_is_done_immediately() -> None:
    async def scenario() -> None:
        base = await _baseline()
        register_turn_hook(_NoRedrive())
        runner = SuspendedHookRunner(_ctx())
        step = await runner.start({"id": "r0"})
        assert step.kind == "done" and step.response is None
        assert runner.rounds == 0
        assert await _baseline() == base

    asyncio.run(scenario())


def test_runner_replacement_is_returned() -> None:
    async def scenario() -> None:
        register_turn_hook(_Replaces())
        runner = SuspendedHookRunner(_ctx())
        step = await runner.start({"id": "orig"})
        assert step.kind == "done"
        assert step.response == {"replaced": True, "from": "orig"}

    asyncio.run(scenario())


def test_runner_hook_that_raises_is_done_with_none() -> None:
    async def scenario() -> None:
        base = await _baseline()
        register_turn_hook(_Raises())
        runner = SuspendedHookRunner(_ctx())
        step = await runner.start({"id": "r0"})
        assert step.kind == "done" and step.response is None
        assert await _baseline() == base

    asyncio.run(scenario())


def test_runner_broken_runner_function_is_done_with_none() -> None:
    """``run_response_hooks`` swallows hook errors; a broken runner itself
    must still resolve to a done step rather than an exception."""

    async def broken(ctx: Any, response: Any, call_model: Any) -> Any:
        raise ValueError("runner exploded")

    async def scenario() -> None:
        base = await _baseline()
        runner = SuspendedHookRunner(_ctx(), run=broken)
        step = await runner.start({"id": "r0"})
        assert step.kind == "done" and step.response is None
        assert await _baseline() == base

    asyncio.run(scenario())


def test_runner_no_hooks_registered_is_inert() -> None:
    async def scenario() -> None:
        runner = SuspendedHookRunner(_ctx())
        step = await runner.start({"id": "r0"})
        assert step.kind == "done" and step.response is None

    asyncio.run(scenario())


def test_runner_start_twice_is_an_error() -> None:
    async def scenario() -> None:
        register_turn_hook(_NoRedrive())
        runner = SuspendedHookRunner(_ctx())
        await runner.start({"id": "r0"})
        with pytest.raises(RuntimeError):
            await runner.start({"id": "r1"})

    asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# PendingTurnRegistry                                                          #
# --------------------------------------------------------------------------- #


def _outcome(request_id: str = "hr_test") -> RequestOutcome:
    return RequestOutcome(
        request_id=request_id,
        provider="compress",
        model="gpt-4o",
        original_tokens=100,
        optimized_tokens=50,
        output_tokens=0,
        tokens_saved=50,
        attempted_input_tokens=100,
    )


def _turn(turn_id: str, now: float, ttl: float, draft: RequestOutcome | None = None) -> PendingTurn:
    return PendingTurn(
        turn_id=turn_id,
        session_key=None,
        session_id=None,
        provider="openai",
        model="gpt-4o",
        body={"model": "gpt-4o", "messages": []},
        ctx=None,
        obligations=["relay_usage"],
        outcome_draft=draft,
        created_at=now,
        deadline=now + ttl,
    )


def test_registry_ttl_expiry_calls_on_expire_with_draft() -> None:
    clock = [1000.0]
    expired: list[PendingTurn] = []
    reg = PendingTurnRegistry(
        ttl_seconds=10, max_entries=100, on_expire=expired.append, clock=lambda: clock[0]
    )
    reg.register(_turn("a", clock[0], 10, _outcome("a")))
    reg.register(_turn("b", clock[0], 10))
    assert len(reg) == 2
    assert reg.get("a") is not None
    clock[0] += 11
    # Lazy sweep on lookup: the expired turn is gone and reported once.
    assert reg.get("b") is None
    assert len(reg) == 0
    assert sorted(t.turn_id for t in expired) == ["a", "b"]
    assert expired[0].outcome_draft is not None or expired[1].outcome_draft is not None
    assert reg.stats()["expired"] == 2


def test_registry_lru_eviction_at_capacity() -> None:
    evicted: list[str] = []
    reg = PendingTurnRegistry(
        ttl_seconds=100,
        max_entries=2,
        on_expire=lambda t: evicted.append(t.turn_id),
        clock=lambda: 0.0,
    )
    for tid in ("a", "b", "c"):
        reg.register(_turn(tid, 0.0, 100))
    assert len(reg) == 2
    assert evicted == ["a"]
    assert reg.get("a") is None and reg.get("c") is not None
    assert reg.stats()["evicted"] == 1


def test_registry_pop_and_in_flight_turns_survive_sweep() -> None:
    clock = [0.0]
    expired: list[str] = []
    reg = PendingTurnRegistry(
        ttl_seconds=5,
        max_entries=10,
        on_expire=lambda t: expired.append(t.turn_id),
        clock=lambda: clock[0],
    )
    t = _turn("a", 0.0, 5)
    reg.register(t)
    t.in_flight = True
    clock[0] = 100.0
    assert reg.sweep() == []  # being driven right now: never yanked mid-step
    t.in_flight = False
    assert [x.turn_id for x in reg.sweep()] == ["a"]
    assert reg.pop("a") is None
    assert expired == ["a"]


def test_registry_on_expire_exceptions_are_swallowed() -> None:
    def bad(turn: PendingTurn) -> None:
        raise RuntimeError("nope")

    clock = [0.0]
    reg = PendingTurnRegistry(ttl_seconds=1, max_entries=10, on_expire=bad, clock=lambda: clock[0])
    reg.register(_turn("a", 0.0, 1))
    clock[0] = 5.0
    assert reg.sweep()  # did not raise


def test_registry_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_TURN_TTL_SECONDS", "7.5")
    monkeypatch.setenv("HEADROOM_GATEWAY_MAX_PENDING_TURNS", "3")
    reg = PendingTurnRegistry()
    assert reg.ttl_seconds == 7.5 and reg.max_entries == 3
    monkeypatch.setenv("HEADROOM_GATEWAY_TURN_TTL_SECONDS", "garbage")
    assert PendingTurnRegistry().ttl_seconds == gt.DEFAULT_TURN_TTL_SECONDS


def test_on_turn_expired_records_draft_on_the_loop_and_cancels_runner() -> None:
    recorded: list[RequestOutcome] = []

    class _Proxy:
        gateway_turns = PendingTurnRegistry(ttl_seconds=1, max_entries=10)

        async def _record_request_outcome(self, outcome: RequestOutcome) -> None:
            recorded.append(outcome)

    proxy = _Proxy()

    async def scenario() -> None:
        base = await _baseline()
        register_turn_hook(_RedriveN(1))
        runner = SuspendedHookRunner(_ctx())
        await runner.start({"id": "r0"})
        turn = _turn("a", 0.0, 1, _outcome("draft"))
        turn.runner = runner
        gt.on_turn_expired(proxy, turn)
        # Both the record and the cancel are tasks; let them run.
        for _ in range(5):
            await asyncio.sleep(0)
        assert [o.request_id for o in recorded] == ["draft"]
        assert turn.outcome_draft is None
        assert runner.state == "done"
        assert await _baseline() == base

    asyncio.run(scenario())


def test_on_turn_expired_without_loop_parks_the_draft_then_flushes() -> None:
    recorded: list[RequestOutcome] = []

    class _Proxy:
        gateway_turns = PendingTurnRegistry(ttl_seconds=1, max_entries=10)

        async def _record_request_outcome(self, outcome: RequestOutcome) -> None:
            recorded.append(outcome)

    proxy = _Proxy()
    # No running loop: the draft is parked, not lost.
    gt.on_turn_expired(proxy, _turn("a", 0.0, 1, _outcome("orphan")))
    assert [o.request_id for o in proxy.gateway_turns.orphaned_outcomes] == ["orphan"]
    assert recorded == []

    async def scenario() -> None:
        gt.on_turn_expired(proxy, _turn("b", 0.0, 1, _outcome("fresh")))
        for _ in range(5):
            await asyncio.sleep(0)

    asyncio.run(scenario())
    assert sorted(o.request_id for o in recorded) == ["fresh", "orphan"]
    assert proxy.gateway_turns.orphaned_outcomes == []


# --------------------------------------------------------------------------- #
# normalize_usage                                                              #
# --------------------------------------------------------------------------- #


def test_normalize_anthropic_shape() -> None:
    u = normalize_usage(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 60,
            "cache_creation_input_tokens": 10,
        }
    )
    assert u == NormalizedUsage(100, 20, 60, 10)
    assert u.has_cache_signal and u.should_apply


def test_normalize_openai_chat_shape() -> None:
    u = normalize_usage(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 64},
        }
    )
    # OpenAI-shaped: prompt_tokens already counts the cached prefix.
    assert u == NormalizedUsage(100, 20, 64, None, input_includes_cache=True)
    assert u.has_cache_signal and u.should_apply


def test_normalize_openai_responses_shape() -> None:
    u = normalize_usage(
        {"input_tokens": 10, "output_tokens": 2, "input_tokens_details": {"cached_tokens": 8}}
    )
    # Responses names the field ``input_tokens`` like Anthropic but, like the
    # rest of OpenAI, already counts the cached prefix inside it.
    assert u == NormalizedUsage(10, 2, 8, None, input_includes_cache=True)


def test_normalize_kong_flat_shape() -> None:
    u = normalize_usage({"prompt_tokens": 50, "completion_tokens": 5, "cached_tokens": 40})
    assert u == NormalizedUsage(50, 5, 40, None, input_includes_cache=True)


def test_normalize_nested_usage_descends_once() -> None:
    u = normalize_usage({"id": "x", "usage": {"prompt_tokens": 7, "completion_tokens": 3}})
    assert u == NormalizedUsage(7, 3, None, None, input_includes_cache=True)
    assert not u.has_cache_signal and not u.should_apply


def test_normalize_no_cache_fields_is_accepted_but_not_applied() -> None:
    u = normalize_usage({"prompt_tokens": 12345})
    assert u.input_tokens == 12345
    assert not u.has_cache_signal
    assert not u.should_apply


def test_normalize_zero_only_read_is_no_signal_but_both_zero_applies() -> None:
    assert not normalize_usage({"prompt_tokens_details": {"cached_tokens": 0}}).should_apply
    assert normalize_usage(
        {"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
    ).should_apply


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": True},
        {"prompt_tokens": -1},
        {"cache_read_input_tokens": False},
        {"prompt_tokens_details": {"cached_tokens": -5}},
        {"cached_tokens": "12"},
        {"output_tokens": 1.5},
    ],
)
def test_normalize_rejects_bools_negatives_and_non_ints(usage: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        normalize_usage(usage)


def test_normalize_empty_and_odd_shapes_never_raise() -> None:
    for usage in ({}, {"usage": None}, {"prompt_tokens_details": "nope"}, {"input_tokens": None}):
        u = normalize_usage(usage)
        assert not u.has_cache_signal


def test_merged_with_sums_billed_and_keeps_latest_cache() -> None:
    a = NormalizedUsage(100, 10, 60, None)
    b = NormalizedUsage(150, 20, 90, None)
    m = a.merged_with(b)
    assert (m.input_tokens, m.output_tokens, m.cache_read) == (250, 30, 90)
    c = a.merged_with(NormalizedUsage(1, 1, None, None))
    assert c.cache_read == 60  # a signal-free round does not erase the last signal


# --------------------------------------------------------------------------- #
# Capabilities, obligations, outcome completion, compaction                   #
# --------------------------------------------------------------------------- #


def test_parse_gateway_block_modes() -> None:
    assert parse_gateway_block({"messages": []}) is None
    caps = parse_gateway_block({"gateway": {}})
    assert caps == GatewayCapabilities(False, False, True, None)
    caps = parse_gateway_block(
        {
            "gateway": {
                "can_redrive": True,
                "session_affinity": False,
                "plugin_version": "0.1.0",
                "x": 1,
            }
        }
    )
    assert caps is not None and caps.can_redrive and not caps.session_affinity
    assert not caps.redrive_allowed
    assert caps.plugin_version == "0.1.0"
    assert caps.echo() == {
        "can_redrive": True,
        "can_relay_response": False,
        "session_affinity": False,
    }
    for bad in ({"gateway": []}, {"gateway": None}, {"gateway": {"can_redrive": "yes"}}):
        with pytest.raises(GatewayRequestError):
            parse_gateway_block(bad)


def test_compute_obligations() -> None:
    both = GatewayCapabilities(can_redrive=True, can_relay_response=True)
    assert compute_obligations(both, redrive_armed=True) == ["redrive", "relay_usage"]
    assert compute_obligations(both, redrive_armed=False) == ["relay_usage"]
    no_affinity = dataclasses.replace(both, session_affinity=False)
    assert compute_obligations(no_affinity, redrive_armed=True) == ["relay_usage"]
    assert compute_obligations(GatewayCapabilities(), redrive_armed=True) == []


def test_complete_outcome_fills_only_existing_fields() -> None:
    draft = dataclasses.replace(_outcome(), total_latency_ms=5.0)
    done = gt.complete_outcome(draft, NormalizedUsage(120, 30, 80, 4), status=200, latency_ms=250)
    assert done.output_tokens == 30
    # Anthropic shape: input_tokens is the uncached remainder, so the billed
    # prompt is that plus the two cache counters.
    assert done.provider_input_tokens == 204
    assert done.uncached_input_tokens == 120
    assert done.optimized_tokens == 50  # never overwritten by the provider count
    assert (done.cache_read_tokens, done.cache_write_tokens) == (80, 4)
    assert done.total_latency_ms == 255.0
    assert done.status_code == 200
    failed = gt.complete_outcome(draft, None, status=529, latency_ms=None)
    assert failed.status_code == 529 and failed.output_tokens == 0


def test_complete_outcome_bills_the_whole_prompt_on_a_cached_anthropic_turn() -> None:
    """A warm agent turn is ~all cache read; billing only the uncached sliver
    made ``/stats`` divide savings by a denominator tens of times too small."""
    usage = normalize_usage(
        {
            "input_tokens": 3,
            "output_tokens": 210,
            "cache_read_input_tokens": 41_000,
            "cache_creation_input_tokens": 1_200,
        }
    )
    done = gt.complete_outcome(_outcome(), usage, status=200, latency_ms=100)
    assert done.provider_input_tokens == 42_203
    assert done.uncached_input_tokens == 3


def test_complete_outcome_does_not_double_count_openai_responses_cached_tokens() -> None:
    """The Responses shape uses Anthropic's field name for an OpenAI-style total."""
    usage = normalize_usage(
        {
            "input_tokens": 12_000,
            "output_tokens": 40,
            "input_tokens_details": {"cached_tokens": 9_000},
        }
    )
    done = gt.complete_outcome(_outcome(), usage, status=200, latency_ms=100)
    assert done.provider_input_tokens == 12_000
    assert done.uncached_input_tokens == 3_000


def test_complete_outcome_does_not_double_count_openai_cached_tokens() -> None:
    """OpenAI's prompt_tokens already contains cached_tokens."""
    usage = normalize_usage(
        {
            "prompt_tokens": 12_000,
            "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 9_000},
        }
    )
    assert usage.input_includes_cache is True
    done = gt.complete_outcome(_outcome(), usage, status=200, latency_ms=100)
    assert done.provider_input_tokens == 12_000
    assert done.uncached_input_tokens == 3_000
    assert done.cache_read_tokens == 9_000


def test_compact_gateway_tools_is_deterministic_and_isolated() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get",
                "description": "Get   things\n\n  now.",
                "parameters": {"$schema": "x", "title": "T", "type": "object", "properties": {}},
            },
        }
    ]
    a, ta = compact_gateway_tools(tools)
    b, tb = compact_gateway_tools(tools)
    assert a == b and ta == tb == ["tool_schema_compaction"]
    assert a is not b  # a hook mutating one turn's tools cannot touch the next
    a[0]["function"]["name"] = "mutated"
    c, _ = compact_gateway_tools(tools)
    assert c[0]["function"]["name"] == "get"
    assert compact_gateway_tools(None) == (None, [])
    assert compact_gateway_tools([]) == ([], [])


def test_request_transformer_runs_hooks_and_arms_redrive() -> None:
    class _Fold:
        name = "fold"
        stream_safe = True

        def on_request(self, ctx: Any) -> None:
            ctx.messages[-1]["content"] = "x"

    class _Redriver:
        name = "redriver"
        stream_safe = False

        def on_request(self, ctx: Any) -> None:
            ctx.tools = [t for t in ctx.tools if t["name"] != "big"]

        async def on_response(self, ctx: Any, response: Any, call_model: Any) -> Any:
            return None

    register_turn_hook(_Fold())
    register_turn_hook(_Redriver())
    tools = [{"name": "big", "description": "b" * 500}, {"name": "small"}]
    tags: dict[str, Any] = {}
    tr = gt.RequestTransformer(
        provider="openai",
        model_name="gpt-4o",
        tools=tools,
        config=None,
        tags=tags,
        caps=GatewayCapabilities(can_redrive=True),
    )
    out = tr.run([{"role": "user", "content": "a much longer message than x"}])
    assert out[-1]["content"] == "x"
    assert tr.result is not None
    assert tr.result.redrive_armed
    assert [t["name"] for t in tr.result.tools] == ["small"]
    assert "turn_hook" in tr.result.transforms
    assert any(t.startswith("turn_hook:tools:") for t in tr.result.transforms)
    assert tags["turn_hook_tools_saved_tokens"] > 0

    # Without redrive the re-driving hook is skipped entirely: tools intact,
    # nothing armed. No shrink without reload.
    tr2 = gt.RequestTransformer(
        provider="openai",
        model_name="gpt-4o",
        tools=tools,
        config=None,
        tags={},
        caps=GatewayCapabilities(can_redrive=False),
    )
    tr2.run([{"role": "user", "content": "hello there"}])
    assert tr2.result is not None
    assert not tr2.result.redrive_armed
    assert [t["name"] for t in tr2.result.tools] == ["big", "small"]


def test_build_provider_body_strips_control_keys() -> None:
    body = {
        "model": "gpt-4o",
        "messages": [1],
        "config": {"session_id": "s"},
        "gateway": {},
        "token_budget": 10,
        "temperature": 0.2,
        "tools": None,
    }
    out = gt.build_provider_body(body, [2], None)
    assert out == {"model": "gpt-4o", "messages": [2], "temperature": 0.2, "tools": None}
    assert gt.build_provider_body(body, [2], [{"name": "t"}])["tools"] == [{"name": "t"}]


# --------------------------------------------------------------------------- #
# CCR redrive (config.mode="ccr")                                              #
# --------------------------------------------------------------------------- #


def _transform_result(tools: Any = None) -> gt.RequestTransformResult:
    ctx = _ctx(tools=tools)
    return gt.RequestTransformResult(
        messages=ctx.messages, tools=tools, ctx=ctx, transforms=[], redrive_armed=False
    )


def test_arm_ccr_redrive_gates_and_injects_once() -> None:
    caps = GatewayCapabilities(can_redrive=True)
    r = _transform_result(tools=[{"type": "function", "function": {"name": "get"}}])
    assert not gt.arm_ccr_redrive(r, provider="openai", caps=caps, mode=None, ccr_hashes=["a"])
    assert not gt.arm_ccr_redrive(r, provider="openai", caps=caps, mode="ccr", ccr_hashes=[])
    assert not gt.arm_ccr_redrive(
        r, provider="openai", caps=GatewayCapabilities(), mode="ccr", ccr_hashes=["a"]
    )
    assert r.tools is not None and len(r.tools) == 1 and r.transforms == []

    assert gt.arm_ccr_redrive(r, provider="openai", caps=caps, mode="ccr", ccr_hashes=["a"])
    names = [t["function"]["name"] for t in r.tools]
    assert names == ["get", "headroom_retrieve"]
    assert r.ctx is not None and r.ctx.tools is r.tools  # live tools for the re-drive
    assert r.transforms == ["ccr_tool_injected"]
    # Already present: no duplicate, no second transform marker.
    assert gt.arm_ccr_redrive(r, provider="openai", caps=caps, mode="ccr", ccr_hashes=["a"])
    assert [t["function"]["name"] for t in r.tools] == ["get", "headroom_retrieve"]
    assert r.transforms == ["ccr_tool_injected"]

    # No tools at all: the CCR tool becomes the whole list, Anthropic shape.
    r2 = _transform_result(tools=None)
    assert gt.arm_ccr_redrive(r2, provider="anthropic", caps=caps, mode="ccr", ccr_hashes=["a"])
    from headroom.ccr.tool_injection import create_ccr_tool_definition

    assert r2.tools == [create_ccr_tool_definition("anthropic")]


class _FakeCCRHandler:
    """Stands in for CCRResponseHandler: one retrieval round via api_call_fn."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict[str, Any]], Any]] = []

    def has_ccr_tool_calls(self, response: dict[str, Any], provider: str) -> bool:
        return bool(response.get("ccr"))

    async def handle_response(self, response, messages, tools, api_call_fn, provider="openai"):
        msgs = list(messages) + [{"role": "tool", "content": "original bytes"}]
        self.calls.append((msgs, tools))
        return await api_call_fn(msgs, tools)


def test_response_runner_drives_ccr_then_hooks_through_one_suspension() -> None:
    class _Proxy:
        ccr_response_handler = _FakeCCRHandler()

    proxy = _Proxy()
    seen: list[str] = []

    class _Observer:
        name = "observer"
        stream_safe = False

        async def on_response(self, ctx: Any, response: Any, call_model: Any) -> Any:
            seen.append(response["id"])
            return None

    async def scenario() -> None:
        base = await _baseline()
        register_turn_hook(_Observer())
        tools = [{"type": "function", "function": {"name": "headroom_retrieve"}}]
        turn = _turn("t", 0.0, 10)
        turn.ctx = _ctx(tools=tools)
        turn.obligations = ["redrive"]
        turn.ccr_armed = True
        runner = SuspendedHookRunner(turn.ctx, run=gt.make_response_runner(proxy, turn))

        step = await runner.start({"id": "r0", "ccr": True})
        assert step.is_redrive
        assert step.messages is not None and step.messages[-1]["content"] == "original bytes"
        assert step.tools == tools
        final = {"id": "r1"}
        step2 = await runner.resume(final)
        assert step2.kind == "done" and step2.response is None
        # CCR ran first, then the hook saw the CCR-resolved response only.
        assert seen == ["r1"]
        assert proxy.ccr_response_handler.calls[0][1] == tools
        assert await _baseline() == base

    asyncio.run(scenario())


def test_response_runner_without_ccr_calls_is_hooks_only() -> None:
    class _Proxy:
        ccr_response_handler = _FakeCCRHandler()

    async def scenario() -> None:
        turn = _turn("t", 0.0, 10)
        turn.ctx = _ctx()
        turn.ccr_armed = True
        runner = SuspendedHookRunner(turn.ctx, run=gt.make_response_runner(_Proxy(), turn))
        step = await runner.start({"id": "plain"})
        assert step.kind == "done" and step.response is None

    asyncio.run(scenario())


def test_response_runner_ccr_failure_falls_back_to_the_response() -> None:
    class _Broken(_FakeCCRHandler):
        async def handle_response(self, *a, **kw):
            raise RuntimeError("store down")

    class _Proxy:
        ccr_response_handler = _Broken()

    async def scenario() -> None:
        base = await _baseline()
        turn = _turn("t", 0.0, 10)
        turn.ctx = _ctx()
        turn.ccr_armed = True
        runner = SuspendedHookRunner(turn.ctx, run=gt.make_response_runner(_Proxy(), turn))
        step = await runner.start({"id": "r0", "ccr": True})
        assert step.kind == "done" and step.response is None
        assert await _baseline() == base

    asyncio.run(scenario())
