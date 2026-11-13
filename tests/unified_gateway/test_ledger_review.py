"""Deterministic reproductions retained from the independent batch-2 review."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from decimal import Decimal

import pytest

from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.context import GatewayPrincipal
from headroom.proxy.gateway.errors import GatewayAuthorizationError
from headroom.proxy.gateway.execution import GatewayOperation
from headroom.proxy.gateway.routing import RetryDecision
from headroom.proxy.gateway.runtime import GatewayRuntime
from headroom.proxy.gateway.usage import (
    CostEvaluation,
    conservative_cost_bound,
    evaluate_cost,
    normalize_usage,
)
from tests.unified_gateway.test_execution import environment, operation, runtime_snapshot
from tests.unified_gateway.test_ledger import charge, controller, request, wait_for_queue
from tests.unified_gateway.test_usage import priced_route, tariff


@pytest.mark.asyncio
async def test_c1_close_cannot_release_a_handed_off_sendable_attempt_unsent(monkeypatch):
    runtime = GatewayRuntime(runtime_snapshot(), environ=environment())
    op = operation(runtime)
    original = runtime.admission.reserve
    closing = []

    async def schedule_close_after_real_admission(admission_request):
        reservation = await original(admission_request)
        closing.append(asyncio.create_task(op.close("cancelled")))
        await asyncio.sleep(0)
        return reservation

    monkeypatch.setattr(runtime.admission, "reserve", schedule_close_after_real_admission)
    sent = False
    try:
        try:
            attempt = await op.start_attempt(
                op.route.credentials[0], CostEvaluation(reserved_upper_micro_usd=10)
            )
        except (GatewayAuthorizationError, asyncio.CancelledError):
            pass
        else:
            try:
                attempt.mark_sending()
            except ValueError:
                pass
            else:
                sent = True
                attempt.usage = charge(7)
        await asyncio.gather(*closing)
        state = runtime.admission.snapshot()
        assert not sent, (state, op.attempts)
        assert state.known_micro_usd == state.unresolved_micro_usd == 0
        assert runtime.admission.active_count == 0
        assert not runtime.active_work
    finally:
        await op.close("cancelled")
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_c1_sending_rejects_operation_close_before_attempt_close_runs():
    runtime = GatewayRuntime(runtime_snapshot(), environ=environment())
    op = operation(runtime)
    try:
        attempt = await op.start_attempt(op.route.credentials[0], CostEvaluation())
        closing = asyncio.create_task(op.close("cancelled"))
        await asyncio.sleep(0)
        try:
            with pytest.raises(ValueError):
                attempt.mark_sending()
        finally:
            await closing
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("qualified", [False, True])
async def test_c2_retry_qualification_is_checked_atomically_after_cap_reload(
    monkeypatch, tmp_path, qualified
):
    raw = runtime_snapshot().model_dump(mode="json")
    raw["routes"][0]["retry"]["max_attempts"] = 2
    if qualified:
        route = priced_route()
        raw["routes"][0]["pricing"] = route.pricing.model_dump(mode="json")
        raw["routes"][0]["model_bounds"] = route.model_bounds.model_dump(mode="json")
    runtime = GatewayRuntime(GatewayConfigSnapshot.model_validate(raw), environ=environment())
    op = operation(runtime)
    cost = (
        conservative_cost_bound(op.route, {}, qualified_contracts=frozenset({"fake-model-v1"}))
        if qualified
        else CostEvaluation(reserved_upper_micro_usd=5)
    )
    entered, resume = asyncio.Event(), asyncio.Event()
    original = runtime.admission.reserve

    async def pause_before_controller_transition(admission_request):
        entered.set()
        await resume.wait()
        return await original(admission_request)

    task = None
    try:
        first = await op.start_attempt(op.route.credentials[0], cost)
        await first.close("failed")
        monkeypatch.setattr(runtime.admission, "reserve", pause_before_controller_transition)
        task = asyncio.create_task(
            op.start_attempt(op.route.credentials[0], cost, retry=RetryDecision(True, "connect"))
        )
        await asyncio.wait_for(entered.wait(), 1)
        raw["admission"] = {"budget_usd": "1", "unknown_cost_policy": "block"}
        if qualified:
            raw["routes"][0]["pricing"]["revision"] = "new-tariff"
        path = tmp_path / "strict.json"
        path.write_text(json.dumps(raw))
        assert (await runtime.reload(path)).applied
        resume.set()
        if qualified:
            second = await task
            assert second.request.generation == 1
            assert second.request.pricing.revision == "fake-v1"
            assert runtime.generation == 2
        else:
            with pytest.raises(GatewayAuthorizationError, match="unknown_cost"):
                await task
            assert runtime.admission.active_count == 0
    finally:
        resume.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await op.close("cancelled")
        await runtime.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("settled", [False, True])
async def test_c2_unqualified_numeric_reservation_or_liability_blocks_enabling_cap(
    tmp_path, settled
):
    runtime = GatewayRuntime(runtime_snapshot(), environ=environment())
    op = operation(runtime)
    try:
        attempt = await op.start_attempt(
            op.route.credentials[0], CostEvaluation(reserved_upper_micro_usd=5)
        )
        if settled:
            attempt.mark_sending()
            await op.close("cancelled")
        path = tmp_path / "strict.json"
        path.write_text(
            runtime_snapshot(budget_usd="1", unknown_cost_policy="block").model_dump_json()
        )
        assert not (await runtime.reload(path)).applied
        assert runtime.generation == 1
    finally:
        await op.close("cancelled")
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_i1_pending_owners_cannot_consume_another_principals_reserved_slot():
    raw = runtime_snapshot(max_concurrency=2, queue_limit=0).model_dump(mode="json")
    first = raw["client_auth"]["principals"][0]
    first["admission"] = {"max_concurrency": 2, "reserved_concurrency": 1, "queue_limit": 0}
    other = {**first, "id": "other", "secret_ref": "env:OTHER_CLIENT_TOKEN"}
    raw["client_auth"]["principals"].append(other)
    runtime = GatewayRuntime(
        GatewayConfigSnapshot.model_validate(raw),
        environ={**environment(), "OTHER_CLIENT_TOKEN": "other-secret"},
    )
    owners = []
    try:
        a = operation(runtime)
        owners.append(a)
        await a.start_attempt(a.route.credentials[0], CostEvaluation())
        try:
            pending_a = operation(runtime)
        except GatewayAuthorizationError:
            pass  # A tenant-aware owner gate may reject this excess owner itself.
        else:
            owners.append(pending_a)
        generation = runtime.capture()
        b = GatewayOperation(
            runtime,
            generation=generation,
            principal=GatewayPrincipal(
                "other", frozenset(other["scopes"]), frozenset(other["routes"])
            ),
            route=generation.snapshot.routes[0],
            ingress_protocol="openai-responses",
            target_protocol="openai-responses",
            dispatch_plan=None,
        )
        owners.append(b)
        await b.start_attempt(b.route.credentials[0], CostEvaluation())
        assert runtime.admission.active_count == 2
    finally:
        await asyncio.gather(*(owner.close("cancelled") for owner in owners))
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_i2_same_principal_arrivals_cannot_bypass_a_busy_account_queue_head():
    ledger = controller(cap=None, maximum=3, accounts={"busy": 1, "free": 1})
    held = await ledger.reserve(request(account="busy"))
    older = asyncio.create_task(ledger.reserve(request(account="busy", timeout=1)))
    await wait_for_queue(ledger, 1)
    nonblocking = await ledger.try_reserve(request(account="free"))
    newer = None
    try:
        assert not nonblocking.allowed, (
            "newer account-free request bypassed its queued principal head"
        )
        newer = asyncio.create_task(ledger.reserve(request(account="free", timeout=1)))
        await wait_for_queue(ledger, 2)
        assert not newer.done()
        await held.release_unsent()
        first, second = await asyncio.gather(older, newer)
        await asyncio.gather(first.release_unsent(), second.release_unsent())
    finally:
        if nonblocking.reservation is not None:
            await nonblocking.reservation.release_unsent()
        for task in (older, newer):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await held.release_unsent()


@pytest.mark.parametrize(
    "protocol,initial,refined",
    [
        (
            "openai-chat",
            {"usage": {"prompt_tokens": 12, "completion_tokens": 3}},
            {
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 2},
                }
            },
        ),
        (
            "openai-responses",
            {"usage": {"input_tokens": 12, "output_tokens": 3}},
            {
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 3,
                    "input_tokens_details": {"cached_tokens": 2},
                }
            },
        ),
        (
            "gemini-generate",
            {"usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3}},
            {
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 3,
                    "cachedContentTokenCount": 2,
                }
            },
        ),
        (
            "vertex-generate",
            {"usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3}},
            {
                "usageMetadata": {
                    "promptTokenCount": 12,
                    "candidatesTokenCount": 3,
                    "cachedContentTokenCount": 2,
                }
            },
        ),
    ],
)
def test_i3_cache_refinement_preserves_coherent_authoritative_totals(protocol, initial, refined):
    first = normalize_usage(protocol, initial)
    merged = normalize_usage(protocol, refined, previous=first)
    assert (merged.input_tokens, merged.cache_read_tokens, merged.output_tokens) == (10, 2, 3)
    assert evaluate_cost(merged, tariff()).known_micro_usd == 34
    assert normalize_usage(protocol, refined, previous=merged) == merged
    assert normalize_usage(protocol, initial, previous=merged) == merged


@pytest.mark.parametrize(
    "protocol,initial,cache_only",
    [
        (
            "openai-chat",
            {"usage": {"prompt_tokens": 12, "completion_tokens": 3}},
            {"usage": {"prompt_tokens_details": {"cached_tokens": 2}}},
        ),
        (
            "openai-responses",
            {"usage": {"input_tokens": 12, "output_tokens": 3}},
            {"usage": {"input_tokens_details": {"cached_tokens": 2}}},
        ),
        (
            "gemini-generate",
            {"usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3}},
            {"usageMetadata": {"cachedContentTokenCount": 2}},
        ),
        (
            "vertex-generate",
            {"usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 3}},
            {"usageMetadata": {"cachedContentTokenCount": 2}},
        ),
    ],
)
def test_i3_cache_only_refinement_uses_the_retained_inclusive_total(protocol, initial, cache_only):
    previous = normalize_usage(protocol, initial)
    refined = normalize_usage(protocol, cache_only, previous=previous)
    assert (refined.input_tokens, refined.cache_read_tokens, refined.output_tokens) == (10, 2, 3)
    assert evaluate_cost(refined, tariff()).known_micro_usd == 34


@pytest.mark.asyncio
async def test_i4_finalization_freezes_one_observation_for_ledger_outcome_and_metrics():
    runtime = GatewayRuntime(runtime_snapshot(), environ=environment())
    op = operation(runtime)
    try:
        attempt = await op.start_attempt(
            op.route.credentials[0], CostEvaluation(reserved_upper_micro_usd=10)
        )
        attempt.mark_sending()
        attempt.usage = replace(charge(4), availability="partial")
        late_rejected = False
        async with runtime.admission._condition:
            closing = asyncio.create_task(attempt.close("success"))
            while attempt.reservation._finalizer is None:
                await asyncio.sleep(0)
            try:
                attempt.usage = charge(7)
            except ValueError:
                late_rejected = True
        await closing
        await op.close("success")
        ledger = runtime.admission.snapshot()
        outcome = op.attempts[0]
        totals = runtime.observability.totals()
        assert late_rejected, (ledger, outcome, totals)
        assert (ledger.known_micro_usd, ledger.unresolved_micro_usd) == (4, 6)
        assert outcome.cost.known_micro_usd == totals["known_micro_usd"] == 4
        assert not outcome.cost.complete
        assert outcome.usage.currency_charge == Decimal("0.000004")
        assert totals["unknown_usage_attempts"] == 1
    finally:
        await op.close("cancelled")
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_i5_cancelled_close_callback_commits_one_logical_terminal_result():
    runtime = GatewayRuntime(runtime_snapshot(), environ=environment())
    op = operation(runtime)
    try:
        attempt = await op.start_attempt(
            op.route.credentials[0], CostEvaluation(reserved_upper_micro_usd=10)
        )
        attempt.mark_sending()

        async def cancelled_close():
            raise asyncio.CancelledError

        attempt.upstream_close = cancelled_close
        try:
            await op.close("success")
        except asyncio.CancelledError:
            pass
        assert op.terminal == "cancelled"
        assert runtime.observability.totals()["logical_requests"] == 1
        assert runtime.observability.totals()["attempts"] == 1
        assert runtime.admission.snapshot().unresolved_micro_usd == 10
        assert not runtime.active_work
        await op.close("cancelled")
        assert runtime.observability.totals()["logical_requests"] == 1
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_i5_cancelling_close_caller_still_cancels_only_the_caller():
    runtime = GatewayRuntime(runtime_snapshot(), environ=environment())
    op = operation(runtime)
    entered, resume = asyncio.Event(), asyncio.Event()
    try:
        attempt = await op.start_attempt(
            op.route.credentials[0], CostEvaluation(reserved_upper_micro_usd=10)
        )
        attempt.mark_sending()

        async def blocked_close():
            entered.set()
            await resume.wait()

        attempt.upstream_close = blocked_close
        caller = asyncio.create_task(op.close("cancelled"))
        await asyncio.wait_for(entered.wait(), 1)
        caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await caller
        resume.set()
        await op.close("cancelled")
        assert op.terminal == "cancelled"
        assert runtime.observability.totals()["logical_requests"] == 1
    finally:
        resume.set()
        await op.close("cancelled")
        await runtime.shutdown()
