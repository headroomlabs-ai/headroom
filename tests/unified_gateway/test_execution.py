from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from decimal import Decimal
from importlib import import_module, util

import pytest

from headroom.proxy.gateway.admission import AdmissionRequest
from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.context import GatewayPrincipal
from headroom.proxy.gateway.errors import GatewayAuthorizationError
from headroom.proxy.gateway.observability import GatewayEvent, GatewayObservability
from headroom.proxy.gateway.runtime import GatewayRuntime
from headroom.proxy.gateway.usage import CostEvaluation, UsageObservation
from tests.unified_gateway.accounting_fixtures import qualified_request
from tests.unified_gateway.test_admission import EXAMPLE


def runtime_snapshot(**admission):
    raw = json.loads(EXAMPLE.read_text())
    raw["admission"] = admission
    return GatewayConfigSnapshot.model_validate(raw)


def environment():
    return {
        "HEADROOM_GATEWAY_CLIENT_TOKEN": "client",
        "OPENAI_API_KEY": "fixture",
        "ANTHROPIC_API_KEY": "fixture",
        "GEMINI_API_KEY": "fixture",
    }


def operation(runtime):
    assert util.find_spec("headroom.proxy.gateway.execution") is not None, (
        "execution owners missing"
    )
    module = import_module("headroom.proxy.gateway.execution")
    generation = runtime.capture()
    principal_config = generation.snapshot.client_auth.principals[0]
    return module.GatewayOperation(
        runtime,
        generation=generation,
        principal=GatewayPrincipal(
            principal_config.id,
            frozenset(principal_config.scopes),
            frozenset(principal_config.routes),
        ),
        route=generation.snapshot.routes[0],
        ingress_protocol="openai-responses",
        target_protocol="openai-responses",
        dispatch_plan=None,
    )


@pytest.mark.asyncio
async def test_runtime_enforces_configured_budget_without_credential_acquisition():
    runtime = GatewayRuntime(
        runtime_snapshot(budget_usd="0.000001", unknown_cost_policy="block"), environ=environment()
    )
    try:
        principal = runtime.snapshot.client_auth.principals[0].id
        result = await runtime.admission.try_reserve(
            qualified_request(AdmissionRequest(principal, estimated_cost=0.000002))
        )
        assert result.allowed is False
        assert result.reason == "budget"
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_attempt_close_is_idempotent_and_records_one_logical_result():
    runtime = GatewayRuntime(runtime_snapshot(), environ=environment())
    try:
        op = operation(runtime)
        assert len(runtime.active_work) == 1
        assert op.deadline > 0
        attempt = await op.start_attempt(
            op.route.credentials[0], CostEvaluation(reserved_upper_micro_usd=10)
        )
        closed = []

        async def close_upstream():
            closed.append(True)

        attempt.upstream_close = close_upstream
        attempt.mark_sending()
        attempt.usage = UsageObservation(
            currency_charge=Decimal("0.000004"), currency="USD", availability="complete"
        )
        await asyncio.gather(attempt.close("success"), attempt.close("success"))
        await asyncio.gather(op.close("success"), op.close("success"))
        assert closed == [True]
        assert runtime.admission.snapshot().known_micro_usd == 4
        assert runtime.admission.active_count == 0
        assert not runtime.active_work
        totals = runtime.observability.totals()
        assert totals["logical_requests"] == 1
        assert totals["attempts"] == 1
        assert totals["known_micro_usd"] == 4
        assert totals["unknown_usage_attempts"] == 0
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_cancelled_owner_closes_transport_and_retains_ambiguous_liability():
    runtime = GatewayRuntime(runtime_snapshot(), environ=environment())
    try:
        op = operation(runtime)
        attempt = await op.start_attempt(
            op.route.credentials[0], CostEvaluation(reserved_upper_micro_usd=10)
        )
        closed = asyncio.Event()

        async def close_upstream():
            closed.set()

        attempt.upstream_close = close_upstream
        attempt.mark_sending()
        await op.cancel()
        assert closed.is_set()
        assert not runtime.active_work
        assert runtime.admission.active_count == 0
        assert runtime.admission.snapshot().unresolved_micro_usd == 10
        assert runtime.observability.totals()["unknown_usage_attempts"] == 1
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_operation_limits_attempts_and_disallows_replay_after_send():
    runtime = GatewayRuntime(runtime_snapshot(), environ=environment())
    try:
        op = operation(runtime)
        attempt = await op.start_attempt(op.route.credentials[0], CostEvaluation())
        attempt.mark_sending()
        await attempt.close("failed")
        with pytest.raises(GatewayAuthorizationError):
            await op.start_attempt(op.route.credentials[0], CostEvaluation())
        await op.close("failed")
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_runtime_reload_preserves_ledger_and_updates_caps_atomically(tmp_path):
    runtime = GatewayRuntime(runtime_snapshot(), environ=environment())
    try:
        op = operation(runtime)
        first = await op.start_attempt(
            op.route.credentials[0], CostEvaluation(reserved_upper_micro_usd=10)
        )
        first.mark_sending()
        first.usage = UsageObservation(
            currency_charge=Decimal("0.000004"), currency="USD", availability="complete"
        )
        await op.close("success")
        path = tmp_path / "policy.json"
        path.write_text(
            runtime_snapshot(budget_usd="0.000004", unknown_cost_policy="block").model_dump_json()
        )
        result = await runtime.reload(path)
        assert result.applied
        assert runtime.admission.snapshot().known_micro_usd == 4
        denied = await runtime.admission.try_reserve(
            qualified_request(AdmissionRequest(op.principal.id, estimated_cost=0.000001))
        )
        assert not denied.allowed and denied.reason == "budget"
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_reload_cannot_hide_unbounded_unknown_liability(tmp_path):
    runtime = GatewayRuntime(runtime_snapshot(), environ=environment())
    try:
        op = operation(runtime)
        first = await op.start_attempt(op.route.credentials[0], CostEvaluation())
        first.mark_sending()
        await op.close("failed")
        path = tmp_path / "policy.json"
        path.write_text(
            runtime_snapshot(budget_usd="1", unknown_cost_policy="block").model_dump_json()
        )
        result = await runtime.reload(path)
        assert not result.applied
        assert runtime.generation == 1
        assert runtime.admission.snapshot().unbounded_charge_count == 1
    finally:
        await runtime.shutdown()


def test_observability_rejects_arbitrary_labels_and_separates_event_kinds():
    event = GatewayEvent(
        "openai-responses", "public-api", "strict-native", "env", "none", "none", "success"
    )
    for field in (
        "ingress_protocol",
        "route_class",
        "adapter",
        "credential_source",
        "failure_origin",
        "retry_reason",
        "terminal_result",
    ):
        with pytest.raises(ValueError):
            replace(event, **{field: "private-sentinel"})
    metrics = GatewayObservability()
    metrics.record(event)
    assert sum(metrics.snapshot().values()) == 1
    metrics.record_attempt(
        event,
        UsageObservation(),
        CostEvaluation(known_micro_usd=5, basis="configured_tariff", complete=True),
    )
    assert metrics.totals()["known_micro_usd_configured_tariff"] == 5
    assert metrics.totals().get("known_micro_usd_provider_reported", 0) == 0


@pytest.mark.asyncio
async def test_retry_accounting_keeps_known_and_unknown_attempts():
    import time

    from headroom.proxy.gateway.routing import ProviderContract, RetryDecision, TransportFailure

    raw = json.loads(EXAMPLE.read_text())
    raw["routes"][0]["retry"]["max_attempts"] = 2
    runtime = GatewayRuntime(GatewayConfigSnapshot.model_validate(raw), environ=environment())
    try:
        op = operation(runtime)
        contract = ProviderContract(
            2, frozenset({"rate_limit"}), rejected_failures=frozenset({"rate_limit"})
        )
        first = await op.start_attempt(
            op.route.credentials[0], CostEvaluation(reserved_upper_micro_usd=10)
        )
        first.mark_sending()
        first.mark_rejected("rate_limit", contract)
        first.usage = UsageObservation(
            currency_charge=Decimal("0.000002"), currency="USD", availability="complete"
        )
        await first.close("rejected", failure_origin="upstream", retry_reason="rate-limit")
        decision = RetryDecision.decide(
            TransportFailure("rate_limit", None),
            first.exposure,
            contract,
            attempt_count=1,
            deadline=op.deadline,
            now=time.monotonic(),
            policy=op.route.retry,
        )
        second = await op.start_attempt(
            op.route.credentials[0], CostEvaluation(reserved_upper_micro_usd=10), retry=decision
        )
        second.mark_sending()
        await op.close("failed", failure_origin="network")
        state = runtime.admission.snapshot()
        assert (state.known_micro_usd, state.unresolved_micro_usd, state.unknown_charge_count) == (
            2,
            10,
            1,
        )
        assert runtime.observability.totals()["logical_requests"] == 1
        assert runtime.observability.totals()["attempts"] == 2
        assert sum(runtime.observability.attempt_snapshot().values()) == 2
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_failed_admission_closes_logical_owner_without_provider_attempt():
    runtime = GatewayRuntime(
        runtime_snapshot(budget_usd="0", unknown_cost_policy="block"), environ=environment()
    )
    try:
        op = operation(runtime)
        with pytest.raises(GatewayAuthorizationError):
            await op.start_attempt(op.route.credentials[0], CostEvaluation())
        assert not runtime.active_work
        assert runtime.observability.totals()["logical_requests"] == 1
        assert runtime.observability.totals().get("attempts", 0) == 0
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_runtime_revoke_denies_queue_and_cancels_owned_attempt():
    runtime = GatewayRuntime(
        runtime_snapshot(max_concurrency=1, queue_timeout_seconds=1), environ=environment()
    )
    entered = asyncio.Event()

    async def worker():
        async with operation(runtime) as op:
            attempt = await op.start_attempt(
                op.route.credentials[0], CostEvaluation(reserved_upper_micro_usd=5)
            )
            attempt.mark_sending()
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(worker())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        principal = runtime.snapshot.client_auth.principals[0].id
        waiter = asyncio.create_task(
            runtime.admission.reserve(AdmissionRequest(principal, queue_timeout=1))
        )
        while runtime.admission.queued_count != 1:
            await asyncio.sleep(0)
        await runtime.revoke(principal_id=principal)
        with pytest.raises(GatewayAuthorizationError):
            await asyncio.wait_for(waiter, 1)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert not runtime.active_work
        assert runtime.admission.snapshot().unresolved_micro_usd == 5
        assert runtime.admission.active_count == 0
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_shutdown_closes_owned_attempt_before_closing_runtime():
    raw = json.loads(EXAMPLE.read_text())
    raw["limits"] = {"shutdown_drain_seconds": 0.01, "shutdown_cleanup_seconds": 1}
    runtime = GatewayRuntime(GatewayConfigSnapshot.model_validate(raw), environ=environment())
    entered = asyncio.Event()

    async def worker():
        async with operation(runtime) as op:
            attempt = await op.start_attempt(
                op.route.credentials[0], CostEvaluation(reserved_upper_micro_usd=5)
            )
            attempt.mark_sending()
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(worker())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        await runtime.shutdown()
        assert not runtime.active_work
        assert runtime.admission.active_count == 0
        assert runtime.admission.snapshot().unresolved_micro_usd == 5
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_metrics_observe_queued_work_and_currency_basis():
    runtime = GatewayRuntime(
        runtime_snapshot(max_concurrency=1, queue_timeout_seconds=1), environ=environment()
    )
    principal = runtime.snapshot.client_auth.principals[0].id
    first = await runtime.admission.reserve(AdmissionRequest(principal))
    waiter = asyncio.create_task(
        runtime.admission.reserve(AdmissionRequest(principal, queue_timeout=1))
    )
    try:
        while runtime.admission.queued_count != 1:
            await asyncio.sleep(0)
        assert runtime.observability.totals()["queued"] == 1
        assert runtime.observability.totals()["active"] == 1
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        await first.release_unsent()
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_ineligible_attempt_closes_logical_owner():
    runtime = GatewayRuntime(runtime_snapshot(), environ=environment())
    try:
        op = operation(runtime)
        with pytest.raises(GatewayAuthorizationError):
            await op.start_attempt("ungranted", CostEvaluation())
        assert not runtime.active_work
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_close_failure_keeps_usage_and_produces_failed_logical_result():
    runtime = GatewayRuntime(runtime_snapshot(), environ=environment())
    try:
        op = operation(runtime)
        attempt = await op.start_attempt(
            op.route.credentials[0], CostEvaluation(reserved_upper_micro_usd=10)
        )
        attempt.mark_sending()
        attempt.usage = UsageObservation(allowance_units=Decimal(3), allowance_unit="credits")

        async def broken_close():
            raise RuntimeError("private-provider-sentinel")

        attempt.upstream_close = broken_close
        await op.close("success")
        assert op.terminal == "failed"
        assert op.attempts[0].usage.allowance_units == 3
        assert runtime.admission.snapshot().unresolved_micro_usd == 10
        assert "sentinel" not in str(runtime.observability.snapshot())
    finally:
        await runtime.shutdown()


@pytest.mark.asyncio
async def test_strict_operation_rejects_a_numeric_bound_without_tariff_provenance():
    runtime = GatewayRuntime(
        runtime_snapshot(budget_usd="1", unknown_cost_policy="block"), environ=environment()
    )
    try:
        op = operation(runtime)
        with pytest.raises(GatewayAuthorizationError):
            await op.start_attempt(
                op.route.credentials[0], CostEvaluation(reserved_upper_micro_usd=5)
            )
        assert runtime.admission.active_count == 0
        assert not runtime.active_work
    finally:
        await runtime.shutdown()
