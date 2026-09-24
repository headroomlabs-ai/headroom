from __future__ import annotations

import asyncio
import time
from decimal import Decimal

import pytest

from headroom.proxy.gateway.admission import AdmissionController, AdmissionRequest
from headroom.proxy.gateway.config import AdmissionPolicy, PrincipalAdmissionPolicy
from headroom.proxy.gateway.errors import GatewayAuthorizationError
from headroom.proxy.gateway.usage import UsageObservation
from tests.unified_gateway.accounting_fixtures import qualified_request


def controller(*, cap="0.000010", maximum=2, principals=None, accounts=None, queue=4):
    assert hasattr(AdmissionController, "from_policy"), "atomic scope ledger missing"
    return AdmissionController.from_policy(
        AdmissionPolicy(
            budget_usd=cap,
            unknown_cost_policy="block" if cap is not None else "allow",
            max_concurrency=maximum,
            queue_limit=queue,
            queue_timeout_seconds=1,
        ),
        principals=principals or {"a": PrincipalAdmissionPolicy(), "b": PrincipalAdmissionPolicy()},
        account_limits=accounts or {"key": 2},
        generation=1,
    )


def request(principal="a", *, upper=5, route="r", account="key", timeout=0, generation=1):
    return qualified_request(
        AdmissionRequest(
            principal,
            route_id=route,
            account_key=account,
            generation=generation,
            reserved_upper_micro_usd=upper,
            deadline=time.monotonic() + 2,
            queue_timeout=timeout,
        )
    )


def charge(micro):
    return UsageObservation(
        currency_charge=Decimal(micro) / 1_000_000,
        currency="USD",
        availability="complete",
        provenance="provider_reported",
    )


async def wait_for_queue(ledger, count):
    async with asyncio.timeout(2):
        while ledger.queued_count != count:
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_atomic_global_principal_account_and_decimal_boundary():
    ledger = controller(
        principals={
            "a": PrincipalAdmissionPolicy(budget_usd="0.000005", unknown_cost_policy="block"),
            "b": PrincipalAdmissionPolicy(),
        }
    )
    results = await asyncio.gather(
        *(ledger.try_reserve(request("a", route=r)) for r in ("r1", "r2"))
    )
    assert sorted(r.allowed for r in results) == [False, True]
    admitted = next(r.reservation for r in results if r.allowed)
    assert ledger.snapshot().reserved_micro_usd == 5
    assert (await ledger.try_reserve(request("b", upper=6))).reason == "budget"
    await admitted.finalize(charge(5), "accepted")
    assert ledger.snapshot("a").known_micro_usd == 5
    assert (await ledger.try_reserve(request("a", upper=1))).reason == "budget"
    second = await ledger.reserve(request("b"))
    await second.release_unsent()
    assert ledger.snapshot().known_micro_usd == 5


@pytest.mark.asyncio
async def test_unknown_partial_and_over_bound_costs_remain_distinct():
    ledger = controller(cap="0.000020")
    first = await ledger.reserve(request(upper=10))
    await first.finalize(
        UsageObservation(
            currency_charge=Decimal("0.000004"), currency="USD", availability="partial"
        ),
        "acceptance_unknown",
    )
    view = ledger.snapshot()
    assert (
        view.known_micro_usd,
        view.unresolved_micro_usd,
        view.reserved_micro_usd,
        view.unknown_charge_count,
    ) == (4, 6, 0, 1)
    await first.finalize(charge(10), "accepted")
    assert ledger.snapshot() == view
    second = await ledger.reserve(request(upper=5))
    await second.finalize(charge(7), "accepted")
    assert ledger.snapshot().known_micro_usd == 11
    assert (await ledger.try_reserve(request(upper=1))).reason == "bound_violated"


@pytest.mark.asyncio
async def test_unbounded_unknown_charge_prevents_enabling_strict_cap():
    ledger = controller(cap=None)
    first = await ledger.reserve(request(upper=None))
    await first.finalize(
        UsageObservation(allowance_units=Decimal(2), allowance_unit="credits"), "accepted"
    )
    assert ledger.snapshot().unknown_charge_count == 1
    assert ledger.snapshot().unbounded_charge_count == 1
    with pytest.raises(ValueError, match="unbounded"):
        await ledger.update_policy(
            AdmissionPolicy(budget_usd="1", unknown_cost_policy="block"),
            principals={"a": PrincipalAdmissionPolicy()},
            account_limits={"key": 1},
            generation=2,
        )
    assert (await ledger.try_reserve(request(upper=None))).allowed


@pytest.mark.asyncio
async def test_reserved_slots_fifo_fair_queue_and_cancellation():
    policies = {
        p: PrincipalAdmissionPolicy(
            max_concurrency=2, reserved_concurrency=1, queue_timeout_seconds=1, queue_limit=2
        )
        for p in ("a", "b")
    }
    ledger = controller(cap=None, principals=policies, accounts={"key": 4}, maximum=2)
    a = await ledger.reserve(request("a"))
    assert (await ledger.try_reserve(request("a"))).reason == "concurrency"
    a1 = asyncio.create_task(ledger.reserve(request("a", timeout=1)))
    await wait_for_queue(ledger, 1)
    a2 = asyncio.create_task(ledger.reserve(request("a", timeout=1)))
    await wait_for_queue(ledger, 2)
    with pytest.raises(GatewayAuthorizationError, match="queue_full"):
        await ledger.reserve(request("a", timeout=1))
    b = await ledger.reserve(request("b"))
    assert ledger.active_count == 2
    a1.cancel()
    with pytest.raises(asyncio.CancelledError):
        await a1
    assert ledger.queued_count == 1
    await a.release_unsent()
    next_a = await asyncio.wait_for(a2, 1)
    assert ledger.queued_count == 0
    await asyncio.gather(next_a.release_unsent(), b.release_unsent())
    assert ledger.active_count == 0


@pytest.mark.asyncio
async def test_ready_tenant_heads_are_round_robin_and_cannot_be_bypassed():
    ledger = controller(
        cap=None,
        maximum=1,
        principals={p: PrincipalAdmissionPolicy(queue_timeout_seconds=1) for p in ("a", "b")},
    )
    first = await ledger.reserve(request())
    tasks = [asyncio.create_task(ledger.reserve(request(p, timeout=1))) for p in ("a", "a", "b")]
    await wait_for_queue(ledger, 3)
    await first.release_unsent()
    a1 = await asyncio.wait_for(tasks[0], 1)
    assert not tasks[1].done() and not tasks[2].done()
    await a1.release_unsent()
    b1 = await asyncio.wait_for(tasks[2], 1)
    assert not tasks[1].done()
    await b1.release_unsent()
    await (await tasks[1]).release_unsent()


@pytest.mark.asyncio
async def test_account_concurrency_and_deadline_bound_queue():
    ledger = controller(cap=None, accounts={"key": 1})
    active = await ledger.reserve(request())
    assert (await ledger.try_reserve(request("b"))).reason == "concurrency"
    expired = AdmissionRequest(
        "b", account_key="key", generation=1, deadline=time.monotonic() + 0.02, queue_timeout=1
    )
    with pytest.raises(GatewayAuthorizationError, match="queue_timeout"):
        await ledger.reserve(expired)
    assert ledger.queued_count == 0
    await active.release_unsent()


@pytest.mark.asyncio
async def test_shutdown_retains_live_liability_until_owner_finalizes():
    ledger = controller()
    active = await ledger.reserve(request())
    await ledger.shutdown()
    assert ledger.active_count == 1
    assert ledger.snapshot().reserved_micro_usd == 5
    await active.finalize(UsageObservation(), "acceptance_unknown")
    assert ledger.active_count == 0
    assert ledger.snapshot().unresolved_micro_usd == 5


@pytest.mark.asyncio
async def test_generation_revocation_and_reload_keep_existing_ledgers():
    ledger = controller()
    active = await ledger.reserve(request())
    await active.finalize(charge(5), "accepted")
    await ledger.update_policy(
        AdmissionPolicy(budget_usd="0.000004", unknown_cost_policy="block"),
        principals={"a": PrincipalAdmissionPolicy()},
        account_limits={"key": 1},
        generation=2,
    )
    assert (await ledger.try_reserve(request())).reason == "configuration_changed"
    assert (await ledger.try_reserve(request(generation=2))).reason == "budget"
    assert ledger.snapshot("a").known_micro_usd == 5
    await ledger.revoke(principal_id="a")
    assert (await ledger.try_reserve(request(generation=2))).reason == "revoked"


@pytest.mark.asyncio
async def test_cannot_enable_strict_cap_while_unbounded_attempt_is_still_active():
    ledger = controller(cap=None)
    active = await ledger.reserve(request(upper=None))
    with pytest.raises(ValueError, match="unbounded"):
        await ledger.update_policy(
            AdmissionPolicy(budget_usd="1", unknown_cost_policy="block"),
            principals={"a": PrincipalAdmissionPolicy()},
            account_limits={"key": 1},
            generation=2,
        )
    await active.release_unsent()


@pytest.mark.asyncio
async def test_cancel_after_queued_admission_releases_the_unsent_reservation():
    ledger = controller(cap=None, maximum=1)
    active = await ledger.reserve(request())
    queued = asyncio.create_task(ledger.reserve(request("b", timeout=1)))
    await wait_for_queue(ledger, 1)
    # Keep the waiter from resuming until the reservation was handed off.
    await ledger._close(active._reservation_id, UsageObservation(), "unsent")
    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    assert ledger.active_count == 0
    assert ledger.queued_count == 0
    assert ledger.snapshot().unknown_charge_count == 0
    await active.release_unsent()


@pytest.mark.asyncio
async def test_reload_rejects_already_admitted_operation_waiting_for_retry_slot():
    from dataclasses import replace

    ledger = controller(cap=None, maximum=1)
    first_request = replace(request(), operation_id="operation")
    first = await ledger.reserve(first_request)
    await first.release_unsent()
    held = await ledger.reserve(request("b"))
    retry = asyncio.create_task(ledger.reserve(replace(first_request, queue_timeout=1)))
    await wait_for_queue(ledger, 1)
    await ledger.update_policy(
        AdmissionPolicy(max_concurrency=1, queue_timeout_seconds=1),
        principals={"a": PrincipalAdmissionPolicy(), "b": PrincipalAdmissionPolicy()},
        account_limits={"key": 2},
        generation=2,
    )
    try:
        with pytest.raises(GatewayAuthorizationError, match="configuration_changed"):
            await asyncio.wait_for(retry, 0.05)
    finally:
        await held.release_unsent()


@pytest.mark.asyncio
async def test_budget_below_one_micro_dollar_does_not_round_cap_up():
    ledger = controller(cap="0.0000009")
    assert (await ledger.try_reserve(request(upper=1))).reason == "budget"
