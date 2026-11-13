"""Process-lifetime atomic admission and truthful per-attempt accounting."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Literal, NoReturn

from headroom.proxy.gateway.config import (
    AdmissionPolicy,
    ModelBounds,
    PricingConfig,
    PrincipalAdmissionPolicy,
)
from headroom.proxy.gateway.errors import GatewayAuthorizationError
from headroom.proxy.gateway.usage import (
    CostEvaluation,
    UsageObservation,
    evaluate_cost,
    usd_to_micro,
)

AcceptanceState = Literal[
    "unsent", "proven_rejected", "acceptance_unknown", "accepted", "output_exposed"
]


@dataclass(frozen=True, slots=True)
class AdmissionRequest:
    principal_id: str
    estimated_cost: float | None = None
    queue_timeout: float = 0.0
    route_id: str = ""
    account_key: str = ""
    generation: int = 0
    revocation_epochs: tuple[int, int, int, int] = (0, 0, 0, 0)
    reserved_upper_micro_usd: int | None = None
    deadline: float = float("inf")
    pricing: PricingConfig | None = None
    operation_id: str | None = None
    cost_bound: CostEvaluation | None = None
    model_bounds: ModelBounds | None = None

    def __post_init__(self) -> None:
        upper = self.reserved_upper_micro_usd
        if upper is not None and (type(upper) is not int or upper < 0):
            raise ValueError("invalid reservation amount")
        if self.estimated_cost is not None:
            if upper is not None:
                raise ValueError("ambiguous reservation amount")
            object.__setattr__(
                self, "reserved_upper_micro_usd", usd_to_micro(Decimal(str(self.estimated_cost)))
            )
        if self.deadline != self.deadline or self.queue_timeout < 0:
            raise ValueError("invalid admission deadline")


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    allowed: bool
    reason: str | None
    reservation: AdmissionReservation | None


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    known_micro_usd: int = 0
    unresolved_micro_usd: int = 0
    reserved_micro_usd: int = 0
    unknown_charge_count: int = 0
    unbounded_charge_count: int = 0
    unqualified_charge_count: int = 0

    @property
    def total_micro_usd(self) -> int:
        return self.known_micro_usd + self.unresolved_micro_usd + self.reserved_micro_usd


class AdmissionReservation:
    """A finalizer may outlive its cancelled caller; the controller owns settlement."""

    def __init__(self, controller: AdmissionController, reservation_id: str) -> None:
        self._controller = controller
        self._reservation_id = reservation_id
        self._finalizer: asyncio.Task[CostEvaluation] | None = None

    async def finalize(
        self,
        usage: UsageObservation | None = None,
        acceptance: AcceptanceState = "accepted",
        *,
        actual_cost: float | None = None,
    ) -> CostEvaluation:
        if acceptance not in {
            "unsent",
            "proven_rejected",
            "acceptance_unknown",
            "accepted",
            "output_exposed",
        }:
            raise ValueError("invalid acceptance state")
        if usage is None:
            usage = (
                UsageObservation()
                if actual_cost is None
                else UsageObservation(
                    currency_charge=Decimal(str(actual_cost)),
                    currency="USD",
                    availability="complete",
                    provenance="provider_reported",
                )
            )
        if self._finalizer is None:
            self._finalizer = asyncio.create_task(
                self._controller._close(self._reservation_id, usage, acceptance)
            )
        return await asyncio.shield(self._finalizer)

    async def release_unsent(self) -> None:
        await self.finalize(UsageObservation(), "unsent")

    async def release(self) -> None:
        """Compatibility for old dispatch: an unclassified close is possibly charged."""
        await self.finalize(UsageObservation(), "acceptance_unknown")

    async def __aenter__(self) -> AdmissionReservation:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        await self.finalize(UsageObservation(), "acceptance_unknown")


@dataclass(slots=True)
class _Waiter:
    request: AdmissionRequest
    future: asyncio.Future[AdmissionResult]


class AdmissionController:
    def __init__(
        self,
        *,
        budget_limit: float | None,
        max_concurrency: int,
        queue_limit: int,
        unknown_cost_policy: Literal["allow", "block"],
    ) -> None:
        self._policy = AdmissionPolicy(
            budget_usd=None if budget_limit is None else str(budget_limit),
            max_concurrency=max_concurrency,
            queue_limit=queue_limit,
            unknown_cost_policy=unknown_cost_policy,
        )
        self._principals: dict[str, PrincipalAdmissionPolicy] | None = None
        self._account_limits: dict[str, int] | None = None
        self._grants: dict[str, frozenset[str]] | None = None
        self._routes: frozenset[str] | None = None
        self._generation = 0
        self._active: dict[str, AdmissionRequest] = {}
        self._ledger = LedgerSnapshot()
        self._principal_ledgers: dict[str, LedgerSnapshot] = {}
        self._queues: dict[str, deque[_Waiter]] = {}
        self._turns: deque[str] = deque()
        self._condition = asyncio.Condition()
        self._shutdown = False
        self._epochs: dict[tuple[str, str], int] = {}
        self._revoked: set[tuple[str, str]] = set()
        self._violated: set[tuple[str, str | None]] = set()
        self._admitted_operations: dict[str, tuple[str, str, int]] = {}
        self._registered_operations: dict[str, str] = {}

    @classmethod
    def from_policy(
        cls,
        policy: AdmissionPolicy,
        *,
        principals: dict[str, PrincipalAdmissionPolicy],
        account_limits: dict[str, int],
        generation: int,
        grants: dict[str, frozenset[str]] | None = None,
        routes: frozenset[str] | None = None,
    ) -> AdmissionController:
        controller = cls(
            budget_limit=None,
            max_concurrency=policy.max_concurrency,
            queue_limit=policy.queue_limit,
            unknown_cost_policy=policy.unknown_cost_policy,
        )
        controller._install(policy, principals, account_limits, generation, grants, routes)
        return controller

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def queued_count(self) -> int:
        return sum(len(queue) for queue in self._queues.values())

    @property
    def committed_cost(self) -> float:
        """Legacy display only; no float participates in admission arithmetic."""
        return self._ledger.known_micro_usd / 1_000_000

    def snapshot(self, principal_id: str | None = None) -> LedgerSnapshot:
        return (
            self._ledger
            if principal_id is None
            else self._principal_ledgers.get(principal_id, LedgerSnapshot())
        )

    def epochs(
        self, principal_id: str, route_id: str, account_key: str
    ) -> tuple[int, int, int, int]:
        return (
            self._epochs.get(("principal", principal_id), 0),
            self._epochs.get(("route", route_id), 0),
            self._epochs.get(("account", account_key), 0),
            self._epochs.get(("grant", principal_id + "/" + route_id), 0),
        )

    def forget_operation(self, operation_id: str) -> None:
        self._admitted_operations.pop(operation_id, None)
        self._registered_operations.pop(operation_id, None)

    def register_operation(self, operation_id: str, principal_id: str) -> None:
        """Bound owners without taking another tenant's reserved allocation.

        This synchronous, no-I/O transition runs on the controller's event loop,
        so policy publication cannot interleave with its checks or publication.
        """
        if self._shutdown:
            self._raise_denied("shutdown")
        if self._principals is not None and principal_id not in self._principals:
            self._raise_denied("revoked")
        if operation_id in self._registered_operations:
            raise ValueError("operation already registered")
        counts = {
            key: sum(owner == key for owner in self._registered_operations.values())
            for key in (self._principals or {})
        }
        principal = self._principal_policy(principal_id)
        if (
            principal is not None
            and counts[principal_id] >= principal.max_concurrency + principal.queue_limit
        ):
            self._raise_denied("queue_full")
        protected = sum(
            max(0, policy.reserved_concurrency - counts[key])
            for key, policy in (self._principals or {}).items()
            if key != principal_id
        )
        if (
            len(self._registered_operations) + 1 + protected
            > self._policy.max_concurrency + self._policy.queue_limit
        ):
            self._raise_denied("queue_full")
        self._registered_operations[operation_id] = principal_id

    async def update_policy(
        self,
        policy: AdmissionPolicy,
        *,
        principals: dict[str, PrincipalAdmissionPolicy],
        account_limits: dict[str, int],
        generation: int,
        grants: dict[str, frozenset[str]] | None = None,
        routes: frozenset[str] | None = None,
        publish: Callable[[], None] | None = None,
        validate: Callable[[], None] | None = None,
    ) -> None:
        async with self._condition:
            if validate is not None:
                validate()
            previous_generation = self._generation
            self._install(policy, principals, account_limits, generation, grants, routes)
            if publish is not None:
                publish()
            if generation != previous_generation:
                for queue in self._queues.values():
                    for waiter in queue:
                        waiter.future.set_result(
                            AdmissionResult(False, "configuration_changed", None)
                        )
                self._queues.clear()
                self._turns.clear()
            self._drain()

    def _install(
        self,
        policy: AdmissionPolicy,
        principals: dict[str, PrincipalAdmissionPolicy],
        account_limits: dict[str, int],
        generation: int,
        grants: dict[str, frozenset[str]] | None,
        routes: frozenset[str] | None,
    ) -> None:
        if policy.budget_usd is not None and (
            self._ledger.unbounded_charge_count
            or any(request.reserved_upper_micro_usd is None for request in self._active.values())
        ):
            raise ValueError("unbounded liabilities prevent strict budget")
        if any(
            p.budget_usd is not None
            and (
                self.snapshot(key).unbounded_charge_count
                or any(
                    request.principal_id == key and request.reserved_upper_micro_usd is None
                    for request in self._active.values()
                )
            )
            for key, p in principals.items()
        ):
            raise ValueError("unbounded principal liabilities prevent strict budget")
        if policy.budget_usd is not None and (
            self._ledger.unqualified_charge_count
            or any(not self._has_qualified_bound(request) for request in self._active.values())
        ):
            raise ValueError("unqualified liabilities prevent strict budget")
        if any(
            p.budget_usd is not None
            and (
                self.snapshot(key).unqualified_charge_count
                or any(
                    request.principal_id == key and not self._has_qualified_bound(request)
                    for request in self._active.values()
                )
            )
            for key, p in principals.items()
        ):
            raise ValueError("unqualified principal liabilities prevent strict budget")
        if sum(p.reserved_concurrency for p in principals.values()) > policy.max_concurrency:
            raise ValueError("reserved concurrency exceeds global capacity")
        if any(type(limit) is not int or limit < 1 for limit in account_limits.values()):
            raise ValueError("invalid account concurrency")
        for kind, before, after in (
            ("principal", self._principals, principals),
            ("account", self._account_limits, account_limits),
            ("route", self._routes, routes),
        ):
            if before is not None and after is not None:
                for key in set(before) - set(after):
                    self._bump(kind, key)
        if self._grants is not None and grants is not None:
            for principal, old in self._grants.items():
                for route in old - grants.get(principal, frozenset()):
                    self._bump("grant", principal + "/" + route)
        self._policy, self._principals, self._account_limits = (
            policy,
            dict(principals),
            dict(account_limits),
        )
        self._generation, self._grants, self._routes = generation, grants, routes

    def _bump(self, kind: str, key: str) -> None:
        self._epochs[(kind, key)] = self._epochs.get((kind, key), 0) + 1

    async def revoke(
        self,
        *,
        principal_id: str | None = None,
        route_id: str | None = None,
        account_key: str | None = None,
    ) -> None:
        selectors = [
            (kind, value)
            for kind, value in (
                ("principal", principal_id),
                ("route", route_id),
                ("account", account_key),
            )
            if value is not None
        ]
        if len(selectors) != 1:
            raise ValueError("one revocation selector required")
        async with self._condition:
            kind, key = selectors[0]
            self._bump(kind, key)
            self._revoked.add((kind, key))
            self._drain()

    def _principal_policy(self, principal_id: str) -> PrincipalAdmissionPolicy | None:
        return self._principals.get(principal_id) if self._principals is not None else None

    @staticmethod
    def _has_qualified_bound(request: AdmissionRequest) -> bool:
        bound, pricing, model = request.cost_bound, request.pricing, request.model_bounds
        if (
            bound is not None
            and bound.provider_contract == "openai-responses-resource-v1"
            and bound.qualified_bound
            and bound.complete
            and bound.basis == "provider_reported"
            and bound.known_micro_usd
            == bound.reserved_upper_micro_usd
            == request.reserved_upper_micro_usd
            == 0
        ):
            return True
        return bool(
            bound is not None
            and pricing is not None
            and model is not None
            and bound.qualified_bound
            and bound.basis == "configured_tariff"
            and bound.reserved_upper_micro_usd is not None
            and bound.reserved_upper_micro_usd == request.reserved_upper_micro_usd
            and bound.tariff_revision == pricing.revision
            and bound.provider_contract == model.provider_contract
            and pricing.cache_read_usd_per_million is not None
            and pricing.cache_create_usd_per_million is not None
        )

    def _denial_reason(self, request: AdmissionRequest) -> str | None:
        if self._shutdown:
            return "shutdown"
        if time.monotonic() >= request.deadline:
            return "deadline"
        if any(
            (kind, key) in self._revoked
            for kind, key in (
                ("principal", request.principal_id),
                ("route", request.route_id),
                ("account", request.account_key),
            )
        ) or request.revocation_epochs != self.epochs(
            request.principal_id, request.route_id, request.account_key
        ):
            return "revoked"
        if (
            self._principals is not None
            and request.principal_id not in self._principals
            or request.account_key
            and self._account_limits is not None
            and request.account_key not in self._account_limits
            or request.route_id
            and self._routes is not None
            and request.route_id not in self._routes
            or request.route_id
            and self._grants is not None
            and request.route_id not in self._grants.get(request.principal_id, ())
        ):
            return "revoked"
        signature = (request.principal_id, request.route_id, request.generation)
        if (
            request.generation
            and request.generation != self._generation
            and (
                request.operation_id is None
                or self._admitted_operations.get(request.operation_id) != signature
            )
        ):
            return "configuration_changed"
        principal = self._principal_policy(request.principal_id)
        for policy, ledger in (
            (self._policy, self._ledger),
            (principal, self.snapshot(request.principal_id)),
        ):
            if policy is None:
                continue
            if not self._has_qualified_bound(request) and (
                policy.unknown_cost_policy == "block" or policy.budget_usd is not None
            ):
                return "unknown_cost"
            if policy.budget_usd is not None:
                if (
                    request.route_id,
                    request.pricing.revision if request.pricing else None,
                ) in self._violated:
                    return "bound_violated"
                if ledger.unbounded_charge_count or ledger.total_micro_usd + (
                    request.reserved_upper_micro_usd or 0
                ) > usd_to_micro(Decimal(policy.budget_usd), reservation=False):
                    return "budget"
        active_p = sum(r.principal_id == request.principal_id for r in self._active.values())
        protected = sum(
            max(
                0,
                p.reserved_concurrency - sum(r.principal_id == key for r in self._active.values()),
            )
            for key, p in (self._principals or {}).items()
            if key != request.principal_id
        )
        if self.active_count + 1 + protected > self._policy.max_concurrency:
            return "concurrency"
        if principal is not None and active_p >= principal.max_concurrency:
            return "concurrency"
        if (
            request.account_key
            and self._account_limits is not None
            and sum(r.account_key == request.account_key for r in self._active.values())
            >= self._account_limits[request.account_key]
        ):
            return "concurrency"
        return None

    async def check_current(self, request: AdmissionRequest) -> None:
        """Recheck authorization epochs after acquisition and immediately before send."""
        async with self._condition:
            reason = self._denial_reason(request)
            if reason not in {None, "concurrency", "budget", "unknown_cost", "bound_violated"}:
                self._raise_denied(reason)

    async def try_reserve(self, request: AdmissionRequest) -> AdmissionResult:
        async with self._condition:
            self._drain()
            reason = self._denial_reason(request)
            if reason is None and self._queues.get(request.principal_id):
                reason = "concurrency"
            if reason is not None:
                return AdmissionResult(False, reason, None)
            return AdmissionResult(True, None, self._admit(request))

    async def reserve(self, request: AdmissionRequest) -> AdmissionReservation:
        async with self._condition:
            self._drain()
            reason = self._denial_reason(request)
            if reason is None and self._queues.get(request.principal_id):
                reason = "concurrency"
            if reason is None:
                return self._admit(request)
            timeout = min(request.queue_timeout, request.deadline - time.monotonic())
            principal = self._principal_policy(request.principal_id)
            for policy in (self._policy, principal):
                if policy is not None and policy.queue_timeout_seconds > 0:
                    timeout = min(timeout, policy.queue_timeout_seconds)
            if reason != "concurrency" or timeout <= 0:
                self._raise_denied(reason)
            queue = self._queues.get(request.principal_id)
            if self.queued_count >= self._policy.queue_limit or (
                principal is not None and len(queue or ()) >= principal.queue_limit
            ):
                self._raise_denied("queue_full")
            waiter = _Waiter(request, asyncio.get_running_loop().create_future())
            if queue is None:
                queue = self._queues[request.principal_id] = deque()
                self._turns.append(request.principal_id)
            queue.append(waiter)
        delivered = False
        try:
            result = await asyncio.wait_for(asyncio.shield(waiter.future), timeout=timeout)
            if result.reservation is None:
                self._raise_denied(result.reason)
            delivered = True
            return result.reservation
        except asyncio.TimeoutError:
            self._raise_denied("queue_timeout")
        finally:

            async def remove() -> None:
                async with self._condition:
                    pending = self._queues.get(request.principal_id)
                    if pending is not None and waiter in pending:
                        pending.remove(waiter)
                        self._trim(request.principal_id)
                    self._drain()
                if not delivered and waiter.future.done() and not waiter.future.cancelled():
                    admitted = waiter.future.result().reservation
                    if admitted is not None:
                        await admitted.release_unsent()

            if not delivered:
                cleanup = asyncio.create_task(remove())
                await asyncio.shield(cleanup)

    def _trim(self, principal: str) -> None:
        if not self._queues.get(principal):
            self._queues.pop(principal, None)
            if principal in self._turns:
                self._turns.remove(principal)

    def _drain(self) -> None:
        while self._turns:
            progressed = False
            for _ in range(len(self._turns)):
                principal = self._turns[0]
                self._turns.rotate(-1)
                waiter = self._queues[principal][0]
                reason = self._denial_reason(waiter.request)
                if reason == "concurrency":
                    continue
                self._queues[principal].popleft()
                self._trim(principal)
                reservation = self._admit(waiter.request) if reason is None else None
                waiter.future.set_result(AdmissionResult(reason is None, reason, reservation))
                progressed = True
                break
            if not progressed:
                return

    def _admit(self, request: AdmissionRequest) -> AdmissionReservation:
        key = uuid.uuid4().hex
        self._active[key] = request
        self._change(request.principal_id, reserved_micro_usd=request.reserved_upper_micro_usd or 0)
        if request.operation_id is not None:
            self._admitted_operations[request.operation_id] = (
                request.principal_id,
                request.route_id,
                request.generation,
            )
        return AdmissionReservation(self, key)

    def _change(self, principal: str, **deltas: int) -> None:
        def changed(ledger: LedgerSnapshot) -> LedgerSnapshot:
            return replace(
                ledger, **{key: getattr(ledger, key) + value for key, value in deltas.items()}
            )

        self._ledger = changed(self._ledger)
        self._principal_ledgers[principal] = changed(self.snapshot(principal))

    async def _close(
        self, key: str, usage: UsageObservation, acceptance: AcceptanceState
    ) -> CostEvaluation:
        async with self._condition:
            request = self._active.pop(key, None)
            if request is None:
                return CostEvaluation()
            cost = evaluate_cost(
                usage, request.pricing, reserved_upper_micro_usd=request.reserved_upper_micro_usd
            )
            free = acceptance == "unsent" or usage.charge_free
            known = 0 if free else cost.known_micro_usd or 0
            unknown = not free and not cost.complete
            unresolved = max(0, (request.reserved_upper_micro_usd or 0) - known) if unknown else 0
            self._change(
                request.principal_id,
                known_micro_usd=known,
                unresolved_micro_usd=unresolved,
                reserved_micro_usd=-(request.reserved_upper_micro_usd or 0),
                unknown_charge_count=int(unknown),
                unbounded_charge_count=int(unknown and request.reserved_upper_micro_usd is None),
                unqualified_charge_count=int(unknown and not self._has_qualified_bound(request)),
            )
            if cost.bound_violated and not free:
                self._violated.add(
                    (request.route_id, request.pricing.revision if request.pricing else None)
                )
            self._drain()
            self._condition.notify_all()
            return (
                replace(
                    cost,
                    known_micro_usd=0,
                    complete=True,
                    basis="provider_reported",
                    bound_violated=False,
                )
                if free
                else cost
            )

    async def shutdown(self) -> None:
        async with self._condition:
            self._shutdown = True
            self._drain()
            self._condition.notify_all()

    @staticmethod
    def _raise_denied(reason: str | None) -> NoReturn:
        raise GatewayAuthorizationError(
            status_code=429
            if reason in {"concurrency", "queue_full", "queue_timeout"}
            else 503
            if reason in {"shutdown", "configuration_changed", "deadline"}
            else 403,
            code=f"gateway_admission_{reason or 'denied'}",
            message=f"Gateway admission denied: {reason or 'unknown'}",
        )
