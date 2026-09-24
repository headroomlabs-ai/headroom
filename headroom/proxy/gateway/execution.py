"""Bounded ownership of one logical request and its provider attempts.

HTTP and WebSocket dispatch integrate this same owner in their transport batches.
No credentials or transport details are retained in terminal attempt metadata.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from headroom.proxy.gateway.admission import AcceptanceState, AdmissionRequest, AdmissionReservation
from headroom.proxy.gateway.config import Protocol, RouteConfig
from headroom.proxy.gateway.context import GatewayPrincipal
from headroom.proxy.gateway.errors import GatewayAuthorizationError
from headroom.proxy.gateway.observability import (
    Adapter,
    CredentialSource,
    FailureOrigin,
    GatewayEvent,
    RetryReason,
    RouteClass,
    TerminalResult,
)
from headroom.proxy.gateway.routing import ExposureState, ProviderContract, RetryDecision
from headroom.proxy.gateway.usage import CostEvaluation, UsageObservation

if TYPE_CHECKING:
    from headroom.proxy.gateway.runtime import GatewayRuntime, RuntimeGeneration


@dataclass(frozen=True, slots=True)
class AttemptOutcome:
    result: TerminalResult
    exposure: ExposureState
    cost: CostEvaluation
    usage: UsageObservation


class GatewayOperation:
    def __init__(
        self,
        runtime: GatewayRuntime,
        *,
        generation: RuntimeGeneration,
        principal: GatewayPrincipal,
        route: RouteConfig,
        ingress_protocol: Protocol,
        target_protocol: Protocol,
        dispatch_plan: object,
    ) -> None:
        self.runtime, self.generation, self.catalog = runtime, generation, generation.catalog
        self.principal, self.route = principal, route
        self.ingress_protocol, self.target_protocol = ingress_protocol, target_protocol
        self.dispatch_plan = dispatch_plan
        self.id = uuid.uuid4().hex
        self.deadline = time.monotonic() + generation.snapshot.limits.request_deadline_seconds
        self.terminal: TerminalResult | None = None
        self.attempts: list[AttemptOutcome] = []
        self.current_attempt: GatewayAttempt | None = None
        self.selected_account_key: str | None = None
        self.owner_task = asyncio.current_task()
        self._close_task: asyncio.Task[None] | None = None
        self._starting = False
        self._pending_reservation: asyncio.Task[AdmissionReservation] | None = None
        self._epochs = {
            account: runtime.admission.epochs(
                principal.id, route.id, generation.account_key(account)
            )
            for account in route.credentials
        }
        runtime.admission.register_operation(self.id, principal.id)
        runtime.active_work[self.id] = self
        runtime.retain(generation)
        runtime._work_empty.clear()

    async def start_attempt(
        self, account_ref: str, cost: CostEvaluation, *, retry: RetryDecision | None = None
    ) -> GatewayAttempt:
        try:
            return await self._start_attempt(account_ref, cost, retry=retry)
        except GatewayAuthorizationError:
            await self.close("rejected", failure_origin="gateway")
            raise

    async def _start_attempt(
        self,
        account_ref: str,
        cost: CostEvaluation,
        *,
        retry: RetryDecision | None = None,
    ) -> GatewayAttempt:
        if (
            self._starting
            or self._close_task is not None
            or self.terminal is not None
            or self.current_attempt is not None
            and not self.current_attempt.closed
            or len(self.attempts) >= min(self.route.retry.max_attempts, 3)
        ):
            self._deny("attempt_limit")
        if self.attempts and (
            retry is None
            or not retry.allowed
            or self.attempts[-1].exposure
            not in {ExposureState.UNSENT, ExposureState.PROVEN_REJECTED}
        ):
            self._deny("retry_forbidden")
        if account_ref not in self.catalog.eligible_accounts(self.route):
            self._deny("account_unavailable")
        account_key = self.generation.account_key(account_ref)
        self.selected_account_key = account_key
        principal_policy = next(
            p.admission
            for p in self.generation.snapshot.client_auth.principals
            if p.id == self.principal.id
        )
        policy = self.generation.snapshot.admission
        request = AdmissionRequest(
            self.principal.id,
            queue_timeout=min(policy.queue_timeout_seconds, principal_policy.queue_timeout_seconds),
            route_id=self.route.id,
            account_key=account_key,
            generation=self.generation.number,
            revocation_epochs=self._epochs[account_ref],
            reserved_upper_micro_usd=cost.reserved_upper_micro_usd,
            deadline=self.deadline,
            pricing=self.route.pricing,
            operation_id=self.id,
            cost_bound=cost,
            model_bounds=self.route.model_bounds,
        )
        self._starting = True
        try:
            pending = asyncio.create_task(self.runtime.admission.reserve(request))
            self._pending_reservation = pending
            reservation = await asyncio.shield(pending)
            # No await in this handoff: close either owns the pending reservation,
            # or sees the fully published attempt. It can never own both.
            if self._close_task is not None or self._pending_reservation is not pending:
                self._deny("closing")
            self.current_attempt = GatewayAttempt(self, account_ref, request, reservation)
            self._pending_reservation = None
            self._update_metrics()
            return self.current_attempt
        except asyncio.CancelledError:
            await self.close("cancelled", failure_origin="gateway")
            raise
        except GatewayAuthorizationError:
            await self.close("rejected", failure_origin="gateway")
            raise
        finally:
            self._starting = False

    def _event(
        self,
        result: TerminalResult,
        *,
        account_ref: str | None = None,
        failure_origin: FailureOrigin = "none",
        retry_reason: RetryReason = "none",
    ) -> GatewayEvent:
        source = next(
            (c.source.kind for c in self.generation.snapshot.credentials if c.id == account_ref),
            "none",
        )
        route_class: RouteClass = (
            "private-compatible"
            if self.route.private_network
            else (
                "cloud-workload" if self.route.provider in {"vertex", "bedrock"} else "public-api"
            )
        )
        adapter = (
            "translated"
            if self.ingress_protocol != self.target_protocol
            else self.route.body_contract
        )
        return GatewayEvent(
            self.ingress_protocol,
            route_class,
            cast(Adapter, adapter),
            cast(CredentialSource, source),
            failure_origin,
            retry_reason,
            result,
        )

    def _update_metrics(self) -> None:
        ledger = self.runtime.admission
        self.runtime.observability.state(
            active=ledger.active_count,
            queued=ledger.queued_count,
            unresolved_micro_usd=ledger.snapshot().unresolved_micro_usd,
        )

    async def close(
        self,
        result: TerminalResult,
        *,
        failure_origin: FailureOrigin = "none",
        cleanup_deadline: float | None = None,
    ) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._finish(result, failure_origin, cleanup_deadline=cleanup_deadline)
            )
        await asyncio.shield(self._close_task)

    async def _finish(
        self,
        result: TerminalResult,
        failure_origin: FailureOrigin,
        *,
        cleanup_deadline: float | None = None,
    ) -> None:
        try:
            pending = self._pending_reservation
            self._pending_reservation = None
            if pending is not None:
                if not pending.done():
                    pending.cancel()
                finished = await asyncio.gather(pending, return_exceptions=True)
                if isinstance(finished[0], AdmissionReservation):
                    await finished[0].release_unsent()
            if self.current_attempt is not None:
                await self.current_attempt.close(result, failure_origin=failure_origin)
                if result == "success" and self.attempts[-1].result != "success":
                    result = self.attempts[-1].result
            self.terminal = result
            self.runtime.observability.record(self._event(result, failure_origin=failure_origin))
        finally:
            self.runtime.admission.forget_operation(self.id)
            self.runtime.active_work.pop(self.id, None)
            await self.runtime.release(self.generation, cleanup_deadline=cleanup_deadline)
            if not self.runtime.active_work:
                self.runtime._work_empty.set()
            self._update_metrics()

    async def cancel(self, *, cancel_owner: bool = True) -> None:
        if (
            cancel_owner
            and self.owner_task is not None
            and self.owner_task is not asyncio.current_task()
            and not self.owner_task.done()
        ):
            self.owner_task.cancel()
        await self.close("cancelled", failure_origin="gateway")

    async def __aenter__(self) -> GatewayOperation:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:  # type: ignore[no-untyped-def]
        await self.close(
            "success"
            if exc_type is None
            else "cancelled"
            if exc_type is asyncio.CancelledError
            else "failed"
        )

    @staticmethod
    def _deny(reason: str) -> None:
        raise GatewayAuthorizationError(
            status_code=503,
            code="gateway_execution_" + reason,
            message="Gateway operation unavailable",
        )


class GatewayAttempt:
    def __init__(
        self,
        operation: GatewayOperation,
        account_ref: str,
        request: AdmissionRequest,
        reservation: AdmissionReservation,
    ) -> None:
        self.operation, self.account_ref, self.request, self.reservation = (
            operation,
            account_ref,
            request,
            reservation,
        )
        self.account_key = request.account_key
        self.lease_generation: int | None = None
        self.exposure = ExposureState.UNSENT
        self._usage = UsageObservation()
        self.upstream_close: Callable[[], Awaitable[None]] | None = None
        self._close_task: asyncio.Task[None] | None = None

    @property
    def closed(self) -> bool:
        return self._close_task is not None and self._close_task.done()

    @property
    def usage(self) -> UsageObservation:
        return self._usage

    @usage.setter
    def usage(self, observation: UsageObservation) -> None:
        if self._close_task is not None:
            raise ValueError("attempt observation is already finalized")
        self._usage = observation

    def mark_sending(self) -> None:
        if (
            self.exposure != ExposureState.UNSENT
            or self._close_task is not None
            or self.operation._close_task is not None
        ):
            raise ValueError("invalid attempt exposure transition")
        self.exposure = ExposureState.ACCEPTANCE_UNKNOWN

    def mark_accepted(self) -> None:
        if self.exposure != ExposureState.ACCEPTANCE_UNKNOWN or self._close_task is not None:
            raise ValueError("invalid attempt exposure transition")
        self.exposure = ExposureState.ACCEPTED

    def mark_unsent_connect_failure(self) -> None:
        """Only the HTTP pre-write connect/TLS failure classifier calls this."""
        if self.exposure != ExposureState.ACCEPTANCE_UNKNOWN or self._close_task is not None:
            raise ValueError("invalid acceptance transition")
        self.exposure = ExposureState.UNSENT

    def mark_output(self) -> None:
        if (
            self.exposure
            not in {
                ExposureState.ACCEPTANCE_UNKNOWN,
                ExposureState.ACCEPTED,
                ExposureState.OUTPUT_EXPOSED,
            }
            or self._close_task is not None
        ):
            raise ValueError("invalid attempt exposure transition")
        self.exposure = ExposureState.OUTPUT_EXPOSED

    def mark_rejected(self, failure: str, contract: ProviderContract) -> None:
        if (
            self.exposure != ExposureState.ACCEPTANCE_UNKNOWN
            or self._close_task is not None
            or failure not in contract.rejected_failures
        ):
            raise ValueError("unproven rejection")
        self.exposure = ExposureState.PROVEN_REJECTED

    async def check_current(self) -> None:
        await self.operation.runtime.admission.check_current(self.request)

    async def close(
        self,
        result: TerminalResult,
        *,
        failure_origin: FailureOrigin = "none",
        retry_reason: RetryReason = "none",
    ) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(
                self._finish(
                    result,
                    failure_origin,
                    retry_reason,
                    self._usage,
                    self.exposure,
                    self.upstream_close,
                )
            )
        await asyncio.shield(self._close_task)

    async def _finish(
        self,
        result: TerminalResult,
        failure_origin: FailureOrigin,
        retry_reason: RetryReason,
        usage: UsageObservation,
        exposure: ExposureState,
        upstream_close: Callable[[], Awaitable[None]] | None,
    ) -> None:
        try:
            if upstream_close is not None:
                try:
                    await asyncio.wait_for(
                        upstream_close(),
                        self.operation.generation.snapshot.limits.shutdown_cleanup_seconds,
                    )
                except asyncio.CancelledError:
                    # Cancellation of the owned callback is a terminal fact. A
                    # cancelled caller still propagates through close's shield.
                    result, failure_origin = "cancelled", "network"
                except Exception:
                    result, failure_origin = "failed", "network"
        finally:
            cost = await self.reservation.finalize(usage, cast(AcceptanceState, exposure.value))
            self.operation.attempts.append(AttemptOutcome(result, exposure, cost, usage))
            self.operation.runtime.observability.record_attempt(
                self.operation._event(
                    result,
                    account_ref=self.account_ref,
                    failure_origin=failure_origin,
                    retry_reason=retry_reason,
                ),
                usage,
                cost,
            )
            self.operation._update_metrics()
