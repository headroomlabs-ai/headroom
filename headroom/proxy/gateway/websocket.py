"""Owned affine Responses sessions with a separate ledger operation per turn."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from typing import Any

from starlette.websockets import WebSocket, WebSocketDisconnect

from headroom.proxy.gateway.auth import GatewayAuthorizer
from headroom.proxy.gateway.capabilities import requested_features
from headroom.proxy.gateway.config import RouteConfig
from headroom.proxy.gateway.context import GatewayPrincipal, GatewayRequestContext
from headroom.proxy.gateway.dispatch import route_target
from headroom.proxy.gateway.egress import build_managed_upstream_headers
from headroom.proxy.gateway.errors import GatewayAuthorizationError, GatewayPublicError
from headroom.proxy.gateway.execution import GatewayOperation
from headroom.proxy.gateway.observability import FailureOrigin, TerminalResult
from headroom.proxy.gateway.resources import ResourceBinding
from headroom.proxy.gateway.runtime import GatewayRuntime, RuntimeGeneration
from headroom.proxy.gateway.streaming import StreamObserver
from headroom.proxy.gateway.transport import websocket_connection
from headroom.proxy.gateway.usage import conservative_cost_bound

_MAX_FRAME_BYTES = 1_048_576


def routed_response_create_frame(frame: str, route: RouteConfig) -> str:
    if route.body_contract == "strict-native" or route.public_model == route.upstream_model:
        return frame
    envelope = json.loads(frame)
    envelope.get("response", envelope)["model"] = route.upstream_model
    return json.dumps(envelope, separators=(",", ":"))


def authorize_response_create_frame(
    frame: str,
    principal: GatewayPrincipal,
    authorizer: GatewayAuthorizer,
    *,
    expected_route_id: str | None = None,
    maximum: int = _MAX_FRAME_BYTES,
) -> tuple[RouteConfig, dict[str, Any]]:
    if len(frame.encode("utf-8")) > maximum:
        raise _invalid("gateway_frame_too_large")
    try:
        envelope = json.loads(frame)
    except (ValueError, RecursionError):
        raise _invalid("gateway_request_invalid") from None
    if not isinstance(envelope, dict) or envelope.get("type") != "response.create":
        raise _invalid("gateway_frame_invalid")
    payload = envelope.get("response", envelope)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
        raise _invalid("gateway_model_required")
    route = authorizer.authorize(
        principal,
        scope="inference",
        protocol="openai-responses",
        public_model=payload["model"],
        transport="websocket",
        features=requested_features("openai-responses", payload),
        defer_cost_to_admission=True,
    )
    if expected_route_id is not None and route.id != expected_route_id:
        raise _invalid("gateway_websocket_affinity", 403)
    return route, payload


def _invalid(code: str, status: int = 400) -> GatewayAuthorizationError:
    return GatewayAuthorizationError(
        status_code=status, code=code, message="WebSocket operation unavailable"
    )


class GatewayWebSocketSession:
    """Own the socket and bounded relay tasks, including time between turns."""

    def __init__(self, websocket: WebSocket, runtime: GatewayRuntime, principal: GatewayPrincipal):
        self.websocket, self.runtime, self.principal = websocket, runtime, principal
        self.id = "ws-" + uuid.uuid4().hex
        self.owner_task = asyncio.current_task()
        self.route: RouteConfig | None = None
        self.generation: RuntimeGeneration | None = None
        self.selected_account_key: str | None = None
        self.binding: ResourceBinding | None = None
        self.operation: GatewayOperation | None = None
        self.observer: StreamObserver | None = None
        self.upstream: Any = None
        self.context: Any = None
        self._context_settled = asyncio.Event()
        self._context_entered = False
        self._finishing = False
        self.tasks: set[asyncio.Task[Any]] = set()
        self.finished = asyncio.Event()
        self._cancel_requested = False
        self._finish_task: asyncio.Task[None] | None = None
        self.runtime.active_work[self.id] = self
        self.runtime._work_empty.clear()

    def spawn(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.create_task(coro)
        self.tasks.add(task)
        task.add_done_callback(self._task_finished)
        return task

    def _task_finished(self, task: asyncio.Task[Any]) -> None:
        self.tasks.discard(task)
        if not task.cancelled():
            task.exception()

    async def wait_owned(self, coro: Any, deadline: float) -> Any:
        task = self.spawn(coro)
        remaining = deadline - time.monotonic()
        try:
            if remaining > 0:
                done, _ = await asyncio.wait({task}, timeout=remaining)
                if task in done:
                    return task.result()
            raise _invalid("gateway_websocket_timeout", 504)
        finally:
            if not task.done():
                task.cancel()

    async def cancel(self, *, cancel_owner: bool = True) -> None:
        if not cancel_owner:
            return
        if (
            cancel_owner
            and not self._cancel_requested
            and self.owner_task is not None
            and not self.owner_task.done()
        ):
            self._cancel_requested = True
            self.owner_task.cancel()
        # Revocation can be invoked by an owned child; don't join its ancestor.
        if asyncio.current_task() not in {*self.tasks, self.owner_task}:
            await self.finished.wait()

    def authenticate(self) -> tuple[RuntimeGeneration, GatewayPrincipal]:
        generation = self.runtime.capture()
        principal = generation.authenticator.authenticate(self.websocket.headers)
        if principal.id != self.principal.id:
            raise _invalid("gateway_route_forbidden", 403)
        return generation, principal

    def authorize(
        self, frame: str
    ) -> tuple[RuntimeGeneration, GatewayPrincipal, RouteConfig, dict[str, Any]]:
        generation, principal = self.authenticate()
        route, payload = authorize_response_create_frame(
            frame,
            principal,
            generation.authorizer,
            expected_route_id=self.route.id if self.route else None,
            maximum=generation.snapshot.limits.max_frame_bytes,
        )
        return generation, principal, route, payload

    async def start_turn(
        self,
        frame: str,
        generation: RuntimeGeneration,
        principal: GatewayPrincipal,
        route: RouteConfig,
        payload: dict[str, Any],
    ) -> None:
        operation = self.operation
        assert operation is not None
        previous = payload.get("previous_response_id")
        binding = self.binding
        if previous is not None:
            if not isinstance(previous, str):
                raise _invalid("gateway_request_invalid")
            owned = await self.runtime.resources.authorize(
                previous, principal_id=principal.id, route_id=route.id, now=time.time()
            )
            generation.validate_binding(owned)
            if binding is not None and owned.account_ref != binding.account_ref:
                raise _invalid("gateway_websocket_affinity", 403)
            binding = owned
        if self.binding is not None:
            generation.validate_binding(self.binding)
        selection = self.runtime.router.select(
            route,
            principal,
            resource_binding=binding,
            eligible_accounts=generation.catalog.eligible_accounts(
                route,
                protocol="openai-responses",
                transport="websocket",
                features=requested_features("openai-responses", payload),
            ),
            authority_keys=dict(generation.authorities),
            target_key=generation.target_key(route.id),
        )
        attempt = await operation.start_attempt(
            selection.account_ref,
            conservative_cost_bound(
                route,
                payload,
                qualified_contracts=self.runtime.dependencies.qualified_cost_contracts,
            ),
        )
        self.selected_account_key = attempt.account_key
        await attempt.check_current()
        lease = await generation.broker.acquire(route, account_ref=selection.account_ref)
        await attempt.check_current()
        attempt.lease_generation = lease.generation
        if self.upstream is None:
            target = route_target(route, "/v1/responses")
            destination = await asyncio.to_thread(
                generation.egress_policy.authorize, lease, target, route=route
            )
            await attempt.check_current()
            headers = build_managed_upstream_headers(
                {
                    name: value
                    for name, value in self.websocket.headers.items()
                    if name.lower()
                    not in {"connection", "upgrade", "content-length", "transfer-encoding"}
                    and not name.lower().startswith("sec-websocket-")
                },
                lease,
                target,
                method="GET",
                resolved_addresses=destination.addresses,
                route=route,
            )
            headers.pop("host", None)
            self.context = websocket_connection(destination, headers, generation.tls_context)
            try:
                self.upstream = await self.context.__aenter__()
                self._context_entered = True
            finally:
                self._context_settled.set()
            if self._finishing:
                raise asyncio.CancelledError
            self.generation, self.route = generation, route
            self.runtime.retain(generation)
            self.binding = ResourceBinding(
                provider_id=self.id,
                principal_id=principal.id,
                route_id=route.id,
                account_ref=selection.account_ref,
                adapter="openai-responses",
                expires_at=None,
                authority_fingerprint=attempt.account_key,
                target_fingerprint=generation.target_key(route.id),
                generation=generation.number,
            )
        self.websocket.scope["gateway"] = GatewayRequestContext(
            principal,
            route,
            "openai-responses",
            "",
            generation,
            generation.catalog,
            selection,
            operation=operation,
        )
        self.observer = StreamObserver(
            "openai-responses", generation.snapshot.limits, operation.deadline
        )
        await attempt.check_current()
        attempt.mark_sending()
        await self.upstream.send(routed_response_create_frame(frame, route))
        attempt.mark_accepted()

    async def observe(self, frame: Any) -> bool:
        operation, observer = self.operation, self.observer
        if operation is None or observer is None or operation.current_attempt is None:
            raise _invalid("gateway_upstream_frame", 502)
        attempt = operation.current_attempt
        if (
            not isinstance(frame, str)
            or len(frame.encode("utf-8")) > operation.generation.snapshot.limits.max_frame_bytes
        ):
            raise _invalid("gateway_frame_too_large", 502)
        try:
            event = json.loads(frame)
        except (ValueError, RecursionError):
            raise _invalid("gateway_upstream_frame", 502) from None
        if not isinstance(event, dict):
            raise _invalid("gateway_upstream_frame", 502)
        try:
            observer.inspect(b"data: " + json.dumps(event).encode() + b"\n\n")
        except GatewayPublicError:
            raise GatewayPublicError(
                status_code=502, code="gateway_upstream_error", message="Upstream request failed"
            ) from None
        finally:
            attempt.usage = observer.usage
        response = event.get("response")
        if isinstance(response, dict) and isinstance(response.get("id"), str):
            await self.runtime.resources.bind(
                ResourceBinding(
                    provider_id=response["id"],
                    principal_id=operation.principal.id,
                    route_id=operation.route.id,
                    account_ref=attempt.account_ref,
                    adapter="openai-responses",
                    expires_at=time.time()
                    + operation.generation.snapshot.limits.resource_ttl_seconds,
                    authority_fingerprint=attempt.account_key,
                    target_fingerprint=operation.generation.target_key(operation.route.id),
                    generation=operation.generation.number,
                )
            )
        attempt.mark_output()
        terminal = observer.terminal is not None
        if terminal:
            await operation.close("success" if observer.terminal == "success" else "failed")
            self.operation = None
            self.observer = None
        # Release the turn slot before a client can react to the terminal event.
        await self.wait_owned(self.websocket.send_text(frame), operation.deadline)
        return terminal

    async def run(self) -> None:
        client = self.spawn(self.websocket.receive_text())
        provider: asyncio.Task[Any] | None = None
        starting: asyncio.Task[Any] | None = None
        idle_deadline = time.monotonic() + self.runtime.snapshot.limits.websocket_idle_seconds
        result: TerminalResult = "failed"
        origin: FailureOrigin = "network"
        try:
            while True:
                deadline = self.operation.deadline if self.operation is not None else idle_deadline
                pending = {
                    client,
                    *([provider] if provider else []),
                    *([starting] if starting else []),
                }
                done, _ = await asyncio.wait(
                    pending,
                    timeout=max(0, deadline - time.monotonic()),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    raise _invalid("gateway_websocket_timeout", 504)
                if starting is not None and starting in done:
                    starting.result()
                    starting = None
                    if provider is None:
                        provider = self.spawn(anext(self.upstream.__aiter__()))
                if provider is not None and provider in done:
                    terminal = await self.observe(provider.result())
                    if terminal:
                        idle_deadline = (
                            time.monotonic() + self.runtime.snapshot.limits.websocket_idle_seconds
                        )
                    provider = self.spawn(anext(self.upstream.__aiter__()))
                if client in done:
                    frame = client.result()
                    if len(frame.encode("utf-8")) > self.runtime.snapshot.limits.max_frame_bytes:
                        raise _invalid("gateway_frame_too_large")
                    try:
                        parsed = json.loads(frame)
                    except (ValueError, RecursionError):
                        raise _invalid("gateway_request_invalid") from None
                    if isinstance(parsed, dict) and parsed.get("type") == "response.create":
                        generation, principal, route, payload = self.authorize(frame)
                        if self.operation is not None or starting is not None:
                            raise _invalid("gateway_generation_in_progress", 409)
                        # Route ownership starts before queueing/acquisition, so
                        # revocation can cancel a not-yet-connected session.
                        self.route = route
                        self.operation = GatewayOperation(
                            self.runtime,
                            generation=generation,
                            principal=principal,
                            route=route,
                            ingress_protocol="openai-responses",
                            target_protocol="openai-responses",
                            dispatch_plan="native-websocket",
                        )
                        # The session owns cancellation of the relay; canceling
                        # both owners must not interrupt the session finalizer.
                        self.operation.owner_task = None
                        starting = self.spawn(
                            self.start_turn(frame, generation, principal, route, payload)
                        )
                    elif isinstance(parsed, dict) and parsed.get("type") == "response.cancel":
                        generation, principal = self.authenticate()
                        if (
                            self.operation is None
                            or self.operation.current_attempt is None
                            or self.upstream is None
                        ):
                            raise _invalid("gateway_frame_invalid")
                        generation.authorizer.authorize(
                            principal,
                            scope="inference",
                            protocol="openai-responses",
                            public_model=self.operation.route.public_model,
                            transport="websocket",
                            defer_cost_to_admission=True,
                        )
                        assert self.binding is not None
                        generation.validate_binding(self.binding)
                        await self.operation.current_attempt.check_current()
                        await asyncio.wait_for(
                            self.upstream.send(frame), max(0, deadline - time.monotonic())
                        )
                    else:
                        raise _invalid("gateway_frame_invalid")
                    client = self.spawn(self.websocket.receive_text())
        except (WebSocketDisconnect, asyncio.CancelledError):
            result, origin = "cancelled", "gateway"
        except GatewayPublicError as exc:
            origin = "gateway" if exc.status_code < 500 else "upstream"
            with contextlib.suppress(Exception):
                await self.wait_owned(
                    self.websocket.send_json(
                        {"type": "error", "error": {"code": exc.code, "message": exc.message}}
                    ),
                    deadline,
                )
        except Exception:
            with contextlib.suppress(Exception):
                await self.wait_owned(
                    self.websocket.send_json(
                        {
                            "type": "error",
                            "error": {
                                "code": "gateway_upstream_error",
                                "message": "Upstream request failed",
                            },
                        },
                    ),
                    deadline,
                )
        finally:
            self._finish_task = asyncio.create_task(self.finish(result, origin))
            await asyncio.shield(self._finish_task)

    async def _close_context(self) -> None:
        while not self._context_settled.is_set():
            try:
                await self._context_settled.wait()
            except asyncio.CancelledError:
                # The caller's cleanup budget may expire before a native
                # connector settles. Keep ownership until late entry can be
                # paired with exactly one exit.
                pass
        if self._context_entered:
            await self.context.__aexit__(None, None, None)
            self._context_entered = False

    async def finish(self, result: TerminalResult, origin: FailureOrigin) -> None:
        self._finishing = True
        deadline = time.monotonic() + self.runtime.snapshot.limits.shutdown_cleanup_seconds
        tasks = set(self.tasks)
        for task in tasks:
            task.cancel()
        if self.context is not None:
            tasks.add(self.spawn(self._close_context()))
        tasks.add(self.spawn(self.websocket.close(code=1000 if result == "cancelled" else 1011)))
        try:
            await asyncio.wait(tasks, timeout=max(0, deadline - time.monotonic()))
        finally:
            # A cancellation-resistant transport stays explicitly owned, but
            # cannot defer settlement or release of the logical operation.
            for task in tasks:
                if not task.done():
                    task.cancel()
                    self.runtime._cleanup_tasks.add(task)
                    task.add_done_callback(self.runtime._cleanup_finished)
            try:
                if self.operation is not None:
                    await self.operation.close(
                        result, failure_origin=origin, cleanup_deadline=deadline
                    )
            finally:
                try:
                    if self.generation is not None:
                        await self.runtime.release(self.generation, cleanup_deadline=deadline)
                finally:
                    self.runtime.active_work.pop(self.id, None)
                    if not self.runtime.active_work:
                        self.runtime._work_empty.set()
                    self.finished.set()


async def dispatch_native_responses_websocket(websocket: WebSocket, proxy: Any) -> None:
    runtime = websocket.app.state.gateway_runtime
    principal = websocket.scope.get("gateway_principal")
    if not isinstance(principal, GatewayPrincipal) or not runtime.status().ready:
        await websocket.close(code=1008, reason="gateway authentication required")
        return
    session = GatewayWebSocketSession(websocket, runtime, principal)
    await websocket.accept()
    await session.run()
