"""Pure native gateway dispatch using existing proxy HTTP transport."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, cast
from urllib.parse import parse_qsl, urlencode, urlparse

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from headroom.proxy.gateway.capabilities import requested_features
from headroom.proxy.gateway.config import Protocol, RouteConfig
from headroom.proxy.gateway.context import GatewayRequestContext
from headroom.proxy.gateway.credentials import CredentialLease
from headroom.proxy.gateway.destinations import path_within, validate_path
from headroom.proxy.gateway.egress import build_managed_upstream_headers
from headroom.proxy.gateway.errors import (
    GatewayAuthorizationError,
    GatewayPublicError,
    protocol_error_payload,
)
from headroom.proxy.gateway.execution import GatewayAttempt, GatewayOperation
from headroom.proxy.gateway.models import Capability
from headroom.proxy.gateway.observability import RetryReason, TerminalResult
from headroom.proxy.gateway.resources import ResourceBinding
from headroom.proxy.gateway.routing import (
    ProviderContract,
    RetryDecision,
    TransportFailure,
)
from headroom.proxy.gateway.streaming import (
    SSEFrames,
    StreamObserver,
    event_data,
    observed_body,
    responses_refusal,
)
from headroom.proxy.gateway.transport import private_transport
from headroom.proxy.gateway.usage import (
    CostEvaluation,
    UsageObservation,
    conservative_cost_bound,
    normalize_usage,
)

_REQUEST_HEADER_DENYLIST = frozenset(
    {"host", "content-length", "connection", "transfer-encoding", "upgrade"}
)
_RESPONSE_HEADER_DENYLIST = frozenset(
    {"content-length", "connection", "transfer-encoding", "content-encoding"}
)
_QUERY_CREDENTIAL_NAMES = frozenset(
    {"api_key", "key", "access_token", "token", "x-api-key", "x-goog-api-key"}
)

DispatchContract = Literal["strict-native", "routed-native", "translated"]


def route_target(route: RouteConfig, path: str) -> str:
    validate_path(path)
    if route.provider == "compatible" and not path_within(path, route.upstream_path_prefix):
        prefix = next((p for p in ("/v1beta/", "/v1/") if path.startswith(p)), None)
        if prefix is None:
            raise ValueError("unsupported compatible route path")
        path = route.upstream_path_prefix.rstrip("/") + "/" + path.removeprefix(prefix)
    return route.upstream_origin.rstrip("/") + path


def rewrite_native_model_path(
    path: str, *, protocol: str, public_model: str, upstream_model: str
) -> str:
    """Patch only the authorized model span, never a structural URL component."""
    if protocol in {"gemini-generate", "vertex-generate"}:
        prefix, separator, endpoint = path.rpartition("/models/")
        model, colon, method = endpoint.rpartition(":")
        if (
            separator
            and colon
            and model == public_model
            and method in {"generateContent", "streamGenerateContent"}
        ):
            return prefix + separator + upstream_model + colon + method
    elif protocol == "bedrock-invoke":
        resource, slash, method = path.rpartition("/")
        if (
            resource == "/model/" + public_model
            and slash
            and method in {"invoke", "invoke-with-response-stream"}
        ):
            return "/model/" + upstream_model + slash + method
    raise GatewayAuthorizationError(
        status_code=400,
        code="gateway_request_invalid",
        message="Model endpoint cannot be represented",
    )


def _upstream_error() -> GatewayPublicError:
    return GatewayPublicError(
        status_code=502, code="gateway_upstream_error", message="Upstream request failed"
    )


def _safe_response_headers(upstream: Any) -> dict[str, str]:
    content_type = upstream.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    return (
        {"content-type": content_type}
        if content_type in {"application/json", "text/event-stream", "application/octet-stream"}
        else {}
    )


async def _send_managed(
    request: Request,
    proxy: Any,
    route: RouteConfig,
    lease: CredentialLease,
    target: str,
    client_headers: dict[str, str],
    body: bytes,
    attempt: GatewayAttempt,
) -> Any:
    destination = await asyncio.to_thread(
        request.state.gateway_generation.egress_policy.authorize, lease, target, route=route
    )
    headers = build_managed_upstream_headers(
        client_headers,
        lease,
        target,
        resolved_addresses=destination.addresses,
        route=route,
        method=request.method,
        body=body,
    )
    if request.state.gateway_generation.http_client is None:
        raise _upstream_error()
    upstream_request = request.state.gateway_generation.http_client.build_request(
        request.method, target, headers=headers, content=body
    )
    upstream_request.extensions["gateway_destination"] = destination
    await attempt.check_current()
    attempt.mark_sending()
    with private_transport():
        upstream = await request.state.gateway_generation.http_client.send(
            upstream_request, stream=True, follow_redirects=False
        )
    attempt.upstream_close = upstream.aclose
    return upstream


@dataclass(frozen=True, slots=True)
class DispatchPlan:
    contract: DispatchContract
    capabilities: frozenset[Capability]
    body: bytes
    mutation_reasons: tuple[str, ...]


class GatewayDispatcher:
    """Resolve explicit native entity mutations before provider I/O."""

    @staticmethod
    def resolve_body(
        body: bytes,
        *,
        public_model: str,
        upstream_model: str,
        declared_contract: DispatchContract,
    ) -> DispatchPlan:
        if declared_contract == "strict-native":
            if public_model != upstream_model:
                raise GatewayAuthorizationError(
                    status_code=500,
                    code="gateway_route_invalid",
                    message="Strict-native route cannot rewrite its model",
                )
            return DispatchPlan(
                contract="strict-native",
                capabilities=frozenset({Capability.GENERATE}),
                body=body,
                mutation_reasons=(),
            )
        if declared_contract == "routed-native":
            rewritten = rewrite_routed_native_model(
                body,
                public_model=public_model,
                upstream_model=upstream_model,
            )
            reasons = ("model_alias",) if rewritten is not body else ()
            return DispatchPlan(
                contract="routed-native",
                capabilities=frozenset({Capability.GENERATE}),
                body=rewritten,
                mutation_reasons=reasons,
            )
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_translation_required",
            message="Translated dispatch requires a qualified protocol adapter",
        )


async def _iter_upstream_bytes(
    upstream: Any, *, frame_security: bool = True
) -> AsyncIterator[bytes]:
    """Yield original, undecoded HTTP entity chunks once."""

    async def raw() -> AsyncIterator[bytes]:
        if upstream.is_stream_consumed:
            yield upstream.content
        else:
            async for chunk in upstream.aiter_raw():
                yield chunk

    is_sse = (
        frame_security
        and getattr(upstream, "headers", {})
        .get("content-type", "")
        .split(";", 1)[0]
        .strip()
        .lower()
        == "text/event-stream"
    )
    frames = SSEFrames(1_048_576)
    try:
        async for chunk in raw():
            if not is_sse:
                yield chunk
                continue
            for frame in frames.feed(chunk):
                data = event_data(frame)
                if data and data != "[DONE]":
                    payload = json.loads(data)
                    if isinstance(payload, dict) and (
                        payload.get("error")
                        or payload.get("type") in {"error", "response.failed", "response.error"}
                    ):
                        raise _upstream_error()
                yield frame
        if frames.pending:
            raise _upstream_error()
    except Exception:
        raise _upstream_error() from None


async def _bind_response(
    request: Request, operation: GatewayOperation, payload: dict[str, Any]
) -> None:
    response = payload.get("response", payload)
    if not isinstance(response, dict) or not isinstance(response.get("id"), str):
        return
    attempt = operation.current_attempt
    assert attempt is not None
    await operation.runtime.resources.bind(
        ResourceBinding(
            provider_id=response["id"],
            principal_id=operation.principal.id,
            route_id=operation.route.id,
            account_ref=attempt.account_ref,
            adapter="openai-responses",
            expires_at=time.time() + operation.generation.snapshot.limits.resource_ttl_seconds,
            authority_fingerprint=operation.generation.account_key(attempt.account_ref),
            target_fingerprint=operation.generation.target_key(operation.route.id),
            generation=operation.generation.number,
        )
    )


def _retry_contract(route: RouteConfig) -> ProviderContract:
    # Only documented API rejection shapes below can prove rejection. Compatible
    # endpoints do not inherit public-provider billing/acceptance guarantees.
    rejected = (
        frozenset({"rate_limit", "unavailable"}) if route.provider == "anthropic" else frozenset()
    )
    return ProviderContract(
        max_attempts=3,
        retryable_failures=frozenset({"connect_failed"}) | rejected,
        rejected_failures=rejected,
        max_retry_after=60,
    )


def _rejection(route: RouteConfig, status: int, payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    if route.provider == "anthropic" and isinstance(error, dict):
        if status == 429 and error.get("type") == "rate_limit_error":
            return "rate_limit"
        if status == 529 and error.get("type") == "overloaded_error":
            return "unavailable"
    return None


async def _run_attempts(
    request: Request,
    proxy: Any,
    operation: GatewayOperation,
    *,
    target: str,
    body: bytes,
    cost: CostEvaluation,
    transport: str,
    features: frozenset[str],
    binding: ResourceBinding | None,
) -> Any:
    generation, route, runtime = operation.generation, operation.route, operation.runtime
    contract = _retry_contract(route)
    retry = None
    while True:
        # Every attempt repeats authorization against the admitted catalog and
        # current revocation epochs; it never recaptures tariff/semantic policy.
        generation.authorizer.authorize(
            operation.principal,
            scope="inference",
            protocol=operation.ingress_protocol,
            defer_cost_to_admission=True,
            public_model=route.public_model,
            transport=transport,
            features=features,
        )
        selection = runtime.router.select(
            route,
            operation.principal,
            resource_binding=binding,
            eligible_accounts=generation.catalog.eligible_accounts(
                route,
                protocol=operation.ingress_protocol,
                transport=transport,
                features=features,
            ),
            authority_keys=dict(generation.authorities),
            target_key=generation.target_key(route.id),
        )
        attempt = await operation.start_attempt(selection.account_ref, cost, retry=retry)
        if cost.provider_contract == "openai-responses-resource-v1":
            attempt.usage = UsageObservation(
                charge_free=True, availability="complete", provenance="resource-control"
            )
        request.state.gateway = GatewayRequestContext(
            principal=operation.principal,
            route=route,
            ingress_protocol=operation.ingress_protocol,
            request_id=request.headers.get("x-request-id", ""),
            generation=generation,
            catalog=generation.catalog,
            account_selection=selection,
            operation=operation,
            mutation_reasons=getattr(request.state, "gateway_mutation_reasons", ()),
        )
        retry = None
        try:
            await attempt.check_current()
            lease = await asyncio.wait_for(
                generation.broker.acquire(route, account_ref=selection.account_ref),
                max(0, operation.deadline - time.monotonic()),
            )
            attempt.lease_generation = lease.generation
            await attempt.check_current()
            headers = {
                name: value
                for name, value in request.headers.items()
                if name.lower() not in _REQUEST_HEADER_DENYLIST
            }
            upstream = await asyncio.wait_for(
                _send_managed(request, proxy, route, lease, target, headers, body, attempt),
                max(0, operation.deadline - time.monotonic()),
            )
            if 200 <= upstream.status_code < 300:
                attempt.mark_accepted()
                return upstream
            raw = await observed_body(
                _iter_upstream_bytes(upstream),
                maximum=generation.snapshot.limits.max_observed_json_bytes,
                deadline=operation.deadline,
            )
            try:
                payload = json.loads(raw)
            except (ValueError, RecursionError):
                payload = {}
            if not isinstance(payload, dict):
                payload = {}
            if cost.provider_contract != "openai-responses-resource-v1":
                attempt.usage = normalize_usage(operation.target_protocol, payload)
            failure = _rejection(route, upstream.status_code, payload)
            if failure is not None:
                attempt.mark_rejected(failure, contract)
                # These rejections are pre-generation. Preserve any reported
                # usage instead of discarding contradictory charge evidence.
                if attempt.usage.availability == "unknown":
                    attempt.usage = UsageObservation(
                        charge_free=True, availability="complete", provenance="provider-rejection"
                    )
                retry = RetryDecision.decide(
                    TransportFailure(failure, upstream.headers.get("retry-after")),
                    attempt.exposure.value,
                    contract,
                    attempt_count=len(operation.attempts) + 1,
                    deadline=operation.deadline,
                    now=time.monotonic(),
                    policy=route.retry,
                )
                runtime.router.cool_down(
                    selection.account_ref,
                    quota_key=route.selection.quota_group or route.id,
                    until=time.time()
                    + (retry.delay if retry.allowed else route.retry.max_retry_after_seconds),
                )
            await attempt.close(
                "rejected",
                failure_origin="upstream",
                retry_reason=cast(RetryReason, retry.reason)
                if retry is not None and retry.allowed
                else "none",
            )
            if retry is None or not retry.allowed:
                raise _upstream_error()
        except (httpx.ConnectError, httpx.ConnectTimeout):
            # HTTPX connect/TLS establishment failure proves no request write.
            attempt.mark_unsent_connect_failure()
            retry = RetryDecision.decide(
                TransportFailure("connect_failed", None),
                attempt.exposure.value,
                contract,
                attempt_count=len(operation.attempts) + 1,
                deadline=operation.deadline,
                now=time.monotonic(),
                policy=route.retry,
            )
            await attempt.close(
                "failed",
                failure_origin="network",
                retry_reason="connect" if retry.allowed else "none",
            )
            if not retry.allowed:
                raise _upstream_error() from None
        if retry is None or not retry.allowed:
            raise _upstream_error()
        # The completed attempt owns no socket/reservation during bounded backoff.
        await asyncio.wait_for(
            asyncio.sleep(retry.delay), max(0, operation.deadline - time.monotonic())
        )


class OwnedStreamingResponse(StreamingResponse):
    """Own admission/upstream even when ASGI never starts body iteration."""

    def __init__(
        self, operation: GatewayOperation, content: AsyncIterator[bytes], **kwargs: Any
    ) -> None:
        super().__init__(content, **kwargs)
        self.operation = operation

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await asyncio.wait_for(
                super().__call__(scope, receive, send),
                max(0, self.operation.deadline - time.monotonic()),
            )
        finally:
            try:
                await cast(Any, self.body_iterator).aclose()
            finally:
                await self.operation.close("cancelled", failure_origin="network")


async def _owned_stream(
    request: Request,
    operation: GatewayOperation,
    upstream: Any,
    *,
    translated: bool,
    expected_candidates: int | None = None,
) -> AsyncIterator[bytes]:
    attempt = operation.current_attempt
    assert attempt is not None
    observer = StreamObserver(
        operation.target_protocol,
        operation.generation.snapshot.limits,
        operation.deadline,
        expected_candidates=expected_candidates,
    )
    result: TerminalResult = "failed"
    operation.owner_task = asyncio.current_task()

    async def observed() -> AsyncGenerator[bytes, None]:
        try:
            async for frame in observer.observe(
                _iter_upstream_bytes(upstream, frame_security=False)
            ):
                attempt.usage = observer.usage
                if (
                    operation.target_protocol == "openai-responses"
                    and observer.last_event is not None
                ):
                    if observer.last_event.get("type") in {
                        "response.created",
                        "response.completed",
                    }:
                        await _bind_response(request, operation, observer.last_event)
                yield frame
        finally:
            if not attempt.closed:
                attempt.usage = observer.usage

    source = observed()
    output = source
    if translated:
        from headroom.proxy.gateway.protocols.events import translate_sse_stream

        output = translate_sse_stream(
            operation.target_protocol,
            operation.ingress_protocol,
            source,
            public_model=operation.route.public_model,
        )
    try:
        async for chunk in output:
            attempt.mark_output()
            yield chunk
        result = "success"
    except asyncio.CancelledError:
        result = "cancelled"
        raise
    except Exception:
        # ASGI middleware may convert an aborted body to normal HTTP EOF. Emit
        # a protocol error (never a success terminal) so clients see the failure.
        error = protocol_error_payload(operation.ingress_protocol, _upstream_error())
        prefix = (
            b"event: error\n"
            if operation.ingress_protocol in {"anthropic-messages", "openai-responses"}
            else b""
        )
        yield prefix + b"data: " + json.dumps(error, separators=(",", ":")).encode() + b"\n\n"
    finally:
        try:
            await output.aclose()
            await source.aclose()
        finally:
            await operation.close(
                result, failure_origin="none" if result == "success" else "network"
            )


def rewrite_routed_native_model(
    body: bytes,
    *,
    public_model: str,
    upstream_model: str,
) -> bytes:
    """Return original bytes for identity routes; patch only model otherwise."""

    if public_model == upstream_model:
        return body
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_request_invalid",
            message="Gateway request body is not valid JSON",
        ) from exc
    if not isinstance(payload, dict) or payload.get("model") != public_model:
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_model_mismatch",
            message="Request model does not match the authorized route model",
        )
    # Walk only top-level JSON members to retain all unrelated entity bytes,
    # including numeric spellings, whitespace, escapes and signed blocks.
    text = body.decode("utf-8")
    decoder = json.JSONDecoder()
    position = text.index("{") + 1
    replacements = []
    while True:
        while text[position].isspace():
            position += 1
        if text[position] == "}":
            break
        key, position = decoder.raw_decode(text, position)
        while text[position].isspace():
            position += 1
        position += 1  # colon; the complete entity has already been validated
        while text[position].isspace():
            position += 1
        start = position
        _, position = decoder.raw_decode(text, position)
        if key == "model":
            replacements.append((start, position))
        while text[position].isspace():
            position += 1
        if text[position] == "}":
            break
        position += 1
    if len(replacements) != 1:
        raise GatewayAuthorizationError(
            status_code=400,
            code="gateway_request_invalid",
            message="Model must appear exactly once",
        )
    start, end = replacements[0]
    return (text[:start] + json.dumps(upstream_model, ensure_ascii=False) + text[end:]).encode()


async def dispatch_native_http(
    request: Request,
    proxy: Any,
    protocol: Protocol,
    *,
    public_model: str | None = None,
) -> Response:
    """Authorize, lease, and forward one native HTTP entity without optimization."""

    operation: GatewayOperation | None = None
    handed_off = False
    try:
        body = await request.body()
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise GatewayAuthorizationError(
                status_code=400,
                code="gateway_request_invalid",
                message="Gateway request body must be a JSON object",
            )
        body_model = payload.get("model")
        if public_model is None and not isinstance(body_model, str):
            raise GatewayAuthorizationError(
                status_code=400,
                code="gateway_model_required",
                message="A string model is required",
            )
        requested_model = public_model if public_model is not None else body_model
        assert isinstance(requested_model, str)
        principal = request.state.gateway_principal
        route = request.state.gateway_generation.authorizer.authorize(
            principal,
            defer_cost_to_admission=True,
            scope="inference",
            protocol=protocol,
            public_model=requested_model,
            transport="http-stream"
            if payload.get("stream") is True
            or request.url.path.endswith((":streamGenerateContent", "/invoke-with-response-stream"))
            else "http-json",
            features=requested_features(protocol, payload),
        )
        resource_binding = None
        if protocol == "openai-responses":
            previous_response_id = payload.get("previous_response_id")
            if previous_response_id is not None:
                if not isinstance(previous_response_id, str):
                    raise GatewayAuthorizationError(
                        status_code=400,
                        code="gateway_request_invalid",
                        message="previous_response_id must be a string",
                    )
                resource_binding = await request.app.state.gateway_runtime.resources.authorize(
                    previous_response_id,
                    principal_id=principal.id,
                    route_id=route.id,
                    now=time.time(),
                )
        if resource_binding is not None:
            request.state.gateway_generation.validate_binding(resource_binding)
        target_protocol = protocol
        streaming = bool(
            payload.get("stream")
            or request.url.path.endswith((":streamGenerateContent", "/invoke-with-response-stream"))
        )
        translated = protocol not in route.native_protocols
        if translated:
            if route.translation != "qualified" or len(route.native_protocols) != 1:
                raise GatewayAuthorizationError(
                    status_code=400,
                    code="gateway_unsupported_capability",
                    message="Route does not qualify this protocol translation",
                )
            target_protocol = route.native_protocols[0]
            from headroom.proxy.gateway.protocols import translate

            translated_payload = translate(protocol, target_protocol, payload)
            if (
                target_protocol == "anthropic-messages"
                and translated_payload.get("max_tokens") is None
            ):
                raise GatewayAuthorizationError(
                    status_code=400,
                    code="gateway_unsupported_capability",
                    message="This route requires an explicit output token limit",
                )
            if streaming:
                translated_payload["stream"] = True
                if target_protocol == "openai-chat":
                    translated_payload["stream_options"] = {"include_usage": True}
            if target_protocol in ("openai-chat", "anthropic-messages"):
                translated_payload["model"] = route.upstream_model
            outbound_body = json.dumps(
                translated_payload, ensure_ascii=False, separators=(",", ":")
            ).encode()
        else:
            outbound_body = body

        upstream_paths = {
            "openai-chat": "/v1/chat/completions",
            "openai-responses": "/v1/responses",
            "anthropic-messages": "/v1/messages",
            "gemini-generate": f"/v1beta/models/{route.upstream_model}:{'streamGenerateContent' if streaming else 'generateContent'}",
        }
        upstream_path = (
            upstream_paths.get(target_protocol, request.url.path)
            if translated
            else request.url.path
        )
        if (
            not translated
            and public_model is not None
            and route.public_model != route.upstream_model
        ):
            upstream_path = rewrite_native_model_path(
                upstream_path,
                protocol=protocol,
                public_model=route.public_model,
                upstream_model=route.upstream_model,
            )
        target = route_target(route, upstream_path)
        mutation_reasons: tuple[str, ...] = ()
        if not translated and urlparse(target).path != request.url.path:
            mutation_reasons = ("endpoint_route",)
        safe_query = [
            (name, value)
            for name, value in parse_qsl(request.url.query, keep_blank_values=True)
            if name.casefold() not in _QUERY_CREDENTIAL_NAMES
        ]
        if safe_query:
            target += "?" + urlencode(safe_query)
        if not translated and public_model is None:
            plan = GatewayDispatcher.resolve_body(
                body,
                public_model=route.public_model,
                upstream_model=route.upstream_model,
                declared_contract=route.body_contract,
            )
            outbound_body = plan.body
            mutation_reasons += plan.mutation_reasons
        request.state.gateway_mutation_reasons = mutation_reasons
        generation = request.state.gateway_generation
        runtime = request.app.state.gateway_runtime
        expected_candidates = None
        if target_protocol in {"gemini-generate", "vertex-generate"}:
            options = payload.get("generationConfig", {})
            expected_candidates = (
                options.get("candidateCount", 1) if isinstance(options, dict) else None
            )
            if type(expected_candidates) is not int or not 1 <= expected_candidates <= 8:
                raise GatewayAuthorizationError(
                    status_code=400,
                    code="gateway_unsupported_capability",
                    message="Candidate count is not supported",
                )
        transport = "http-stream" if streaming else "http-json"
        operation = GatewayOperation(
            runtime,
            generation=generation,
            principal=principal,
            route=route,
            ingress_protocol=protocol,
            target_protocol=target_protocol,
            dispatch_plan=outbound_body,
        )
        cost = conservative_cost_bound(
            route, payload, qualified_contracts=runtime.dependencies.qualified_cost_contracts
        )
        upstream = await _run_attempts(
            request,
            proxy,
            operation,
            target=target,
            body=outbound_body,
            cost=cost,
            transport=transport,
            features=requested_features(protocol, payload),
            binding=resource_binding,
        )
        response_headers = _safe_response_headers(upstream)
        if transport == "http-stream":
            if (
                upstream.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                != "text/event-stream"
            ):
                raise _upstream_error()
            handed_off = True
            return OwnedStreamingResponse(
                operation,
                _owned_stream(
                    request,
                    operation,
                    upstream,
                    translated=translated,
                    expected_candidates=expected_candidates,
                ),
                status_code=upstream.status_code,
                headers=response_headers,
                media_type=None,
            )
        upstream_body = await observed_body(
            _iter_upstream_bytes(upstream),
            maximum=generation.snapshot.limits.max_observed_json_bytes,
            deadline=operation.deadline,
        )
        try:
            response_payload = json.loads(upstream_body)
        except (ValueError, RecursionError):
            raise _upstream_error() from None
        if not isinstance(response_payload, dict):
            raise _upstream_error()
        assert operation.current_attempt is not None
        operation.current_attempt.usage = normalize_usage(target_protocol, response_payload)
        if (
            response_payload.get("error")
            or (
                target_protocol == "openai-responses"
                and (
                    response_payload.get("status") in {"failed", "incomplete", "cancelled"}
                    or responses_refusal(response_payload)
                )
            )
            or (
                target_protocol == "openai-chat"
                and any(
                    choice.get("finish_reason") == "content_filter"
                    or choice.get("message", {}).get("refusal")
                    for choice in response_payload.get("choices", [])
                )
            )
            or (
                target_protocol == "anthropic-messages"
                and response_payload.get("stop_reason")
                in {"refusal", "model_context_window_exceeded"}
            )
            or (
                target_protocol in {"gemini-generate", "vertex-generate"}
                and any(
                    candidate.get("finishReason") not in {None, "STOP", "MAX_TOKENS"}
                    for candidate in response_payload.get("candidates", [])
                )
            )
        ):
            raise _upstream_error()
        if protocol == "openai-responses":
            await _bind_response(request, operation, response_payload)
        if translated:
            from headroom.proxy.gateway.protocols import translate_response

            response: Response = JSONResponse(
                translate_response(
                    target_protocol, protocol, response_payload, public_model=route.public_model
                ),
                status_code=upstream.status_code,
            )
        else:
            response = Response(
                content=upstream_body, status_code=upstream.status_code, headers=response_headers
            )
        await operation.close("success")
        return response
    except GatewayPublicError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content=protocol_error_payload(protocol, exc),
        )
    except Exception:
        return JSONResponse(
            status_code=502,
            content=protocol_error_payload(protocol, _upstream_error()),
        )
    finally:
        if operation is not None and not handed_off:
            await operation.close("failed", failure_origin="network")


def gateway_model_catalog(
    request: Request, model_id: str | None = None, *, protocol: str | None = None
) -> JSONResponse:
    if protocol is None and "anthropic-version" in request.headers:
        protocol = "anthropic-messages"
    principal = request.state.gateway_principal
    catalog = request.state.gateway_generation.catalog

    def catalog_error(status: int, code: str) -> JSONResponse:
        if protocol == "anthropic-messages":
            return JSONResponse(
                status_code=status,
                content={
                    "type": "error",
                    "error": {
                        "type": "not_found_error" if status == 404 else "permission_error",
                        "message": "Model unavailable" if status == 404 else "Scope denied",
                    },
                },
            )
        return JSONResponse(status_code=status, content={"error": {"code": code}})

    if "models" not in principal.scopes:
        return catalog_error(403, "gateway_scope_forbidden")
    routes = catalog.visible_routes(principal)
    if protocol:
        routes = tuple(r for r in routes if protocol in r.protocols)
    if model_id is not None:
        routes = tuple(r for r in routes if r.id == model_id)
        if not routes:
            return catalog_error(404, "gateway_model_unavailable")
    data = []
    for route in routes:
        item = {
            "id": route.id,
            "object": "model",
            "owned_by": "headroom-gateway",
            "headroom": {
                "protocols": route.protocols,
                "body_contract": route.body_contract,
                "catalog_revision": catalog.revision,
                "generation": catalog.generation,
                "provenance": route.provenance,
                "state": route.state,
                "capabilities": [
                    {"protocol": p, "transport": t, "features": f} for p, t, f in route.capabilities
                ],
                "cost_available": route.tariff_revision is not None,
                "tariff_revision": route.tariff_revision,
            },
        }
        if protocol == "anthropic-messages":
            item.pop("object")
            item.pop("owned_by")
            item.update(
                type="model",
                display_name=route.id,
                created_at=datetime.fromtimestamp(catalog.published_at, timezone.utc).isoformat(),
            )
            item["headroom"]["created_at_provenance"] = "gateway_catalog_publication"
        elif protocol == "gemini-generate":
            item["name"] = "models/" + route.id
            item["supportedGenerationMethods"] = ["generateContent"] + (
                ["streamGenerateContent"]
                if any(t == "http-stream" for p, t, _f in route.capabilities if p == protocol)
                else []
            )
        data.append(item)
    if protocol == "anthropic-messages" and model_id is None:
        return JSONResponse(
            {
                "data": data,
                "has_more": False,
                "first_id": data[0]["id"] if data else None,
                "last_id": data[-1]["id"] if data else None,
            }
        )
    return JSONResponse(
        data[0]
        if model_id is not None
        else {"models": data}
        if protocol == "gemini-generate"
        else {"object": "list", "data": data}
    )


async def dispatch_stateful_response_http(
    request: Request,
    proxy: Any,
    sub_path: str,
) -> Response:
    """Affine metadata/control work owns concurrency but no new generation charge."""
    operation: GatewayOperation | None = None
    try:
        response_id = sub_path.split("/", 1)[0]
        principal = request.state.gateway_principal
        generation = request.state.gateway_generation
        runtime = request.app.state.gateway_runtime
        if "inference" not in principal.scopes:
            raise GatewayAuthorizationError(
                status_code=403, code="gateway_scope_denied", message="Gateway scope denied"
            )
        binding = await runtime.resources.authorize(
            response_id,
            principal_id=principal.id,
            route_id=None,
            allowed_route_ids=principal.routes,
            now=time.time(),
        )
        route = generation.catalog.route_for_id(binding.route_id)
        if route is None:
            raise GatewayAuthorizationError(
                status_code=404,
                code="gateway_resource_not_found",
                message="Stateful resource not found",
            )
        generation.validate_binding(binding)
        operation = GatewayOperation(
            runtime,
            generation=generation,
            principal=principal,
            route=route,
            ingress_protocol="openai-responses",
            target_protocol="openai-responses",
            dispatch_plan="resource-control",
        )
        upstream = await _run_attempts(
            request,
            proxy,
            operation,
            target=route_target(route, request.url.path),
            body=await request.body(),
            cost=CostEvaluation(
                known_micro_usd=0,
                reserved_upper_micro_usd=0,
                complete=True,
                basis="provider_reported",
                provider_contract="openai-responses-resource-v1",
                qualified_bound=True,
            ),
            transport="http-json",
            features=frozenset({"text"}),
            binding=binding,
        )
        body = await observed_body(
            _iter_upstream_bytes(upstream),
            maximum=generation.snapshot.limits.max_observed_json_bytes,
            deadline=operation.deadline,
        )
        assert operation.current_attempt is not None
        operation.current_attempt.usage = UsageObservation(
            charge_free=True, availability="complete", provenance="resource-control"
        )
        if request.method == "DELETE":
            await runtime.resources.delete(
                response_id, principal_id=principal.id, route_id=route.id
            )
        await operation.close("success")
        return Response(
            content=body, status_code=upstream.status_code, headers=_safe_response_headers(upstream)
        )
    except GatewayPublicError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"type": "gateway_error", "code": exc.code, "message": exc.message}},
        )
    except Exception:
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "type": "gateway_error",
                    "code": "gateway_upstream_error",
                    "message": "Upstream request failed",
                }
            },
        )
    finally:
        if operation is not None:
            await operation.close("failed", failure_origin="network")
