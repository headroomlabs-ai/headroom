# mypy: disable-error-code=no-untyped-def
"""Provider-specific proxy route registration."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse, Response

from headroom.exceptions import ProviderError
from headroom.log_safety import redact_url
from headroom.providers.bedrock_models import newest_snapshot
from headroom.providers.claude import DEFAULT_API_URL as DEFAULT_ANTHROPIC_API_URL
from headroom.providers.cloudcode import (
    cloudcode_passthrough_target,
    normalize_cloudcode_passthrough_path,
)
from headroom.providers.codex.endpoints import codex_backend_url
from headroom.providers.codex.headers import drop_header
from headroom.providers.codex.live import (
    CODEX_LIVE_ROUTE_PATHS,
    handle_codex_live_http,
    handle_codex_live_websocket,
)
from headroom.providers.codex.responses import handle_chatgpt_codex_responses_subpath
from headroom.providers.codex.runtime import resolve_codex_routing
from headroom.providers.hermes import SCOPED_PATH_PREFIXES as HERMES_SCOPED_PATH_PREFIXES
from headroom.providers.model_metadata import (
    MODEL_METADATA_LIST_ENDPOINT,
    handle_model_metadata_endpoint,
    model_metadata_get_endpoint,
)
from headroom.providers.openai_images import (
    OPENAI_IMAGE_ENDPOINTS,
    OpenAIImageEndpoint,
    handle_openai_image_endpoint,
)
from headroom.providers.openai_responses import (
    OPENAI_RESPONSES_ROOT_PATHS,
    OPENAI_RESPONSES_SUBPATH_ROUTES,
    OPENAI_RESPONSES_WEBSOCKET_PATHS,
    OpenAIResponsesSubpathRoute,
    handle_openai_responses_subpath,
)
from headroom.providers.proxy_targets import (
    api_target as _api_target,
)
from headroom.providers.proxy_targets import (
    openai_compatible_base_url as _openai_compatible_base_url,
)
from headroom.providers.proxy_targets import (
    select_passthrough_base_url as _select_passthrough_base_url,
)
from headroom.providers.proxy_targets import (
    vertex_target_for_location as _vertex_target_for_location,
)
from headroom.providers.route_specs import (
    PROVIDER_HANDLER_ROUTES,
    PROVIDER_PASSTHROUGH_ROUTES,
    ProviderHandlerRoute,
    ProviderPassthroughRoute,
)
from headroom.providers.vertex import (
    VERTEX_ANTHROPIC_PROVIDER_NAME,
    VERTEX_COUNT_TOKENS,
    VERTEX_GENERATE_CONTENT,
    VERTEX_GOOGLE_PROVIDER_NAME,
    VERTEX_RAW_PREDICT,
    VERTEX_STREAM_GENERATE_CONTENT,
    VERTEX_STREAM_RAW_PREDICT,
    is_vertex_anthropic_publisher,
    is_vertex_google_publisher,
    vertex_anthropic_target,
    vertex_publisher_provider_name,
    with_vertex_diagnostics,
)
from headroom.proxy.buffered_ccr_response import ANTHROPIC_ERROR_TYPES
from headroom.proxy.passthrough import (
    custom_base_passthrough_telemetry as _custom_base_passthrough_telemetry,
)
from headroom.proxy.request_scope import normalize_request_path
from headroom.proxy.upstream_guard import is_safe_upstream_url_async

logger = logging.getLogger("headroom.proxy.routes")


async def _handle_chatgpt_codex_alpha_search(request: Request, proxy: Any) -> Response | None:
    upstream_headers = dict(request.headers.items())
    drop_header(upstream_headers, "host")
    drop_header(upstream_headers, "accept-encoding")
    from headroom.proxy.helpers import _strip_internal_headers

    decision = resolve_codex_routing(_strip_internal_headers(upstream_headers))
    if not decision.is_chatgpt_auth:
        return None

    body = await request.body()
    assert proxy.http_client is not None
    resp = await proxy.http_client.request(
        request.method,
        codex_backend_url("/alpha/search", request.url.query),
        headers=decision.headers,
        content=body,
        timeout=120.0,
    )
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
    )


def _same_origin(a: str, b: str) -> bool:
    """True when two URLs name the same scheme, host, port and path (default ports implied)."""

    def key(url: str) -> tuple[str, str, int | None, str]:
        parts = urlsplit(url)
        scheme = parts.scheme.lower()
        port = parts.port if parts.port is not None else {"https": 443, "http": 80}.get(scheme)
        return scheme, (parts.hostname or "").lower(), port, parts.path.rstrip("/")

    return key(a) == key(b)


async def _count_tokens_on_backend(proxy: Any, request: Request) -> Response | None:
    """Answer ``/v1/messages/count_tokens`` from the Anthropic backend, or None to pass through.

    In Bedrock mode there is no Anthropic account behind the placeholder key
    clients send, so the passthrough would 401; the backend counts instead.
    """
    backend = getattr(proxy, "anthropic_backend", None)
    if backend is None:
        return None
    try:
        body = await request.json()
    except ValueError:
        return None
    if not isinstance(body, dict):
        return None
    try:
        tokens = await backend.count_tokens(body, dict(request.headers))
    except ProviderError as exc:
        status = int(exc.details.get("status", 502))
        logger.warning("count_tokens on backend %s failed: %s", backend.name, exc)
        return JSONResponse(
            {
                "type": "error",
                "error": {
                    "type": ANTHROPIC_ERROR_TYPES.get(status, "api_error"),
                    "message": str(exc),
                },
            },
            status_code=status,
        )
    if tokens is None:
        return None
    return JSONResponse({"input_tokens": tokens})


def _register_provider_passthrough_route(
    app: FastAPI,
    proxy: Any,
    spec: ProviderPassthroughRoute,
) -> None:
    async def provider_passthrough(request: Request):
        # A reachability probe the token gate let through without a credential
        # (/api/hello): answer it here, never relay it upstream anonymously.
        if getattr(request.state, "headroom_local_hello_probe", False):
            return Response(status_code=200)

        # Honor the per-request upstream override like /v1/messages and the
        # catch-all do. Clients that speak one wire format but authenticate
        # against a custom OpenAI-compatible gateway must reach that gateway
        # for embeddings/audio/metadata too, not the default provider target
        # (the same gap as the model-metadata routes, #4302). Telemetry keeps
        # the route spec labels: they are bounded by this module, so the
        # request-controlled host never becomes a stats label (#3759), while
        # `handle_passthrough` still records the outcome for these known routes.
        custom_base = request.headers.get("x-headroom-base-url", "").strip()
        if custom_base:
            if not await is_safe_upstream_url_async(custom_base):
                logger.warning("rejecting unsafe x-headroom-base-url: %r", custom_base)
                raise HTTPException(status_code=400, detail="Rejected unsafe upstream base URL")
            return await proxy.handle_passthrough(
                request,
                custom_base.rstrip("/"),
                spec.sub_path,
                spec.provider_name,
            )
        # Only where the passthrough would reach api.anthropic.com: an operator
        # who points the Anthropic target at a gateway keeps getting its counts.
        if (
            spec.provider_name == "anthropic"
            and spec.sub_path == "count_tokens"
            and _same_origin(_api_target(proxy, "anthropic"), DEFAULT_ANTHROPIC_API_URL)
        ):
            counted = await _count_tokens_on_backend(proxy, request)
            if counted is not None:
                return counted
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, spec.provider_name),
            spec.sub_path,
            spec.provider_name,
        )

    provider_passthrough.__name__ = (
        f"{spec.provider_name}_{spec.sub_path.replace('/', '_')}_{spec.method.lower()}_passthrough"
    )
    app.api_route(spec.path, methods=[spec.method])(provider_passthrough)


def _register_provider_passthrough_routes(app: FastAPI, proxy: Any) -> None:
    for spec in PROVIDER_PASSTHROUGH_ROUTES:
        _register_provider_passthrough_route(app, proxy, spec)


def _register_provider_handler_route(app: FastAPI, proxy: Any, spec: ProviderHandlerRoute) -> None:
    async def provider_handler(
        request: Request,
        batch_id: str = "",
        batch_name: str = "",
        model: str = "",
    ):
        handler = getattr(proxy, spec.handler_name)
        handler_kwargs: dict[str, str] = {}
        custom_base = request.headers.get("x-headroom-base-url", "").strip()
        if custom_base and spec.supports_custom_base_url:
            if not await is_safe_upstream_url_async(custom_base):
                logger.warning("rejecting unsafe x-headroom-base-url: %r", custom_base)
                raise HTTPException(status_code=400, detail="Rejected unsafe upstream base URL")
            handler_kwargs["upstream_base_url"] = custom_base.rstrip("/")
        if spec.path_param is None:
            return await handler(request, **handler_kwargs)
        path_args = {
            "batch_id": batch_id,
            "batch_name": batch_name,
            "model": model,
        }
        return await handler(request, path_args[spec.path_param], **handler_kwargs)

    provider_handler.__name__ = (
        spec.handler_name.replace("handle_", "")
        + "_"
        + spec.method.lower()
        + "_"
        + spec.path.strip("/").replace("/", "_").replace("-", "_")
    )
    app.api_route(spec.path, methods=[spec.method])(provider_handler)


def _register_provider_handler_routes(app: FastAPI, proxy: Any) -> None:
    for spec in PROVIDER_HANDLER_ROUTES:
        _register_provider_handler_route(app, proxy, spec)


def _register_openai_responses_root_route(app: FastAPI, proxy: Any, path: str) -> None:
    async def openai_responses_root(request: Request):
        return await proxy.handle_openai_responses(request)

    openai_responses_root.__name__ = path.strip("/").replace("/", "_").replace("-", "_") + "_root"
    app.post(path)(openai_responses_root)


def _register_openai_responses_websocket_route(app: FastAPI, proxy: Any, path: str) -> None:
    async def openai_responses_ws(websocket: WebSocket):
        await proxy.handle_openai_responses_ws(websocket)

    openai_responses_ws.__name__ = path.strip("/").replace("/", "_").replace("-", "_") + "_ws"
    app.websocket(path)(openai_responses_ws)


def _register_openai_responses_subpath_route(
    app: FastAPI,
    proxy: Any,
    spec: OpenAIResponsesSubpathRoute,
) -> None:
    async def openai_responses_subpath(request: Request, sub_path: str):
        codex_subpath_client = getattr(proxy, "http_client_h1", None) or proxy.http_client
        chatgpt_response = await handle_chatgpt_codex_responses_subpath(
            codex_subpath_client,
            request,
            sub_path,
        )
        if chatgpt_response is not None:
            return chatgpt_response

        return await handle_openai_responses_subpath(
            proxy.http_client,
            request,
            _api_target(proxy, "openai"),
            sub_path,
        )

    openai_responses_subpath.__name__ = (
        spec.path.strip("/")
        .replace("/", "_")
        .replace("-", "_")
        .replace("{sub_path:path}", "subpath")
    )
    app.api_route(spec.path, methods=list(spec.methods))(openai_responses_subpath)


def _register_openai_responses_routes(app: FastAPI, proxy: Any) -> None:
    for path in OPENAI_RESPONSES_ROOT_PATHS:
        _register_openai_responses_root_route(app, proxy, path)
    for path in OPENAI_RESPONSES_WEBSOCKET_PATHS:
        _register_openai_responses_websocket_route(app, proxy, path)
    for spec in OPENAI_RESPONSES_SUBPATH_ROUTES:
        _register_openai_responses_subpath_route(app, proxy, spec)


def _register_codex_live_routes(app: FastAPI, proxy: Any) -> None:
    for path in CODEX_LIVE_ROUTE_PATHS:

        async def codex_live_http(request: Request, route_path: str = path):
            response = await handle_codex_live_http(
                request,
                proxy.http_client,
                _api_target(proxy, "openai"),
                route_path,
            )
            if response is not None:
                return response
            return await proxy.handle_passthrough(
                request,
                _api_target(proxy, "openai"),
                route_path,
                "openai",
            )

        codex_live_http.__name__ = path.strip("/").replace("/", "_") + "_live_http"
        app.post(path)(codex_live_http)

        def register_websocket_route(route_path: str) -> None:
            async def codex_live_websocket(websocket: WebSocket):
                await handle_codex_live_websocket(
                    websocket,
                    proxy,
                    _api_target(proxy, "openai"),
                    route_path,
                )

            codex_live_websocket.__name__ = (
                route_path.strip("/").replace("/", "_") + "_live_websocket"
            )
            app.websocket(route_path)(codex_live_websocket)

        register_websocket_route(path)


def _register_openai_image_route(app: FastAPI, proxy: Any, endpoint: OpenAIImageEndpoint) -> None:
    async def openai_image_endpoint(request: Request):
        return await handle_openai_image_endpoint(
            proxy,
            request,
            openai_api_base_url=_api_target(proxy, "openai"),
            endpoint=endpoint,
        )

    openai_image_endpoint.__name__ = endpoint.sub_path.replace("/", "_") + "_post"
    app.post(endpoint.route_path)(openai_image_endpoint)


def _register_openai_image_routes(app: FastAPI, proxy: Any) -> None:
    for endpoint in OPENAI_IMAGE_ENDPOINTS:
        _register_openai_image_route(app, proxy, endpoint)


def register_provider_routes(app: FastAPI, proxy: Any) -> None:
    """Register provider-specific proxy endpoints."""

    async def _default_passthrough(request: Request, path: str):
        """Catch-all forwarding for paths no provider route claims.

        ``path`` is the request path without its leading slash, as the
        ``/{path:path}`` catch-all captures it.
        """
        custom_base = request.headers.get("x-headroom-base-url")
        if custom_base:
            if not await is_safe_upstream_url_async(custom_base):
                logger.warning("rejecting unsafe x-headroom-base-url: %r", custom_base)
                raise HTTPException(status_code=400, detail="Rejected unsafe upstream base URL")
            base_url = custom_base.rstrip("/")
            endpoint_name, provider_name = _custom_base_passthrough_telemetry(
                request.method,
                path,
                base_url,
            )
            return await proxy.handle_passthrough(
                request,
                base_url,
                endpoint_name,
                provider_name,
            )

        normalized_cloudcode_path = normalize_cloudcode_passthrough_path(path)
        if normalized_cloudcode_path is not None:
            normalize_request_path(request, normalized_cloudcode_path)

            return await proxy.handle_passthrough(
                request,
                # Antigravity's non-generateContent calls (fetchAvailableModels,
                # loadCodeAssist) land here, not in the Gemini handler, so the
                # daily-sandbox routing has to happen at this seam too (#4957).
                cloudcode_passthrough_target(
                    request.headers.get("user-agent"),
                    _api_target(proxy, "cloudcode"),
                ),
            )

        return await proxy.handle_passthrough(
            request,
            # The path matters here: this is where unrouted paths land, and
            # Copilot's inline completions are one of them (#3076).
            _select_passthrough_base_url(proxy, dict(request.headers), request.url.path),
        )

    async def vertex_publisher_passthrough(request: Request, publisher: str, action: str):
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, "vertex"),
            action,
            vertex_publisher_provider_name(publisher),
        )

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        # Honor the per-request upstream override so clients that speak the
        # Anthropic Messages wire format but authenticate against a
        # non-Anthropic gateway route correctly, consistent with the
        # OpenAI-compatible and generic passthrough routes.
        custom_base = request.headers.get("x-headroom-base-url", "").strip()
        if custom_base:
            if not await is_safe_upstream_url_async(custom_base):
                logger.warning("rejecting unsafe x-headroom-base-url: %r", custom_base)
                raise HTTPException(status_code=400, detail="Rejected unsafe upstream base URL")
            return await proxy.handle_anthropic_messages(
                request, upstream_base_url=custom_base.rstrip("/")
            )
        return await proxy.handle_anthropic_messages(request)

    @app.post("/omlx/v1/messages")
    async def omlx_messages(request: Request):
        from headroom.config import resolve_omlx_target_api_url

        omlx_target = (
            getattr(proxy.config, "omlx_target_api_url", None)
            or resolve_omlx_target_api_url(environ=os.environ)
            or "http://127.0.0.1:8888"
        )
        return await proxy.handle_anthropic_messages(
            request, upstream_base_url=omlx_target.rstrip("/")
        )

    @app.post("/anthropic/v1/messages")
    async def foundry_anthropic_messages(request: Request):
        normalize_request_path(request, "/v1/messages")
        return await proxy.handle_anthropic_messages(request, _api_target(proxy, "anthropic"))

    # AWS Bedrock InvokeModel passthrough. Registered ONLY when an upstream is
    # configured (`--bedrock-api-url` / BEDROCK_TARGET_API_URL): without it,
    # `/model/{id}/invoke` keeps falling through to the catch-all (verbatim,
    # signature-intact) so existing behavior is unchanged. The `{model_id:path}`
    # converter captures inference-profile ids that contain dots, colons and
    # slashes (e.g. `us.anthropic.claude-sonnet-4-5-20250929-v1:0`). See
    # headroom/proxy/handlers/bedrock.py for the SigV4 caveat.
    if getattr(proxy.config, "bedrock_api_url", None):

        @app.post("/model/{model_id:path}/invoke")
        async def bedrock_invoke(request: Request, model_id: str):
            return await proxy.handle_bedrock_invoke(request, model_id, stream=False)

        @app.post("/model/{model_id:path}/invoke-with-response-stream")
        async def bedrock_invoke_stream(request: Request, model_id: str):
            return await proxy.handle_bedrock_invoke(request, model_id, stream=True)

    # Factory Droid. Registered ONLY when an upstream is configured
    # (`--factory-api-url` / FACTORY_TARGET_API_URL, set by `headroom wrap
    # droid`). Droid sends everything under `/api/`; nothing outside `/api/`
    # ever goes to Factory, so another tool still pointed at this port (a
    # leftover Codex/Claude config) keeps its usual upstream and its
    # credentials never reach Factory.
    factory_api_url = getattr(proxy.config, "factory_api_url", None)
    if factory_api_url:
        factory_base = factory_api_url.rstrip("/")

        @app.post("/api/llm/a/v1/messages")
        async def factory_anthropic_messages(request: Request):
            # Anthropic-shaped inference: compressed, then forwarded to
            # `{factory}/api/llm/a/v1/messages`. A client-sent
            # `x-headroom-base-url` is deliberately ignored (the handler strips
            # `x-headroom-*`), and Factory is not a trusted host, so operator
            # Anthropic extra headers are withheld. `force_direct` keeps a
            # configured translating backend or route advice from taking it.
            return await proxy.handle_anthropic_messages(
                request,
                upstream_base_url=factory_base,
                provider_name="factory",
                force_direct=True,
            )

        @app.api_route(
            "/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"]
        )
        async def factory_passthrough(request: Request, path: str):
            # Hermes Studio's scoped proxies and callers that name their own
            # upstream are not Droid traffic: keep the catch-all behaviour.
            if request.headers.get(
                "x-headroom-base-url", ""
            ).strip() or request.url.path.startswith(HERMES_SCOPED_PATH_PREFIXES):
                return await _default_passthrough(request, request.url.path[1:])
            # Factory REST (and, for now, OpenAI-shaped `/api/llm/o/*`):
            # verbatim and buffered, even with `compress_passthrough` on. No
            # endpoint name, so no LLM telemetry.
            return await proxy.handle_passthrough(
                request, factory_base, None, "factory", verbatim=True
            )

    _register_openai_responses_routes(app, proxy)

    _register_provider_handler_routes(app, proxy)

    @app.post(
        "/{api_version}/projects/{project}/locations/{location}/publishers/{publisher}/models/{model}:generateContent"
    )
    @with_vertex_diagnostics
    async def vertex_generate_content(
        request: Request,
        api_version: str,
        project: str,
        location: str,
        publisher: str,
        model: str,
    ):
        del api_version, project
        if is_vertex_google_publisher(publisher):
            return await proxy.handle_gemini_generate_content(
                request,
                model,
                _vertex_target_for_location(proxy, location),
                VERTEX_GOOGLE_PROVIDER_NAME,
            )
        return await vertex_publisher_passthrough(request, publisher, VERTEX_GENERATE_CONTENT.name)

    @app.post(
        "/{api_version}/projects/{project}/locations/{location}/publishers/{publisher}/models/{model}:streamGenerateContent"
    )
    @with_vertex_diagnostics
    async def vertex_stream_generate_content(
        request: Request,
        api_version: str,
        project: str,
        location: str,
        publisher: str,
        model: str,
    ):
        del api_version, project
        if is_vertex_google_publisher(publisher):
            return await proxy.handle_gemini_generate_content(
                request,
                model,
                _vertex_target_for_location(proxy, location),
                VERTEX_GOOGLE_PROVIDER_NAME,
            )
        return await vertex_publisher_passthrough(
            request,
            publisher,
            VERTEX_STREAM_GENERATE_CONTENT.name,
        )

    @app.post(
        "/{api_version}/projects/{project}/locations/{location}/publishers/{publisher}/models/{model}:countTokens"
    )
    @with_vertex_diagnostics
    async def vertex_count_tokens(
        request: Request,
        api_version: str,
        project: str,
        location: str,
        publisher: str,
        model: str,
    ):
        del api_version, project
        if is_vertex_google_publisher(publisher):
            return await proxy.handle_gemini_count_tokens(
                request,
                model,
                _vertex_target_for_location(proxy, location),
                VERTEX_GOOGLE_PROVIDER_NAME,
            )
        return await vertex_publisher_passthrough(request, publisher, VERTEX_COUNT_TOKENS.name)

    @app.post(
        "/{api_version}/projects/{project}/locations/{location}/publishers/{publisher}/models/{model}:rawPredict"
    )
    @with_vertex_diagnostics
    async def vertex_raw_predict(
        request: Request,
        api_version: str,
        project: str,
        location: str,
        publisher: str,
        model: str,
    ):
        del api_version, project
        if is_vertex_anthropic_publisher(publisher):
            return await proxy.handle_anthropic_messages(
                request,
                _vertex_target_for_location(proxy, location),
                VERTEX_ANTHROPIC_PROVIDER_NAME,
                model,
            )
        return await vertex_publisher_passthrough(request, publisher, VERTEX_RAW_PREDICT.name)

    @app.post(
        "/projects/{project}/locations/{location}/publishers/{publisher}/models/{model}:rawPredict"
    )
    @with_vertex_diagnostics
    async def vertex_raw_predict_no_version(
        request: Request,
        project: str,
        location: str,
        publisher: str,
        model: str,
    ):
        if is_vertex_anthropic_publisher(publisher):
            del project
            target = vertex_anthropic_target(
                _vertex_target_for_location(proxy, location),
                versionless_route=True,
            )
            return await proxy.handle_anthropic_messages(
                request,
                target,
                VERTEX_ANTHROPIC_PROVIDER_NAME,
                model,
            )
        return await vertex_publisher_passthrough(request, publisher, VERTEX_RAW_PREDICT.name)

    @app.post(
        "/{api_version}/projects/{project}/locations/{location}/publishers/{publisher}/models/{model}:streamRawPredict"
    )
    @with_vertex_diagnostics
    async def vertex_stream_raw_predict(
        request: Request,
        api_version: str,
        project: str,
        location: str,
        publisher: str,
        model: str,
    ):
        del api_version, project
        if is_vertex_anthropic_publisher(publisher):
            return await proxy.handle_anthropic_messages(
                request,
                _vertex_target_for_location(proxy, location),
                VERTEX_ANTHROPIC_PROVIDER_NAME,
                model,
                VERTEX_STREAM_RAW_PREDICT.force_stream,
            )
        return await vertex_publisher_passthrough(
            request,
            publisher,
            VERTEX_STREAM_RAW_PREDICT.name,
        )

    @app.post(
        "/projects/{project}/locations/{location}/publishers/{publisher}/models/{model}:streamRawPredict"
    )
    @with_vertex_diagnostics
    async def vertex_stream_raw_predict_no_version(
        request: Request,
        project: str,
        location: str,
        publisher: str,
        model: str,
    ):
        if is_vertex_anthropic_publisher(publisher):
            del project
            target = vertex_anthropic_target(
                _vertex_target_for_location(proxy, location),
                versionless_route=True,
            )
            return await proxy.handle_anthropic_messages(
                request,
                target,
                VERTEX_ANTHROPIC_PROVIDER_NAME,
                model,
                VERTEX_STREAM_RAW_PREDICT.force_stream,
            )
        return await vertex_publisher_passthrough(
            request,
            publisher,
            VERTEX_STREAM_RAW_PREDICT.name,
        )

    async def _model_metadata_route(request: Request) -> tuple[str, str]:
        # A caller-named upstream wins, as in the catch-all and the chat
        # handlers: model discovery must reach the same gateway as inference.
        custom_base = request.headers.get("x-headroom-base-url", "").strip()
        if custom_base:
            if not await is_safe_upstream_url_async(custom_base):
                logger.warning("rejecting unsafe x-headroom-base-url: %s", redact_url(custom_base))
                raise HTTPException(status_code=400, detail="Rejected unsafe upstream base URL")
            # Only the destination is overridden: keep the provider the request's
            # auth selects, so Anthropic-auth requests still get model-id
            # sanitization in handle_passthrough.
            provider_name = proxy.provider_runtime.model_metadata_provider(dict(request.headers))
            return provider_name, custom_base.rstrip("/")
        headers = dict(request.headers)
        provider_name = proxy.provider_runtime.model_metadata_provider(headers)
        base_url = (
            _openai_compatible_base_url(proxy, headers)
            if provider_name == "openai"
            else _api_target(proxy, provider_name)
        )
        return provider_name, base_url

    def _bedrock_model_ids(request: Request, provider_name: str) -> list[str] | None:
        # In Bedrock mode an Anthropic-format model lookup has no Anthropic
        # account behind it (Claude Desktop in gateway mode sends a placeholder
        # key); answer from the Bedrock models the backend discovered. A
        # caller-named upstream still gets the passthrough.
        if provider_name != "anthropic" or request.headers.get("x-headroom-base-url", "").strip():
            return None
        backend = getattr(proxy, "anthropic_backend", None)
        return backend.bedrock_model_ids() if backend is not None else None

    def _anthropic_model_entry(model_id: str) -> dict[str, str]:
        return {
            "type": "model",
            "id": model_id,
            "display_name": model_id,
            "created_at": "1970-01-01T00:00:00Z",
        }

    def _anthropic_model_page(ids: list[str], params: Mapping[str, str]) -> dict[str, Any]:
        # Anthropic's list paging: ``limit`` (default 20, 1-1000), and cursors
        # ``after_id`` / ``before_id`` naming a model id.
        try:
            limit = min(max(int(params.get("limit", "20")), 1), 1000)
        except ValueError:
            limit = 20
        cursors = [params[c] for c in ("after_id", "before_id") if c in params]
        if any(cursor not in ids for cursor in cursors):
            # A cursor naming a model no longer listed: an empty page, never a
            # silent restart from the top.
            return {"data": [], "has_more": False, "first_id": None, "last_id": None}
        start, end = 0, len(ids)
        if params.get("after_id") in ids:
            start = ids.index(params["after_id"]) + 1
        if params.get("before_id") in ids:
            end = ids.index(params["before_id"])
            start = max(start, end - limit)
        page = ids[start:end][:limit]
        return {
            "data": [_anthropic_model_entry(m) for m in page],
            # More pages in the direction being paged.
            "has_more": start > 0 if "before_id" in params else start + len(page) < end,
            "first_id": page[0] if page else None,
            "last_id": page[-1] if page else None,
        }

    @app.get("/v1/models")
    async def list_models(request: Request):
        provider_name, base_url = await _model_metadata_route(request)
        bedrock_ids = _bedrock_model_ids(request, provider_name)
        if bedrock_ids is not None:
            return JSONResponse(_anthropic_model_page(bedrock_ids, request.query_params))
        return await handle_model_metadata_endpoint(
            proxy,
            request,
            endpoint=MODEL_METADATA_LIST_ENDPOINT,
            provider_api_base_url=base_url,
            provider_name=provider_name,
        )

    @app.get("/v1/models/{model_id}")
    async def get_model(request: Request, model_id: str):
        provider_name, base_url = await _model_metadata_route(request)
        bedrock_ids = _bedrock_model_ids(request, provider_name)
        if bedrock_ids is not None:
            # Same names inference accepts: a "[1m]" context suffix and an
            # undated alias of a listed snapshot.
            # Local import: the module pulls in litellm, which proxy startup avoids.
            from headroom.providers.anthropic import sanitize_anthropic_model_id

            clean_id = sanitize_anthropic_model_id(model_id)
            if clean_id in bedrock_ids or newest_snapshot(clean_id, bedrock_ids):
                return JSONResponse(_anthropic_model_entry(model_id))
            return JSONResponse(
                {
                    "type": "error",
                    "error": {
                        "type": "not_found_error",
                        "message": f"Model {model_id!r} is not available on this Bedrock backend.",
                    },
                },
                status_code=404,
            )
        return await handle_model_metadata_endpoint(
            proxy,
            request,
            endpoint=model_metadata_get_endpoint(model_id),
            provider_api_base_url=base_url,
            provider_name=provider_name,
        )

    @app.post("/v1/alpha/search")
    async def codex_alpha_search(request: Request):
        chatgpt_response = await _handle_chatgpt_codex_alpha_search(request, proxy)
        if chatgpt_response is not None:
            return chatgpt_response
        # A caller-named upstream is checked once, here, and that checked value
        # is the destination -- as in the catch-all. Re-deriving it later would
        # resolve the name a second time and, on a different answer, send the
        # caller's credential to a host it did not name.
        custom_base = request.headers.get("x-headroom-base-url", "").strip()
        if custom_base:
            if not await is_safe_upstream_url_async(custom_base):
                logger.warning("rejecting unsafe x-headroom-base-url: %r", custom_base)
                raise HTTPException(status_code=400, detail="Rejected unsafe upstream base URL")
            return await proxy.handle_passthrough(request, custom_base.rstrip("/"))
        return await proxy.handle_passthrough(
            request,
            _select_passthrough_base_url(proxy, dict(request.headers)),
        )

    _register_openai_image_routes(app, proxy)

    _register_codex_live_routes(app, proxy)

    _register_provider_passthrough_routes(app, proxy)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD"])
    async def passthrough(request: Request, path: str):
        return await _default_passthrough(request, path)
