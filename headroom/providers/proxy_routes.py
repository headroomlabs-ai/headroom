# mypy: disable-error-code=no-untyped-def
"""Provider-specific proxy route registration."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import Response

from headroom.copilot_auth import copilot_bearer_upstream as _copilot_bearer_upstream
from headroom.providers.cloudcode import normalize_cloudcode_passthrough_path
from headroom.providers.codex.endpoints import codex_backend_url
from headroom.providers.codex.headers import drop_header
from headroom.providers.codex.live import (
    CODEX_LIVE_ROUTE_PATHS,
    handle_codex_live_websocket,
)
from headroom.providers.codex.responses import handle_chatgpt_codex_responses_subpath
from headroom.providers.codex.runtime import resolve_codex_routing
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
)
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


def _register_provider_passthrough_route(
    app: FastAPI,
    proxy: Any,
    spec: ProviderPassthroughRoute,
) -> None:
    async def provider_passthrough(request: Request):
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
        if spec.path_param is None:
            return await handler(request)
        path_args = {
            "batch_id": batch_id,
            "batch_name": batch_name,
            "model": model,
        }
        return await handler(request, path_args[spec.path_param])

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
        assert proxy.http_client is not None
        chatgpt_response = await handle_chatgpt_codex_responses_subpath(
            proxy.http_client,
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
        # Claude models on Copilot travel this route with GitHub's bearer token.
        # On a shared proxy whose Anthropic target is the stock host that token
        # cannot succeed; hand the handler the Copilot base the same way a
        # per-request override would, so auth, `/v1` handling and provider
        # labelling all follow the existing Copilot path.
        copilot_base = _copilot_bearer_upstream(dict(request.headers), proxy.ANTHROPIC_API_URL)
        if copilot_base is not None:
            return await proxy.handle_anthropic_messages(
                request, upstream_base_url=copilot_base, copilot_redirect=True
            )
        return await proxy.handle_anthropic_messages(request)

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

    _register_openai_responses_routes(app, proxy)

    _register_provider_handler_routes(app, proxy)

    @app.post(
        "/{api_version}/projects/{project}/locations/{location}/publishers/{publisher}/models/{model}:generateContent"
    )
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

    def _model_metadata_target(request: Request, provider_name: str) -> str:
        target = _api_target(proxy, provider_name)
        if provider_name == "openai":
            # Same rule as the unprefixed `/models` passthrough: a Copilot
            # credential against the stock OpenAI target belongs at Copilot.
            return _copilot_bearer_upstream(dict(request.headers), target) or target
        return target

    @app.get("/v1/models")
    async def list_models(request: Request):
        provider_name = proxy.provider_runtime.model_metadata_provider(dict(request.headers))
        return await handle_model_metadata_endpoint(
            proxy,
            request,
            endpoint=MODEL_METADATA_LIST_ENDPOINT,
            provider_api_base_url=_model_metadata_target(request, provider_name),
            provider_name=provider_name,
        )

    @app.get("/v1/models/{model_id}")
    async def get_model(request: Request, model_id: str):
        provider_name = proxy.provider_runtime.model_metadata_provider(dict(request.headers))
        return await handle_model_metadata_endpoint(
            proxy,
            request,
            endpoint=model_metadata_get_endpoint(model_id),
            provider_api_base_url=_model_metadata_target(request, provider_name),
            provider_name=provider_name,
        )

    @app.post("/v1/alpha/search")
    async def codex_alpha_search(request: Request):
        chatgpt_response = await _handle_chatgpt_codex_alpha_search(request, proxy)
        if chatgpt_response is not None:
            return chatgpt_response
        # This route resolves a caller-named upstream like the catch-all does,
        # so it needs the same rejection. Without it a client could point the
        # proxy at loopback/RFC1918/cloud-metadata and read the response back
        # (CVE-2026-77775).
        custom_base = request.headers.get("x-headroom-base-url", "").strip()
        if custom_base and not await is_safe_upstream_url_async(custom_base):
            logger.warning("rejecting unsafe x-headroom-base-url: %r", custom_base)
            raise HTTPException(status_code=400, detail="Rejected unsafe upstream base URL")
        return await proxy.handle_passthrough(
            request,
            _select_passthrough_base_url(proxy, dict(request.headers)),
        )

    _register_openai_image_routes(app, proxy)

    _register_codex_live_routes(app, proxy)

    _register_provider_passthrough_routes(app, proxy)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD"])
    async def passthrough(request: Request, path: str):
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
                _api_target(proxy, "cloudcode"),
            )

        return await proxy.handle_passthrough(
            request,
            # The path matters here: this is where unrouted paths land, and
            # Copilot's inline completions are one of them (#3076).
            _select_passthrough_base_url(proxy, dict(request.headers), request.url.path),
        )
