# mypy: disable-error-code=no-untyped-def
"""Provider-specific proxy route registration."""

from __future__ import annotations

import logging
from typing import Any, cast

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response

from headroom.providers.codex import model_metadata as codex_model_metadata
from headroom.providers.codex.model_metadata import (
    fetch_chatgpt_codex_model_get_response,
    fetch_chatgpt_codex_model_ids,
    fetch_chatgpt_codex_models_response,
    handle_chatgpt_model_metadata,
    models_list_response,
    normalize_codex_registry_headers,
    synthetic_model_get_response,
    synthetic_models_list_response,
)
from headroom.providers.codex.runtime import resolve_codex_routing_headers
from headroom.providers.vertex import dispatch_vertex_publisher_request

logger = logging.getLogger("headroom.proxy.routes")
codex_model_metadata.logger = logger

_resolve_codex_routing_headers = resolve_codex_routing_headers
_normalize_codex_registry_headers = normalize_codex_registry_headers
_models_list_response = models_list_response
_synthetic_models_list_response = synthetic_models_list_response
_synthetic_model_get_response = synthetic_model_get_response


def _api_target(proxy: Any, provider_name: str) -> str:
    legacy_attrs = {
        "anthropic": "ANTHROPIC_API_URL",
        "openai": "OPENAI_API_URL",
        "gemini": "GEMINI_API_URL",
        "cloudcode": "CLOUDCODE_API_URL",
        "vertex": "VERTEX_API_URL",
    }
    legacy_attr = legacy_attrs[provider_name]
    return cast(str, getattr(proxy, legacy_attr, proxy.provider_runtime.api_target(provider_name)))


def _select_passthrough_base_url(proxy: Any, headers: dict[str, str]) -> str:
    # Codex CLI subscription mode hits a wide surface under
    # `/backend-api/*` (rate-limit polling, agent identity, JWT
    # refresh, cloud tasks). Without this branch the catchall
    # routes those to api.openai.com which 404s, and Codex
    # interprets the failure as "session invalid" and refuses
    # to use subscription auth at all. The check is a no-op
    # for non-ChatGPT-authed requests.
    _, is_chatgpt_auth = _resolve_codex_routing_headers(headers)
    if is_chatgpt_auth:
        return "https://chatgpt.com"
    if headers.get("x-goog-api-key"):
        return _api_target(proxy, "gemini")
    if headers.get("api-key"):
        azure_base = headers.get("x-headroom-base-url", "")
        if azure_base:
            return azure_base.rstrip("/")
    provider_name = proxy.provider_runtime.model_metadata_provider(headers)
    return _api_target(proxy, provider_name)


async def _fetch_chatgpt_codex_model_ids(
    proxy: Any,
    headers: dict[str, str],
    requested_client_version: str | None,
) -> tuple[str, ...] | None:
    assert proxy.http_client is not None
    return await fetch_chatgpt_codex_model_ids(
        proxy.http_client,
        headers,
        requested_client_version,
    )


async def _fetch_chatgpt_codex_models_response(
    proxy: Any,
    headers: dict[str, str],
    requested_client_version: str | None,
) -> Response | None:
    """Build a dynamic `/v1/models` response from the Codex registry when available."""
    assert proxy.http_client is not None
    return await fetch_chatgpt_codex_models_response(
        proxy.http_client,
        headers,
        requested_client_version,
    )


async def _fetch_chatgpt_codex_model_get_response(
    proxy: Any,
    headers: dict[str, str],
    model_id: str,
    requested_client_version: str | None,
) -> Response | None:
    """Build a dynamic `/v1/models/{id}` response from the Codex registry when available."""
    assert proxy.http_client is not None
    return await fetch_chatgpt_codex_model_get_response(
        proxy.http_client,
        headers,
        model_id,
        requested_client_version,
    )


async def _handle_chatgpt_model_metadata(
    proxy: Any,
    request: Request,
    upstream_path: str,
) -> Response | None:
    assert proxy.http_client is not None
    return await handle_chatgpt_model_metadata(proxy.http_client, request, upstream_path)


def register_provider_routes(app: FastAPI, proxy: Any) -> None:
    """Register provider-specific proxy endpoints."""

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        return await proxy.handle_anthropic_messages(request)

    @app.post("/v1/messages/count_tokens")
    async def anthropic_count_tokens(request: Request):
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, "anthropic"),
            "count_tokens",
            "anthropic",
        )

    @app.post("/v1/messages/batches")
    async def anthropic_batch_create(request: Request):
        return await proxy.handle_anthropic_batch_create(request)

    @app.get("/v1/messages/batches")
    async def anthropic_batch_list(request: Request):
        return await proxy.handle_anthropic_batch_passthrough(request)

    @app.get("/v1/messages/batches/{batch_id}")
    async def anthropic_batch_get(request: Request, batch_id: str):
        return await proxy.handle_anthropic_batch_passthrough(request, batch_id)

    @app.get("/v1/messages/batches/{batch_id}/results")
    async def anthropic_batch_results(request: Request, batch_id: str):
        return await proxy.handle_anthropic_batch_results(request, batch_id)

    @app.post("/v1/messages/batches/{batch_id}/cancel")
    async def anthropic_batch_cancel(request: Request, batch_id: str):
        return await proxy.handle_anthropic_batch_passthrough(request, batch_id)

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request):
        return await proxy.handle_openai_chat(request)

    @app.post("/v1/responses")
    async def openai_responses(request: Request):
        return await proxy.handle_openai_responses(request)

    @app.post("/v1/codex/responses")
    async def openai_v1_codex_responses(request: Request):
        return await proxy.handle_openai_responses(request)

    @app.post("/backend-api/responses")
    async def openai_codex_responses(request: Request):
        return await proxy.handle_openai_responses(request)

    @app.post("/backend-api/codex/responses")
    async def openai_codex_nested_responses(request: Request):
        return await proxy.handle_openai_responses(request)

    @app.websocket("/v1/responses")
    async def openai_responses_ws(websocket: WebSocket):
        await proxy.handle_openai_responses_ws(websocket)

    @app.websocket("/v1/codex/responses")
    async def openai_v1_codex_responses_ws(websocket: WebSocket):
        await proxy.handle_openai_responses_ws(websocket)

    @app.api_route("/v1/responses/{sub_path:path}", methods=["GET", "POST", "DELETE"])
    async def openai_responses_sub(request: Request, sub_path: str):
        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers, is_chatgpt_auth = _resolve_codex_routing_headers(headers)

        if is_chatgpt_auth:
            url = f"https://chatgpt.com/backend-api/codex/responses/{sub_path}"
        else:
            url = f"{_api_target(proxy, 'openai')}/v1/responses/{sub_path}"

        if request.url.query:
            url = f"{url}?{request.url.query}"

        body = await request.body()
        try:
            assert proxy.http_client is not None
            resp = await proxy.http_client.request(
                request.method,
                url,
                headers=headers,
                content=body,
                timeout=120.0,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )
        except Exception as exc:
            logger.error("Passthrough /v1/responses/%s failed: %s", sub_path, exc)
            return Response(content=str(exc), status_code=502)

    @app.api_route("/v1/codex/responses/{sub_path:path}", methods=["GET", "POST", "DELETE"])
    async def openai_v1_codex_responses_sub(request: Request, sub_path: str):
        return await openai_responses_sub(request, sub_path)

    @app.websocket("/backend-api/responses")
    async def openai_codex_responses_ws(websocket: WebSocket):
        await proxy.handle_openai_responses_ws(websocket)

    @app.websocket("/backend-api/codex/responses")
    async def openai_codex_nested_responses_ws(websocket: WebSocket):
        await proxy.handle_openai_responses_ws(websocket)

    @app.api_route("/backend-api/responses/{sub_path:path}", methods=["GET", "POST", "DELETE"])
    async def openai_codex_responses_sub(request: Request, sub_path: str):
        return await openai_responses_sub(request, sub_path)

    @app.api_route(
        "/backend-api/codex/responses/{sub_path:path}", methods=["GET", "POST", "DELETE"]
    )
    async def openai_codex_nested_responses_sub(request: Request, sub_path: str):
        return await openai_responses_sub(request, sub_path)

    @app.post("/v1/batches")
    async def create_batch(request: Request):
        return await proxy.handle_batch_create(request)

    @app.get("/v1/batches")
    async def list_batches(request: Request):
        return await proxy.handle_batch_list(request)

    @app.get("/v1/batches/{batch_id}")
    async def get_batch(request: Request, batch_id: str):
        return await proxy.handle_batch_get(request, batch_id)

    @app.post("/v1/batches/{batch_id}/cancel")
    async def cancel_batch(request: Request, batch_id: str):
        return await proxy.handle_batch_cancel(request, batch_id)

    @app.post("/v1beta/models/{model}:generateContent")
    async def gemini_generate_content(request: Request, model: str):
        return await proxy.handle_gemini_generate_content(request, model)

    @app.post("/v1beta/models/{model}:streamGenerateContent")
    async def gemini_stream_generate_content(request: Request, model: str):
        return await proxy.handle_gemini_stream_generate_content(request, model)

    @app.post("/v1beta/models/{model}:countTokens")
    async def gemini_count_tokens(request: Request, model: str):
        return await proxy.handle_gemini_count_tokens(request, model)

    @app.post("/v1internal:streamGenerateContent")
    async def google_cloudcode_stream_generate_content(request: Request):
        return await proxy.handle_google_cloudcode_stream(request)

    @app.post("/v1/v1internal:streamGenerateContent")
    async def google_cloudcode_stream_generate_content_v1(request: Request):
        return await proxy.handle_google_cloudcode_stream(request)

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
        del api_version, project, location
        return await dispatch_vertex_publisher_request(
            proxy,
            request,
            publisher=publisher,
            action="generateContent",
            model=model,
            upstream_base_url=_api_target(proxy, "vertex"),
        )

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
        del api_version, project, location
        return await dispatch_vertex_publisher_request(
            proxy,
            request,
            publisher=publisher,
            action="streamGenerateContent",
            model=model,
            upstream_base_url=_api_target(proxy, "vertex"),
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
        del api_version, project, location
        return await dispatch_vertex_publisher_request(
            proxy,
            request,
            publisher=publisher,
            action="countTokens",
            model=model,
            upstream_base_url=_api_target(proxy, "vertex"),
        )

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
        del api_version, project, location
        return await dispatch_vertex_publisher_request(
            proxy,
            request,
            publisher=publisher,
            action="rawPredict",
            model=model,
            upstream_base_url=_api_target(proxy, "vertex"),
        )

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
        del api_version, project, location
        return await dispatch_vertex_publisher_request(
            proxy,
            request,
            publisher=publisher,
            action="streamRawPredict",
            model=model,
            upstream_base_url=_api_target(proxy, "vertex"),
        )

    @app.get("/v1/models")
    async def list_models(request: Request):
        chatgpt_response = await _handle_chatgpt_model_metadata(
            proxy,
            request,
            "/backend-api/models",
        )
        if chatgpt_response is not None:
            return chatgpt_response

        provider_name = proxy.provider_runtime.model_metadata_provider(dict(request.headers))
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, provider_name),
            "models",
            provider_name,
        )

    @app.get("/v1/models/{model_id}")
    async def get_model(request: Request, model_id: str):
        chatgpt_response = await _handle_chatgpt_model_metadata(
            proxy,
            request,
            f"/backend-api/models/{model_id}",
        )
        if chatgpt_response is not None:
            return chatgpt_response

        provider_name = proxy.provider_runtime.model_metadata_provider(dict(request.headers))
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, provider_name),
            "models",
            provider_name,
        )

    @app.post("/v1/embeddings")
    async def openai_embeddings(request: Request):
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, "openai"),
            "embeddings",
            "openai",
        )

    @app.post("/v1/moderations")
    async def openai_moderations(request: Request):
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, "openai"),
            "moderations",
            "openai",
        )

    @app.post("/v1/images/generations")
    async def openai_images_generations(request: Request):
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, "openai"),
            "images/generations",
            "openai",
        )

    @app.post("/v1/audio/transcriptions")
    async def openai_audio_transcriptions(request: Request):
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, "openai"),
            "audio/transcriptions",
            "openai",
        )

    @app.post("/v1/audio/speech")
    async def openai_audio_speech(request: Request):
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, "openai"),
            "audio/speech",
            "openai",
        )

    @app.get("/v1beta/models")
    async def gemini_list_models(request: Request):
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, "gemini"),
            "models",
            "gemini",
        )

    @app.get("/v1beta/models/{model_name}")
    async def gemini_get_model(request: Request, model_name: str):
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, "gemini"),
            "models",
            "gemini",
        )

    @app.post("/v1beta/models/{model}:embedContent")
    async def gemini_embed_content(request: Request, model: str):
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, "gemini"),
            "embedContent",
            "gemini",
        )

    @app.post("/v1beta/models/{model}:batchEmbedContents")
    async def gemini_batch_embed_contents(request: Request, model: str):
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, "gemini"),
            "batchEmbedContents",
            "gemini",
        )

    @app.post("/v1beta/models/{model}:batchGenerateContent")
    async def gemini_batch_create(request: Request, model: str):
        return await proxy.handle_google_batch_create(request, model)

    @app.get("/v1beta/batches/{batch_name}")
    async def gemini_batch_get(request: Request, batch_name: str):
        return await proxy.handle_google_batch_results(request, batch_name)

    @app.post("/v1beta/batches/{batch_name}:cancel")
    async def gemini_batch_cancel(request: Request, batch_name: str):
        return await proxy.handle_google_batch_passthrough(request, batch_name)

    @app.delete("/v1beta/batches/{batch_name}")
    async def gemini_batch_delete(request: Request, batch_name: str):
        return await proxy.handle_google_batch_passthrough(request, batch_name)

    @app.post("/v1beta/cachedContents")
    async def gemini_create_cached_content(request: Request):
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, "gemini"),
            "cachedContents",
            "gemini",
        )

    @app.get("/v1beta/cachedContents")
    async def gemini_list_cached_contents(request: Request):
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, "gemini"),
            "cachedContents",
            "gemini",
        )

    @app.get("/v1beta/cachedContents/{cache_id}")
    async def gemini_get_cached_content(request: Request, cache_id: str):
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, "gemini"),
            "cachedContents",
            "gemini",
        )

    @app.delete("/v1beta/cachedContents/{cache_id}")
    async def gemini_delete_cached_content(request: Request, cache_id: str):
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, "gemini"),
            "cachedContents",
            "gemini",
        )

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def passthrough(request: Request, path: str):
        custom_base = request.headers.get("x-headroom-base-url")
        if custom_base:
            return await proxy.handle_passthrough(request, custom_base.rstrip("/"))
        return await proxy.handle_passthrough(
            request,
            _select_passthrough_base_url(proxy, dict(request.headers)),
        )
