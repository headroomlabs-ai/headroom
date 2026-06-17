# mypy: disable-error-code=no-untyped-def
"""Provider-specific proxy route registration."""

from __future__ import annotations

import json
import logging
from typing import Any, cast
from urllib.parse import quote

from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response

from headroom.proxy.handlers.openai import _resolve_codex_routing_headers

logger = logging.getLogger("headroom.proxy.routes")


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


# Codex ChatGPT-subscription auth doesn't have access to
# `chatgpt.com/backend-api/models` — that endpoint returns 403 to OAuth
# bearer tokens (issue #478). Codex polls `/v1/models` every few seconds
# to populate its model-picker UI, so the 403 storm is noisy and breaks
# refresh. The fix: when Codex hits `/v1/models` under ChatGPT auth,
# fetch the Codex-specific registry first and synthesize an
# OpenAI-compatible response from its slugs. If that registry is
# unavailable, fall back to the known-supported static set.
#
# The list mirrors what Codex itself ships in its built-in model
# registry (the same models its provider config exposes); it's the
# safe-by-construction set since these are what `/v1/responses` actually
# accepts under ChatGPT auth.
_CHATGPT_AUTH_CODEX_MODELS: tuple[str, ...] = (
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.3",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
)


def _codex_client_version(requested_client_version: str | None = None) -> str:
    """Return the Codex client version to use for model-registry requests."""
    if requested_client_version:
        return requested_client_version
    return "0.130.0"


_CODEX_REASONING_LEVELS: tuple[dict[str, str], ...] = (
    {"effort": "low", "description": "Fast responses with lighter reasoning"},
    {
        "effort": "medium",
        "description": "Balances speed and reasoning depth for everyday tasks",
    },
    {"effort": "high", "description": "Greater reasoning depth for complex problems"},
    {"effort": "xhigh", "description": "Extra high reasoning depth for complex problems"},
)


def _display_name_from_model_id(model_id: str) -> str:
    return "-".join(
        part.upper() if part == "gpt" else part.capitalize() for part in model_id.split("-")
    )


def _codex_model_registry_entry(
    model_id: str,
    upstream_entry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Codex app-server model metadata with required registry fields."""
    entry = dict(upstream_entry or {})
    entry["slug"] = model_id
    entry.setdefault("display_name", _display_name_from_model_id(model_id))
    entry.setdefault("description", "Codex model available through ChatGPT subscription auth.")
    entry.setdefault("default_reasoning_level", "medium")
    entry.setdefault("supported_reasoning_levels", list(_CODEX_REASONING_LEVELS))
    entry.setdefault("shell_type", "shell_command")
    entry.setdefault("visibility", "list")
    entry.setdefault("supported_in_api", True)
    entry.setdefault("priority", 50)
    entry.setdefault("additional_speed_tiers", ["fast"])
    entry.setdefault(
        "service_tiers",
        [{"id": "priority", "name": "Fast", "description": "1.5x speed, increased usage"}],
    )
    entry.setdefault("availability_nux", None)
    entry.setdefault("upgrade", None)
    entry.setdefault("context_window", 272000)
    entry.setdefault("max_context_window", 272000)
    entry.setdefault("effective_context_window_percent", 95)
    entry.setdefault("experimental_supported_tools", [])
    entry.setdefault("input_modalities", ["text", "image"])
    entry.setdefault("supports_search_tool", True)
    entry.setdefault("use_responses_lite", False)
    entry.setdefault("support_verbosity", True)
    entry.setdefault("default_verbosity", "low")
    entry.setdefault("apply_patch_tool_type", "freeform")
    entry.setdefault("web_search_tool_type", "text_and_image")
    entry.setdefault("truncation_policy", {"mode": "tokens", "limit": 10000})
    entry.setdefault("supports_image_detail_original", True)
    entry.setdefault("supports_parallel_tool_calls", True)
    entry.setdefault("supports_reasoning_summaries", True)
    entry.setdefault("default_reasoning_summary", "none")
    return entry


def _models_list_response_from_entries(model_entries: tuple[dict[str, Any], ...]) -> Response:
    model_ids = tuple(
        slug
        for entry in model_entries
        for slug in (entry.get("slug"),)
        if isinstance(slug, str) and slug
    )
    payload = {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "openai",
            }
            for model_id in model_ids
        ],
        "models": list(model_entries),
    }
    return Response(
        content=json.dumps(payload),
        status_code=200,
        headers={"content-type": "application/json"},
    )


def _synthetic_models_list_response() -> Response:
    """OpenAI-compatible `/v1/models` payload for Codex ChatGPT auth."""
    return _models_list_response_from_entries(
        tuple(_codex_model_registry_entry(model_id) for model_id in _CHATGPT_AUTH_CODEX_MODELS)
    )


def _synthetic_model_get_response(model_id: str) -> Response:
    """OpenAI-compatible `/v1/models/{id}` payload."""
    if model_id not in _CHATGPT_AUTH_CODEX_MODELS:
        return Response(
            content=json.dumps(
                {
                    "error": {
                        "message": f"Model {model_id!r} not available under ChatGPT auth",
                        "type": "invalid_request_error",
                        "code": "model_not_found",
                    }
                }
            ),
            status_code=404,
            headers={"content-type": "application/json"},
        )
    return Response(
        content=json.dumps(
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": "openai",
            }
        ),
        status_code=200,
        headers={"content-type": "application/json"},
    )


def _normalize_codex_registry_headers(headers: dict[str, str]) -> dict[str, str]:
    """Prepare inbound ChatGPT auth headers for the Codex model registry."""
    upstream_headers = dict(headers)
    upstream_headers.pop("host", None)
    account_id = (
        upstream_headers.get("chatgpt-account-id")
        or upstream_headers.get("ChatGPT-Account-ID")
        or ""
    )
    if account_id:
        upstream_headers["chatgpt-account-id"] = account_id
        upstream_headers.pop("ChatGPT-Account-ID", None)
    upstream_headers["accept"] = "application/json"
    upstream_headers.pop("Accept", None)
    return upstream_headers


async def _fetch_chatgpt_codex_model_entries(
    proxy: Any,
    headers: dict[str, str],
    requested_client_version: str | None,
) -> tuple[dict[str, Any], ...] | None:
    """Fetch Codex model metadata from ChatGPT, returning None when fallback should apply."""
    client_version = _codex_client_version(requested_client_version)
    upstream_headers = _normalize_codex_registry_headers(headers)
    url = (
        "https://chatgpt.com/backend-api/codex/models"
        f"?client_version={quote(client_version, safe='')}"
    )
    try:
        assert proxy.http_client is not None
        resp = await proxy.http_client.get(
            url,
            headers=upstream_headers,
            timeout=15.0,
        )
        if resp.status_code >= 400:
            logger.warning(
                "Codex model registry fetch failed: HTTP %s: %s",
                resp.status_code,
                resp.text[:300],
            )
            return None

        data = resp.json()
        models_raw = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models_raw, list):
            logger.warning("Codex model registry response did not contain models[]")
            return None

        model_entries = tuple(
            _codex_model_registry_entry(slug, entry)
            for entry in models_raw
            if isinstance(entry, dict)
            for slug in (entry.get("slug"),)
            if isinstance(slug, str) and slug
        )
        if not model_entries:
            logger.warning("Codex model registry returned no model slugs")
            return None

        model_ids = [entry["slug"] for entry in model_entries]
        logger.info("Fetched %d Codex models from upstream model registry", len(model_entries))
        logger.debug("Fetched Codex model IDs from upstream model registry: %s", model_ids)
        return model_entries
    except Exception:
        logger.exception("Codex model registry fetch failed")
        return None


async def _fetch_chatgpt_codex_models_response(
    proxy: Any,
    headers: dict[str, str],
    requested_client_version: str | None,
) -> Response | None:
    """Build a dynamic `/v1/models` response from the Codex registry when available."""
    model_entries = await _fetch_chatgpt_codex_model_entries(
        proxy, headers, requested_client_version
    )
    if model_entries is None:
        return None
    return _models_list_response_from_entries(model_entries)


async def _fetch_chatgpt_codex_model_get_response(
    proxy: Any,
    headers: dict[str, str],
    model_id: str,
    requested_client_version: str | None,
) -> Response | None:
    """Build a dynamic `/v1/models/{id}` response from the Codex registry when available."""
    model_entries = await _fetch_chatgpt_codex_model_entries(
        proxy, headers, requested_client_version
    )
    if model_entries is None:
        return None
    model_ids = tuple(
        slug
        for entry in model_entries
        for slug in (entry.get("slug"),)
        if isinstance(slug, str) and slug
    )
    if model_id in model_ids:
        return Response(
            content=json.dumps(
                {
                    "id": model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "openai",
                }
            ),
            status_code=200,
            headers={"content-type": "application/json"},
        )
    return Response(
        content=json.dumps(
            {
                "error": {
                    "message": f"Model {model_id!r} not available under ChatGPT auth",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            }
        ),
        status_code=404,
        headers={"content-type": "application/json"},
    )


async def _handle_chatgpt_model_metadata(
    proxy: Any,
    request: Request,
    upstream_path: str,
) -> Response | None:
    headers = dict(request.headers.items())
    headers.pop("host", None)
    headers, is_chatgpt_auth = _resolve_codex_routing_headers(headers)
    if not is_chatgpt_auth:
        return None

    # Avoid generic `/backend-api/models[/{id}]`, which returns 403 for
    # OAuth tokens, but prefer the Codex-specific registry when available.
    requested_client_version = request.query_params.get("client_version")
    if upstream_path == "/backend-api/models":
        upstream_response = await _fetch_chatgpt_codex_models_response(
            proxy,
            headers,
            requested_client_version,
        )
        if upstream_response is not None:
            return upstream_response
        return _synthetic_models_list_response()
    if upstream_path.startswith("/backend-api/models/"):
        model_id = upstream_path[len("/backend-api/models/") :]
        upstream_response = await _fetch_chatgpt_codex_model_get_response(
            proxy,
            headers,
            model_id,
            requested_client_version,
        )
        if upstream_response is not None:
            return upstream_response
        return _synthetic_model_get_response(model_id)

    url = f"https://chatgpt.com{upstream_path}"
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
        logger.error("Passthrough %s failed: %s", upstream_path, exc)
        return Response(content=str(exc), status_code=502)


def register_provider_routes(app: FastAPI, proxy: Any) -> None:
    """Register provider-specific proxy endpoints."""

    async def vertex_publisher_passthrough(request: Request, publisher: str, action: str):
        return await proxy.handle_passthrough(
            request,
            _api_target(proxy, "vertex"),
            action,
            f"vertex:{publisher}",
        )

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request):
        return await proxy.handle_anthropic_messages(request)

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
        if publisher == "google":
            return await proxy.handle_gemini_generate_content(
                request,
                model,
                _api_target(proxy, "vertex"),
                "vertex:google",
            )
        return await vertex_publisher_passthrough(request, publisher, "generateContent")

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
        if publisher == "google":
            return await proxy.handle_gemini_generate_content(
                request,
                model,
                _api_target(proxy, "vertex"),
                "vertex:google",
            )
        return await vertex_publisher_passthrough(request, publisher, "streamGenerateContent")

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
        if publisher == "google":
            return await proxy.handle_gemini_count_tokens(
                request,
                model,
                _api_target(proxy, "vertex"),
                "vertex:google",
            )
        return await vertex_publisher_passthrough(request, publisher, "countTokens")

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
        if publisher == "anthropic":
            return await proxy.handle_anthropic_messages(
                request,
                _api_target(proxy, "vertex"),
                "vertex:anthropic",
                model,
            )
        return await vertex_publisher_passthrough(request, publisher, "rawPredict")

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
        if publisher == "anthropic":
            return await proxy.handle_anthropic_messages(
                request,
                _api_target(proxy, "vertex"),
                "vertex:anthropic",
                model,
                True,
            )
        return await vertex_publisher_passthrough(request, publisher, "streamRawPredict")

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

        # Intercept Code Assist authentication and onboarding routes
        clean_path = path.lstrip("/")
        if clean_path.startswith(("v1internal:", "v1/v1internal:")):
            # Normalize path (remove v1/ prefix if present to avoid 404 on cloudcode-pa upstream)
            normalized_path = clean_path
            if normalized_path.startswith("v1/"):
                normalized_path = normalized_path[3:]
            normalized_path = f"/{normalized_path}"

            # Mutate request scope so handle_passthrough uses the normalized path
            request.scope["path"] = normalized_path
            if "raw_path" in request.scope:
                from urllib.parse import quote

                request.scope["raw_path"] = quote(normalized_path).encode("ascii")
            if hasattr(request, "_url"):
                delattr(request, "_url")

            return await proxy.handle_passthrough(
                request,
                _api_target(proxy, "cloudcode"),
            )

        return await proxy.handle_passthrough(
            request,
            _select_passthrough_base_url(proxy, dict(request.headers)),
        )
