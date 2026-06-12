"""Codex ChatGPT-subscription model metadata handling."""

from __future__ import annotations

import json
import logging
from typing import Any, Protocol
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import Response

from .runtime import resolve_codex_routing_headers

logger = logging.getLogger("headroom.providers.codex.model_metadata")


class CodexModelRegistryHttpClient(Protocol):
    """HTTP client surface needed to fetch the Codex model registry."""

    async def get(self, url: str, **kwargs: Any) -> Any:
        """Issue a GET request and return an httpx-like response."""
        ...

    async def request(self, method: str, url: str, **kwargs: Any) -> Any:
        """Issue a generic request and return an httpx-like response."""
        ...


# Codex ChatGPT-subscription auth cannot call `chatgpt.com/backend-api/models`
# with OAuth bearer tokens. These are known-good Codex model slugs used when
# the provider registry is unavailable.
CHATGPT_AUTH_CODEX_MODELS: tuple[str, ...] = (
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.3",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
)


def codex_client_version(requested_client_version: str | None = None) -> str:
    """Return the Codex client version to use for model-registry requests."""
    if requested_client_version:
        return requested_client_version
    return "0.130.0"


def models_list_response(model_ids: tuple[str, ...]) -> Response:
    """Build an OpenAI-compatible model-list response."""
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
    }
    return Response(
        content=json.dumps(payload),
        status_code=200,
        headers={"content-type": "application/json"},
    )


def synthetic_models_list_response() -> Response:
    """OpenAI-compatible `/v1/models` payload for Codex ChatGPT auth."""
    return models_list_response(CHATGPT_AUTH_CODEX_MODELS)


def synthetic_model_get_response(model_id: str) -> Response:
    """OpenAI-compatible `/v1/models/{id}` payload."""
    if model_id not in CHATGPT_AUTH_CODEX_MODELS:
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


def normalize_codex_registry_headers(headers: dict[str, str]) -> dict[str, str]:
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


async def fetch_chatgpt_codex_model_ids(
    http_client: CodexModelRegistryHttpClient,
    headers: dict[str, str],
    requested_client_version: str | None,
) -> tuple[str, ...] | None:
    """Fetch Codex model slugs from ChatGPT, returning None when fallback should apply."""
    client_version = codex_client_version(requested_client_version)
    upstream_headers = normalize_codex_registry_headers(headers)
    url = (
        "https://chatgpt.com/backend-api/codex/models"
        f"?client_version={quote(client_version, safe='')}"
    )
    try:
        resp = await http_client.get(
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

        model_ids = tuple(
            slug
            for entry in models_raw
            if isinstance(entry, dict)
            for slug in (entry.get("slug"),)
            if isinstance(slug, str) and slug
        )
        if not model_ids:
            logger.warning("Codex model registry returned no model slugs")
            return None

        logger.info("Fetched %d Codex models from upstream model registry", len(model_ids))
        logger.debug("Fetched Codex model IDs from upstream model registry: %s", list(model_ids))
        return model_ids
    except Exception:
        logger.exception("Codex model registry fetch failed")
        return None


async def fetch_chatgpt_codex_models_response(
    http_client: CodexModelRegistryHttpClient,
    headers: dict[str, str],
    requested_client_version: str | None,
) -> Response | None:
    """Build a dynamic `/v1/models` response from the Codex registry when available."""
    model_ids = await fetch_chatgpt_codex_model_ids(http_client, headers, requested_client_version)
    if model_ids is None:
        return None
    return models_list_response(model_ids)


async def fetch_chatgpt_codex_model_get_response(
    http_client: CodexModelRegistryHttpClient,
    headers: dict[str, str],
    model_id: str,
    requested_client_version: str | None,
) -> Response | None:
    """Build a dynamic `/v1/models/{id}` response from the Codex registry when available."""
    model_ids = await fetch_chatgpt_codex_model_ids(http_client, headers, requested_client_version)
    if model_ids is None:
        return None
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


async def handle_chatgpt_model_metadata(
    http_client: CodexModelRegistryHttpClient,
    request: Request,
    upstream_path: str,
) -> Response | None:
    """Handle Codex ChatGPT-auth model metadata or return None for normal routing."""
    headers = dict(request.headers.items())
    headers.pop("host", None)
    headers, is_chatgpt_auth = resolve_codex_routing_headers(headers)
    if not is_chatgpt_auth:
        return None

    requested_client_version = request.query_params.get("client_version")
    if upstream_path == "/backend-api/models":
        upstream_response = await fetch_chatgpt_codex_models_response(
            http_client,
            headers,
            requested_client_version,
        )
        if upstream_response is not None:
            return upstream_response
        return synthetic_models_list_response()
    if upstream_path.startswith("/backend-api/models/"):
        model_id = upstream_path[len("/backend-api/models/") :]
        upstream_response = await fetch_chatgpt_codex_model_get_response(
            http_client,
            headers,
            model_id,
            requested_client_version,
        )
        if upstream_response is not None:
            return upstream_response
        return synthetic_model_get_response(model_id)

    url = f"https://chatgpt.com{upstream_path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    body = await request.body()
    try:
        resp = await http_client.request(
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
