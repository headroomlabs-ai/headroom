"""Codex ChatGPT-subscription model metadata handling."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote

from fastapi import Request
from fastapi.responses import Response

from .endpoints import chatgpt_backend_url, codex_backend_url
from .headers import drop_header, header_name
from .runtime import resolve_codex_routing

logger = logging.getLogger("headroom.providers.codex.model_metadata")
DEFAULT_CODEX_CLIENT_VERSION = "0.130.0"


@dataclass(frozen=True, slots=True)
class CodexModelRegistryOptions:
    """Configuration for ChatGPT Codex model-registry lookups."""

    default_client_version: str = DEFAULT_CODEX_CLIENT_VERSION
    timeout_seconds: float = 15.0


class CodexModelRegistryResponse(Protocol):
    """HTTP response surface used by Codex model metadata helpers."""

    status_code: int
    text: str
    content: bytes
    headers: Mapping[str, str]

    def json(self) -> Any:
        """Parse response JSON."""
        ...


class CodexModelRegistryHttpClient(Protocol):
    """HTTP client surface needed to fetch the Codex model registry."""

    async def get(self, url: str, **kwargs: Any) -> CodexModelRegistryResponse:
        """Issue a GET request and return an httpx-like response."""
        ...

    async def request(self, method: str, url: str, **kwargs: Any) -> CodexModelRegistryResponse:
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


CODEX_REASONING_LEVELS: tuple[dict[str, str], ...] = (
    {"effort": "low", "description": "Fast responses with lighter reasoning"},
    {
        "effort": "medium",
        "description": "Balances speed and reasoning depth for everyday tasks",
    },
    {"effort": "high", "description": "Greater reasoning depth for complex problems"},
    {"effort": "xhigh", "description": "Extra high reasoning depth for complex problems"},
)


def codex_client_version(
    requested_client_version: str | None = None,
    options: CodexModelRegistryOptions = CodexModelRegistryOptions(),
) -> str:
    """Return the Codex client version to use for model-registry requests."""
    if requested_client_version:
        return requested_client_version
    return options.default_client_version


def _json_response(payload: dict[str, Any], status_code: int = 200) -> Response:
    return Response(
        content=json.dumps(payload),
        status_code=status_code,
        headers={"content-type": "application/json"},
    )


def _model_payload(model_id: str) -> dict[str, Any]:
    return {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "openai",
    }


def display_name_from_model_id(model_id: str) -> str:
    """Return the Codex display name for a model slug."""
    return "-".join(
        part.upper() if part == "gpt" else part.capitalize() for part in model_id.split("-")
    )


def codex_model_registry_entry(
    model_id: str,
    upstream_entry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return Codex app-server model metadata with required registry fields."""
    entry = dict(upstream_entry or {})
    entry["slug"] = model_id
    entry.setdefault("display_name", display_name_from_model_id(model_id))
    entry.setdefault("description", "Codex model available through ChatGPT subscription auth.")
    entry.setdefault("default_reasoning_level", "medium")
    entry.setdefault("supported_reasoning_levels", list(CODEX_REASONING_LEVELS))
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


def _model_not_found_response(model_id: str) -> Response:
    return _json_response(
        {
            "error": {
                "message": f"Model {model_id!r} not available under ChatGPT auth",
                "type": "invalid_request_error",
                "code": "model_not_found",
            }
        },
        status_code=404,
    )


def models_list_response_from_entries(model_entries: tuple[dict[str, Any], ...]) -> Response:
    """Build an OpenAI-compatible model-list response."""
    model_ids = tuple(
        slug
        for entry in model_entries
        for slug in (entry.get("slug"),)
        if isinstance(slug, str) and slug
    )
    return _json_response(
        {
            "object": "list",
            "data": [_model_payload(model_id) for model_id in model_ids],
            "models": list(model_entries),
        }
    )


def models_list_response(model_ids: tuple[str, ...]) -> Response:
    """Build a model-list response from model slugs."""
    return models_list_response_from_entries(
        tuple(codex_model_registry_entry(model_id) for model_id in model_ids)
    )


def synthetic_models_list_response() -> Response:
    """OpenAI-compatible `/v1/models` payload for Codex ChatGPT auth."""
    return models_list_response(CHATGPT_AUTH_CODEX_MODELS)


def synthetic_model_get_response(model_id: str) -> Response:
    """OpenAI-compatible `/v1/models/{id}` payload."""
    if model_id not in CHATGPT_AUTH_CODEX_MODELS:
        return _model_not_found_response(model_id)
    return _json_response(_model_payload(model_id))


def normalize_codex_registry_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Prepare inbound ChatGPT auth headers for the Codex model registry."""
    upstream_headers = dict(headers)
    drop_header(upstream_headers, "host")

    account_header = header_name(upstream_headers, "chatgpt-account-id")
    account_id = upstream_headers.get(account_header, "") if account_header else ""
    if account_header is not None and account_id:
        upstream_headers["chatgpt-account-id"] = account_id
        if account_header != "chatgpt-account-id":
            upstream_headers.pop(account_header, None)

    upstream_headers["accept"] = "application/json"
    for existing_header_name in list(upstream_headers):
        if existing_header_name.lower() == "accept" and existing_header_name != "accept":
            upstream_headers.pop(existing_header_name, None)
    return upstream_headers


async def fetch_chatgpt_codex_model_entries(
    http_client: CodexModelRegistryHttpClient,
    headers: Mapping[str, str],
    requested_client_version: str | None,
    options: CodexModelRegistryOptions = CodexModelRegistryOptions(),
) -> tuple[dict[str, Any], ...] | None:
    """Fetch Codex model metadata from ChatGPT, returning None when fallback should apply."""
    client_version = codex_client_version(requested_client_version, options)
    upstream_headers = normalize_codex_registry_headers(headers)
    url = codex_backend_url("models", f"client_version={quote(client_version, safe='')}")
    try:
        resp = await http_client.get(
            url,
            headers=upstream_headers,
            timeout=options.timeout_seconds,
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
            codex_model_registry_entry(slug, entry)
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


async def fetch_chatgpt_codex_model_ids(
    http_client: CodexModelRegistryHttpClient,
    headers: Mapping[str, str],
    requested_client_version: str | None,
    options: CodexModelRegistryOptions = CodexModelRegistryOptions(),
) -> tuple[str, ...] | None:
    """Fetch Codex model slugs from ChatGPT, returning None when fallback should apply."""
    model_entries = await fetch_chatgpt_codex_model_entries(
        http_client,
        headers,
        requested_client_version,
        options,
    )
    if model_entries is None:
        return None
    return tuple(
        slug
        for entry in model_entries
        for slug in (entry.get("slug"),)
        if isinstance(slug, str) and slug
    )


async def fetch_chatgpt_codex_models_response(
    http_client: CodexModelRegistryHttpClient,
    headers: Mapping[str, str],
    requested_client_version: str | None,
) -> Response | None:
    """Build a dynamic `/v1/models` response from the Codex registry when available."""
    model_entries = await fetch_chatgpt_codex_model_entries(
        http_client, headers, requested_client_version
    )
    if model_entries is None:
        return None
    return models_list_response_from_entries(model_entries)


async def fetch_chatgpt_codex_model_get_response(
    http_client: CodexModelRegistryHttpClient,
    headers: Mapping[str, str],
    model_id: str,
    requested_client_version: str | None,
) -> Response | None:
    """Build a dynamic `/v1/models/{id}` response from the Codex registry when available."""
    model_entries = await fetch_chatgpt_codex_model_entries(
        http_client, headers, requested_client_version
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
        return _json_response(_model_payload(model_id))
    return _model_not_found_response(model_id)


async def handle_chatgpt_model_metadata(
    http_client: CodexModelRegistryHttpClient,
    request: Request,
    upstream_path: str,
) -> Response | None:
    """Handle Codex ChatGPT-auth model metadata or return None for normal routing."""
    headers = dict(request.headers.items())
    drop_header(headers, "host")
    routing = resolve_codex_routing(headers)
    if not routing.is_chatgpt_auth:
        return None
    headers = routing.headers

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

    url = chatgpt_backend_url(upstream_path, request.url.query)

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
    except Exception:
        logger.exception("Passthrough %s failed", upstream_path)
        return Response(content="Upstream request failed.", status_code=502)
