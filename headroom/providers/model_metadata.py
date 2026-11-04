"""Provider model metadata route helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, NoReturn, cast

from fastapi import Request
from fastapi.responses import Response

from headroom.providers.codex.model_metadata import handle_chatgpt_model_metadata
from headroom.providers.grok.model_metadata import (
    is_xai_model_list_target,
    normalize_xai_model_metadata,
)
from headroom.proxy.helpers import sanitize_forwarded_response_headers


@dataclass(frozen=True, slots=True)
class ModelMetadataEndpoint:
    """OpenAI-compatible model metadata endpoint shape."""

    route_path: str
    upstream_path: str
    passthrough_sub_path: str = "models"


MODEL_METADATA_LIST_ENDPOINT = ModelMetadataEndpoint("/v1/models", "/backend-api/models")


def _reject_non_standard_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-standard JSON constant: {value}")


def model_metadata_get_endpoint(model_id: str) -> ModelMetadataEndpoint:
    """Return the single-model metadata endpoint for ``model_id``."""
    return ModelMetadataEndpoint(
        "/v1/models/{model_id}",
        f"/backend-api/models/{model_id}",
    )


async def handle_model_metadata_endpoint(
    proxy: Any,
    request: Request,
    *,
    endpoint: ModelMetadataEndpoint,
    provider_api_base_url: str,
    provider_name: str,
) -> Response:
    """Handle OpenAI-compatible model metadata with Codex ChatGPT-auth support."""
    assert proxy.http_client is not None
    chatgpt_response = await handle_chatgpt_model_metadata(
        proxy.http_client,
        request,
        endpoint.upstream_path,
    )
    if chatgpt_response is not None:
        return chatgpt_response

    response = cast(
        Response,
        await proxy.handle_passthrough(
            request,
            provider_api_base_url,
            endpoint.passthrough_sub_path,
            provider_name,
        ),
    )
    if (
        endpoint == MODEL_METADATA_LIST_ENDPOINT
        and 200 <= response.status_code < 300
        and is_xai_model_list_target(provider_api_base_url)
    ):
        normalized_content: bytes | None = None
        try:
            payload = json.loads(
                response.body,
                parse_constant=_reject_non_standard_json_constant,
            )
            normalized_payload = normalize_xai_model_metadata(payload)
            if normalized_payload is not None:
                normalized_content = json.dumps(
                    normalized_payload,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
        except (TypeError, UnicodeError, ValueError):
            normalized_payload = None
        if normalized_payload is not None and normalized_content is not None:
            headers = sanitize_forwarded_response_headers(
                response.headers,
                "etag",
                "last-modified",
                "cache-control",
            )
            headers["content-type"] = "application/json"
            return Response(
                content=normalized_content,
                status_code=response.status_code,
                headers=headers,
            )
    return response
