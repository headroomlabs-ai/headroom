"""Provider model metadata route helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from fastapi import Request
from fastapi.responses import Response

from headroom.providers.codex.model_metadata import handle_chatgpt_model_metadata


@dataclass(frozen=True, slots=True)
class ModelMetadataEndpoint:
    """OpenAI-compatible model metadata endpoint shape."""

    route_path: str
    upstream_path: str
    passthrough_sub_path: str = "models"


MODEL_METADATA_LIST_ENDPOINT = ModelMetadataEndpoint("/v1/models", "/backend-api/models")


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

    return cast(
        Response,
        await proxy.handle_passthrough(
            request,
            provider_api_base_url,
            endpoint.passthrough_sub_path,
            provider_name,
        ),
    )
