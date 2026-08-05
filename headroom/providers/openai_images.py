"""OpenAI image endpoint routing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from fastapi import Request
from fastapi.responses import Response

from headroom.providers.codex.images import handle_chatgpt_codex_images


@dataclass(frozen=True, slots=True)
class OpenAIImageEndpoint:
    """An OpenAI image endpoint with a possible Codex ChatGPT-auth override."""

    route_path: str
    sub_path: str


OPENAI_IMAGE_ENDPOINTS: tuple[OpenAIImageEndpoint, ...] = (
    OpenAIImageEndpoint("/v1/images/generations", "images/generations"),
    OpenAIImageEndpoint("/v1/images/edits", "images/edits"),
)


def codex_image_subpath(openai_image_sub_path: str) -> str:
    """Return the Codex image backend subpath for an OpenAI image endpoint."""
    return openai_image_sub_path.removeprefix("images/")


def select_codex_image_client(proxy: Any) -> Any:
    """Return the HTTP client used for ChatGPT-auth image forwarding."""
    return getattr(proxy, "http_client_h1", None) or getattr(proxy, "http_client", None)


async def handle_openai_image_endpoint(
    proxy: Any,
    request: Request,
    *,
    openai_api_base_url: str,
    endpoint: OpenAIImageEndpoint,
) -> Response:
    """Handle an OpenAI image endpoint, including Codex ChatGPT-auth routing."""
    chatgpt_response = await handle_chatgpt_codex_images(
        select_codex_image_client(proxy),
        request,
        codex_image_subpath(endpoint.sub_path),
    )
    if chatgpt_response is not None:
        return chatgpt_response

    return cast(
        Response,
        await proxy.handle_passthrough(
            request,
            openai_api_base_url,
            endpoint.sub_path,
            "openai",
        ),
    )
