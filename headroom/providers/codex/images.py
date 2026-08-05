"""Codex ChatGPT-subscription image forwarding."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any, Protocol

from fastapi import Request
from fastapi.responses import Response
from starlette.requests import ClientDisconnect

from headroom.proxy.helpers import _strip_internal_headers

from .endpoints import codex_backend_url
from .headers import drop_header
from .runtime import resolve_codex_routing

logger = logging.getLogger("headroom.providers.codex.images")


class CodexImageForwardResponse(Protocol):
    """HTTP response surface used by Codex image forwarding."""

    status_code: int
    content: bytes
    headers: Mapping[str, str]


class CodexImageForwardHttpClient(Protocol):
    """HTTP client surface needed to forward Codex image requests."""

    async def request(self, method: str, url: str, **kwargs: Any) -> CodexImageForwardResponse:
        """Issue an HTTP request and return an httpx-like response."""
        ...


def normalize_codex_image_headers(headers: Mapping[str, str]) -> tuple[dict[str, str], bool]:
    """Prepare inbound headers for ChatGPT Codex image forwarding."""
    upstream_headers = dict(headers)
    drop_header(upstream_headers, "host")
    drop_header(upstream_headers, "accept-encoding")
    upstream_headers = _strip_internal_headers(upstream_headers)
    decision = resolve_codex_routing(upstream_headers)
    return decision.headers, decision.is_chatgpt_auth


def codex_image_url(sub_path: str, query: str = "") -> str:
    """Return the ChatGPT Codex image upstream URL for a route subpath."""
    return codex_backend_url(f"/images/{sub_path}", query)


def sanitize_codex_image_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Drop stale compression/framing headers after materializing response bytes."""
    response_headers = dict(headers)
    drop_header(response_headers, "content-encoding")
    drop_header(response_headers, "content-length")
    drop_header(response_headers, "server")
    return response_headers


def codex_image_forward_error_response() -> Response:
    """Return the stable client-facing error for failed Codex image forwarding."""
    return Response(
        content=json.dumps(
            {
                "error": {
                    "type": "upstream_error",
                    "message": "Failed to forward Codex image request",
                }
            }
        ),
        status_code=502,
        media_type="application/json",
    )


async def handle_chatgpt_codex_images(
    http_client: CodexImageForwardHttpClient | None,
    request: Request,
    sub_path: str,
) -> Response | None:
    """Forward Codex OAuth image requests to ChatGPT's Codex image backend."""
    headers, is_chatgpt_auth = normalize_codex_image_headers(dict(request.headers.items()))
    if not is_chatgpt_auth:
        return None

    try:
        body = await request.body()
    except ClientDisconnect:
        logger.debug("Client disconnected during body read for passthrough")
        return Response(status_code=204)
    try:
        if http_client is None:
            raise RuntimeError("No HTTP client configured for Codex image forwarding")
        resp = await http_client.request(
            request.method,
            codex_image_url(sub_path, request.url.query),
            headers=headers,
            content=body,
            timeout=120.0,
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=sanitize_codex_image_response_headers(resp.headers),
        )
    except Exception as exc:
        logger.error("Passthrough /v1/images/%s failed: %s", sub_path, exc)
        return codex_image_forward_error_response()
