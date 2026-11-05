"""Codex ChatGPT-subscription Responses passthrough helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Protocol

from fastapi import Request
from fastapi.responses import Response
from starlette.requests import ClientDisconnect

from .endpoints import codex_backend_url, codex_backend_ws_url
from .headers import drop_header, header_name
from .runtime import resolve_codex_routing

logger = logging.getLogger("headroom.providers.codex.responses")


class CodexResponsesPassthroughResponse(Protocol):
    """HTTP response surface used by Codex Responses passthrough."""

    status_code: int
    content: bytes
    headers: Mapping[str, str]


class CodexResponsesPassthroughHttpClient(Protocol):
    """HTTP client surface needed to forward Codex Responses subpaths."""

    async def request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> CodexResponsesPassthroughResponse:
        """Issue an HTTP request and return an httpx-like response."""
        ...


def normalize_codex_responses_headers(headers: Mapping[str, str]) -> tuple[dict[str, str], bool]:
    """Prepare inbound headers for ChatGPT Codex Responses passthrough."""
    upstream_headers = dict(headers)
    drop_header(upstream_headers, "host")
    # Drop the client's Accept-Encoding so the upstream returns plaintext. httpx
    # decodes the body before we read ``resp.content``, so if we let the upstream
    # gzip we would forward a decoded body under a ``content-encoding: gzip`` /
    # compressed ``content-length`` (see sanitize below) -- the #3019 framing
    # corruption. Mirrors the image passthrough.
    drop_header(upstream_headers, "accept-encoding")
    decision = resolve_codex_routing(upstream_headers)
    return decision.headers, decision.is_chatgpt_auth


def sanitize_codex_responses_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Drop stale compression/framing headers after materializing response bytes.

    ``resp.content`` is already-decoded plaintext, so the upstream's
    ``content-encoding`` and ``content-length`` (measured on the compressed
    bytes) no longer describe what we forward; replaying them makes the client
    fail to gunzip or truncate the body. Starlette recomputes ``content-length``.
    """
    response_headers = dict(headers)
    drop_header(response_headers, "content-encoding")
    drop_header(response_headers, "content-length")
    drop_header(response_headers, "server")
    return response_headers


def codex_responses_subpath_url(sub_path: str, query: str = "") -> str:
    """Return the ChatGPT Codex Responses upstream URL for a route subpath."""
    return codex_backend_url(f"/responses/{sub_path}", query)


def codex_responses_http_url(query: str = "") -> str:
    """Return the ChatGPT Codex Responses HTTP upstream URL."""
    return codex_backend_url("/responses", query)


def codex_responses_websocket_url() -> str:
    """Return the ChatGPT Codex Responses WebSocket upstream URL."""
    return codex_backend_ws_url("/responses")


def has_chatgpt_account_header(headers: Mapping[str, str]) -> bool:
    """Return whether resolved headers contain a ChatGPT account routing hint."""
    return header_name(headers, "chatgpt-account-id") is not None


async def handle_chatgpt_codex_responses_subpath(
    http_client: CodexResponsesPassthroughHttpClient,
    request: Request,
    sub_path: str,
) -> Response | None:
    """Forward ChatGPT-auth Codex Responses subpaths or return None for OpenAI fallback."""
    headers, is_chatgpt_auth = normalize_codex_responses_headers(dict(request.headers.items()))
    if not is_chatgpt_auth:
        return None

    try:
        body = await request.body()
    except ClientDisconnect:
        logger.debug("Client disconnected during body read for /v1/responses passthrough")
        return Response(status_code=204)
    try:
        resp = await http_client.request(
            request.method,
            codex_responses_subpath_url(sub_path, request.url.query),
            headers=headers,
            content=body,
            timeout=120.0,
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=sanitize_codex_responses_response_headers(resp.headers),
        )
    except Exception:
        logger.exception("Passthrough /v1/responses/%s failed", sub_path)
        return Response(content="Upstream request failed.", status_code=502)
