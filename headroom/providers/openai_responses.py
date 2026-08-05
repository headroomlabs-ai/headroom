"""OpenAI Responses API passthrough helpers."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from fastapi.responses import Response
from starlette.requests import ClientDisconnect

logger = logging.getLogger("headroom.providers.openai.responses")


def _sanitize_for_log(value: str) -> str:
    """Return a log-safe single-line representation of untrusted text."""
    return value.replace("\r", "").replace("\n", "")


@dataclass(frozen=True, slots=True)
class OpenAIResponsesSubpathRoute:
    """Responses API subpath alias exposed by provider route registration."""

    path: str
    methods: tuple[str, ...]


OPENAI_RESPONSES_ROOT_PATHS: tuple[str, ...] = (
    "/v1/responses",
    "/v1/codex/responses",
    "/backend-api/responses",
    "/backend-api/codex/responses",
)

OPENAI_RESPONSES_WEBSOCKET_PATHS: tuple[str, ...] = OPENAI_RESPONSES_ROOT_PATHS

OPENAI_RESPONSES_SUBPATH_ROUTES: tuple[OpenAIResponsesSubpathRoute, ...] = (
    OpenAIResponsesSubpathRoute("/v1/responses/{sub_path:path}", ("GET", "POST", "DELETE")),
    OpenAIResponsesSubpathRoute("/v1/codex/responses/{sub_path:path}", ("GET", "POST", "DELETE")),
    OpenAIResponsesSubpathRoute(
        "/backend-api/responses/{sub_path:path}",
        ("GET", "POST", "DELETE"),
    ),
    OpenAIResponsesSubpathRoute(
        "/backend-api/codex/responses/{sub_path:path}",
        ("GET", "POST", "DELETE"),
    ),
)


def openai_responses_subpath_url(api_base_url: str, sub_path: str, query: str = "") -> str:
    """Build an OpenAI Responses API subpath URL."""
    url = f"{api_base_url.rstrip('/')}/v1/responses/{sub_path}"
    if query:
        url = f"{url}?{query}"
    return url


def normalize_openai_responses_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Return request headers suitable for upstream OpenAI forwarding."""
    upstream_headers = dict(headers)
    upstream_headers.pop("host", None)
    return upstream_headers


async def handle_openai_responses_subpath(
    http_client: Any,
    request: Request,
    api_base_url: str,
    sub_path: str,
) -> Response:
    """Forward a Responses API subpath request to the configured OpenAI upstream."""
    url = openai_responses_subpath_url(api_base_url, sub_path, request.url.query)
    try:
        body = await request.body()
    except ClientDisconnect:
        logger.debug("Client disconnected during body read for codex responses passthrough")
        return Response(status_code=204)
    try:
        resp = await http_client.request(
            request.method,
            url,
            headers=normalize_openai_responses_headers(dict(request.headers.items())),
            content=body,
            timeout=120.0,
        )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=dict(resp.headers),
        )
    except Exception as exc:
        logger.error(
            "Passthrough /v1/responses/%s failed: %s",
            _sanitize_for_log(sub_path),
            exc,
        )
        return Response(content="Upstream request failed.", status_code=502)
