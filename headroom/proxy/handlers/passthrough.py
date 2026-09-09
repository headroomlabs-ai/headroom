"""Generic passthrough handler mixin for HeadroomProxy.

Forwards requests that do not hit the compressed ``/v1/messages`` path
(``/v1/messages/count_tokens``, ``/v1/models``, and the catch-all route)
upstream unchanged, recording an outcome so the dashboard still sees them.

Extracted from the removed OpenAI handler so the Anthropic-only build keeps a
working catch-all. Also hosts two small helpers the Anthropic handler shares.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, unquote

import httpx

from headroom.proxy.auth_mode import classify_client
from headroom.proxy.helpers import extract_tags, sanitize_forwarded_response_headers
from headroom.proxy.outcome import RequestOutcome

if TYPE_CHECKING:
    from fastapi import Request
    from fastapi.responses import Response

logger = logging.getLogger("headroom.proxy")


def _usage_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _passthrough_usage_from_json(payload: Any) -> dict[str, int]:
    """Normalize usage from pass-through provider response shapes."""
    if not isinstance(payload, dict):
        return {}

    usage = payload.get("usage")
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens")
        if input_tokens is None:
            input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("output_tokens")
        if output_tokens is None:
            output_tokens = usage.get("completion_tokens")
        details = usage.get("prompt_tokens_details") or usage.get("input_tokens_details") or {}
        cache_read = details.get("cached_tokens") if isinstance(details, dict) else None
        return {
            "input_tokens": _usage_int(input_tokens),
            "output_tokens": _usage_int(output_tokens),
            "cache_read_input_tokens": _usage_int(usage.get("cache_read_input_tokens", cache_read)),
            "cache_creation_input_tokens": _usage_int(usage.get("cache_creation_input_tokens")),
        }

    return {}


def _passthrough_model_from_path(path: str, endpoint_name: str) -> str:
    marker = "/models/"
    if marker in path:
        model_part = path.split(marker, 1)[1].split("/", 1)[0]
        model = model_part.split(":", 1)[0]
        if model:
            return model
    return f"passthrough:{endpoint_name}"


class PassthroughHandlerMixin:
    """Passthrough forwarding plus helpers shared with the Anthropic handler."""

    async def _count_tokens_offloaded(self, model, messages):  # noqa: ANN001, ANN201
        from headroom.proxy.token_counting import count_tokens_offloaded

        return await count_tokens_offloaded(self, model, messages)

    @staticmethod
    def _strict_previous_turn_frozen_count(
        messages: list[dict[str, Any]],
        base_frozen_count: int,
    ) -> int:
        """Freeze all prior turns in cache mode; only the final OBSERVATION turn
        is mutable (the newest delta we compress-once-then-freeze).

        The newest observation may arrive as ``role:"user"`` (Claude Code,
        text harnesses) or as ``role:"tool"`` / ``role:"function"``
        (function-calling harnesses). Treat any of those as the mutable tail;
        assistant/system endings freeze everything (they are not observations).
        """
        if not messages:
            return base_frozen_count
        final_idx = len(messages) - 1
        if messages[final_idx].get("role") in ("user", "tool", "function"):
            return final_idx
        return len(messages)

    async def handle_passthrough(
        self,
        request: Request,
        base_url: str,
        endpoint_name: str | None = None,
        provider: str | None = None,
    ) -> Response:
        """Pass through request unchanged.

        Args:
            request: The incoming request
            base_url: The upstream API base URL
            endpoint_name: Optional name for stats tracking (e.g., "models", "count_tokens")
            provider: Optional provider name for stats (e.g., "anthropic")
        """
        from fastapi.responses import Response

        start_time = time.time()
        path = request.url.path
        if provider == "anthropic" and endpoint_name == "models" and path.startswith("/v1/models/"):
            from headroom.providers.anthropic import sanitize_anthropic_model_id

            raw_model_id = path[len("/v1/models/") :]
            clean_model_id = sanitize_anthropic_model_id(unquote(raw_model_id))
            if clean_model_id != unquote(raw_model_id):
                path = "/v1/models/" + quote(clean_model_id, safe="")
        url = f"{base_url.rstrip('/')}{path}"

        # Preserve query string parameters
        if request.url.query:
            url = f"{url}?{request.url.query}"

        headers = dict(request.headers.items())
        headers.pop("host", None)
        headers.pop("accept-encoding", None)
        client = classify_client(headers)
        tags = extract_tags(headers)
        # Strip internal x-headroom-* before forwarding upstream.
        from headroom.proxy.helpers import (
            _strip_internal_headers,
            log_outbound_headers,
            request_with_transient_retry,
        )

        _pre_strip_count_pt = sum(1 for k in headers if k.lower().startswith("x-headroom-"))
        headers = _strip_internal_headers(headers)
        log_outbound_headers(
            forwarder="passthrough",
            stripped_count=_pre_strip_count_pt,
            request_id=None,
        )

        from starlette.requests import ClientDisconnect

        try:
            body = await request.body()
        except ClientDisconnect:
            logger.debug("Client disconnected during body read for passthrough")
            return Response(status_code=204)

        try:
            # Retry once on a transient keep-alive close (httpx
            # RemoteProtocolError / "incomplete chunked read"): the upstream
            # closed a pooled connection httpx then reused. See GH #1112.
            response = await request_with_transient_retry(
                self.http_client,  # type: ignore[attr-defined]
                method=request.method,
                url=url,
                headers=headers,
                content=body,
            )
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            logger.warning(
                "Passthrough request failed before upstream response: %s %s -> %s: %s",
                request.method,
                path,
                url,
                e,
            )
            return Response(
                content=json.dumps(
                    {
                        "error": {
                            "type": "connection_error",
                            "message": f"Failed to connect to upstream API: {e}",
                        }
                    }
                ),
                status_code=502,
                media_type="application/json",
            )
        except httpx.RemoteProtocolError as e:
            logger.warning(
                "Passthrough upstream closed connection without a complete "
                "response after retry: %s %s -> %s: %s",
                request.method,
                path,
                url,
                e,
            )
            return Response(
                content=json.dumps(
                    {
                        "error": {
                            "type": "upstream_protocol_error",
                            "message": (
                                "Upstream closed the connection without sending "
                                "a complete response."
                            ),
                        }
                    }
                ),
                status_code=502,
                media_type="application/json",
            )

        # Remove compression headers since httpx already decompressed the response
        response_headers = sanitize_forwarded_response_headers(response.headers)
        response_content = response.content

        if provider == "anthropic" and endpoint_name == "models":
            from headroom.providers.anthropic import sanitize_anthropic_model_metadata

            try:
                payload = response.json()
                sanitized_payload = sanitize_anthropic_model_metadata(payload)
            except (TypeError, ValueError):
                sanitized_payload = None
            if sanitized_payload is not None and sanitized_payload != payload:
                response_content = json.dumps(
                    sanitized_payload,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
                response_headers["content-type"] = "application/json"

        # Passthrough request: forwarded upstream with no transforms. Still
        # recorded so dashboards see traffic on the passthrough endpoints.
        if endpoint_name and provider:
            latency_ms = (time.time() - start_time) * 1000
            request_id = await self._next_request_id()  # type: ignore[attr-defined]
            usage: dict[str, int] = {}
            if response.headers.get("content-type", "").lower().startswith("application/json"):
                try:
                    usage = _passthrough_usage_from_json(response.json())
                except (json.JSONDecodeError, ValueError, TypeError):
                    usage = {}
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cache_read_tokens = usage.get("cache_read_input_tokens", 0)
            cache_write_tokens = usage.get("cache_creation_input_tokens", 0)
            uncached_input_tokens = max(0, input_tokens - cache_read_tokens - cache_write_tokens)
            await self._record_request_outcome(  # type: ignore[attr-defined]
                RequestOutcome(
                    request_id=request_id,
                    provider=provider,
                    model=_passthrough_model_from_path(path, endpoint_name),
                    status_code=response.status_code,
                    original_tokens=input_tokens,
                    optimized_tokens=input_tokens,
                    output_tokens=output_tokens,
                    tokens_saved=0,
                    attempted_input_tokens=input_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    uncached_input_tokens=uncached_input_tokens,
                    total_latency_ms=latency_ms,
                    tags=tags,
                    client=client,
                )
            )

        return Response(
            content=response_content,
            status_code=response.status_code,
            headers=response_headers,
        )
