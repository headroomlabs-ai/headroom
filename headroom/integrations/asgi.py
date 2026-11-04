"""ASGI Middleware — add Headroom compression to any Python proxy.

Drop-in middleware for FastAPI, Starlette, LiteLLM proxy, or any ASGI app.
Intercepts LLM requests, compresses messages, forwards the smaller payload.

Local mode (compression runs in-process):

    from headroom.integrations.asgi import CompressionMiddleware
    app.add_middleware(CompressionMiddleware)

Cloud mode (managed CCR, TOIN, analytics via Headroom Cloud):

    app.add_middleware(CompressionMiddleware, api_key="hdr_xxx")

Usage with LiteLLM proxy:

    from litellm.proxy.proxy_server import app
    from headroom.integrations.asgi import CompressionMiddleware

    app.add_middleware(CompressionMiddleware)  # local
    # OR
    app.add_middleware(CompressionMiddleware, api_key="hdr_xxx")  # cloud

Cloud mode requires httpx: pip install httpx
"""

from __future__ import annotations

import json
import logging
from collections.abc import MutableMapping
from typing import Any

from starlette.types import ASGIApp, Receive, Scope, Send

# _DEFAULT_CLOUD_URL and guard_egress stay importable from this module.
from headroom.integrations._compress_backend import (  # noqa: F401
    _DEFAULT_CLOUD_URL,
    CompressBackend,
)
from headroom.offline import guard_egress  # noqa: F401

logger = logging.getLogger(__name__)

# Paths that contain LLM messages to compress
_LLM_PATHS = (
    "/v1/messages",  # Anthropic
    "/v1/chat/completions",  # OpenAI
    "/v1/responses",  # OpenAI Responses API
    "/chat/completions",  # LiteLLM (without /v1 prefix)
)


class CompressionMiddleware(CompressBackend):
    """ASGI middleware that compresses LLM request messages.

    Two modes:
    - Local (default): Compresses in-process using headroom.compress().
    - Cloud (api_key set): Calls Headroom Cloud API for managed compression
      with org-scoped CCR, TOIN learning, and analytics dashboards.

    Response headers include compression metrics:
    - x-headroom-tokens-before: original token count
    - x-headroom-tokens-after: compressed token count
    - x-headroom-tokens-saved: tokens removed
    - x-headroom-compressed: "true" if compression occurred
    """

    _log = logger

    def __init__(
        self,
        app: ASGIApp,
        min_tokens: int = 500,
        model_limit: int = 200000,
        hooks: Any = None,
        api_key: str | None = None,
        api_url: str | None = None,
    ) -> None:
        self.app = app
        self._min_tokens = min_tokens
        self._init_compress_backend(model_limit, hooks, api_key, api_url)

    async def aclose(self) -> None:
        """Close the underlying httpx.AsyncClient, if one was created."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "GET")

        # Only intercept POST to LLM endpoints
        if method != "POST" or not any(path.endswith(p) or path == p for p in _LLM_PATHS):
            await self.app(scope, receive, send)
            return

        # Buffer the request body
        body_chunks: list[bytes] = []

        async def buffering_receive() -> MutableMapping[str, Any]:
            message: MutableMapping[str, Any] = await receive()
            if message["type"] == "http.request":
                chunk = message.get("body", b"")
                if chunk:
                    body_chunks.append(chunk)
            return message

        # Read the full body
        while True:
            msg = await buffering_receive()
            if msg.get("type") == "http.request":
                if not msg.get("more_body", False):
                    break

        full_body = b"".join(body_chunks)

        # Parse and compress
        tokens_saved = 0
        tokens_before = 0
        tokens_after = 0
        try:
            body_json = json.loads(full_body)
            messages = body_json.get("messages", [])
            model = body_json.get("model", "")

            if messages:
                if self._api_key:
                    result = await self._cloud_compress(messages, model)
                else:
                    result = self._local_compress(messages, model)

                if result and result.get("tokens_saved", 0) > 0 and "messages" in result:
                    body_json["messages"] = result["messages"]
                    full_body = json.dumps(body_json).encode("utf-8")
                    tokens_saved = result["tokens_saved"]
                    tokens_before = result.get("tokens_before", 0)
                    tokens_after = result.get("tokens_after", 0)

                    logger.info(
                        "Headroom%s: %d→%d tokens (saved %d, %.0f%%)",
                        " Cloud" if self._api_key else "",
                        tokens_before,
                        tokens_after,
                        tokens_saved,
                        result.get("compression_ratio", 0) * 100,
                    )

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.debug("Headroom middleware: skipping non-JSON request: %s", e)

        # Create a new receive that returns the (possibly modified) body
        body_sent = False

        async def modified_receive() -> MutableMapping[str, Any]:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": full_body, "more_body": False}
            result: MutableMapping[str, Any] = await receive()
            return result

        # Wrap send to inject compression headers
        async def metrics_send(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start" and tokens_saved > 0:
                headers = list(message.get("headers", []))
                headers.append((b"x-headroom-compressed", b"true"))
                headers.append((b"x-headroom-tokens-before", str(tokens_before).encode()))
                headers.append((b"x-headroom-tokens-after", str(tokens_after).encode()))
                headers.append((b"x-headroom-tokens-saved", str(tokens_saved).encode()))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, modified_receive, metrics_send)
