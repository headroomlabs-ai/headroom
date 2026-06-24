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
import os
from collections.abc import MutableMapping
from typing import Any

from starlette.types import ASGIApp, Receive, Scope, Send

logger = logging.getLogger(__name__)

_DEFAULT_CLOUD_URL = "https://api.headroomlabs.ai"

# Paths that contain LLM messages to compress
_LLM_PATHS = (
    "/v1/messages",  # Anthropic
    "/v1/chat/completions",  # OpenAI
    "/v1/responses",  # OpenAI Responses API
    "/chat/completions",  # LiteLLM (without /v1 prefix)
)


class CompressionMiddleware:
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
        self._model_limit = model_limit
        self._hooks = hooks

        # Cloud mode: if api_key is set, compress via Headroom Cloud API
        self._api_key = api_key or os.environ.get("HEADROOM_API_KEY", "").strip() or None
        self._api_url = (
            api_url or os.environ.get("HEADROOM_API_URL", "").strip() or _DEFAULT_CLOUD_URL
        ).rstrip("/")
        self._client: Any = None  # Lazy-initialized httpx.AsyncClient

    @property
    def cloud_mode(self) -> bool:
        """Whether cloud compression is enabled."""
        return self._api_key is not None

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

    def _local_compress(self, messages: list[dict], model: str) -> dict[str, Any] | None:
        """Compress locally using headroom.compress()."""
        from headroom.compress import compress

        result = compress(
            messages=messages,
            model=model or "claude-sonnet-4-5-20250929",
            model_limit=self._model_limit,
            hooks=self._hooks,
        )
        return {
            "messages": result.messages,
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
            "tokens_saved": result.tokens_saved,
            "compression_ratio": result.compression_ratio,
        }

    async def _cloud_compress(self, messages: list[dict], model: str) -> dict[str, Any] | None:
        """Compress via Headroom Cloud API (managed CCR, TOIN, analytics)."""
        if self._client is None:
            try:
                import httpx
            except ImportError as e:
                raise ImportError(
                    "httpx is required for Headroom Cloud mode: pip install httpx"
                ) from e
            self._client = httpx.AsyncClient(timeout=30.0)

        client = self._client
        assert client is not None
        resp = await client.post(
            f"{self._api_url}/v1/saas/compress",
            headers={
                "X-Headroom-Key": self._api_key,
                "Content-Type": "application/json",
            },
            content=json.dumps(
                {
                    "messages": messages,
                    "model": model or "claude-sonnet-4-5-20250929",
                    "model_limit": self._model_limit,
                }
            ),
        )

        if resp.status_code != 200:
            logger.warning("Headroom Cloud API error: %d %s", resp.status_code, resp.text[:200])
            return None

        result: dict[str, Any] = resp.json()
        return result
