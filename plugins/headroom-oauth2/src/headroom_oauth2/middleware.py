"""ASGI middleware that injects a refreshed OAuth2 bearer on each upstream request.

Headroom's litellm backend forwards the request's `Authorization` bearer to the
upstream as the API key, so setting it here makes the minted token reach the
backend with no core changes.
"""

from __future__ import annotations

import asyncio
import json
import logging

from .provider import OAuth2Error

log = logging.getLogger("headroom_oauth2")


class OAuth2Middleware:
    """ASGI middleware that replaces the request Authorization with a minted bearer."""

    def __init__(self, app, provider):
        self.app = app
        self.provider = provider

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        # Hot path: a cached, still-valid token needs no thread hop. Only mint (blocking
        # urllib) off the event loop when the cache is empty/expired.
        token = self.provider.cached()
        if token is None:
            try:
                loop = asyncio.get_running_loop()
                token = await loop.run_in_executor(None, self.provider.token)
            except OAuth2Error as e:
                log.warning("oauth2: token mint failed: %s", e)
                await self._error(
                    send, 502, "upstream_auth_error", "could not obtain upstream credentials"
                )
                return
        headers = [(k, v) for (k, v) in scope.get("headers", []) if k.lower() != b"authorization"]
        headers.append((b"authorization", b"Bearer " + token.encode()))
        await self.app(dict(scope, headers=headers), receive, send)

    @staticmethod
    async def _error(send, status, etype, message):
        body = json.dumps({"type": "error", "error": {"type": etype, "message": message}}).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    (b"cache-control", b"no-store"),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
