"""ASGI middleware that injects a refreshed OAuth2 bearer on each upstream request.

Headroom's litellm backend forwards the request's `Authorization` bearer to the
upstream as the API key, so setting it here makes the minted token reach the
backend with no core changes.

Requests the proxy answers itself are left untouched. Health probes, ``/stats``,
``/metrics``, the compress/retrieve/telemetry routes and any extension's own
``/ext/...`` routes never go upstream, so there is nothing to inject and no
reason to mint. Before this rule ``/health`` returned 502 whenever the IdP was
unreachable, and every management request cost a mint when the cache was cold.

Inbound authentication is not this layer's job: on headroom-ai >= 0.40 extension
middleware runs inside the ``HEADROOM_PROXY_TOKEN`` gate, so an unauthenticated
request never reaches it.
"""

from __future__ import annotations

import asyncio
import json
import logging

from headroom.proxy.project_policy import split_project_path

from .provider import OAuth2Error

log = logging.getLogger("headroom_oauth2")

# Exact paths the proxy answers itself and that must stay reachable when the
# IdP is down (orchestrator probes).
LOCAL_ROUTE_PATHS: frozenset[str] = frozenset(
    {"/health", "/healthz", "/livez", "/readyz", "/favicon.ico", "/subscription-window"}
)

# Prefixes of routes the proxy (or a co-installed extension) serves locally,
# never forwarding to a provider. Everything NOT matched here is treated as an
# upstream request, including the provider passthrough catch-all, so a new
# provider route needs no change here; a new *local* route does.
LOCAL_ROUTE_PREFIXES: tuple[str, ...] = (
    "/stats",  # /stats, /stats-history, /stats-lifetime
    "/metrics",
    "/quota",
    "/settings",
    "/dashboard",
    "/admin/",
    "/debug/",
    "/transformations/",
    "/v1/compress",  # incl. /v1/compress/response
    "/v1/usage",
    "/v1/retrieve",
    "/v1/telemetry",
    "/v1/toin",
    "/v1/feedback",
    "/ext/",  # convention for routes registered by other extensions
)


def is_local_route(path: str) -> bool:
    """True when ``path`` is served by the proxy itself and never goes upstream."""
    path = split_project_path(path or "/")[1]  # drop a /p/<project> base-URL prefix
    return path in LOCAL_ROUTE_PATHS or path.startswith(LOCAL_ROUTE_PREFIXES)


class OAuth2Middleware:
    """ASGI middleware that replaces the request Authorization with a minted bearer."""

    def __init__(self, app, provider):
        self.app = app
        self.provider = provider

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        if is_local_route(scope.get("path") or "/"):
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
