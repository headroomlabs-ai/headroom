from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from headroom.proxy import helpers
from headroom.proxy.gateway import dispatch
from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.resources import ResourceBinding
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [False, True], ids=["chunked", "declared"])
@pytest.mark.parametrize("path", ["/v1/responses", "/v1/responses/resp_owned/cancel"])
async def test_gateway_body_ceiling_precedes_dispatch(monkeypatch, declared, path):
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    monkeypatch.setattr(helpers, "MAX_REQUEST_BODY_SIZE", 64)
    example = (
        Path(__file__).parents[2]
        / "docs/proposals/unified-api-gateway/examples/gateway.api-keys.json"
    )
    snapshot = GatewayConfigSnapshot.load(example)
    principal = snapshot.client_auth.principals[0]
    principal = principal.model_copy(update={"scopes": (*principal.scopes, "resources")})
    snapshot = snapshot.model_copy(
        update={"client_auth": snapshot.client_auth.model_copy(update={"principals": (principal,)})}
    )
    app = create_app(ProxyConfig(gateway=snapshot))
    generation = app.state.gateway_runtime.capture()
    route = next(route for route in snapshot.routes if "openai-responses" in route.native_protocols)
    await app.state.gateway_runtime.resources.bind(
        ResourceBinding(
            provider_id="resp_owned",
            principal_id=principal.id,
            route_id=route.id,
            account_ref=route.credentials[0],
            adapter="openai-responses",
            expires_at=None,
            authority_fingerprint=generation.account_key(route.credentials[0]),
            target_fingerprint=generation.target_key(route.id),
        )
    )
    attempts = AsyncMock(side_effect=AssertionError("oversize body reached upstream dispatch"))
    monkeypatch.setattr(dispatch, "_run_attempts", attempts)
    headers = [(b"host", b"127.0.0.1:8787"), (b"authorization", b"Bearer client-secret")]
    if declared:
        headers.append((b"content-length", b"120"))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8787),
    }
    received = 0
    sent = []

    async def receive():
        nonlocal received
        received += 1
        if received <= 3:
            return {"type": "http.request", "body": b"x" * 40, "more_body": True}
        await asyncio.Event().wait()

    async def send(message):
        sent.append(message)

    await asyncio.wait_for(app(scope, receive, send), timeout=5)
    assert [message["status"] for message in sent if message["type"] == "http.response.start"] == [
        413
    ]
    assert b"request_too_large" in b"".join(message.get("body", b"") for message in sent)
    assert received == (0 if declared else 2)
    attempts.assert_not_awaited()
