from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.proxy.gateway.config import GatewayConfigSnapshot, PrincipalConfig
from headroom.proxy.gateway.errors import GatewayAuthorizationError
from headroom.proxy.gateway.resources import ResourceBinding, ResourceRegistry
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app


@pytest.mark.asyncio
@pytest.mark.parametrize("reuse_id", [False, True])
async def test_bind_reclaims_expired_capacity_and_identity(monkeypatch, reuse_id):
    import time

    now = [10.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    registry = ResourceRegistry(max_entries=1)
    first = ResourceBinding("old", "a", "route", "account", "openai-responses", 11.0)
    await registry.bind(first)
    now[0] = 12.0
    replacement = ResourceBinding(
        "old" if reuse_id else "new", "b", "route", "account", "openai-responses", 20.0
    )
    await registry.bind(replacement)
    assert (
        await registry.authorize(
            replacement.provider_id, principal_id="b", route_id="route", now=12.0
        )
        == replacement
    )


EXAMPLE = (
    Path(__file__).parents[2]
    / "docs"
    / "proposals"
    / "unified-api-gateway"
    / "examples"
    / "gateway.api-keys.json"
)


@pytest.mark.asyncio
async def test_resource_binding_is_owned_by_principal_route_and_account() -> None:
    registry = ResourceRegistry(max_entries=4)
    binding = ResourceBinding(
        provider_id="resp_1",
        principal_id="principal-a",
        route_id="route-a",
        account_ref="account-a",
        adapter="openai-responses",
        expires_at=200.0,
    )

    await registry.bind(binding)

    assert (
        await registry.authorize(
            "resp_1", principal_id="principal-a", route_id="route-a", now=100.0
        )
        == binding
    )
    with pytest.raises(GatewayAuthorizationError, match="not found"):
        await registry.authorize(
            "resp_1", principal_id="principal-b", route_id="route-a", now=100.0
        )


@pytest.mark.asyncio
async def test_expired_or_deleted_binding_is_never_reused() -> None:
    registry = ResourceRegistry(max_entries=4)
    await registry.bind(
        ResourceBinding(
            provider_id="resp_1",
            principal_id="principal-a",
            route_id="route-a",
            account_ref="account-a",
            adapter="openai-responses",
            expires_at=10.0,
        )
    )

    assert await registry.expire(now=11.0) == 1
    with pytest.raises(GatewayAuthorizationError, match="not found"):
        await registry.authorize("resp_1", principal_id="principal-a", route_id="route-a", now=11.0)

    await registry.bind(
        ResourceBinding(
            provider_id="resp_2",
            principal_id="principal-a",
            route_id="route-a",
            account_ref="account-a",
            adapter="openai-responses",
            expires_at=None,
        )
    )
    assert await registry.delete("resp_2", principal_id="principal-a", route_id="route-a")
    with pytest.raises(GatewayAuthorizationError, match="not found"):
        await registry.authorize("resp_2", principal_id="principal-a", route_id="route-a", now=11.0)


@pytest.mark.asyncio
async def test_registry_capacity_fails_closed_without_evicting_live_owners() -> None:
    registry = ResourceRegistry(max_entries=1)
    first = ResourceBinding(
        provider_id="resp_1",
        principal_id="principal-a",
        route_id="route-a",
        account_ref="account-a",
        adapter="openai-responses",
        expires_at=None,
    )
    await registry.bind(first)

    with pytest.raises(GatewayAuthorizationError, match="capacity"):
        await registry.bind(
            ResourceBinding(
                provider_id="resp_2",
                principal_id="principal-a",
                route_id="route-a",
                account_ref="account-a",
                adapter="openai-responses",
                expires_at=None,
            )
        )

    assert (
        await registry.authorize("resp_1", principal_id="principal-a", route_id="route-a", now=0.0)
        == first
    )


def test_cross_principal_response_lookup_never_reaches_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-a")
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN_B", "client-b")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    upstream_requests: list[httpx.Request] = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        upstream_requests.append(request)
        if request.method == "POST":
            return httpx.Response(
                200,
                json={"id": "resp_owned", "object": "response", "output": []},
            )
        return httpx.Response(200, json={"id": "resp_owned", "object": "response"})

    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    principal_a = snapshot.client_auth.principals[0]
    principal_b = PrincipalConfig(
        id="local-app-b",
        secret_ref="env:HEADROOM_GATEWAY_CLIENT_TOKEN_B",
        scopes=principal_a.scopes,
        routes=principal_a.routes,
    )
    snapshot = snapshot.model_copy(
        update={
            "client_auth": snapshot.client_auth.model_copy(
                update={"principals": (principal_a, principal_b)}
            )
        }
    )
    app = create_app(ProxyConfig(gateway=snapshot))
    app.state.gateway_runtime.dependencies.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream)
    )
    client = TestClient(app)
    common = {"host": "127.0.0.1:8787"}

    created = client.post(
        "/v1/responses",
        headers={**common, "authorization": "Bearer client-a"},
        json={"model": "REPLACE_WITH_ENABLED_OPENAI_MODEL", "input": "hello"},
    )
    denied = client.get(
        "/v1/responses/resp_owned",
        headers={**common, "authorization": "Bearer client-b"},
    )
    denied_continuation = client.post(
        "/v1/responses",
        headers={**common, "authorization": "Bearer client-b"},
        json={
            "model": "REPLACE_WITH_ENABLED_OPENAI_MODEL",
            "previous_response_id": "resp_owned",
            "input": "continue",
        },
    )
    owner = client.get(
        "/v1/responses/resp_owned",
        headers={**common, "authorization": "Bearer client-a"},
    )

    assert created.status_code == 200
    assert denied.status_code == 404
    assert denied_continuation.status_code == 404
    assert owner.status_code == 200
    assert len(upstream_requests) == 2
    assert all(
        request.headers["authorization"] == "Bearer provider-secret"
        for request in upstream_requests
    )


def test_streamed_response_id_is_bound_before_later_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-a")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    requests: list[httpx.Request] = []

    async def upstream(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=(
                    b'data: {"type":"response.created","response":'
                    b'{"id":"resp_streamed","object":"response"}}\n\n'
                    b'data: {"type":"response.completed"}\n\n'
                ),
            )
        return httpx.Response(200, json={"id": "resp_streamed", "object": "response"})

    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.load(EXAMPLE)))
    app.state.gateway_runtime.dependencies.http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream)
    )
    client = TestClient(app)
    headers = {"host": "127.0.0.1:8787", "authorization": "Bearer client-a"}

    streamed = client.post(
        "/v1/responses",
        headers=headers,
        json={
            "model": "REPLACE_WITH_ENABLED_OPENAI_MODEL",
            "input": "hello",
            "stream": True,
        },
    )
    lookup = client.get("/v1/responses/resp_streamed", headers=headers)

    assert streamed.status_code == 200
    assert lookup.status_code == 200
    assert len(requests) == 2
