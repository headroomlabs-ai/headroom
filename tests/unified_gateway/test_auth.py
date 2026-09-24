from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.datastructures import Headers
from starlette.websockets import WebSocketDisconnect

from headroom.proxy.gateway.auth import GatewayAuthenticator
from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.errors import GatewayAuthError
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app

EXAMPLE = (
    Path(__file__).parents[2]
    / "docs"
    / "proposals"
    / "unified-api-gateway"
    / "examples"
    / "gateway.api-keys.json"
)


@pytest.fixture
def authenticator() -> GatewayAuthenticator:
    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    return GatewayAuthenticator(snapshot, {"HEADROOM_GATEWAY_CLIENT_TOKEN": "client-secret"})


@pytest.mark.parametrize("headers", [Headers(), Headers({"authorization": "Bearer wrong"})])
def test_missing_or_wrong_client_token_is_rejected(
    authenticator: GatewayAuthenticator, headers: Headers
) -> None:
    with pytest.raises(GatewayAuthError) as exc_info:
        authenticator.authenticate(headers)

    assert exc_info.value.status_code == 401
    assert exc_info.value.code in {"gateway_auth_required", "gateway_auth_invalid"}


def test_normal_protocol_auth_slots_accept_the_headroom_client_token(
    authenticator: GatewayAuthenticator,
) -> None:
    bearer = authenticator.authenticate(Headers({"authorization": "Bearer client-secret"}))
    api_key = authenticator.authenticate(Headers({"x-api-key": "client-secret"}))

    assert bearer.id == "local-app"
    assert api_key.id == "local-app"


def test_conflicting_protocol_credentials_are_rejected(authenticator: GatewayAuthenticator) -> None:
    with pytest.raises(GatewayAuthError) as exc_info:
        authenticator.authenticate(
            Headers({"authorization": "Bearer client-secret", "x-api-key": "different"})
        )

    assert exc_info.value.code == "gateway_auth_conflict"


def test_query_string_credentials_are_never_authentication_input(
    authenticator: GatewayAuthenticator,
) -> None:
    with pytest.raises(GatewayAuthError, match="required"):
        authenticator.authenticate(Headers(), query_string=b"key=client-secret")


def test_real_gateway_app_requires_auth_even_on_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    app = create_app(ProxyConfig(gateway=snapshot))

    response = TestClient(app).get("/v1/models", headers={"host": "127.0.0.1:8787"})

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "gateway_auth_required"


def test_real_gateway_websocket_requires_auth_even_on_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    app = create_app(ProxyConfig(gateway=snapshot))

    with pytest.raises(WebSocketDisconnect) as exc_info:
        with TestClient(app).websocket_connect("/v1/responses", headers={"host": "127.0.0.1:8787"}):
            pass

    assert exc_info.value.code == 1008
