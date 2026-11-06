from __future__ import annotations

from pathlib import Path

import pytest
from starlette.datastructures import Headers

from headroom.proxy.gateway.auth import GatewayAuthenticator, GatewayAuthorizer
from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.errors import GatewayAuthorizationError

EXAMPLE = (
    Path(__file__).parents[2]
    / "docs"
    / "proposals"
    / "unified-api-gateway"
    / "examples"
    / "gateway.api-keys.json"
)


@pytest.fixture
def auth_pair() -> tuple[GatewayAuthenticator, GatewayAuthorizer]:
    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    return (
        GatewayAuthenticator(snapshot, {"HEADROOM_GATEWAY_CLIENT_TOKEN": "client-secret"}),
        GatewayAuthorizer(snapshot),
    )


def test_principal_can_only_use_granted_route_and_protocol(auth_pair) -> None:
    authenticator, authorizer = auth_pair
    principal = authenticator.authenticate(Headers({"x-api-key": "client-secret"}))

    route = authorizer.authorize(
        principal,
        scope="inference",
        protocol="openai-responses",
        public_model="REPLACE_WITH_ENABLED_OPENAI_MODEL",
    )

    assert route.id == "openai-native"


def test_unknown_model_is_rejected_before_any_downstream_callback(auth_pair) -> None:
    authenticator, authorizer = auth_pair
    principal = authenticator.authenticate(Headers({"x-api-key": "client-secret"}))
    downstream_calls = 0

    with pytest.raises(GatewayAuthorizationError) as exc_info:
        authorizer.authorize(
            principal,
            scope="inference",
            protocol="openai-responses",
            public_model="not-granted",
        )

    assert exc_info.value.code == "gateway_model_unavailable"
    assert downstream_calls == 0


def test_protocol_mismatch_is_rejected(auth_pair) -> None:
    authenticator, authorizer = auth_pair
    principal = authenticator.authenticate(Headers({"x-api-key": "client-secret"}))

    with pytest.raises(GatewayAuthorizationError) as exc_info:
        authorizer.authorize(
            principal,
            scope="inference",
            protocol="anthropic-messages",
            public_model="REPLACE_WITH_ENABLED_OPENAI_MODEL",
        )

    assert exc_info.value.code == "gateway_protocol_unavailable"
