from __future__ import annotations

import pytest

from headroom.proxy.gateway.credentials import CredentialLease, SecretHandle
from headroom.proxy.gateway.egress import EgressPolicy, build_managed_upstream_headers
from headroom.proxy.gateway.errors import GatewayEgressDenied


@pytest.mark.asyncio
async def test_offline_gateway_forwarding_client_construction_never_dials(monkeypatch):
    import socket
    from pathlib import Path

    from headroom.proxy.gateway.config import GatewayConfigSnapshot
    from headroom.proxy.gateway.transport import http_client

    monkeypatch.setenv("HEADROOM_OFFLINE", "1")

    def forbidden(*args, **kwargs):
        raise AssertionError("constructing a gateway client opened a connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    example = (
        Path(__file__).parents[2]
        / "docs/proposals/unified-api-gateway/examples/gateway.api-keys.json"
    )
    client = http_client(GatewayConfigSnapshot.load(example))
    await client.aclose()


@pytest.fixture
def lease() -> CredentialLease:
    return CredentialLease(
        credential_id="openai-api",
        provider="openai",
        account_ref="openai-api",
        allowed_origins=("https://api.openai.com:443",),
        allowed_path_prefixes=("/v1/",),
        expires_at=None,
        generation=1,
        secret=SecretHandle("secret"),
    )


def test_exact_origin_and_path_are_authorized(lease: CredentialLease) -> None:
    EgressPolicy().authorize(lease, "https://api.openai.com/v1/responses")


@pytest.mark.parametrize(
    "url",
    [
        "http://api.openai.com/v1/responses",
        "https://api.openai.com.evil.example/v1/responses",
        "https://api.openai.com:444/v1/responses",
        "https://api.openai.com/other",
        "https://user@api.openai.com/v1/responses",
    ],
)
def test_origin_path_and_userinfo_confusion_are_denied(lease: CredentialLease, url: str) -> None:
    with pytest.raises(GatewayEgressDenied):
        EgressPolicy().authorize(lease, url)


@pytest.mark.parametrize("address", ["127.0.0.1", "169.254.169.254", "::1", "fe80::1"])
def test_public_credentials_cannot_reach_private_or_metadata_addresses(
    lease: CredentialLease, address: str
) -> None:
    with pytest.raises(GatewayEgressDenied):
        EgressPolicy().authorize(
            lease,
            "https://api.openai.com/v1/responses",
            resolved_addresses=(address,),
        )


def test_client_auth_headers_are_removed_before_provider_credential_is_added(
    lease: CredentialLease,
) -> None:
    headers = build_managed_upstream_headers(
        {
            "Authorization": "Bearer client-secret",
            "X-Api-Key": "client-secret",
            "X-Goog-Api-Key": "client-secret",
            "Content-Type": "application/json",
        },
        lease,
        "https://api.openai.com/v1/responses",
    )

    assert headers == {
        "content-type": "application/json",
        "authorization": "Bearer secret",
    }
