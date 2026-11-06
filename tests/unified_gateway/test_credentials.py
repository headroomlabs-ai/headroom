from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.credential_sources.aws import AwsChainCredentialSource
from headroom.proxy.gateway.credential_sources.gcp import GcpAdcCredentialSource
from headroom.proxy.gateway.credentials import CredentialBroker
from headroom.proxy.gateway.errors import GatewayCredentialUnavailable
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app

PROPOSAL = Path(__file__).parents[2] / "docs" / "proposals" / "unified-api-gateway"
EXAMPLE = PROPOSAL / "examples" / "gateway.api-keys.json"


@pytest.mark.asyncio
async def test_environment_lease_is_opaque_and_bound_to_provider_audience() -> None:
    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    broker = CredentialBroker.from_snapshot(
        snapshot,
        environ={"OPENAI_API_KEY": "upstream-secret-sentinel"},
    )

    lease = await broker.acquire(snapshot.routes[0])

    assert lease.credential_id == "openai-api"
    assert lease.provider == "openai"
    assert lease.allowed_origins == ("https://api.openai.com:443",)
    assert "upstream-secret-sentinel" not in repr(lease)
    assert "upstream-secret-sentinel" not in json.dumps(lease.redacted_dict())
    assert lease.authorization_headers() == {"authorization": "Bearer upstream-secret-sentinel"}


@pytest.mark.asyncio
async def test_missing_environment_secret_is_specific_unavailable_error() -> None:
    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    broker = CredentialBroker.from_snapshot(snapshot, environ={})

    with pytest.raises(GatewayCredentialUnavailable) as exc_info:
        await broker.acquire(snapshot.routes[0])

    assert exc_info.value.code == "credential_unavailable"
    assert "OPENAI_API_KEY" not in exc_info.value.message


@pytest.mark.asyncio
async def test_gcp_source_binds_token_to_configured_project() -> None:
    snapshot = GatewayConfigSnapshot.load(PROPOSAL / "examples" / "gateway.cloud-identities.json")
    config = snapshot.credentials[0]

    async def resolve():
        return "gcp-token", 1234.0, "REPLACE_WITH_AUTHORIZED_PROJECT"

    lease = await GcpAdcCredentialSource(config, resolve=resolve).acquire(now=1000.0)

    assert lease.provider == "vertex"
    assert lease.expires_at == 1234.0
    assert lease.authorization_headers() == {"authorization": "Bearer gcp-token"}


@pytest.mark.asyncio
async def test_aws_source_keeps_sdk_credentials_opaque() -> None:
    snapshot = GatewayConfigSnapshot.load(PROPOSAL / "examples" / "gateway.cloud-identities.json")
    config = snapshot.credentials[1]
    frozen_credentials = object()

    async def resolve():
        return frozen_credentials, 1234.0

    lease = await AwsChainCredentialSource(config, resolve=resolve).acquire(now=1000.0)

    assert lease.provider == "bedrock"
    assert lease.expires_at == 1234.0
    assert "object at" not in repr(lease)


def test_gateway_app_owns_one_credential_broker(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    snapshot = GatewayConfigSnapshot.load(EXAMPLE)

    app = create_app(ProxyConfig(gateway=snapshot))

    assert isinstance(app.state.gateway_runtime.capture().broker, CredentialBroker)
