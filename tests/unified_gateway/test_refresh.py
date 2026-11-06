from __future__ import annotations

import asyncio

import pytest

from headroom.proxy.gateway.credentials import CredentialBroker, CredentialLease, SecretHandle


class CountingSource:
    def __init__(self) -> None:
        self.calls = 0

    async def acquire(self, *, now: float) -> CredentialLease:
        self.calls += 1
        await asyncio.sleep(0)
        return CredentialLease(
            credential_id="credential-a",
            provider="openai",
            account_ref="account-a",
            allowed_origins=("https://api.openai.com:443",),
            allowed_path_prefixes=("/v1/",),
            expires_at=now + 3600,
            generation=self.calls,
            secret=SecretHandle("secret"),
        )

    async def invalidate(self, lease: CredentialLease, reason: str) -> None:
        del lease, reason


@pytest.mark.asyncio
async def test_concurrent_acquisition_is_single_flight(route_config) -> None:
    source = CountingSource()
    broker = CredentialBroker({"credential-a": source})
    route = route_config.model_copy(update={"credentials": ("credential-a",)})

    leases = await asyncio.gather(*(broker.acquire(route) for _ in range(20)))

    assert source.calls == 1
    assert {lease.generation for lease in leases} == {1}


@pytest.fixture
def route_config():
    from headroom.proxy.gateway.config import BillingConfig, RetryConfig, RouteConfig

    return RouteConfig(
        id="route-a",
        public_model="public",
        upstream_model="upstream",
        provider="openai",
        upstream_origin="https://api.openai.com:443",
        upstream_path_prefix="/v1/",
        credentials=("credential-a",),
        ingress_protocols=("openai-responses",),
        native_protocols=("openai-responses",),
        translation="disabled",
        body_contract="strict-native",
        private_network=False,
        retry=RetryConfig(max_attempts=1, ambiguous_commit="never", after_output="never"),
        billing=BillingConfig(allow_paid_fallback=False),
        capabilities={"openai-responses": {"http-json": {"features": ["text"]}}},
    )
