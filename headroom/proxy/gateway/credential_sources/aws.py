"""AWS SDK credential-chain source for Bedrock request signing."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from headroom.proxy.gateway.config import AwsChainSource, CredentialConfig
from headroom.proxy.gateway.credentials import CredentialLease, SecretHandle
from headroom.proxy.gateway.errors import GatewayCredentialUnavailable

AwsResolver = Callable[[], Awaitable[tuple[Any, float | None]]]


class AwsChainCredentialSource:
    def __init__(self, config: CredentialConfig, *, resolve: AwsResolver | None = None) -> None:
        if not isinstance(config.source, AwsChainSource):
            raise TypeError("AWS chain source requires AwsChainSource configuration")
        self._config = config
        self._source = config.source
        self._resolve = resolve or self._resolve_default
        self._generation = 0

    async def acquire(self, *, now: float) -> CredentialLease:
        del now
        try:
            credentials, expires_at = await self._resolve()
        except Exception as exc:
            raise GatewayCredentialUnavailable(
                status_code=503,
                code="credential_unavailable",
                message="AWS workload identity is unavailable",
            ) from exc
        if credentials is None:
            raise GatewayCredentialUnavailable(
                status_code=503,
                code="credential_unavailable",
                message="AWS workload identity is unavailable",
            )
        self._generation += 1
        return CredentialLease(
            credential_id=self._config.id,
            provider=self._config.provider,
            account_ref=self._config.id,
            allowed_origins=self._config.allowed_origins,
            allowed_path_prefixes=self._config.allowed_path_prefixes,
            expires_at=expires_at,
            generation=self._generation,
            secret=SecretHandle(credentials),
            source_kind="aws-chain",
            region=self._source.region,
        )

    async def invalidate(self, lease: CredentialLease, reason: str) -> None:
        del lease, reason

    async def _resolve_default(self) -> tuple[Any, float | None]:
        return await asyncio.to_thread(self._resolve_default_sync)

    def _resolve_default_sync(self) -> tuple[Any, float | None]:
        import boto3  # type: ignore[import-not-found]

        session = boto3.Session(
            profile_name=self._source.profile,
            region_name=self._source.region,
        )
        credentials = session.get_credentials()
        if credentials is None:
            raise RuntimeError("AWS chain returned no credentials")
        frozen = credentials.get_frozen_credentials()
        expiry = getattr(credentials, "_expiry_time", None)
        expires_at = expiry.timestamp() if expiry is not None else None
        return frozen, expires_at
