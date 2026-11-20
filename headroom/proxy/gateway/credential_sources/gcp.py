"""Google Application Default Credentials source."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from headroom.proxy.gateway.config import CredentialConfig, GcpAdcSource
from headroom.proxy.gateway.credentials import CredentialLease, SecretHandle
from headroom.proxy.gateway.errors import GatewayCredentialUnavailable

GcpResolver = Callable[[], Awaitable[tuple[str, float | None, str | None]]]


class GcpAdcCredentialSource:
    def __init__(self, config: CredentialConfig, *, resolve: GcpResolver | None = None) -> None:
        if not isinstance(config.source, GcpAdcSource):
            raise TypeError("GCP ADC source requires GcpAdcSource configuration")
        self._config = config
        self._source = config.source
        self._resolve = resolve or self._resolve_default
        self._generation = 0

    async def acquire(self, *, now: float) -> CredentialLease:
        del now
        try:
            token, expires_at, project = await self._resolve()
        except Exception as exc:
            raise GatewayCredentialUnavailable(
                status_code=503,
                code="credential_unavailable",
                message="Google workload identity is unavailable",
            ) from exc
        if not token or project != self._source.project:
            raise GatewayCredentialUnavailable(
                status_code=503,
                code="credential_unavailable",
                message="Google workload identity project is unavailable",
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
            secret=SecretHandle(token),
            source_kind="gcp-adc",
            project=self._source.project,
        )

    async def invalidate(self, lease: CredentialLease, reason: str) -> None:
        del lease, reason

    async def _resolve_default(self) -> tuple[str, float | None, str | None]:
        return await asyncio.to_thread(self._resolve_default_sync)

    @staticmethod
    def _resolve_default_sync() -> tuple[str, float | None, str | None]:
        import google.auth  # type: ignore[import-not-found]
        from google.auth.transport.requests import Request  # type: ignore[import-not-found]

        credentials, project = google.auth.default(
            scopes=("https://www.googleapis.com/auth/cloud-platform",)
        )
        if not getattr(credentials, "valid", False):
            credentials.refresh(Request())
        token = getattr(credentials, "token", None)
        if not isinstance(token, str) or not token:
            raise RuntimeError("ADC returned no access token")
        expiry: Any = getattr(credentials, "expiry", None)
        expires_at = expiry.timestamp() if isinstance(expiry, datetime) else None
        return token, expires_at, project
