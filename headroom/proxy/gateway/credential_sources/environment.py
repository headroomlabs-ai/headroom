"""Explicit environment-reference credential source."""

from __future__ import annotations

from collections.abc import Mapping

from headroom.proxy.gateway.config import CredentialConfig, EnvironmentSource
from headroom.proxy.gateway.credentials import CredentialLease, SecretHandle
from headroom.proxy.gateway.errors import GatewayCredentialUnavailable


class EnvironmentCredentialSource:
    """Resolve exactly one configured environment variable at lease time."""

    def __init__(self, config: CredentialConfig, environ: Mapping[str, str]) -> None:
        if not isinstance(config.source, EnvironmentSource):
            raise TypeError("environment source requires EnvironmentSource configuration")
        self._config = config
        self._source = config.source
        self._environment = environ

    async def acquire(self, *, now: float) -> CredentialLease:
        del now
        value = self._environment.get(self._source.ref)
        if not value:
            raise GatewayCredentialUnavailable(
                status_code=503,
                code="credential_unavailable",
                message="Configured provider credential is unavailable",
            )
        return CredentialLease(
            credential_id=self._config.id,
            provider=self._config.provider,
            account_ref=self._config.id,
            allowed_origins=self._config.allowed_origins,
            allowed_path_prefixes=self._config.allowed_path_prefixes,
            expires_at=None,
            generation=1,
            secret=SecretHandle(value),
        )

    async def invalidate(self, lease: CredentialLease, reason: str) -> None:
        del lease, reason
