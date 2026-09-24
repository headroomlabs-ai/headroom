"""Bounded metadata-only ownership registry for stateful provider resources."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from headroom.proxy.gateway.errors import GatewayAuthorizationError


@dataclass(frozen=True, slots=True)
class ResourceBinding:
    provider_id: str
    principal_id: str
    route_id: str
    account_ref: str
    adapter: str
    expires_at: float | None
    authority_fingerprint: str | None = None
    target_fingerprint: str | None = None
    generation: int | None = None


class ResourceRegistry:
    """Keep only ownership metadata; request and response content never enters it."""

    def __init__(self, *, max_entries: int = 10_000) -> None:
        if max_entries < 1:
            raise ValueError("resource registry capacity must be positive")
        self._max_entries = max_entries
        self._bindings: dict[str, ResourceBinding] = {}
        self._lock = asyncio.Lock()

    async def clear(self) -> None:
        """Release all gateway-owned metadata during shutdown."""
        async with self._lock:
            self._bindings.clear()

    async def bind(self, binding: ResourceBinding) -> None:
        async with self._lock:
            self._expire_locked(time.time())
            existing = self._bindings.get(binding.provider_id)
            if existing is not None and existing != binding:
                raise GatewayAuthorizationError(
                    status_code=409,
                    code="gateway_resource_conflict",
                    message="Stateful resource ownership conflict",
                )
            if existing is None and len(self._bindings) >= self._max_entries:
                raise GatewayAuthorizationError(
                    status_code=503,
                    code="gateway_resource_capacity",
                    message="Stateful resource registry capacity reached",
                )
            self._bindings[binding.provider_id] = binding

    async def authorize(
        self,
        provider_id: str,
        *,
        principal_id: str,
        route_id: str | None,
        now: float,
        allowed_route_ids: frozenset[str] | None = None,
    ) -> ResourceBinding:
        async with self._lock:
            binding = self._bindings.get(provider_id)
            if binding is not None and binding.expires_at is not None and binding.expires_at <= now:
                self._bindings.pop(provider_id, None)
                binding = None
            if (
                binding is None
                or binding.principal_id != principal_id
                or (route_id is not None and binding.route_id != route_id)
                or (allowed_route_ids is not None and binding.route_id not in allowed_route_ids)
            ):
                raise GatewayAuthorizationError(
                    status_code=404,
                    code="gateway_resource_not_found",
                    message="Stateful resource not found",
                )
            return binding

    async def delete(self, provider_id: str, *, principal_id: str, route_id: str) -> bool:
        async with self._lock:
            binding = self._bindings.get(provider_id)
            if (
                binding is None
                or binding.principal_id != principal_id
                or binding.route_id != route_id
            ):
                return False
            self._bindings.pop(provider_id, None)
            return True

    async def expire(self, *, now: float) -> int:
        async with self._lock:
            return self._expire_locked(now)

    def _expire_locked(self, now: float) -> int:
        expired = [
            provider_id
            for provider_id, binding in self._bindings.items()
            if binding.expires_at is not None and binding.expires_at <= now
        ]
        for provider_id in expired:
            self._bindings.pop(provider_id, None)
        return len(expired)
