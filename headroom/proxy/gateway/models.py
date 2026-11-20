"""Immutable, principal-filtered catalog snapshots with explicit provenance."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from headroom.proxy.gateway.config import GatewayConfigSnapshot, Protocol, RouteConfig
from headroom.proxy.gateway.context import GatewayPrincipal


class Capability(str, Enum):
    GENERATE = "generate"
    STREAM = "stream"


@dataclass(frozen=True, slots=True)
class RouteCapabilities:
    protocols: tuple[Protocol, ...]
    operations: frozenset[Capability]


@dataclass(frozen=True, slots=True)
class ProviderModelMetadata:
    id: str
    operations: frozenset[str]
    features: frozenset[str]


@dataclass(frozen=True, slots=True)
class AccountAvailability:
    route_id: str
    account_ref: str
    authority: str
    source_state: Literal["available", "unavailable", "unknown"]
    entitlement: Literal["allowed", "denied", "unknown"]
    provenance: Literal["configured", "provider"]
    observed_at: float | None = None
    expires_at: float | None = None
    stale_until: float | None = None
    metadata_available: bool = False
    refresh_failed: bool = False
    features: frozenset[str] | None = None
    operations: frozenset[str] | None = None

    def state(self, now: float) -> str:
        if (
            self.source_state == "unavailable"
            or self.entitlement != "allowed"
            or not self.metadata_available
        ):
            return "unavailable"
        if self.provenance == "configured":
            return "configured"
        if self.expires_at is not None and now < self.expires_at and not self.refresh_failed:
            return "fresh"
        if self.stale_until is not None and now < self.stale_until:
            return "stale"
        return "unavailable"


@dataclass(frozen=True, slots=True)
class PublishedRoute:
    id: str
    route_id: str
    protocols: tuple[Protocol, ...]
    body_contract: str
    provenance: str
    state: str
    capabilities: tuple[tuple[str, str, tuple[str, ...]], ...]
    tariff_revision: str | None


@dataclass(frozen=True, slots=True)
class ModelRegistry:
    _routes: tuple[RouteConfig, ...]
    accounts: tuple[AccountAvailability, ...]
    revision: int
    generation: int
    captured_at: float
    published_at: float

    def __init__(
        self,
        snapshot: GatewayConfigSnapshot,
        *,
        accounts: tuple[AccountAvailability, ...] | None = None,
        revision: int = 1,
        generation: int = 1,
        now: float | None = None,
        published_at: float | None = None,
    ):
        object.__setattr__(self, "_routes", snapshot.routes)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "captured_at", time.time() if now is None else now)
        object.__setattr__(
            self, "published_at", self.captured_at if published_at is None else published_at
        )
        if accounts is None:
            credentials = {item.id: item for item in snapshot.credentials}
            accounts = tuple(
                AccountAvailability(
                    route.id,
                    account,
                    "",
                    ("available" if credentials[account].source.kind == "none" else "unknown")
                    if credentials[account].enabled
                    else "unavailable",
                    route.catalog.entitlements.get(account, "unknown"),
                    route.catalog.source,
                    metadata_available=route.catalog.source == "configured",
                )
                for route in snapshot.routes
                for account in route.credentials
            )
        object.__setattr__(self, "accounts", accounts)

    def eligible_accounts(
        self,
        route: RouteConfig,
        *,
        protocol: str | None = None,
        transport: str = "http-json",
        features: frozenset[str] = frozenset(),
    ) -> frozenset[str]:
        if not route.enabled:
            return frozenset()
        declaration = next(
            (
                declaration
                for p, transports in route.capabilities.items()
                for t, declaration in transports.items()
                if p == protocol and t == transport
            ),
            None,
        )
        if protocol and (declaration is None or not features <= set(declaration.features)):
            return frozenset()
        return frozenset(
            record.account_ref
            for record in self.accounts
            if record.route_id == route.id
            and record.state(self.captured_at) != "unavailable"
            and (record.features is None or features <= record.features)
            and (
                record.operations is None
                or ("stream" if transport in {"http-stream", "websocket"} else "generate")
                in record.operations
            )
        )

    def visible_routes(self, principal: GatewayPrincipal) -> tuple[PublishedRoute, ...]:
        result = []
        for route in sorted(self._routes, key=lambda item: item.public_model):
            if (
                route.id not in principal.routes
                or "models" not in principal.scopes
                or not self.eligible_accounts(route)
            ):
                continue
            declarations = []
            for protocol, transports in route.capabilities.items():
                for transport, declaration in transports.items():
                    accounts = self.eligible_accounts(
                        route, protocol=protocol, transport=transport, features=frozenset({"text"})
                    )
                    if not accounts:
                        continue
                    supported = set(declaration.features)
                    for record in self.accounts:
                        if (
                            record.route_id == route.id
                            and record.account_ref in accounts
                            and record.features is not None
                        ):
                            supported.intersection_update(record.features)
                    declarations.append((protocol, transport, tuple(sorted(supported))))
            capabilities = tuple(declarations)
            if not capabilities:
                continue
            states = {
                record.state(self.captured_at)
                for record in self.accounts
                if record.route_id == route.id
                and record.account_ref in self.eligible_accounts(route)
            }
            result.append(
                PublishedRoute(
                    route.public_model,
                    route.id,
                    tuple(
                        p for p in route.ingress_protocols if any(c[0] == p for c in capabilities)
                    ),
                    route.body_contract,
                    route.catalog.source,
                    "stale"
                    if states == {"stale"}
                    else ("configured" if route.catalog.source == "configured" else "fresh"),
                    capabilities,
                    route.pricing.revision if route.pricing else None,
                )
            )
        return tuple(result)

    def route_for_model(self, public_model: str) -> RouteConfig | None:
        return next(
            (
                r
                for r in self._routes
                if r.public_model == public_model and self.eligible_accounts(r)
            ),
            None,
        )

    def route_for_id(self, route_id: str) -> RouteConfig | None:
        return next(
            (r for r in self._routes if r.id == route_id and self.eligible_accounts(r)), None
        )


CatalogSnapshot = ModelRegistry
