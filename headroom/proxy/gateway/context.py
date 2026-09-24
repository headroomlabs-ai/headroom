"""Request-scoped gateway authorization state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from headroom.proxy.gateway.config import Protocol, RouteConfig

if TYPE_CHECKING:
    from headroom.proxy.gateway.execution import GatewayOperation
    from headroom.proxy.gateway.models import CatalogSnapshot
    from headroom.proxy.gateway.routing import AccountSelection
    from headroom.proxy.gateway.runtime import RuntimeGeneration


@dataclass(frozen=True, slots=True)
class GatewayPrincipal:
    id: str
    scopes: frozenset[str]
    routes: frozenset[str]


@dataclass(frozen=True, slots=True)
class GatewayRequestContext:
    principal: GatewayPrincipal
    route: RouteConfig
    ingress_protocol: Protocol
    request_id: str
    generation: RuntimeGeneration
    catalog: CatalogSnapshot
    account_selection: AccountSelection | None = None
    operation: GatewayOperation | None = None
    mutation_reasons: tuple[str, ...] = ()

    @property
    def snapshot_generation(self) -> int:
        return self.generation.number
