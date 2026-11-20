"""Fail-closed caller authentication and route authorization."""

from __future__ import annotations

import hmac
from collections.abc import Mapping

from starlette.datastructures import Headers

from headroom.proxy.gateway.config import GatewayConfigSnapshot, Protocol, RouteConfig
from headroom.proxy.gateway.context import GatewayPrincipal
from headroom.proxy.gateway.errors import GatewayAuthError, GatewayAuthorizationError
from headroom.proxy.gateway.models import CatalogSnapshot
from headroom.proxy.loopback_guard import is_loopback_host_header


class GatewayAuthenticator:
    """Authenticate configured principals without exposing which secret matched."""

    def __init__(self, snapshot: GatewayConfigSnapshot, environ: Mapping[str, str]) -> None:
        resolved: list[tuple[str, str, GatewayPrincipal]] = []
        for principal in snapshot.client_auth.principals:
            if not principal.enabled:
                continue
            env_name = principal.secret_ref.removeprefix("env:")
            secret = environ.get(env_name)
            if not secret:
                raise ValueError(f"client principal {principal.id} secret reference is unavailable")
            resolved.append(
                (
                    principal.id,
                    secret,
                    GatewayPrincipal(
                        id=principal.id,
                        scopes=frozenset(principal.scopes),
                        routes=frozenset(principal.routes),
                    ),
                )
            )
        self._principals = tuple(resolved)

    def authenticate(self, headers: Headers, *, query_string: bytes = b"") -> GatewayPrincipal:
        del query_string  # Query credentials are deliberately never parsed.
        supplied = _client_tokens(headers)
        if not supplied:
            raise GatewayAuthError(
                status_code=401,
                code="gateway_auth_required",
                message="Headroom gateway authentication required",
            )
        if len(set(supplied)) != 1:
            raise GatewayAuthError(
                status_code=401,
                code="gateway_auth_conflict",
                message="Conflicting gateway credentials",
            )

        candidate = supplied[0]
        matched: GatewayPrincipal | None = None
        # Compare every configured secret so match position does not shortcut the loop.
        for _principal_id, expected, principal in self._principals:
            if hmac.compare_digest(candidate, expected):
                matched = principal
        if matched is None:
            raise GatewayAuthError(
                status_code=401,
                code="gateway_auth_invalid",
                message="Invalid Headroom gateway credential",
            )
        return matched


class GatewayAuthorizer:
    """Resolve only principal-visible routes and protocols."""

    def __init__(
        self, snapshot: GatewayConfigSnapshot, catalog: CatalogSnapshot | None = None
    ) -> None:
        from headroom.proxy.gateway.models import ModelRegistry

        self.catalog = catalog if catalog is not None else ModelRegistry(snapshot)
        self._routes_by_model = {route.public_model: route for route in snapshot.routes}
        self._unpriced_blocked = frozenset(
            p.id
            for p in snapshot.client_auth.principals
            if snapshot.admission.unknown_cost_policy == "block"
            or p.admission.unknown_cost_policy == "block"
        )

    def authorize(
        self,
        principal: GatewayPrincipal,
        *,
        scope: str,
        protocol: Protocol,
        public_model: str,
        transport: str = "http-json",
        features: frozenset[str] = frozenset({"text"}),
        defer_cost_to_admission: bool = False,
    ) -> RouteConfig:
        if scope not in principal.scopes:
            raise GatewayAuthorizationError(
                status_code=403,
                code="gateway_scope_forbidden",
                message="Gateway scope is not granted",
            )
        route = self.catalog.route_for_model(public_model)
        if route is None or route.id not in principal.routes:
            raise GatewayAuthorizationError(
                status_code=404,
                code="gateway_model_unavailable",
                message="Model is unavailable",
            )
        if protocol not in route.ingress_protocols:
            raise GatewayAuthorizationError(
                status_code=400,
                code="gateway_protocol_unavailable",
                message="Model is unavailable for this protocol",
            )
        if not self.catalog.eligible_accounts(
            route, protocol=protocol, transport=transport, features=features
        ):
            raise GatewayAuthorizationError(
                status_code=400,
                code="gateway_unsupported_capability",
                message="Requested capability is unavailable",
            )
        # HTTP delegates qualified cost admission to its operation owner.
        # Other transports remain closed under strict budgets until integrated.
        if (
            scope == "inference"
            and principal.id in self._unpriced_blocked
            and not defer_cost_to_admission
        ):
            raise GatewayAuthorizationError(
                status_code=429,
                code="gateway_admission_unknown_cost",
                message="Generation cost bound is unavailable",
            )
        return route


def validate_gateway_browser_request(headers: Headers) -> None:
    """Reject browser and forwarding ambiguity for local gateway v1."""

    if not is_loopback_host_header(headers.get("host")):
        raise GatewayAuthError(
            status_code=400,
            code="gateway_host_invalid",
            message="Gateway Host must be loopback",
        )
    if headers.get("origin") is not None:
        raise GatewayAuthError(
            status_code=403,
            code="gateway_browser_origin_denied",
            message="Browser origins are not allowed",
        )
    if headers.get("forwarded") is not None or headers.get("x-forwarded-host") is not None:
        raise GatewayAuthError(
            status_code=400,
            code="gateway_forwarding_denied",
            message="Forwarded gateway requests are not allowed",
        )


def _client_tokens(headers: Headers) -> tuple[str, ...]:
    values: list[str] = []
    authorization = headers.get("authorization")
    if authorization:
        scheme, separator, token = authorization.partition(" ")
        if separator and scheme.casefold() == "bearer" and token:
            values.append(token)
    for name in ("x-api-key", "x-goog-api-key", "x-headroom-proxy-token"):
        value = headers.get(name)
        if value:
            values.append(value)
    return tuple(values)
