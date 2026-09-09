"""Declarative route specifications for provider passthrough endpoints."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderPassthroughRoute:
    """A direct provider passthrough route handled by ``HeadroomProxy.handle_passthrough``."""

    method: str
    path: str
    provider_name: str
    sub_path: str


@dataclass(frozen=True, slots=True)
class ProviderHandlerRoute:
    """A route that delegates directly to a named proxy handler."""

    method: str
    path: str
    handler_name: str
    path_param: str | None = None


ANTHROPIC_PASSTHROUGH_ROUTES: tuple[ProviderPassthroughRoute, ...] = (
    ProviderPassthroughRoute("POST", "/v1/messages/count_tokens", "anthropic", "count_tokens"),
)


PROVIDER_PASSTHROUGH_ROUTES: tuple[ProviderPassthroughRoute, ...] = (
    *ANTHROPIC_PASSTHROUGH_ROUTES,
)


ANTHROPIC_HANDLER_ROUTES: tuple[ProviderHandlerRoute, ...] = (
    ProviderHandlerRoute("POST", "/v1/messages", "handle_anthropic_messages"),
)


PROVIDER_HANDLER_ROUTES: tuple[ProviderHandlerRoute, ...] = (*ANTHROPIC_HANDLER_ROUTES,)
