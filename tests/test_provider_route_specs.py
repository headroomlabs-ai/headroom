from __future__ import annotations

from headroom.providers.route_specs import (
    ANTHROPIC_HANDLER_ROUTES,
    ANTHROPIC_PASSTHROUGH_ROUTES,
    PROVIDER_HANDLER_ROUTES,
    PROVIDER_PASSTHROUGH_ROUTES,
    ProviderHandlerRoute,
    ProviderPassthroughRoute,
)


def test_provider_passthrough_route_specs_are_unique() -> None:
    route_keys = {(spec.method, spec.path) for spec in PROVIDER_PASSTHROUGH_ROUTES}

    assert len(route_keys) == len(PROVIDER_PASSTHROUGH_ROUTES)


def test_provider_handler_route_specs_are_unique() -> None:
    route_keys = {(spec.method, spec.path) for spec in PROVIDER_HANDLER_ROUTES}

    assert len(route_keys) == len(PROVIDER_HANDLER_ROUTES)


def test_anthropic_passthrough_routes_model_endpoint_intent() -> None:
    assert ANTHROPIC_PASSTHROUGH_ROUTES == (
        ProviderPassthroughRoute(
            "POST",
            "/v1/messages/count_tokens",
            "anthropic",
            "count_tokens",
        ),
    )


def test_direct_handler_routes_model_endpoint_intent() -> None:
    assert ANTHROPIC_HANDLER_ROUTES == (
        ProviderHandlerRoute("POST", "/v1/messages", "handle_anthropic_messages"),
    )


def test_only_anthropic_routes_are_registered() -> None:
    assert PROVIDER_PASSTHROUGH_ROUTES == ANTHROPIC_PASSTHROUGH_ROUTES
    assert PROVIDER_HANDLER_ROUTES == ANTHROPIC_HANDLER_ROUTES
