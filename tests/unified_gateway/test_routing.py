from __future__ import annotations

from pathlib import Path

import pytest

from headroom.proxy.gateway.config import GatewayConfigSnapshot, SelectionConfig
from headroom.proxy.gateway.context import GatewayPrincipal
from headroom.proxy.gateway.errors import GatewayCredentialUnavailable
from headroom.proxy.gateway.resources import ResourceBinding
from headroom.proxy.gateway.routing import AccountRouter

EXAMPLE = (
    Path(__file__).parents[2]
    / "docs"
    / "proposals"
    / "unified-api-gateway"
    / "examples"
    / "gateway.api-keys.json"
)


def test_account_router_round_robins_only_route_credentials() -> None:
    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    route = snapshot.routes[0].model_copy(update={"credentials": ("openai-api", "openai-api-2")})
    router = AccountRouter(available_accounts={"openai-api", "openai-api-2", "anthropic-api"})
    principal = GatewayPrincipal(
        id="principal-a",
        scopes=frozenset({"inference"}),
        routes=frozenset({route.id}),
    )

    assert router.select(route, principal).account_ref == "openai-api"
    assert router.select(route, principal).account_ref == "openai-api-2"
    assert router.select(route, principal).account_ref == "openai-api"


def test_resource_binding_denies_cooled_account_without_migrating() -> None:
    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    route = snapshot.routes[0].model_copy(update={"credentials": ("openai-api", "openai-api-2")})
    router = AccountRouter(available_accounts={"openai-api", "openai-api-2"})
    router.cool_down("openai-api", quota_key=route.id, until=200.0)
    binding = ResourceBinding(
        provider_id="resp_1",
        principal_id="principal-a",
        route_id=route.id,
        account_ref="openai-api",
        adapter="openai-responses",
        expires_at=None,
    )
    principal = GatewayPrincipal(
        id="principal-a",
        scopes=frozenset({"inference"}),
        routes=frozenset({route.id}),
    )

    with pytest.raises(GatewayCredentialUnavailable):
        router.select(route, principal, resource_binding=binding, now=100.0)


def test_unbound_selection_skips_cooled_account() -> None:
    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    route = snapshot.routes[0].model_copy(update={"credentials": ("openai-api", "openai-api-2")})
    router = AccountRouter(available_accounts={"openai-api", "openai-api-2"})
    router.cool_down("openai-api", quota_key=route.id, until=200.0)
    principal = GatewayPrincipal(
        id="principal-a",
        scopes=frozenset({"inference"}),
        routes=frozenset({route.id}),
    )

    assert router.select(route, principal, now=100.0).account_ref == "openai-api-2"


def test_priority_weighted_and_round_robin_with_eligibility() -> None:
    route = (
        GatewayConfigSnapshot.load(EXAMPLE)
        .routes[0]
        .model_copy(update={"credentials": ("a", "b", "c")})
    )
    principal = GatewayPrincipal("p", frozenset({"inference"}), frozenset({route.id}))
    expected = {
        "round_robin": ["a", "b", "a", "b"],
        "priority": ["b"] * 4,
        "weighted": ["a", "b", "b", "a"],
    }
    for strategy, sequence in expected.items():
        selection = SelectionConfig(
            strategy=strategy, priority={"a": 1, "b": 3, "c": 9}, weight={"a": 1, "b": 2, "c": 9}
        )
        configured = route.model_copy(update={"selection": selection})
        router = AccountRouter(available_accounts={"a", "b", "c"})
        assert [
            router.select(
                configured, principal, eligible_accounts=frozenset({"a", "b"})
            ).account_ref
            for _ in range(4)
        ] == sequence


def test_cooldown_is_scoped_and_does_not_transfer_to_new_authority() -> None:
    route = GatewayConfigSnapshot.load(EXAMPLE).routes[0]
    principal = GatewayPrincipal("p", frozenset({"inference"}), frozenset({route.id}))
    router = AccountRouter(available_accounts=set(route.credentials))
    router.update_accounts(set(route.credentials), {route.credentials[0]: "old"})
    router.cool_down(route.credentials[0], quota_key="other-route", until=200)
    assert router.select(route, principal, now=100).account_ref == route.credentials[0]
    router.cool_down(route.credentials[0], quota_key=route.id, until=200)
    with pytest.raises(GatewayCredentialUnavailable):
        router.select(route, principal, now=100)
    router.update_accounts(set(route.credentials), {route.credentials[0]: "new"})
    assert router.select(route, principal, now=100).account_ref == route.credentials[0]


def test_router_rejects_nonequivalent_accounts_and_residency() -> None:
    snapshot = GatewayConfigSnapshot.load(EXAMPLE)
    a = snapshot.credentials[0].model_copy(
        update={"id": "a", "owner_group": "owner", "billing_group": "bill", "residency": "us"}
    )
    b = a.model_copy(update={"id": "b", "billing_group": "other"})
    route = snapshot.routes[0].model_copy(
        update={"credentials": ("a", "b"), "selection": SelectionConfig(required_residency="us")}
    )
    principal = GatewayPrincipal("p", frozenset({"inference"}), frozenset({route.id}))
    router = AccountRouter(available_accounts={"a", "b"})
    router.update_accounts({"a", "b"}, credentials={"a": a, "b": b})
    assert router.select(route, principal).account_ref == "a"
    with pytest.raises(GatewayCredentialUnavailable):
        router.select(route, principal, eligible_accounts=frozenset({"b"}))
    router.update_accounts({"a"}, credentials={"a": a.model_copy(update={"residency": "eu"})})
    with pytest.raises(GatewayCredentialUnavailable):
        router.select(route, principal)
