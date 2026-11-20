"""Runtime policy publication must not reset process ownership."""

import json
from pathlib import Path

import pytest
from starlette.datastructures import Headers

from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.runtime import GatewayRuntime

EXAMPLE = (
    Path(__file__).parents[2] / "docs/proposals/unified-api-gateway/examples/gateway.api-keys.json"
)


def policy():
    raw = json.loads(EXAMPLE.read_text())
    raw["client_auth"]["principals"].append(
        {"id": "operator", "secret_ref": "env:OPERATOR_TOKEN", "scopes": ["admin"], "routes": []}
    )
    return raw


def runtime(raw=None, **kwargs):
    return GatewayRuntime(
        GatewayConfigSnapshot.model_validate(raw or policy()),
        environ={
            "HEADROOM_GATEWAY_CLIENT_TOKEN": "client-secret",
            "OPERATOR_TOKEN": "admin-secret",
        },
        **kwargs,
    )


@pytest.mark.asyncio
async def test_reload_preserves_ownership_and_inflight_catalog(tmp_path):
    from headroom.proxy.gateway.resources import ResourceBinding

    service = runtime()
    before = service.capture()
    resources, router, admission, telemetry = (
        service.resources,
        service.router,
        service.admission,
        service.observability,
    )
    await resources.bind(
        ResourceBinding(
            "response-a", "local-app", "openai-native", "openai-api", "openai-responses", None
        )
    )
    raw = policy()
    raw["routes"][0]["public_model"] = "new-alias"
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(raw))
    assert (await service.reload(path)).applied
    assert service.capture().number == 2
    assert service.resources is resources and service.router is router
    assert service.admission is admission and service.observability is telemetry
    principal = before.authenticator.authenticate(
        Headers({"authorization": "Bearer client-secret"})
    )
    assert before.catalog.route_for_model("REPLACE_WITH_ENABLED_OPENAI_MODEL") is not None
    assert service.capture().catalog.route_for_model("new-alias") is not None
    assert (
        await resources.authorize(
            "response-a", principal_id=principal.id, route_id="openai-native", now=0
        )
    ).account_ref == "openai-api"


def test_disabled_and_denied_routes_do_not_enter_catalog():
    raw = policy()
    raw["credentials"][0]["enabled"] = False
    raw["routes"][1]["catalog"] = {"entitlements": {"anthropic-api": "denied"}}
    service = runtime(raw)
    generation = service.capture()
    principal = generation.authenticator.authenticate(
        Headers({"authorization": "Bearer client-secret"})
    )
    assert [route.id for route in generation.catalog.visible_routes(principal)] == [
        "REPLACE_WITH_ENABLED_GEMINI_MODEL"
    ]


def test_explicit_capability_rejects_before_acquisition():
    from headroom.proxy.gateway.errors import GatewayAuthorizationError

    service = runtime()
    generation = service.capture()
    principal = generation.authenticator.authenticate(
        Headers({"authorization": "Bearer client-secret"})
    )
    with pytest.raises(GatewayAuthorizationError, match="capability"):
        generation.authorizer.authorize(
            principal,
            scope="inference",
            protocol="openai-chat",
            public_model="REPLACE_WITH_ENABLED_OPENAI_MODEL",
            transport="http-json",
            features=frozenset({"hosted_tools"}),
        )


def test_finite_principal_policy_cannot_admit_an_unpriced_generation():
    from headroom.proxy.gateway.errors import GatewayAuthorizationError

    raw = policy()
    raw["client_auth"]["principals"][0]["admission"] = {
        "budget_usd": "1",
        "unknown_cost_policy": "block",
    }
    generation = runtime(raw).capture()
    principal = generation.authenticator.authenticate(
        Headers({"authorization": "Bearer client-secret"})
    )
    with pytest.raises(GatewayAuthorizationError, match="cost"):
        generation.authorizer.authorize(
            principal,
            scope="inference",
            protocol="openai-chat",
            public_model="REPLACE_WITH_ENABLED_OPENAI_MODEL",
        )


def test_authority_and_target_identity_change_without_transferring_resources():
    service = runtime()
    before = service.capture()
    raw = policy()
    raw["credentials"][0]["source"]["ref"] = "OTHER_ACCOUNT"
    raw["routes"][0]["upstream_model"] = "other-model"
    after = runtime(raw).capture()
    assert before.account_key("openai-api") != after.account_key("openai-api")
    assert before.target_key("openai-native") != after.target_key("openai-native")


@pytest.mark.asyncio
async def test_reused_account_id_does_not_inherit_old_authority_cooldown(tmp_path):
    service = runtime()
    service.router.cool_down("openai-api", quota_key="group-a", until=100)
    raw = policy()
    raw["credentials"][0]["source"]["ref"] = "OTHER_ACCOUNT"
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(raw))
    assert (await service.reload(path)).applied
    generation = service.capture()
    principal = generation.authenticator.authenticate(
        Headers({"authorization": "Bearer client-secret"})
    )
    assert (
        service.router.select(generation.snapshot.routes[0], principal, now=0).account_ref
        == "openai-api"
    )
