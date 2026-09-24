"""Provider catalog evidence, completeness, and native metadata contracts."""

from collections.abc import Mapping

import httpx
import pytest
from anthropic.types import ModelInfo
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.runtime import GatewayRuntime, RuntimeDependencies
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app
from tests.unified_gateway.test_runtime_policy import policy


def provider_runtime(provider, payloads):
    raw = policy()
    route = next(r for r in raw["routes"] if r["provider"] == provider)
    raw["routes"] = [route]
    raw["credentials"] = [c for c in raw["credentials"] if c["id"] in route["credentials"]]
    raw["client_auth"]["principals"][0]["routes"] = [route["id"]]
    route["catalog"].update(source="provider", ttl_seconds=10, stale_if_error_seconds=5)
    now, requests = [100.0], []

    async def upstream(request):
        requests.append(request)
        return httpx.Response(200, json=payloads.pop(0))

    client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    runtime = GatewayRuntime(
        GatewayConfigSnapshot.model_validate(raw),
        environ={
            "HEADROOM_GATEWAY_CLIENT_TOKEN": "client-secret",
            "OPERATOR_TOKEN": "admin-secret",
            "GEMINI_API_KEY": "gemini-secret",
            "ANTHROPIC_API_KEY": "anthropic-secret",
        },
        dependencies=RuntimeDependencies(clock=lambda: now[0], http_client=client),
    )
    return runtime, client, now, requests, route


@pytest.mark.asyncio
@pytest.mark.parametrize("methods", [["embedContent"], [], None])
async def test_provider_generation_requires_positive_operation_evidence(methods):
    model = {"name": "models/REPLACE_WITH_ENABLED_GEMINI_MODEL"}
    if methods is not None:
        model["supportedGenerationMethods"] = methods
    runtime, client, _, requests, _ = provider_runtime("gemini", [{"models": [model]}])
    try:
        assert (await runtime.refresh_catalog())["refreshed"] == 1
        generation = runtime.capture()
        principal = generation.authenticator.authenticate(
            Headers({"authorization": "Bearer client-secret"})
        )
        assert generation.catalog.visible_routes(principal) == ()
        assert generation.catalog.route_for_model("REPLACE_WITH_ENABLED_GEMINI_MODEL") is None
        assert len(requests) == 1
    finally:
        await runtime.shutdown()
        await client.aclose()


@pytest.mark.asyncio
async def test_provider_features_intersect_configured_capabilities():
    from headroom.proxy.gateway.errors import GatewayAuthorizationError

    runtime, client, _, _, route = provider_runtime(
        "gemini",
        [
            {
                "models": [
                    {
                        "name": "models/REPLACE_WITH_ENABLED_GEMINI_MODEL",
                        "supportedGenerationMethods": ["generateContent"],
                        "supportedFeatures": ["text"],
                    }
                ]
            }
        ],
    )
    try:
        await runtime.refresh_catalog()
        generation = runtime.capture()
        principal = generation.authenticator.authenticate(
            Headers({"authorization": "Bearer client-secret"})
        )
        published = generation.catalog.visible_routes(principal)
        assert len(published) == 1
        assert published[0].capabilities == (("gemini-generate", "http-json", ("text",)),)
        assert generation.catalog.accounts[0].features == frozenset({"text"})
        for transport, features in [
            ("http-json", frozenset({"tools"})),
            ("http-stream", frozenset({"text"})),
        ]:
            with pytest.raises(GatewayAuthorizationError):
                generation.authorizer.authorize(
                    principal,
                    scope="inference",
                    protocol="gemini-generate",
                    public_model=route["public_model"],
                    transport=transport,
                    features=features,
                )
    finally:
        await runtime.shutdown()
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["gemini", "anthropic"])
async def test_incomplete_provider_page_preserves_prior_stale_snapshot(provider):
    if provider == "gemini":
        complete = {
            "models": [
                {
                    "name": "models/REPLACE_WITH_ENABLED_GEMINI_MODEL",
                    "supportedGenerationMethods": ["generateContent"],
                }
            ]
        }
        incomplete = {"models": [], "nextPageToken": "next-page"}
    else:
        complete = {
            "data": [{"id": "REPLACE_WITH_ENABLED_ANTHROPIC_MODEL", "type": "model"}],
            "has_more": False,
        }
        incomplete = {"data": [], "has_more": True, "last_id": "other-model"}
    runtime, client, now, requests, route = provider_runtime(provider, [complete, incomplete])
    try:
        assert (await runtime.refresh_catalog())["refreshed"] == 1
        assert runtime.capture().catalog.route_for_model(route["public_model"]) is not None
        now[0] = 111
        result = await runtime.refresh_catalog()
        assert result["failed"] == 1 and result["refreshed"] == 0
        record = runtime.capture().catalog.accounts[0]
        assert record.metadata_available and record.state(111) == "stale"
        assert record.observed_at == 100
        now[0] = 116
        assert runtime.capture().catalog.route_for_model(route["public_model"]) is None
        assert len(requests) == 2
    finally:
        await runtime.shutdown()
        await client.aclose()


def test_anthropic_catalog_models_and_errors_parse_as_native_protocol(monkeypatch):
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    monkeypatch.setenv("OPERATOR_TOKEN", "admin-secret")
    client = TestClient(
        create_app(ProxyConfig(gateway=GatewayConfigSnapshot.model_validate(policy())))
    )
    headers = {"host": "127.0.0.1", "x-api-key": "client-secret", "anthropic-version": "2023-06-01"}
    response = client.get("/anthropic/v1/models", headers=headers)
    assert response.status_code == 200
    listing = response.json()
    models = [ModelInfo.model_validate(item) for item in listing["data"]]
    assert [m.id for m in models] == ["REPLACE_WITH_ENABLED_ANTHROPIC_MODEL"]
    assert listing["has_more"] is False
    assert listing["first_id"] == listing["last_id"] == models[0].id
    one = client.get("/anthropic/v1/models/" + models[0].id, headers=headers)
    assert ModelInfo.model_validate(one.json()).id == models[0].id
    # Ordinary Anthropic SDKs use /v1/models with the protocol headers.
    ordinary = client.get("/v1/models", headers=headers)
    assert [ModelInfo.model_validate(item).id for item in ordinary.json()["data"]] == [models[0].id]
    unknown = client.get("/anthropic/v1/models/unknown", headers=headers)
    assert unknown.status_code == 404
    assert unknown.json()["type"] == "error"
    assert unknown.json()["error"]["type"] == "not_found_error"
    assert unknown.json()["error"]["message"]


@pytest.mark.parametrize("present, expected", [(False, "unknown"), (True, "available")])
def test_environment_source_status_uses_presence_without_reading_value(present, expected):
    class PresenceOnly(Mapping):
        def __getitem__(self, key):
            if key == "HEADROOM_GATEWAY_CLIENT_TOKEN":
                return "client-secret"
            if key == "OPERATOR_TOKEN":
                return "admin-secret"
            raise AssertionError("metadata status read a provider secret")

        def __iter__(self):
            return iter(
                [
                    "HEADROOM_GATEWAY_CLIENT_TOKEN",
                    "OPERATOR_TOKEN",
                    *(["OPENAI_API_KEY"] if present else []),
                ]
            )

        def __len__(self):
            return 3 if present else 2

        def __contains__(self, key):
            return key in set(iter(self))

    runtime = GatewayRuntime(GatewayConfigSnapshot.model_validate(policy()), environ=PresenceOnly())
    record = next(r for r in runtime.capture().catalog.accounts if r.account_ref == "openai-api")
    assert record.source_state == expected
