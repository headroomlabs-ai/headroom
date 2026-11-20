from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.gateway.models import ProviderModelMetadata
from headroom.proxy.gateway.runtime import RuntimeDependencies
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app
from tests.unified_gateway.test_runtime_policy import policy, runtime

EXAMPLE = (
    Path(__file__).parents[2]
    / "docs"
    / "proposals"
    / "unified-api-gateway"
    / "examples"
    / "gateway.api-keys.json"
)


def test_alias_collision_rejected():
    raw = policy()
    raw["routes"][1]["public_model"] = raw["routes"][0]["public_model"]
    with pytest.raises(ValidationError, match="duplicate public_model"):
        GatewayConfigSnapshot.model_validate(raw)


@pytest.mark.asyncio
async def test_refresh_singleflight_and_obsolete_publish_rejected(tmp_path):
    raw = policy()
    raw["routes"][0]["catalog"]["source"] = "provider"
    raw["routes"][1]["catalog"]["source"] = "provider"
    started, release = asyncio.Event(), asyncio.Event()
    probes = []

    async def metadata(generation, route, account):
        probes.append((route.id, account))
        if route.id == "anthropic-native":
            started.set()
            await release.wait()
        return (
            ProviderModelMetadata(
                route.upstream_model, frozenset({"generate"}), frozenset({"text"})
            ),
        )

    service = runtime(raw, dependencies=RuntimeDependencies(metadata_reader=metadata))
    revision = service.capture().catalog.revision
    pending = [asyncio.create_task(service.refresh_catalog()) for _ in range(8)]
    await started.wait()
    await asyncio.sleep(0)
    assert service.capture().catalog.revision == revision, "partial refresh must not publish"
    changed = policy()
    changed["routes"][0]["public_model"] = "replacement"
    path = tmp_path / "new.json"
    path.write_text(json.dumps(changed))
    assert (await service.reload(path)).applied
    release.set()
    await asyncio.gather(*pending)
    assert probes == [("openai-native", "openai-api"), ("anthropic-native", "anthropic-api")]
    assert service.capture().catalog.route_for_model("replacement") is not None
    assert service.capture().number == 2
    await service.shutdown()


def test_captured_policy_is_deeply_immutable():
    generation = runtime().capture()
    with pytest.raises(TypeError):
        generation.snapshot.routes[0].catalog.entitlements["openai-api"] = "denied"
    with pytest.raises(TypeError):
        generation.snapshot.routes[0].capabilities["openai-chat"].clear()


def test_model_catalog_contains_only_principal_granted_routes(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.load(EXAMPLE)))

    response = TestClient(app).get(
        "/v1/models",
        headers={"host": "127.0.0.1:8787", "authorization": "Bearer client-secret"},
    )

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == [
        "REPLACE_WITH_ENABLED_ANTHROPIC_MODEL",
        "REPLACE_WITH_ENABLED_GEMINI_MODEL",
        "REPLACE_WITH_ENABLED_OPENAI_MODEL",
    ]
