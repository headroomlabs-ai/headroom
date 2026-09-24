from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from headroom.proxy.gateway.config import GatewayConfigSnapshot
from headroom.proxy.models import ProxyConfig
from headroom.proxy.server import create_app

EXAMPLE = (
    Path(__file__).parents[2]
    / "docs"
    / "proposals"
    / "unified-api-gateway"
    / "examples"
    / "gateway.api-keys.json"
)


def test_admin_controls_are_scoped_and_reload_uses_startup_path(monkeypatch, tmp_path):
    import json

    raw = json.loads(EXAMPLE.read_text())
    raw["client_auth"]["principals"].append(
        {"id": "operator", "secret_ref": "env:OPERATOR_TOKEN", "scopes": ["admin"], "routes": []}
    )
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    monkeypatch.setenv("OPERATOR_TOKEN", "admin-secret")
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(raw))
    app = create_app(
        ProxyConfig(gateway=GatewayConfigSnapshot.model_validate(raw), gateway_config_path=path)
    )
    client = TestClient(app)
    headers = {"host": "127.0.0.1", "authorization": "Bearer admin-secret"}
    assert (
        client.get(
            "/admin/gateway/status", headers={**headers, "authorization": "Bearer client-secret"}
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/admin/gateway/reload", headers=headers, json={"path": "other.json"}
        ).status_code
        == 400
    )
    raw["routes"][0]["public_model"] = "updated-model"
    path.write_text(json.dumps(raw))
    assert client.post("/admin/gateway/reload", headers=headers).json()["applied"] is True
    assert client.get("/admin/gateway/status", headers=headers).json()["generation"] == 2
    assert (
        client.get(
            "/admin/gateway/status", headers={**headers, "origin": "https://hostile.example"}
        ).status_code
        == 403
    )


def test_metadata_single_and_protocol_views_share_catalog(monkeypatch):
    import json

    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    raw = json.loads(EXAMPLE.read_text())
    raw["routes"][0]["catalog"]["entitlements"]["openai-api"] = "denied"
    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.model_validate(raw)))
    client = TestClient(app)
    headers = {"host": "127.0.0.1", "authorization": "Bearer client-secret"}
    assert (
        client.get("/v1/models/REPLACE_WITH_ENABLED_OPENAI_MODEL", headers=headers).status_code
        == 404
    )
    result = client.get("/v1beta/models", headers=headers)
    assert result.status_code == 200
    assert [m["name"] for m in result.json()["models"]] == [
        "models/REPLACE_WITH_ENABLED_GEMINI_MODEL"
    ]


def test_gateway_readyz_is_local_and_non_secret(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret-sentinel")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    app = create_app(ProxyConfig(gateway=GatewayConfigSnapshot.load(EXAMPLE)))

    response = TestClient(app).get("/readyz", headers={"host": "127.0.0.1:8787"})

    assert response.status_code == 200
    assert response.json()["service"] == "headroom"
    assert response.json()["profile"] == "gateway"
    assert "sentinel" not in response.text


def test_gateway_disables_runtime_mutation_and_settings(monkeypatch) -> None:
    monkeypatch.setenv("HEADROOM_GATEWAY_CLIENT_TOKEN", "client-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secret")
    client = TestClient(create_app(ProxyConfig(gateway=GatewayConfigSnapshot.load(EXAMPLE))))
    headers = {"host": "127.0.0.1:8787", "authorization": "Bearer client-secret"}

    assert client.post("/admin/runtime-env", headers=headers, json={}).status_code == 404
    assert client.post("/settings", headers=headers, json={}).status_code == 404
