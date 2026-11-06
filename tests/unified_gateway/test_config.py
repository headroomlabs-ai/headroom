from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from headroom.proxy.gateway.config import GatewayConfigSnapshot

PROPOSAL = Path(__file__).parents[2] / "docs" / "proposals" / "unified-api-gateway"


def test_admission_schema_and_runtime_agree():
    raw = _example()
    raw["admission"] = {
        "max_concurrency": 16,
        "budget_usd": "10.00",
        "unknown_cost_policy": "block",
    }
    raw["client_auth"]["principals"].append(
        {"id": "operator", "secret_ref": "env:OPERATOR_TOKEN", "scopes": ["admin"], "routes": []}
    )
    raw["credentials"][0]["enabled"] = False
    schema = GatewayConfigSnapshot.model_json_schema()
    jsonschema.validate(raw, schema)
    snapshot = GatewayConfigSnapshot.model_validate(raw)
    assert str(snapshot.admission.budget_usd) == "10.00"
    for money in ["-1", "NaN", "Infinity", 1.5]:
        invalid = json.loads(json.dumps(raw))
        invalid["admission"]["budget_usd"] = money
        with pytest.raises(ValidationError):
            GatewayConfigSnapshot.model_validate(invalid)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)
    for change in [{"unknown_cost_policy": "allow"}, {"max_concurrency": 0}, {"typo": 1}]:
        invalid = json.loads(json.dumps(raw))
        invalid["admission"].update(change)
        with pytest.raises(ValidationError):
            GatewayConfigSnapshot.model_validate(invalid)


def test_multiple_candidates_require_explicit_equivalent_authority():
    raw = _example()
    second = {
        **raw["credentials"][0],
        "id": "second-api",
        "source": {"kind": "env", "ref": "SECOND_API_KEY"},
    }
    raw["credentials"].append(second)
    route = raw["routes"][0]
    route["credentials"].append("second-api")
    route["catalog"]["entitlements"]["second-api"] = "allowed"
    with pytest.raises(ValidationError, match="equivalent"):
        GatewayConfigSnapshot.model_validate(raw)
    for credential in (raw["credentials"][0], second):
        credential.update(
            owner_group="same-owner", billing_group="same-billing", residency="region-a"
        )
    assert GatewayConfigSnapshot.model_validate(raw).routes[0].credentials == (
        "openai-api",
        "second-api",
    )


def test_check_config_new_policies_has_no_readers(monkeypatch):
    import socket

    from headroom.proxy.gateway.credentials import CredentialBroker

    def forbidden(*args, **kwargs):
        raise AssertionError("offline validation attempted a reader")

    raw = _example()
    raw["limits"] = {"request_deadline_seconds": 30}
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(CredentialBroker, "from_snapshot", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    assert GatewayConfigSnapshot.model_validate(raw).limits.request_deadline_seconds == 30


@pytest.mark.parametrize(
    "policy",
    [
        {"retry": {"max_attempts": 4, "ambiguous_commit": "never", "after_output": "never"}},
        {"capabilities": {"openai-chat": {"http-json": {"features": ["hosted_tools"]}}}},
        {"capabilities": {}},
    ],
)
def test_rejects_unimplemented_or_undeclared_capabilities_and_retry(policy):
    raw = _example()
    raw["routes"][0].update(policy)
    with pytest.raises(ValidationError):
        GatewayConfigSnapshot.model_validate(raw)


def test_non_native_ingress_requires_a_qualified_translation_pair() -> None:
    raw = _example()
    anthropic = raw["routes"][1]  # type: ignore[index]
    anthropic["ingress_protocols"] = ["openai-chat", "anthropic-messages"]

    with pytest.raises(ValidationError, match="translation"):
        GatewayConfigSnapshot.model_validate(raw)

    anthropic["translation"] = "qualified"
    snapshot = GatewayConfigSnapshot.model_validate(raw)
    assert snapshot.routes[1].translation == "qualified"


def _example() -> dict[str, object]:
    return json.loads((PROPOSAL / "examples" / "gateway.api-keys.json").read_text())


def test_loads_version_one_example_as_immutable_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(_example()), encoding="utf-8")

    snapshot = GatewayConfigSnapshot.load(path)

    assert snapshot.version == 1
    assert snapshot.runtime.profile == "gateway"
    assert snapshot.routes[0].ingress_protocols == ("openai-chat", "openai-responses")
    with pytest.raises(ValidationError):
        snapshot.runtime.port = 9999  # type: ignore[misc]


def test_rejects_unknown_security_setting(tmp_path: Path) -> None:
    raw = _example()
    raw["client_auth"]["allow_anonymous"] = True  # type: ignore[index]
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValidationError, match="allow_anonymous"):
        GatewayConfigSnapshot.load(path)


def test_rejects_duplicate_public_model_and_missing_credential_reference(tmp_path: Path) -> None:
    raw = _example()
    routes = raw["routes"]  # type: ignore[index]
    routes.append(
        {
            **routes[0],
            "id": "duplicate-route",
            "credentials": ["missing"],
            "catalog": {"entitlements": {"missing": "allowed"}},
        }
    )
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ValidationError, match="duplicate public_model|unknown credential"):
        GatewayConfigSnapshot.load(path)


def test_redacted_dict_contains_references_not_resolved_secret_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "provider-secret-sentinel")
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps(_example()), encoding="utf-8")

    rendered = json.dumps(GatewayConfigSnapshot.load(path).redacted_dict())

    assert "OPENAI_API_KEY" in rendered
    assert "provider-secret-sentinel" not in rendered
