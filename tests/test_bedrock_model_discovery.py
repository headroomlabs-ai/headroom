"""Model discovery and short model names in Bedrock mode.

Claude Desktop in third-party gateway mode asks the gateway for its models
(``GET /v1/models``) with a placeholder key. Behind a Bedrock backend there is
no Anthropic account to ask, so the proxy answers from the Bedrock models it
discovered. Clients also send undated names ("claude-haiku-4-5") for models
Bedrock only lists dated.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")
pytest.importorskip("fastapi")

from headroom.backends.litellm import LiteLLMBackend  # noqa: E402

_PROFILES = {
    "claude-haiku-4-5-20251001": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-haiku-4-5-20250601": "bedrock/us.anthropic.claude-haiku-4-5-20250601-v1:0",
    "claude-sonnet-4-6": "bedrock/global.anthropic.claude-sonnet-4-6",
    "claude-opus-4-1-20250805": "bedrock/us.anthropic.claude-opus-4-1-20250805-v1:0",
}

_ANTHROPIC_HEADERS = {"x-api-key": "placeholder", "anthropic-version": "2023-06-01"}


def _bedrock_backend() -> LiteLLMBackend:
    with patch(
        "headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value=_PROFILES
    ):
        return LiteLLMBackend(provider="bedrock", region="us-east-1")


def test_undated_name_maps_to_the_newest_dated_bedrock_profile() -> None:
    backend = _bedrock_backend()

    assert (
        backend.map_model_id("claude-haiku-4-5")
        == "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0"
    )
    assert backend.map_model_id("claude-sonnet-4-6") == "bedrock/global.anthropic.claude-sonnet-4-6"


def test_alias_of_an_override_only_snapshot_uses_the_override(monkeypatch) -> None:
    """Discovery lists override keys, so an alias that resolves in the model
    lookup must reach the override at inference, not a made-up profile ID."""
    monkeypatch.setenv(
        "HEADROOM_BEDROCK_MODEL_MAP", "claude-opus-4-9-20260101=us.anthropic.app-profile-x"
    )
    backend = _bedrock_backend()

    assert "claude-opus-4-9-20260101" in (backend.bedrock_model_ids() or [])
    assert backend.map_model_id("claude-opus-4-9") == backend.map_model_id(
        "claude-opus-4-9-20260101"
    )
    assert "app-profile-x" in backend.map_model_id("claude-opus-4-9")


def test_a_version_number_is_not_mistaken_for_a_snapshot_date() -> None:
    # "claude-sonnet-4" must not silently become "claude-sonnet-4-6".
    assert "sonnet-4-6" not in _bedrock_backend().map_model_id("claude-sonnet-4")


def _client(backend: LiteLLMBackend | None):
    from fastapi.testclient import TestClient

    from headroom.proxy.loopback_guard import require_loopback
    from headroom.proxy.server import ProxyConfig, create_app

    app = create_app(ProxyConfig(cache_enabled=False, rate_limit_enabled=False))
    app.dependency_overrides[require_loopback] = lambda: None
    client = TestClient(app)
    client.__enter__()
    app.state.proxy.anthropic_backend = backend
    return client


def test_model_list_is_answered_from_bedrock_without_an_upstream_call() -> None:
    client = _client(_bedrock_backend())
    try:
        with patch.object(
            client.app.state.proxy, "handle_passthrough", side_effect=AssertionError("upstream")
        ):
            listed = client.get("/v1/models", headers=_ANTHROPIC_HEADERS)
            one = client.get("/v1/models/claude-sonnet-4-6", headers=_ANTHROPIC_HEADERS)
            alias = client.get("/v1/models/claude-haiku-4-5", headers=_ANTHROPIC_HEADERS)
            long_context = client.get(
                "/v1/models/claude-sonnet-4-6[1m]", headers=_ANTHROPIC_HEADERS
            )
            first = client.get("/v1/models?limit=2", headers=_ANTHROPIC_HEADERS).json()
            rest = client.get(
                f"/v1/models?limit=2&after_id={first['last_id']}", headers=_ANTHROPIC_HEADERS
            ).json()
            stale = client.get(
                "/v1/models?after_id=claude-gone-1", headers=_ANTHROPIC_HEADERS
            ).json()
            missing = client.get("/v1/models/claude-opus-9", headers=_ANTHROPIC_HEADERS)
    finally:
        client.__exit__(None, None, None)

    assert listed.status_code == 200
    assert [m["id"] for m in listed.json()["data"]] == sorted(_PROFILES)
    assert one.status_code == 200 and one.json()["id"] == "claude-sonnet-4-6"
    assert missing.status_code == 404
    assert alias.status_code == 200 and long_context.status_code == 200
    assert [m["id"] for m in first["data"]] == sorted(_PROFILES)[:2] and first["has_more"]
    assert [m["id"] for m in rest["data"]] == sorted(_PROFILES)[2:] and not rest["has_more"]
    assert stale["data"] == [] and not stale["has_more"]


def test_model_list_without_bedrock_still_passes_through() -> None:
    from fastapi.responses import Response

    client = _client(None)
    try:
        with patch.object(
            client.app.state.proxy,
            "handle_passthrough",
            return_value=Response(b'{"data":[]}', media_type="application/json"),
        ) as passthrough:
            resp = client.get("/v1/models", headers=_ANTHROPIC_HEADERS)
    finally:
        client.__exit__(None, None, None)

    assert resp.status_code == 200
    assert passthrough.called
