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
            missing = client.get("/v1/models/claude-opus-9", headers=_ANTHROPIC_HEADERS)
    finally:
        client.__exit__(None, None, None)

    assert listed.status_code == 200
    assert [m["id"] for m in listed.json()["data"]] == sorted(_PROFILES)
    assert one.status_code == 200 and one.json()["id"] == "claude-sonnet-4-6"
    assert missing.status_code == 404


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
