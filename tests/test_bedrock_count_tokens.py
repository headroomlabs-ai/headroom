"""``POST /v1/messages/count_tokens`` in Bedrock mode.

Bedrock-mode clients (Claude Code with ``ANTHROPIC_BASE_URL`` on the proxy,
Claude Desktop in gateway mode) send a placeholder key, so passing the count
through to Anthropic 401s. The proxy asks Bedrock's CountTokens API instead.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")
pytest.importorskip("fastapi")
botocore_exceptions = pytest.importorskip("botocore.exceptions")

from headroom.backends.litellm import LiteLLMBackend  # noqa: E402

_PROFILES = {
    "claude-haiku-4-5-20251001": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4-6": "bedrock/global.anthropic.claude-sonnet-4-6",
}
_HEADERS = {"x-api-key": "placeholder", "anthropic-version": "2023-06-01"}
_BODY = {
    "model": "claude-sonnet-4-6[1m]",
    "system": "You are terse.",
    "messages": [{"role": "user", "content": "hello"}],
    "tools": [{"name": "read", "input_schema": {"type": "object"}}],
    "thinking": {"type": "enabled", "budget_tokens": 1024, "display": "updates"},
    "context_management": {"edits": []},
}


def _bedrock_backend(runtime: MagicMock) -> LiteLLMBackend:
    with patch(
        "headroom.backends.litellm._fetch_bedrock_inference_profiles", return_value=_PROFILES
    ):
        backend = LiteLLMBackend(provider="bedrock", region="us-east-1")
    backend._bedrock_runtime = runtime
    return backend


def _post(backend, passthrough):
    from fastapi.testclient import TestClient

    from headroom.proxy.loopback_guard import require_loopback
    from headroom.proxy.server import ProxyConfig, create_app

    app = create_app(ProxyConfig(cache_enabled=False, rate_limit_enabled=False))
    app.dependency_overrides[require_loopback] = lambda: None
    with TestClient(app) as client:
        app.state.proxy.anthropic_backend = backend
        with patch.object(app.state.proxy, "handle_passthrough", passthrough):
            return client.post("/v1/messages/count_tokens", headers=_HEADERS, json=_BODY)


def test_bedrock_counts_tokens_without_an_upstream_call() -> None:
    runtime = MagicMock()
    runtime.count_tokens.return_value = {"inputTokens": 42}

    resp = _post(_bedrock_backend(runtime), MagicMock(side_effect=AssertionError("upstream")))

    assert resp.status_code == 200
    assert resp.json() == {"input_tokens": 42}
    call = runtime.count_tokens.call_args.kwargs
    # CountTokens rejects inference-profile IDs; it takes the base model ID.
    assert call["modelId"] == "anthropic.claude-sonnet-4-6"
    sent = json.loads(call["input"]["invokeModel"]["body"])
    assert sent["anthropic_version"] == "bedrock-2023-05-31"
    assert sent["max_tokens"] == 1  # required by Bedrock, absent from count requests
    assert sent["system"] == _BODY["system"] and sent["tools"] == _BODY["tools"]
    assert "model" not in sent and "context_management" not in sent
    assert "display" not in sent["thinking"]
    assert isinstance(call["input"]["invokeModel"]["body"], bytes)  # boto base64-encodes it


def test_without_bedrock_count_tokens_still_passes_through() -> None:
    from fastapi.responses import Response

    passthrough = MagicMock(
        return_value=Response(b'{"input_tokens": 7}', media_type="application/json")
    )

    async def forward(*args, **kwargs):
        return passthrough(*args, **kwargs)

    resp = _post(None, forward)

    assert resp.status_code == 200
    assert resp.json() == {"input_tokens": 7}
    assert passthrough.called


@pytest.mark.parametrize(
    ("code", "status", "error_type"),
    [
        ("AccessDeniedException", 403, "permission_error"),
        ("ValidationException", 400, "invalid_request_error"),
    ],
)
def test_bedrock_error_maps_to_an_anthropic_error(code, status, error_type) -> None:
    runtime = MagicMock()
    runtime.count_tokens.side_effect = botocore_exceptions.ClientError(
        {
            "Error": {"Code": code, "Message": "nope"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "CountTokens",
    )

    resp = _post(_bedrock_backend(runtime), MagicMock(side_effect=AssertionError("upstream")))

    assert resp.status_code == status
    error = resp.json()["error"]
    assert error["type"] == error_type
    assert code in error["message"] and "anthropic.claude-sonnet-4-6" in error["message"]
