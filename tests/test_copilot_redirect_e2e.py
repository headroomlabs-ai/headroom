"""End-to-end: a Copilot credential on a stock-target proxy reaches Copilot.

These drive the real application (``create_app`` + the dedicated chat, responses
and messages handlers) with a capturing transport, so a regression in the
handlers' own wiring — not just in the helper the handlers call — fails here:

- the request leaves for ``api.githubcopilot.com`` with the client's own token,
- the operator's gateway secrets (``*_extra_headers``) stay home,
- the ``Copilot-Session-Token`` header Auto depends on survives the hop,
- the Anthropic surface gets the same model-aware system relocation as a
  configured Copilot target.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("fastapi")

COPILOT_API = "https://api.githubcopilot.com"
COPILOT_TOKEN = (
    "tid=0123456789abcdef0123456789abcdef;exp=1893456000;sku=copilot_for_business_seat:9f8e7d6c"
)
GATEWAY_HEADERS = {"X-Gateway-Key": "operator-secret"}

_CHAT_OK = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
}
_RESPONSES_OK = {
    "id": "resp_test",
    "object": "response",
    "status": "completed",
    "model": "gpt-5.5",
    "output": [],
    "usage": {"input_tokens": 3, "output_tokens": 1},
}
_MESSAGES_OK = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 3, "output_tokens": 1},
}


class _CapturingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = b""
        async for chunk in request.stream:  # type: ignore[union-attr]
            body += chunk
        request.extensions["captured_body"] = body
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/chat/completions"):
            payload: dict[str, Any] = _CHAT_OK
        elif path.endswith("/responses"):
            payload = _RESPONSES_OK
        else:
            payload = _MESSAGES_OK
        return httpx.Response(200, json=payload)


@pytest.fixture
def stock_target_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A shared proxy with both targets on the stock hosts and no Copilot seed."""
    for var in (
        "OPENAI_TARGET_API_URL",
        "ANTHROPIC_TARGET_API_URL",
        "GITHUB_COPILOT_API_URL",
        "GITHUB_COPILOT_ENTERPRISE_URL",
        "GITHUB_COPILOT_ENTERPRISE_DOMAIN",
        "GITHUB_COPILOT_API_TOKEN",
        "GITHUB_COPILOT_REFRESH_OAUTH_TOKEN",
        "COPILOT_PROVIDER_BEARER_TOKEN",
        "HEADROOM_STRIP_INTERNAL_HEADERS",
    ):
        monkeypatch.delenv(var, raising=False)


def _app():
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        openai_extra_headers=dict(GATEWAY_HEADERS),
        anthropic_extra_headers=dict(GATEWAY_HEADERS),
    )
    app = create_app(config)
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        proxy = app.state.proxy
        assert proxy.OPENAI_API_URL.startswith("https://api.openai.com")
        assert proxy.ANTHROPIC_API_URL.startswith("https://api.anthropic.com")
        transport = _CapturingTransport()
        proxy.http_client = httpx.AsyncClient(transport=transport)
        proxy.http_client_h1 = httpx.AsyncClient(transport=transport)
        yield client, transport


def _copilot_headers() -> dict[str, str]:
    return {
        "content-type": "application/json",
        "authorization": f"Bearer {COPILOT_TOKEN}",
        "copilot-session-token": "sess_auto_abc",
        "x-initiator": "user",
        "user-agent": "GitHubCopilotCLI/1.0.82",
    }


def _sent(transport: _CapturingTransport) -> httpx.Request:
    assert len(transport.requests) == 1, [str(r.url) for r in transport.requests]
    return transport.requests[0]


def test_chat_completions_with_a_copilot_token_go_to_copilot(stock_target_env: None) -> None:
    for client, transport in _app():
        response = client.post(
            "/v1/chat/completions",
            headers=_copilot_headers(),
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200, response.text
        sent = _sent(transport)
        assert str(sent.url) == f"{COPILOT_API}/chat/completions"
        assert sent.headers["authorization"] == f"Bearer {COPILOT_TOKEN}"
        assert sent.headers["copilot-session-token"] == "sess_auto_abc"
        assert "x-gateway-key" not in sent.headers


def test_chat_completions_with_an_openai_key_stay_on_openai_with_the_gateway_header(
    stock_target_env: None,
) -> None:
    for client, transport in _app():
        response = client.post(
            "/v1/chat/completions",
            headers={
                "content-type": "application/json",
                "authorization": "Bearer sk-test-0000000000000000000000000000000000000000000",
            },
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200, response.text
        sent = _sent(transport)
        assert sent.url.host == "api.openai.com"
        assert sent.headers["x-gateway-key"] == "operator-secret"


def test_responses_with_a_copilot_token_go_to_copilot(stock_target_env: None) -> None:
    for client, transport in _app():
        response = client.post(
            "/v1/responses",
            headers=_copilot_headers(),
            json={"model": "gpt-5.5", "input": "hi"},
        )
        assert response.status_code == 200, response.text
        sent = _sent(transport)
        assert str(sent.url) == f"{COPILOT_API}/responses"
        assert sent.headers["authorization"] == f"Bearer {COPILOT_TOKEN}"
        assert sent.headers["copilot-session-token"] == "sess_auto_abc"
        assert "x-gateway-key" not in sent.headers


def test_responses_with_an_openai_key_stay_on_openai_with_the_gateway_header(
    stock_target_env: None,
) -> None:
    for client, transport in _app():
        client.post(
            "/v1/responses",
            headers={
                "content-type": "application/json",
                "authorization": "Bearer sk-test-0000000000000000000000000000000000000000000",
            },
            json={"model": "gpt-5.5", "input": "hi"},
        )
        sent = _sent(transport)
        assert sent.url.host == "api.openai.com"
        assert sent.headers["x-gateway-key"] == "operator-secret"


def test_messages_with_a_copilot_token_go_to_copilot_and_keep_anthropic_semantics(
    stock_target_env: None,
) -> None:
    """A mid-conversation system turn is legal for claude-sonnet-5; a redirected
    request must be judged against the model, not hoisted blindly as an
    unknown upstream would be."""
    for client, transport in _app():
        response = client.post(
            "/v1/messages",
            headers=_copilot_headers(),
            json={
                "model": "claude-sonnet-5",
                "max_tokens": 16,
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "system", "content": "stay terse"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "second"},
                ],
            },
        )
        assert response.status_code == 200, response.text
        sent = _sent(transport)
        assert str(sent.url) == f"{COPILOT_API}/v1/messages"
        assert sent.headers["authorization"] == f"Bearer {COPILOT_TOKEN}"
        assert "x-gateway-key" not in sent.headers
        body = json.loads(sent.extensions["captured_body"])
        assert [m["role"] for m in body["messages"]] == ["user", "system", "assistant", "user"]
        assert "system" not in body


def test_messages_with_an_anthropic_key_stay_on_anthropic_with_the_gateway_header(
    stock_target_env: None,
) -> None:
    for client, transport in _app():
        client.post(
            "/v1/messages",
            headers={
                "content-type": "application/json",
                "x-api-key": "sk-ant-test",
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-sonnet-5",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        sent = _sent(transport)
        assert sent.url.host == "api.anthropic.com"
        assert sent.headers["x-gateway-key"] == "operator-secret"
