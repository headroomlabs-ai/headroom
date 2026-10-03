"""Copilot Enterprise model selection survives the VS Code proxy round trip."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom import copilot_auth
from headroom.copilot_auth import CopilotAPIToken
from headroom.proxy.server import ProxyConfig, create_app

ENTERPRISE_CAPI = "https://copilot-api.acme.ghe.com"
INTEGRATION_ID = "vscode-chat"
MODELS = ("claude-sonnet-4.5", "gpt-5.2")


class _EnterpriseTokenProvider:
    def __init__(self) -> None:
        self.minted_for: list[str | None] = []

    async def get_api_token(self, *, integration_id: str | None = None) -> CopilotAPIToken:
        self.minted_for.append(integration_id)
        return CopilotAPIToken(
            token=f"tid_minted_for_{integration_id}",
            expires_at=9_999_999_999.0,
            api_url=ENTERPRISE_CAPI,
        )


class _CapturingCopilotTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict[str, object], httpx.Headers]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(await request.aread())
        self.requests.append((str(request.url), payload, request.headers))
        model = str(payload["model"])
        if payload.get("stream"):
            chunk = {
                "id": f"chatcmpl-{len(self.requests)}",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            }
            content = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
            return httpx.Response(
                200,
                request=request,
                headers={"content-type": "text/event-stream"},
                content=content,
            )
        return httpx.Response(
            200,
            request=request,
            json={
                "id": f"chatcmpl-{len(self.requests)}",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )


@pytest.mark.parametrize(
    "path",
    [
        # VS Code builds this path from overrideCapiUrl.
        "/chat/completions",
        # Copilot clients using the standard OpenAI-compatible route.
        "/v1/chat/completions",
    ],
)
@pytest.mark.parametrize("stream", [False, True], ids=["buffered", "streaming"])
def test_enterprise_copilot_preserves_each_selected_model_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
    path: str,
    stream: bool,
) -> None:
    """Changing the picker changes the next wire request; no model is pinned.

    VS Code's proxy override uses HMAC mode and can send an unusable placeholder
    credential. Headroom must replace it with an integration-bound enterprise
    token without changing the model selected in the request body. Driving two
    model families through one live proxy instance also guards against a model
    value leaking through request or token caches.
    """

    monkeypatch.setenv("GITHUB_COPILOT_API_URL", ENTERPRISE_CAPI)
    provider = _EnterpriseTokenProvider()
    monkeypatch.setattr(copilot_auth, "get_copilot_token_provider", lambda: provider)

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
            log_requests=False,
            ccr_inject_tool=False,
            ccr_handle_responses=False,
            ccr_context_tracking=False,
            image_optimize=False,
        )
    )
    transport = _CapturingCopilotTransport()

    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        proxy = app.state.proxy
        proxy.OPENAI_API_URL = ENTERPRISE_CAPI
        proxy.http_client = httpx.AsyncClient(transport=transport)
        proxy.http_client_h1 = httpx.AsyncClient(transport=transport)

        for model in MODELS:
            response = client.post(
                path,
                headers={
                    "Authorization": "Bearer empty",
                    "Copilot-Integration-Id": INTEGRATION_ID,
                    "Content-Type": "application/json",
                    "User-Agent": "GitHubCopilotChat/0.40.1",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": f"use {model}"}],
                    "stream": stream,
                },
            )
            assert response.status_code == 200, response.text

    assert [request[1]["model"] for request in transport.requests] == list(MODELS)
    assert [request[1]["stream"] for request in transport.requests] == [stream, stream]
    assert all(
        request[0] == f"{ENTERPRISE_CAPI}/chat/completions" for request in transport.requests
    )
    assert provider.minted_for == [INTEGRATION_ID, INTEGRATION_ID]
    assert all(
        request[2]["authorization"] == f"Bearer tid_minted_for_{INTEGRATION_ID}"
        for request in transport.requests
    )
    assert all(
        request[2]["copilot-integration-id"] == INTEGRATION_ID for request in transport.requests
    )
