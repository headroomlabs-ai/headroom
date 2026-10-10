from __future__ import annotations

import httpx
import pytest
from starlette.requests import Request
from starlette.responses import StreamingResponse

import headroom.proxy.handlers.openai as openai_mod
from headroom import paths
from headroom.proxy.server import HeadroomProxy, ProxyConfig


def test_openai_chat_routes_copilot_requests_per_model() -> None:

    copilot_base = "https://api.githubcopilot.com"
    gpt54_mini_url = openai_mod.build_copilot_upstream_url(
        copilot_base,
        openai_mod._resolve_openai_handler_path(
            {},
            handler_path=openai_mod._resolve_openai_chat_handler_path(copilot_base, "gpt-5.4-mini"),
        ),
    )
    claude_url = openai_mod.build_copilot_upstream_url(
        copilot_base,
        openai_mod._resolve_openai_handler_path(
            {},
            handler_path=openai_mod._resolve_openai_chat_handler_path(
                copilot_base, "claude-sonnet-5"
            ),
        ),
    )
    openai_url = openai_mod.build_copilot_upstream_url(
        "https://api.openai.com",
        openai_mod._resolve_openai_handler_path(
            {},
            handler_path=openai_mod._resolve_openai_chat_handler_path(
                "https://api.openai.com", "gpt-5.4-mini"
            ),
        ),
    )

    assert gpt54_mini_url == "https://api.githubcopilot.com/responses"
    assert claude_url == "https://api.githubcopilot.com/chat/completions"
    assert openai_url == "https://api.openai.com/v1/chat/completions"


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["passthrough", "streaming"])
@pytest.mark.parametrize(
    ("path", "expected_auth"),
    [
        (
            "/v1/engines/gpt-41-copilot/completions",
            "Bearer ghu_editor_seat_fixture",
        ),
        ("/chat/completions", "Bearer tid_operator_seat_fixture"),
    ],
    ids=["native-editor-seat", "chat-operator-seat"],
)
async def test_real_forwarders_enforce_credential_and_secret_boundaries(
    monkeypatch: pytest.MonkeyPatch, entrypoint: str, path: str, expected_auth: str
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_TOKEN", "tid_operator_seat_fixture")
    monkeypatch.setenv("GITHUB_COPILOT_PROXY_URL", "https://gateway.example.invalid/v1")
    monkeypatch.delenv("GITHUB_COPILOT_REFRESH_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(paths, "_PROCESS_STATELESS", paths._PROCESS_STATELESS)
    for name in (
        "ANTHROPIC_API_URL",
        "OPENAI_API_URL",
        "GEMINI_API_URL",
        "CLOUDCODE_API_URL",
        "VERTEX_API_URL",
    ):
        monkeypatch.setattr(HeadroomProxy, name, getattr(HeadroomProxy, name))
    proxy = HeadroomProxy(
        ProxyConfig(
            stateless=True,
            optimize=False,
            disable_kompress=True,
            discover_pipeline_extensions=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
    )
    content = b'data: {"choices":[{"text":"accepted"}]}\n\ndata: [DONE]\n\n'

    def credential_gate(request: httpx.Request) -> httpx.Response:
        admitted = (
            request.headers.get("authorization") == expected_auth
            and request.headers.get("copilot-integration-id") == "vscode"
            and "x-api-key" not in request.headers
        )
        return httpx.Response(
            200 if admitted else 401,
            content=content if admitted else b'{"error":"wrong credential or leaked secret"}',
            headers={"content-type": "text/event-stream" if admitted else "application/json"},
        )

    headers = {
        "authorization": "Bearer ghu_editor_seat_fixture",
        "copilot-integration-id": "vscode",
        "x-api-key": "unrelated-provider-secret",
    }
    base = (
        "https://proxy.business.githubcopilot.com"
        if path.startswith("/v1/engines/")
        else "https://api.githubcopilot.com"
    )
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(credential_gate)) as client:
            proxy.http_client = client
            if entrypoint == "passthrough":

                async def receive() -> dict[str, object]:
                    return {"type": "http.request", "body": b"{}", "more_body": False}

                request = Request(
                    {
                        "type": "http",
                        "method": "POST",
                        "scheme": "http",
                        "server": ("proxy.test", 80),
                        "path": path,
                        "query_string": b"",
                        "headers": [(k.encode(), v.encode()) for k, v in headers.items()],
                    },
                    receive,
                )
                response = await proxy.handle_passthrough(request, base)
            else:
                response = await proxy._stream_response(
                    url=base + path,
                    headers=headers,
                    body={"model": "gpt-41-copilot"},
                    provider="openai",
                    model="gpt-41-copilot",
                    request_id="inline-auth-regression",
                    original_tokens=0,
                    optimized_tokens=0,
                    tokens_saved=0,
                    transforms_applied=[],
                    tags={},
                    optimization_latency=0.0,
                )

            assert response.status_code == 200
            if isinstance(response, StreamingResponse):
                received = b"".join([chunk async for chunk in response.body_iterator])
                if response.background is not None:
                    await response.background()
            else:
                received = response.body
            assert received == content
    finally:
        proxy._compression_executor.shutdown()
        proxy._background_compression_executor.shutdown()
