"""Requests from a browser page the proxy does not trust never reach the upstream.

The proxy adds operator-held credentials (``*_extra_headers``, the Copilot
token, provider env keys) to what it forwards, and loopback callers skip
``HEADROOM_PROXY_TOKEN``. A page open in the user's browser reaches the proxy as
a loopback caller and can send a ``text/plain`` JSON body without a CORS
preflight, so the HTTP routes apply the same Origin rule as the WebSocket
routes. Clients that send no ``Origin`` (CLI tools, SDKs, coding agents) and
loopback pages (the dashboard) are unaffected.
"""

from __future__ import annotations

import json

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

OPERATOR_SECRET = "operator-held-secret"
FOREIGN_ORIGIN = "https://page.example"


@pytest.fixture(autouse=True)
def _default_origin_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_WS_ORIGINS", raising=False)
    monkeypatch.delenv("HEADROOM_CORS_ORIGINS", raising=False)
    monkeypatch.delenv("HEADROOM_PROXY_TOKEN", raising=False)


class _Capturing(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async for _ in request.stream:
            pass
        self.requests.append(request)
        if request.url.path.endswith("/chat/completions"):
            body = {
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }
        else:
            body = {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 5, "output_tokens": 1},
            }
        return httpx.Response(200, json=body)


def _client() -> tuple[TestClient, _Capturing]:
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
        anthropic_extra_headers={"x-api-key": OPERATOR_SECRET},
        openai_extra_headers={"Authorization": f"Bearer {OPERATOR_SECRET}"},
    )
    app = create_app(config)
    transport = _Capturing()
    app.state.proxy.http_client = httpx.AsyncClient(transport=transport)
    # The browser and the proxy share the host, so the peer is loopback.
    return TestClient(app, client=("127.0.0.1", 51000)), transport


_ROUTES = {
    "/v1/messages": {
        "model": "claude-sonnet-4-6",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    },
    "/v1/chat/completions": {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
    },
}


def _simple_post(client: TestClient, path: str, headers: dict[str, str]) -> httpx.Response:
    # A browser "simple" request: text/plain body, no custom headers, no preflight.
    return client.post(
        path,
        content=json.dumps(_ROUTES[path]),
        headers={"Content-Type": "text/plain", **headers},
    )


@pytest.mark.parametrize("path", sorted(_ROUTES))
@pytest.mark.parametrize(
    "browser_headers",
    [
        {"Origin": FOREIGN_ORIGIN},
        {"Origin": "null"},
        {"Origin": "http://localhost.page.example"},
    ],
)
def test_foreign_page_request_is_refused_before_the_upstream(
    path: str, browser_headers: dict[str, str]
) -> None:
    client, transport = _client()

    resp = _simple_post(client, path, browser_headers)

    assert resp.status_code == 403
    assert transport.requests == []


@pytest.mark.parametrize("path", sorted(_ROUTES))
def test_request_without_origin_still_forwards(path: str) -> None:
    client, transport = _client()

    resp = _simple_post(client, path, {})

    assert resp.status_code == 200, resp.text
    assert len(transport.requests) == 1
    assert any(OPERATOR_SECRET in value for value in transport.requests[0].headers.values())


@pytest.mark.parametrize("origin", ["http://127.0.0.1:8787", "http://localhost:3000"])
def test_loopback_page_still_forwards(origin: str) -> None:
    client, transport = _client()

    resp = _simple_post(client, "/v1/messages", {"Origin": origin})

    assert resp.status_code == 200, resp.text
    assert len(transport.requests) == 1


def test_configured_origin_still_forwards(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_WS_ORIGINS", "https://dash.example")
    client, transport = _client()

    allowed = _simple_post(client, "/v1/messages", {"Origin": "https://dash.example"})
    refused = _simple_post(client, "/v1/messages", {"Origin": FOREIGN_ORIGIN})

    assert allowed.status_code == 200, allowed.text
    assert refused.status_code == 403
    assert len(transport.requests) == 1


def test_cross_site_subresource_without_origin_is_refused() -> None:
    client, transport = _client()

    resp = client.get(
        "/v1/models",
        headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors"},
    )

    assert resp.status_code == 403
    assert transport.requests == []


def test_cross_site_navigation_to_the_dashboard_still_loads() -> None:
    client, _ = _client()

    resp = client.get(
        "/health",
        headers={"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate"},
    )

    assert resp.status_code == 200


def test_loopback_peer_with_a_repointed_hostname_is_refused() -> None:
    # Origin matches Host, but Host is not loopback while the peer is.
    client, transport = _client()

    resp = _simple_post(
        client,
        "/v1/messages",
        {"Origin": "http://page.example:8787", "Host": "page.example:8787"},
    )

    assert resp.status_code == 403
    assert transport.requests == []
