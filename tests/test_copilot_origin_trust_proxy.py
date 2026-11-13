"""Through the running proxy: which upstream origins receive the operator's Copilot credential.

``tests/test_copilot_origin_trust.py`` pins ``apply_copilot_api_auth`` on its own. This file
drives a real ``create_app`` proxy with ``x-headroom-base-url`` and captures what is about to
be sent upstream (the ``_retry_request`` seam, after the handler has applied its auth), so the
decision is checked where a client can actually reach it. No request leaves the process and no
real token is minted.
"""

from __future__ import annotations

import socket

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom import copilot_auth  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

MINTED = "tid_minted_for_test"
CLIENT_AUTH = "Bearer client-own-token"
BODY = {
    "model": "gpt-4o",
    "stream": False,
    "messages": [{"role": "user", "content": "hi"}],
}


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mint a fixed token, and make every host resolve to a public address."""

    class _Token:
        token = MINTED

    class _Provider:
        async def get_api_token(self, integration_id: str | None = None):
            return _Token()

    monkeypatch.setattr(copilot_auth, "get_copilot_token_provider", lambda: _Provider())

    def public_resolution(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 443))]

    monkeypatch.setattr(socket, "getaddrinfo", public_resolution)
    for name in (
        "GITHUB_COPILOT_API_URL",
        "GITHUB_COPILOT_PROXY_URL",
        "GITHUB_COPILOT_API_TOKEN",
        "HEADROOM_ALLOWED_BASE_URLS",
    ):
        monkeypatch.delenv(name, raising=False)


class _Upstream:
    """Record the URL and headers the handler was about to put on the wire."""

    def __init__(self) -> None:
        self.url: str | None = None
        self.headers: dict[str, str] = {}

    async def __call__(self, method, url, headers, body, **kwargs):
        self.url = url
        self.headers = {k.lower(): v for k, v in dict(headers).items()}
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
            request=httpx.Request(method, url),
        )


@pytest.fixture(scope="module")
def proxy_client():
    """One running proxy for the module; building the app is the slow part."""
    app = create_app(
        ProxyConfig(optimize=False, cache_enabled=False, rate_limit_enabled=False, mode="token")
    )
    with TestClient(app) as client:
        yield client


def _through_proxy(client: TestClient, base_url: str) -> _Upstream:
    upstream = _Upstream()
    client.app.state.proxy._retry_request = upstream
    response = client.post(
        "/v1/chat/completions",
        json=BODY,
        headers={
            "x-headroom-base-url": base_url,
            "authorization": CLIENT_AUTH,
            "x-api-key": "sk-unrelated",
        },
    )
    assert response.status_code == 200, response.text
    assert upstream.url is not None, "the request never reached the upstream seam"
    return upstream


def test_trusted_https_default_port_receives_the_operator_credential(proxy_client) -> None:
    upstream = _through_proxy(proxy_client, "https://api.githubcopilot.com")

    assert upstream.url is not None and upstream.url.startswith("https://api.githubcopilot.com/")
    assert upstream.headers["authorization"] == f"Bearer {MINTED}"
    assert "x-api-key" not in upstream.headers


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.githubcopilot.com",
        "http://api.githubcopilot.com:8080",
        "https://api.githubcopilot.com:8443",
    ],
)
def test_untrusted_origin_receives_no_operator_credential(proxy_client, base_url: str) -> None:
    upstream = _through_proxy(proxy_client, base_url)

    assert MINTED not in str(upstream.headers)
    assert upstream.headers["authorization"] == CLIENT_AUTH
    assert "x-api-key" not in upstream.headers


def test_an_operator_configured_origin_keeps_its_credential(
    proxy_client,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://api.githubcopilot.com:8443")
    upstream = _through_proxy(proxy_client, "https://api.githubcopilot.com:8443")

    assert upstream.headers["authorization"] == f"Bearer {MINTED}"
