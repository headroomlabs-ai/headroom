"""``GET /v1/models/{id}`` with ``x-headroom-base-url`` keeps the selected provider.

The header only overrides the destination. An Anthropic-authenticated request
(``x-api-key``) must still go through ``handle_passthrough``'s Anthropic
model-id sanitization, so Claude Code's ``[1m]`` context-tier suffix never
reaches the named gateway (#4302). These tests run the real
``handle_passthrough`` and fake only the upstream HTTP transport, asserting the
outbound URL.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

import httpx  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

GATEWAY = "https://gateway.example"


@pytest.fixture
def real_passthrough(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")
    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", "gateway.example")
    outbound: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        outbound.append(request)
        return httpx.Response(200, json={"data": []})

    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    with TestClient(create_app(config)) as c:
        c.app.state.proxy.http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        yield c, outbound


def test_anthropic_auth_with_header_upstream_sanitizes_model_id(real_passthrough) -> None:
    client, outbound = real_passthrough
    resp = client.get(
        "/v1/models/claude-opus-4-8%5B1m%5D",
        headers={"x-api-key": "sk-ant-test", "x-headroom-base-url": GATEWAY},
    )
    assert resp.status_code == 200
    assert [str(r.url) for r in outbound] == [f"{GATEWAY}/v1/models/claude-opus-4-8"]


def test_bearer_auth_with_header_upstream_leaves_model_id_untouched(real_passthrough) -> None:
    client, outbound = real_passthrough
    resp = client.get(
        "/v1/models/some-model%5B1m%5D",
        headers={"authorization": "Bearer k", "x-headroom-base-url": GATEWAY},
    )
    assert resp.status_code == 200
    assert [str(r.url) for r in outbound] == [f"{GATEWAY}/v1/models/some-model[1m]"]
