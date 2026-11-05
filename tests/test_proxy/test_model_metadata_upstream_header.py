"""``GET /v1/models`` and ``/v1/models/{id}`` must honor ``x-headroom-base-url``.

The catch-all passthrough and the chat handlers already route to the
caller-named upstream; model discovery used to ignore the header and hit the
default OpenAI upstream (#4302). An unsafe named upstream is rejected with 400
exactly like the catch-all.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi import Response  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app  # noqa: E402

GATEWAY = "https://gateway.example"


_CALLS: list[tuple[str, str, str]] = []


@pytest.fixture(scope="module")
def client():
    # One app for the module: proxy shutdown dominates the runtime otherwise.
    mp = pytest.MonkeyPatch()
    mp.setenv("HEADROOM_REQUIRE_RUST_CORE", "false")

    async def fake_passthrough(self, request, base_url, sub_path=None, *args, **kwargs):
        _CALLS.append((base_url, sub_path, request.url.path))
        return Response(content=b'{"data": []}', media_type="application/json")

    mp.setattr(HeadroomProxy, "handle_passthrough", fake_passthrough)
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    try:
        with TestClient(create_app(config)) as c:
            yield c
    finally:
        mp.undo()


@pytest.fixture
def captured(client, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str, str]]:
    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", "gateway.example")
    _CALLS.clear()
    return _CALLS


def test_list_models_routes_to_header_upstream(client, captured) -> None:
    resp = client.get(
        "/v1/models",
        headers={"authorization": "Bearer k", "x-headroom-base-url": GATEWAY + "/"},
    )
    assert resp.status_code == 200
    assert [c[0] for c in captured] == [GATEWAY]


def test_get_model_routes_to_header_upstream(client, captured) -> None:
    resp = client.get(
        "/v1/models/some-model",
        headers={"authorization": "Bearer k", "x-headroom-base-url": GATEWAY},
    )
    assert resp.status_code == 200
    assert [c[0] for c in captured] == [GATEWAY]


def test_no_header_keeps_default_upstream(client, captured) -> None:
    resp = client.get("/v1/models", headers={"authorization": "Bearer k"})
    assert resp.status_code == 200
    assert captured and captured[0][0] != GATEWAY


@pytest.mark.parametrize("path", ["/v1/models", "/v1/models/some-model"])
def test_unsafe_header_upstream_rejected(client, captured, monkeypatch, path) -> None:
    monkeypatch.delenv("HEADROOM_ALLOWED_BASE_URLS", raising=False)
    resp = client.get(
        path,
        headers={"authorization": "Bearer k", "x-headroom-base-url": "http://127.0.0.1:9"},
    )
    assert resp.status_code == 400
    assert captured == []
