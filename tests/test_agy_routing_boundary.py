"""Cloud Code Host routing belongs to the verified AGY dispatch boundary."""

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from headroom.proxy.agy_dispatch import make_host_guard
from headroom.proxy.agy_terminator import DEFAULT_ALLOWLIST
from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app


@pytest.mark.parametrize("path", ["/opaque-control-plane", "/v1internal:loadCodeAssist"])
def test_shared_proxy_cannot_be_redirected_by_real_cloudcode_host(monkeypatch, path):
    async def passthrough(self, request, base_url, **kw):
        return JSONResponse({"base": base_url, "auth": request.headers.get("authorization")})

    monkeypatch.setattr(HeadroomProxy, "handle_passthrough", passthrough)
    app = create_app(
        ProxyConfig(
            openai_api_url="https://operator-openai.test",
            cloudcode_api_url="https://operator-cloudcode.test",
        )
    )
    response = TestClient(app).post(
        path,
        headers={
            "host": "daily-cloudcode-pa.googleapis.com",
            "authorization": "Bearer local-caller",
            "x-headroom-agy-dispatch": "true",
        },
    )
    assert response.status_code == 200
    expected = "operator-cloudcode" if "loadCodeAssist" in path else "operator-openai"
    assert response.json() == {"base": f"https://{expected}.test", "auth": "Bearer local-caller"}


@pytest.mark.parametrize("path", ["/opaque-control-plane", "/v1internal:loadCodeAssist"])
def test_verified_dispatch_keeps_original_cloudcode_destination(monkeypatch, path):
    async def passthrough(self, request, base_url, **kw):
        return JSONResponse({"base": base_url, "auth": request.headers.get("authorization")})

    monkeypatch.setattr(HeadroomProxy, "handle_passthrough", passthrough)
    app = create_app(
        ProxyConfig(
            openai_api_url="https://operator-openai.test",
            cloudcode_api_url="https://operator-cloudcode.test",
        )
    )
    guarded = make_host_guard(app, DEFAULT_ALLOWLIST)
    response = TestClient(guarded).post(
        path,
        headers={
            "host": "daily-cloudcode-pa.googleapis.com",
            "authorization": "Bearer cloudcode-caller",
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "base": "https://daily-cloudcode-pa.googleapis.com",
        "auth": "Bearer cloudcode-caller",
    }
