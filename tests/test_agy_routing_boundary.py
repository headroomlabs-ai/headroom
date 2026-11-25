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


_STREAM_ANTIGRAVITY_BODY = {
    "project": "demo-project",
    "model": "gemini-3-flash",
    "requestType": "agent",
    "request": {
        "contents": [{"role": "user", "parts": [{"text": "ping"}]}],
    },
}


def test_shared_proxy_streaming_ignores_untrusted_cloudcode_host(monkeypatch):
    """Ordinary shared-app streaming requests must not accept Host-based Cloud Code steering."""

    async def fake_stream(self, url, headers, _body, provider, model, *_args, **_kwargs):
        return JSONResponse({"url": url, "auth": headers.get("authorization")})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", fake_stream)
    app = create_app(ProxyConfig(optimize=False))
    response = TestClient(app).post(
        "/v1internal:streamGenerateContent",
        params={"alt": "sse"},
        headers={
            "host": "cloudcode-pa.googleapis.com",
            "authorization": "Bearer local-caller",
            "x-headroom-agy-dispatch": "true",
        },
        json=_STREAM_ANTIGRAVITY_BODY,
    )
    assert response.status_code == 200
    assert response.json() == {
        "url": "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal:streamGenerateContent?alt=sse",
        "auth": "Bearer local-caller",
    }


@pytest.mark.parametrize(
    "connect_host",
    ["cloudcode-pa.googleapis.com", "daily-cloudcode-pa.googleapis.com"],
)
def test_verified_dispatch_streaming_preserves_cloudcode_host(monkeypatch, connect_host):
    """Requests validated by make_host_guard preserve their original Cloud Code destination."""

    async def fake_stream(self, url, headers, _body, provider, model, *_args, **_kwargs):
        return JSONResponse({"url": url, "auth": headers.get("authorization")})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", fake_stream)
    app = create_app(ProxyConfig(optimize=False))
    guarded = make_host_guard(app, DEFAULT_ALLOWLIST)
    response = TestClient(guarded).post(
        "/v1internal:streamGenerateContent",
        params={"alt": "sse"},
        headers={
            "host": connect_host,
            "authorization": "Bearer cloudcode-caller",
        },
        json=_STREAM_ANTIGRAVITY_BODY,
    )
    assert response.status_code == 200
    assert response.json() == {
        "url": f"https://{connect_host}/v1internal:streamGenerateContent?alt=sse",
        "auth": "Bearer cloudcode-caller",
    }


def test_shared_proxy_antigravity_stream_preserves_sandbox_default(monkeypatch):
    """Unwrapped shared-proxy Antigravity streams keep the sandbox default URL."""

    async def fake_stream(self, url, headers, _body, provider, model, *_args, **_kwargs):
        return JSONResponse({"url": url, "provider": provider, "model": model})

    monkeypatch.delenv("HEADROOM_ANTIGRAVITY_API_URL", raising=False)
    monkeypatch.setattr(HeadroomProxy, "_stream_response", fake_stream)
    app = create_app(ProxyConfig(optimize=False))
    response = TestClient(app).post(
        "/v1internal:streamGenerateContent",
        params={"alt": "sse"},
        json=_STREAM_ANTIGRAVITY_BODY,
    )
    assert response.status_code == 200
    assert response.json() == {
        "url": (
            "https://daily-cloudcode-pa.sandbox.googleapis.com"
            "/v1internal:streamGenerateContent?alt=sse"
        ),
        "provider": "gemini",
        "model": "gemini-3-flash",
    }


@pytest.mark.parametrize(
    "connect_host",
    ["cloudcode-pa.googleapis.com", "daily-cloudcode-pa.googleapis.com"],
)
def test_verified_dispatch_cloudcode_host_takes_precedence_over_antigravity_ua(
    monkeypatch, connect_host
):
    """Guarded AGY dispatch keeps the connected Cloud Code host even with antigravity/ UA."""

    async def passthrough(self, request, base_url, **kw):
        return JSONResponse({"base": base_url, "auth": request.headers.get("authorization")})

    monkeypatch.setattr(HeadroomProxy, "handle_passthrough", passthrough)
    app = create_app(
        ProxyConfig(
            openai_api_url="https://operator-openai.test",
            cloudcode_api_url="https://operator-cloudcode.test",
        )
    )
    headers = {
        "host": connect_host,
        "user-agent": "antigravity/1.0",
        "authorization": "Bearer cloudcode-caller",
    }

    guarded = make_host_guard(app, DEFAULT_ALLOWLIST)
    guarded_resp = TestClient(guarded).post("/v1internal:loadCodeAssist", headers=headers)
    assert guarded_resp.status_code == 200
    assert guarded_resp.json() == {
        "base": f"https://{connect_host}",
        "auth": "Bearer cloudcode-caller",
    }

    # Without validated AGY dispatch state, an antigravity/ UA still uses the daily sandbox route.
    shared_resp = TestClient(app).post("/v1internal:loadCodeAssist", headers=headers)
    assert shared_resp.status_code == 200
    assert shared_resp.json() == {
        "base": "https://daily-cloudcode-pa.sandbox.googleapis.com",
        "auth": "Bearer cloudcode-caller",
    }
