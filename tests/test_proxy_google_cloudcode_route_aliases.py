from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app

CLOUDCODE_BODY = {
    "project": "test-project",
    "model": "gemini-3.1-pro-high",
    "userAgent": "pi-coding-agent",
    "request": {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Reply with pong."}],
            }
        ]
    },
}

ANTIGRAVITY_BODY = {
    "project": "test-project",
    "model": "claude-sonnet-4-6",
    "requestType": "agent",
    "userAgent": "antigravity",
    "request": {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Reply with pong."}],
            }
        ]
    },
}


def test_google_cloudcode_alias_routes_delegate_to_handler(monkeypatch):
    async def fake_handle(self, request):  # type: ignore[no-untyped-def]
        return JSONResponse({"ok": True, "path": request.url.path})

    monkeypatch.setattr(HeadroomProxy, "handle_google_cloudcode_stream", fake_handle)

    with TestClient(create_app(ProxyConfig())) as client:
        for path in (
            "/v1internal:streamGenerateContent",
            "/v1/v1internal:streamGenerateContent",
        ):
            response = client.post(path, params={"alt": "sse"}, json=CLOUDCODE_BODY)
            assert response.status_code == 200
            assert response.json() == {"ok": True, "path": path}


def test_antigravity_cloudcode_route_uses_daily_endpoint(monkeypatch):
    async def fake_stream(self, url, _headers, _body, provider, model, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return JSONResponse({"url": url, "provider": provider, "model": model})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", fake_stream)

    with TestClient(create_app(ProxyConfig(optimize=False))) as client:
        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            json=ANTIGRAVITY_BODY,
        )

    assert response.status_code == 200
    assert response.json() == {
        "url": "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal:streamGenerateContent?alt=sse",
        "provider": "gemini",
        "model": "claude-sonnet-4-6",
    }


def test_cloudcode_route_uses_default_cloudcode_endpoint(monkeypatch):
    async def fake_stream(self, url, _headers, _body, provider, model, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return JSONResponse({"url": url, "provider": provider, "model": model})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", fake_stream)

    with TestClient(create_app(ProxyConfig(optimize=False))) as client:
        response = client.post(
            "/v1/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            json=CLOUDCODE_BODY,
        )

    assert response.status_code == 200
    assert response.json() == {
        "url": "https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse",
        "provider": "gemini",
        "model": "gemini-3.1-pro-high",
    }


def test_cloudcode_route_uses_cloudcode_api_override(monkeypatch):
    async def fake_stream(self, url, _headers, _body, provider, model, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return JSONResponse({"url": url, "provider": provider, "model": model})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", fake_stream)

    with TestClient(
        create_app(ProxyConfig(optimize=False, cloudcode_api_url="https://cloudcode-proxy.test/v1"))
    ) as client:
        response = client.post(
            "/v1/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            json=CLOUDCODE_BODY,
        )

    assert response.status_code == 200
    assert response.json() == {
        "url": "https://cloudcode-proxy.test/v1internal:streamGenerateContent?alt=sse",
        "provider": "gemini",
        "model": "gemini-3.1-pro-high",
    }


def test_antigravity_header_detection_is_case_insensitive(monkeypatch):
    async def fake_stream(self, url, _headers, _body, provider, model, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return JSONResponse({"url": url, "provider": provider, "model": model})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", fake_stream)

    body = {
        **CLOUDCODE_BODY,
        "model": "claude-opus-4-6-thinking",
    }

    with TestClient(create_app(ProxyConfig(optimize=False))) as client:
        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            headers={"User-Agent": "Antigravity/1.2.3 Darwin/arm64"},
            json=body,
        )

    assert response.status_code == 200
    assert response.json() == {
        "url": "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal:streamGenerateContent?alt=sse",
        "provider": "gemini",
        "model": "claude-opus-4-6-thinking",
    }


def test_antigravity_route_does_not_cross_route_to_cloudcode_override(monkeypatch):
    async def fake_stream(self, url, _headers, _body, provider, model, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return JSONResponse({"url": url, "provider": provider, "model": model})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", fake_stream)

    with TestClient(
        create_app(ProxyConfig(optimize=False, cloudcode_api_url="https://cloudcode-proxy.test"))
    ) as client:
        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            json=ANTIGRAVITY_BODY,
        )

    assert response.status_code == 200
    assert response.json() == {
        "url": "https://daily-cloudcode-pa.sandbox.googleapis.com/v1internal:streamGenerateContent?alt=sse",
        "provider": "gemini",
        "model": "claude-sonnet-4-6",
    }


def test_cloudcode_override_does_not_leak_between_app_instances(monkeypatch):
    async def fake_stream(self, url, _headers, _body, provider, model, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return JSONResponse({"url": url, "provider": provider, "model": model})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", fake_stream)

    with TestClient(
        create_app(ProxyConfig(optimize=False, cloudcode_api_url="https://cloudcode-proxy.test"))
    ) as client:
        first = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            json=CLOUDCODE_BODY,
        )

    with TestClient(create_app(ProxyConfig(optimize=False))) as client:
        second = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            json=CLOUDCODE_BODY,
        )

    assert first.status_code == 200
    assert (
        first.json()["url"]
        == "https://cloudcode-proxy.test/v1internal:streamGenerateContent?alt=sse"
    )
    assert second.status_code == 200
    assert (
        second.json()["url"]
        == "https://cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse"
    )
