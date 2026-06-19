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
        "url": "https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse",
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
        "url": "https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse",
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
        "url": "https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse",
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


# ---------------------------------------------------------------------------
# T4: agy agent-model + body detection; env override; Pi/OpenClaw non-regression
# ---------------------------------------------------------------------------

AGY_AGENT_BODY = {
    "project": "my-gcp-project",
    "model": "gemini-3-flash-agent",
    "request": {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "Hello from agy"}],
            }
        ]
    },
}


def test_agy_agent_model_body_routes_to_daily_endpoint(monkeypatch):
    """agy traffic with agent-model name + project + request.contents hits non-sandbox daily host."""

    async def fake_stream(self, url, _headers, _body, provider, model, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return JSONResponse({"url": url, "provider": provider, "model": model})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", fake_stream)

    # No antigravity/ UA header on purpose: detection must rest SOLELY on the
    # agy body shape (agent-model name + project + request.contents), so this
    # test fails if the body-shape detection branch is removed.
    with TestClient(create_app(ProxyConfig(optimize=False))) as client:
        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            json=AGY_AGENT_BODY,
        )

    assert response.status_code == 200
    assert response.json() == {
        "url": "https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse",
        "provider": "gemini",
        "model": "gemini-3-flash-agent",
    }


def test_headroom_antigravity_api_url_env_override(monkeypatch):
    """HEADROOM_ANTIGRAVITY_API_URL env var overrides the corrected default for antigravity traffic."""

    async def fake_stream(self, url, _headers, _body, provider, model, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return JSONResponse({"url": url, "provider": provider, "model": model})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", fake_stream)
    monkeypatch.setenv("HEADROOM_ANTIGRAVITY_API_URL", "https://my-custom-agy.example.com")

    with TestClient(create_app(ProxyConfig(optimize=False))) as client:
        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            json=ANTIGRAVITY_BODY,
        )

    assert response.status_code == 200
    assert (
        response.json()["url"]
        == "https://my-custom-agy.example.com/v1internal:streamGenerateContent?alt=sse"
    )


def test_pi_openclaw_requesttype_agent_still_detected(monkeypatch):
    """Pi/OpenClaw requestType=='agent' detection is not broken by new agy checks."""

    async def fake_stream(self, url, _headers, _body, provider, model, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        return JSONResponse({"url": url, "provider": provider, "model": model})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", fake_stream)

    pi_body = {
        "project": "pi-project",
        "model": "gemini-1.5-pro",
        "requestType": "agent",
        "userAgent": "pi-coding-agent",
        "request": {"contents": [{"role": "user", "parts": [{"text": "ping"}]}]},
    }

    with TestClient(create_app(ProxyConfig(optimize=False))) as client:
        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            json=pi_body,
        )

    assert response.status_code == 200
    assert (
        response.json()["url"]
        == "https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse"
    )


def test_agy_control_plane_passthrough_routes_to_cloudcode_host(monkeypatch):
    """agy's non-streamGenerateContent control-plane calls (loadCodeAssist,
    setUserSettings, …) reach the catch-all and MUST be proxied back to the
    Cloud Code host agy addressed — not the generic Gemini endpoint that the
    x-goog-api-key header would otherwise select. Without this the MITM dispatch
    404s agy's onboarding and agy never issues a generateContent call."""

    async def fake_passthrough(self, request, base_url, *args, **kwargs):  # type: ignore[no-untyped-def]
        return JSONResponse({"base_url": base_url, "path": request.url.path})

    monkeypatch.setattr(HeadroomProxy, "handle_passthrough", fake_passthrough)

    with TestClient(create_app(ProxyConfig(optimize=False))) as client:
        response = client.post(
            "/v1internal:loadCodeAssist",
            headers={
                "host": "daily-cloudcode-pa.googleapis.com",
                # agy sends x-goog-api-key; this previously forced the generic
                # Gemini host. The Cloud Code host check must win over it.
                "x-goog-api-key": "test-key",
            },
            json={"metadata": {"pluginType": "GEMINI"}},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["path"] == "/v1internal:loadCodeAssist"
    assert body["base_url"] == "https://daily-cloudcode-pa.googleapis.com"


def test_agy_control_plane_passthrough_rejects_non_allowlisted_host(monkeypatch):
    """The v1internal control-plane branch forwards to the incoming Host only
    when it is an allowlisted Cloud Code host. A look-alike host (e.g. a suffix
    match like ``evilcloudcode-pa.googleapis.com``) must NOT be used as the
    upstream — it falls back to the static cloudcode target, so a forged Host
    header cannot steer the MITM passthrough to an attacker-controlled origin."""

    async def fake_passthrough(self, request, base_url, *args, **kwargs):  # type: ignore[no-untyped-def]
        return JSONResponse({"base_url": base_url, "path": request.url.path})

    monkeypatch.setattr(HeadroomProxy, "handle_passthrough", fake_passthrough)

    with TestClient(create_app(ProxyConfig(optimize=False))) as client:
        response = client.post(
            "/v1internal:loadCodeAssist",
            headers={
                "host": "evilcloudcode-pa.googleapis.com",
                "x-goog-api-key": "test-key",
            },
            json={"metadata": {"pluginType": "GEMINI"}},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["base_url"] != "https://evilcloudcode-pa.googleapis.com"
