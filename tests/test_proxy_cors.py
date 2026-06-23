"""CORS scoping tests.

The proxy binds on localhost and serves content endpoints (e.g. ``/v1/retrieve``
returns raw, uncompressed tool outputs). A wildcard CORS origin combined with
``allow_credentials=True`` let any web page the user had open read those
responses via a cross-origin fetch to ``127.0.0.1`` (CWE-346). The default
policy must allow only loopback origins — on *any* port, since the bound port
lives in the CLI/uvicorn layer, not in ``ProxyConfig`` — while still offering an
explicit override for Docker / remote-dashboard deployments. See #863 / #864.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


def _make_client() -> TestClient:
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
    )
    return TestClient(create_app(config))


def _preflight(client: TestClient, origin: str) -> httpx.Response:
    """Send a CORS preflight; CORSMiddleware answers it directly."""
    return client.options(
        "/v1/messages",
        headers={"Origin": origin, "Access-Control-Request-Method": "POST"},
    )


@pytest.mark.parametrize(
    "origin",
    [
        "http://localhost:8787",
        "http://127.0.0.1:8787",
        "http://localhost:9000",  # non-default port — must still be allowed
        "http://127.0.0.1:54321",
        "https://localhost:8787",
        "http://[::1]:8787",  # IPv6 loopback
        "http://localhost",  # no explicit port
    ],
)
def test_loopback_origins_allowed_on_any_port(monkeypatch: pytest.MonkeyPatch, origin: str) -> None:
    monkeypatch.delenv("HEADROOM_CORS_ORIGINS", raising=False)
    resp = _preflight(_make_client(), origin)
    assert resp.status_code == 200, resp.text
    assert resp.headers.get("access-control-allow-origin") == origin
    # The original vulnerability was wildcard + credentials; credentials stay off.
    assert resp.headers.get("access-control-allow-credentials") != "true"


@pytest.mark.parametrize(
    "origin",
    [
        "http://evil.com",
        "https://attacker.example",
        "http://localhost.evil.com",  # suffix smuggling
        "http://127.0.0.1.evil.com",
        "http://notlocalhost",  # prefix smuggling
    ],
)
def test_cross_origin_pages_rejected(monkeypatch: pytest.MonkeyPatch, origin: str) -> None:
    monkeypatch.delenv("HEADROOM_CORS_ORIGINS", raising=False)
    resp = _preflight(_make_client(), origin)
    # A disallowed origin is never echoed back, so the browser blocks the read.
    assert resp.headers.get("access-control-allow-origin") != origin


def test_explicit_allowlist_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_CORS_ORIGINS", "https://dash.example.com, http://10.0.0.5:3000")
    client = _make_client()

    allowed = _preflight(client, "https://dash.example.com")
    assert allowed.status_code == 200
    assert allowed.headers.get("access-control-allow-origin") == "https://dash.example.com"

    # Once an explicit list is set, loopback is no longer implicitly trusted.
    blocked = _preflight(client, "http://localhost:8787")
    assert blocked.headers.get("access-control-allow-origin") != "http://localhost:8787"


def test_wildcard_optback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_CORS_ORIGINS", "*")
    resp = _preflight(_make_client(), "http://evil.com")
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "*"
    # Wildcard opt-back must not silently re-enable credentialed reads.
    assert resp.headers.get("access-control-allow-credentials") != "true"
