"""Tests that the ``/v1/messages`` route honors ``x-headroom-base-url``.

The Anthropic Messages route must forward the per-request upstream
override header to ``handle_anthropic_messages`` so clients that speak
the Anthropic wire format but authenticate against a non-Anthropic
gateway (e.g. OpenCode Zen's "Go" tier) route correctly — consistent
with the OpenAI-compatible routes and the generic passthrough route,
which already honor it (see ``providers/proxy_routes.py``).

Contract pinned here:
- header present            → its value is passed as ``upstream_base_url``
- header absent             → no override (``upstream_base_url`` unset/None)
- header empty/whitespace   → no override (must not blank the upstream)
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

MESSAGES = "/v1/messages"
BODY = {"model": "glm-5.2", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}


def _make_config(**overrides) -> ProxyConfig:
    base = {
        "optimize": False,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "mode": "token",
    }
    base.update(overrides)
    return ProxyConfig(**base)


def _install_handler_spy(proxy) -> AsyncMock:
    """Replace handle_anthropic_messages with a spy returning a 200."""
    spy = AsyncMock(return_value=JSONResponse({"ok": True}))
    proxy.handle_anthropic_messages = spy
    return spy


def _base_url_kwarg(spy: AsyncMock):
    call = spy.call_args
    return call.kwargs.get("upstream_base_url")


def test_header_passed_as_upstream_base_url():
    app = create_app(_make_config())
    with TestClient(app) as client:
        spy = _install_handler_spy(client.app.state.proxy)
        resp = client.post(
            MESSAGES,
            json=BODY,
            headers={"x-headroom-base-url": "https://opencode.ai/zen/go"},
        )

    assert resp.status_code == 200
    assert _base_url_kwarg(spy) == "https://opencode.ai/zen/go"


def test_missing_header_leaves_upstream_unset():
    app = create_app(_make_config())
    with TestClient(app) as client:
        spy = _install_handler_spy(client.app.state.proxy)
        resp = client.post(MESSAGES, json=BODY)

    assert resp.status_code == 200
    assert _base_url_kwarg(spy) is None


def test_empty_or_whitespace_header_leaves_upstream_unset():
    for value in ("", "   "):
        app = create_app(_make_config())
        with TestClient(app) as client:
            spy = _install_handler_spy(client.app.state.proxy)
            resp = client.post(MESSAGES, json=BODY, headers={"x-headroom-base-url": value})

        assert resp.status_code == 200
        assert _base_url_kwarg(spy) is None


def test_header_value_is_trimmed_and_trailing_slash_stripped():
    app = create_app(_make_config())
    with TestClient(app) as client:
        spy = _install_handler_spy(client.app.state.proxy)
        resp = client.post(
            MESSAGES,
            json=BODY,
            headers={"x-headroom-base-url": "  https://opencode.ai/zen/go/  "},
        )

    assert resp.status_code == 200
    assert _base_url_kwarg(spy) == "https://opencode.ai/zen/go"
