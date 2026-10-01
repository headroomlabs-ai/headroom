"""Tests for the Factory Droid routes registered when ``factory_api_url`` is set.

``headroom wrap droid`` points Droid's ``FACTORY_API_BASE_URL`` at the proxy.
With a Factory upstream configured, the proxy:

1. compresses ``POST /api/llm/a/v1/messages`` and forwards it to
   ``{factory}/api/llm/a/v1/messages`` (never to a client-named base);
2. forwards every other ``/api/*`` request verbatim to the Factory upstream,
   except Hermes Studio's scoped prefixes and requests that name their own
   ``x-headroom-base-url``, which keep the normal catch-all behaviour;
3. never sends traffic outside ``/api/*`` to Factory (the leak guard).

Forwarding is intercepted with ``respx`` so no real upstream is contacted.
"""

from __future__ import annotations

import json
import runpy
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlsplit

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")
respx = pytest.importorskip("respx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.loopback_guard import require_loopback  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

UPSTREAM = "https://factory.example"
UPSTREAM_HOST = "factory.example"
GATEWAY = "https://gateway.example"
MESSAGES = "/api/llm/a/v1/messages"


@pytest.fixture(autouse=True)
def _allow_reserved_test_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Admit the reserved, intentionally unresolvable client-named gateway."""
    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", "gateway.example")
    monkeypatch.delenv("FACTORY_TARGET_API_URL", raising=False)


class _FakeResult:
    """Stand-in for TransformPipeline.apply's TransformResult."""

    def __init__(self, messages: list[dict], tokens_before: int, tokens_after: int) -> None:
        self.messages = messages
        self.tokens_before = tokens_before
        self.tokens_after = tokens_after
        self.transforms_applied = ["smartcrush"]
        self.timing = {"total": 1.0}


def _make_config(**overrides: Any) -> ProxyConfig:
    base: dict[str, Any] = {
        "factory_api_url": UPSTREAM,
        "optimize": True,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "cost_tracking_enabled": False,
        "log_requests": False,
        "mode": "token",
    }
    base.update(overrides)
    return ProxyConfig(**base)


def _make_app(**overrides: Any):
    app = create_app(_make_config(**overrides))
    app.dependency_overrides[require_loopback] = lambda: None
    return app


def _anthropic_reply() -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }


# Hosts the proxy contacts on its own at startup (pricing table fetch,
# subscription usage poll with the machine's own credentials); unrelated to
# the request under test.
_BACKGROUND_HOSTS = frozenset({"raw.githubusercontent.com", "api.anthropic.com"})


class _Recorder:
    """Capture every upstream request the proxy makes, on any host."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path.endswith("/v1/messages"):
            return httpx.Response(200, json=_anthropic_reply())
        return httpx.Response(200, json={"ok": True, "path": request.url.path})

    def to_host(self, host: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.host == host]

    def proxied(self) -> list[httpx.Request]:
        """Requests forwarded on a client's behalf, minus proxy startup polls."""
        return [r for r in self.requests if r.url.host not in _BACKGROUND_HOSTS]

    def only(self, path: str) -> httpx.Request:
        hits = [r for r in self.requests if r.url.path == path]
        assert len(hits) == 1, [str(r.url) for r in self.requests]
        return hits[0]


@pytest.fixture
def recorder():
    rec = _Recorder()
    with respx.mock(assert_all_called=False) as router:
        router.route().mock(side_effect=rec)
        yield rec


def _big_messages_body() -> dict[str, Any]:
    return {
        "model": "claude-sonnet-4-5",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "lorem ipsum " * 600}],
    }


# ── route gating ──────────────────────────────────────────────────────


def _paths(cfg: ProxyConfig) -> list[str]:
    return [r.path for r in create_app(cfg).routes if hasattr(r, "path")]


def test_routes_absent_when_factory_api_url_unset():
    paths = _paths(ProxyConfig())
    assert MESSAGES not in paths
    assert "/api/{path:path}" not in paths


def test_routes_present_and_ordered_before_catch_all():
    paths = _paths(_make_config())
    assert MESSAGES in paths
    assert "/api/{path:path}" in paths
    assert paths.index(MESSAGES) < paths.index("/api/{path:path}")
    assert paths.index("/api/{path:path}") < paths.index("/{path:path}")


# ── inference route ───────────────────────────────────────────────────


def test_messages_route_compresses_and_reaches_factory(recorder):
    app = _make_app()
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy._record_request_outcome = AsyncMock()
        compressed = [{"role": "user", "content": "compressed"}]
        proxy.anthropic_pipeline.apply = MagicMock(return_value=_FakeResult(compressed, 900, 100))
        response = client.post(
            MESSAGES,
            headers={"authorization": "Bearer fk-test", "anthropic-version": "2023-06-01"},
            json=_big_messages_body(),
        )

    assert response.status_code == 200, response.text
    sent = recorder.only(MESSAGES)
    assert str(sent.url) == f"{UPSTREAM}{MESSAGES}"
    assert sent.headers["authorization"] == "Bearer fk-test"
    assert json.loads(sent.content)["messages"] == compressed
    assert proxy.anthropic_pipeline.apply.called
    outcome = proxy._record_request_outcome.await_args.args[0]
    assert outcome.provider == "factory"


def test_spoofed_headroom_headers_do_not_redirect_inference(recorder):
    app = _make_app()
    with TestClient(app) as client:
        response = client.post(
            MESSAGES,
            headers={
                "authorization": "Bearer fk-test",
                "anthropic-version": "2023-06-01",
                "x-headroom-base-url": "http://evil.example",
                "x-headroom-provider": "openai",
                "x-headroom-bypass": "true",
            },
            json=_big_messages_body(),
        )

    assert response.status_code == 200, response.text
    assert [r.url.host for r in recorder.proxied()] == [UPSTREAM_HOST]
    assert not recorder.to_host("evil.example")
    sent = recorder.only(MESSAGES)
    assert not [k for k in sent.headers if k.lower().startswith("x-headroom-")]
    assert sent.headers["authorization"] == "Bearer fk-test"


def test_operator_extra_headers_withheld_from_factory(recorder):
    app = _make_app(anthropic_extra_headers={"x-gateway-secret": "s3cret"})
    with TestClient(app) as client:
        response = client.post(
            MESSAGES,
            headers={"authorization": "Bearer fk-test", "anthropic-version": "2023-06-01"},
            json=_big_messages_body(),
        )

    assert response.status_code == 200, response.text
    sent = recorder.only(MESSAGES)
    assert "x-gateway-secret" not in {k.lower() for k in sent.headers}


# ── REST passthrough ──────────────────────────────────────────────────


def test_rest_get_passes_through_with_query_and_auth(recorder):
    app = _make_app()
    with TestClient(app) as client:
        response = client.get(
            "/api/sessions?limit=5",
            headers={"authorization": "Bearer fk-test"},
        )

    assert response.status_code == 200, response.text
    sent = recorder.only("/api/sessions")
    assert str(sent.url) == f"{UPSTREAM}/api/sessions?limit=5"
    assert sent.headers["authorization"] == "Bearer fk-test"


def test_rest_patch_passes_through_byte_identical(recorder):
    app = _make_app()
    payload = b'{"title":"renamed"}'
    with TestClient(app) as client:
        response = client.patch(
            "/api/sessions/s1",
            headers={"authorization": "Bearer fk-test", "content-type": "application/json"},
            content=payload,
        )

    assert response.status_code == 200, response.text
    sent = recorder.only("/api/sessions/s1")
    assert sent.method == "PATCH"
    assert sent.url.host == UPSTREAM_HOST
    assert sent.content == payload


# ── leak guard: unmatched traffic must never go to Factory ────────────


@pytest.mark.parametrize(
    ("method", "path", "headers", "expected_host"),
    [
        ("GET", "/backend-api/me", {"chatgpt-account-id": "acct"}, "chatgpt.com"),
        ("POST", "/v1/engines/x/completions", {"authorization": "Bearer gho"}, None),
        (
            "POST",
            "/v1beta/models/m:generateContent",
            {"x-goog-api-key": "g-key"},
            "generativelanguage.googleapis.com",
        ),
        (
            "POST",
            "/api/codex-proxy/s1/v1/responses",
            {"authorization": "Bearer sk-hermes"},
            "api.openai.com",
        ),
        (
            "POST",
            "/api/codex-proxy/s1/other",
            {"authorization": "Bearer sk-hermes"},
            "api.openai.com",
        ),
        (
            "POST",
            "/api/claude-code-proxy/s1/v1/messages",
            {"authorization": "Bearer sk-hermes", "x-headroom-base-url": GATEWAY},
            "gateway.example",
        ),
        (
            "GET",
            "/api/anything",
            {"authorization": "Bearer other", "x-headroom-base-url": GATEWAY},
            "gateway.example",
        ),
        ("GET", "/totally/unrouted", {"authorization": "Bearer sk-x"}, "api.openai.com"),
    ],
)
def test_non_factory_traffic_never_reaches_factory(
    recorder, method: str, path: str, headers: dict[str, str], expected_host: str | None
):
    app = _make_app()
    with TestClient(app) as client:
        body = {"model": "m", "input": "hi"} if method == "POST" else None
        client.request(method, path, headers=headers, json=body)

    forwarded = [r for r in recorder.requests if urlsplit(str(r.url)).path.endswith(path)]
    assert forwarded, [str(r.url) for r in recorder.requests]
    assert not recorder.to_host(UPSTREAM_HOST), [str(r.url) for r in recorder.requests]
    if expected_host is not None:
        assert forwarded[0].url.host == expected_host


def test_unsafe_client_base_on_api_path_is_still_rejected(recorder):
    app = _make_app()
    with TestClient(app) as client:
        response = client.get(
            "/api/anything",
            headers={"x-headroom-base-url": "http://169.254.169.254"},
        )

    assert response.status_code == 400
    assert recorder.proxied() == []


# ── guard: unchanged behaviour without Factory configured ─────────────


def test_without_factory_messages_path_falls_through_to_catch_all(recorder):
    app = create_app(
        ProxyConfig(
            optimize=True, cache_enabled=False, rate_limit_enabled=False, log_requests=False
        )
    )
    app.dependency_overrides[require_loopback] = lambda: None
    with TestClient(app) as client:
        response = client.post(
            MESSAGES,
            headers={"authorization": "Bearer fk-test"},
            json=_big_messages_body(),
        )

    assert response.status_code == 200, response.text
    sent = recorder.only(MESSAGES)
    assert sent.url.host != UPSTREAM_HOST
    assert sent.url.host == "api.openai.com"


# ── config plumbing ───────────────────────────────────────────────────


def _loopback_client(app) -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 12345))


def test_health_reports_factory_api_url():
    app = _make_app()
    with _loopback_client(app) as client:
        payload = client.get("/health").json()
    assert payload["config"]["factory_api_url"] == UPSTREAM


def test_health_reports_none_without_factory():
    app = create_app(ProxyConfig())
    with _loopback_client(app) as client:
        payload = client.get("/health").json()
    assert payload["config"]["factory_api_url"] is None


def test_env_var_feeds_config(monkeypatch):
    from headroom.proxy.server import _proxy_config_from_env

    monkeypatch.delenv("HEADROOM_PROXY_CONFIG_JSON", raising=False)
    monkeypatch.setenv("FACTORY_TARGET_API_URL", UPSTREAM)
    assert _proxy_config_from_env().factory_api_url == UPSTREAM


class _Captured(Exception):
    pass


def test_standalone_argparse_flag_feeds_config(monkeypatch):
    """``python -m headroom.proxy.server --factory-api-url`` populates ProxyConfig."""
    from headroom.proxy import models

    real = models.ProxyConfig
    seen: dict[str, Any] = {}

    def _capture(*args: Any, **kwargs: Any):
        if "factory_api_url" in kwargs:
            seen.update(kwargs)
            raise _Captured
        return real(*args, **kwargs)

    monkeypatch.setattr(sys, "argv", ["server", "--factory-api-url", UPSTREAM])
    monkeypatch.setattr(models, "ProxyConfig", _capture)
    with pytest.raises(_Captured):
        runpy.run_module("headroom.proxy.server", run_name="__main__")
    assert seen["factory_api_url"] == UPSTREAM


def test_cli_proxy_flag_feeds_config():
    from click.testing import CliRunner

    from headroom.cli.main import main

    captured: dict[str, Any] = {}

    def _run_server(config, **kwargs):
        captured["config"] = config

    with patch("headroom.proxy.server.run_server", _run_server):
        result = CliRunner().invoke(
            main, ["proxy", "--factory-api-url", UPSTREAM], catch_exceptions=False
        )

    assert result.exit_code == 0, result.output
    assert captured["config"].factory_api_url == UPSTREAM
    assert "Factory Droid" in result.output


def test_cli_proxy_env_feeds_config():
    from click.testing import CliRunner

    from headroom.cli.main import main

    captured: dict[str, Any] = {}

    def _run_server(config, **kwargs):
        captured["config"] = config

    with patch("headroom.proxy.server.run_server", _run_server):
        result = CliRunner().invoke(
            main, ["proxy"], env={"FACTORY_TARGET_API_URL": UPSTREAM}, catch_exceptions=False
        )

    assert result.exit_code == 0, result.output
    assert captured["config"].factory_api_url == UPSTREAM
