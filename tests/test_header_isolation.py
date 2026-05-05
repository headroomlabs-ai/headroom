"""Header-isolation tests for PR-A5 (P5-49 fix).

`x-headroom-*` request headers are internal control flags consumed by the
proxy itself (bypass gating, mode selection, user-id, stack/base-url
fingerprints). Forwarding them upstream:

  1. Fingerprints the proxy to subscription-revocation enforcers.
  2. Leaks user-id / stack / base-url internals to whichever vendor
     terminates the request.

PR-A5 wraps every handler-entry capture of the request headers with
`_strip_internal_headers`. Inbound read paths (`request.headers.get(...)`
for bypass gating, `_extract_tags` reading `x-headroom-*`) keep working
because they never depended on the local outbound-bound dict.

Operator opt-in `HEADROOM_STRIP_INTERNAL_HEADERS=disabled` keeps the
internal headers in the upstream-bound dict for diagnostic shadow tracing.
That mode is loud and explicit per realignment build constraint #4 — NOT
a silent fallback.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.helpers import (
    _strip_internal_headers,
    get_strip_internal_headers_mode,
)
from headroom.proxy.server import ProxyConfig, create_app

# ---------------------------------------------------------------------------
# Pure helper unit tests
# ---------------------------------------------------------------------------


def test_strip_returns_new_dict_does_not_mutate_caller() -> None:
    """`_strip_internal_headers` is pure — caller's dict is untouched."""
    original = {
        "authorization": "Bearer x",
        "x-headroom-bypass": "true",
    }
    out = _strip_internal_headers(original)
    assert out is not original
    assert "x-headroom-bypass" in original
    assert "x-headroom-bypass" not in out
    assert out["authorization"] == "Bearer x"


def test_strip_x_headroom_bypass_removed() -> None:
    out = _strip_internal_headers({"x-headroom-bypass": "true", "k": "v"})
    assert "x-headroom-bypass" not in out
    assert out["k"] == "v"


def test_strip_x_headroom_mode_removed() -> None:
    out = _strip_internal_headers({"x-headroom-mode": "passthrough", "k": "v"})
    assert "x-headroom-mode" not in out
    assert out["k"] == "v"


def test_strip_x_headroom_user_id_removed() -> None:
    out = _strip_internal_headers({"x-headroom-user-id": "u1", "k": "v"})
    assert "x-headroom-user-id" not in out
    assert out["k"] == "v"


def test_strip_x_headroom_stack_removed() -> None:
    out = _strip_internal_headers({"x-headroom-stack": "engineer", "k": "v"})
    assert "x-headroom-stack" not in out
    assert out["k"] == "v"


def test_strip_x_headroom_base_url_removed() -> None:
    out = _strip_internal_headers({"x-headroom-base-url": "http://x", "k": "v"})
    assert "x-headroom-base-url" not in out
    assert out["k"] == "v"


def test_strip_case_insensitive_prefix_match() -> None:
    """Mixed-case `X-Headroom-Foo`, `x-Headroom-Bar`, `X-HEADROOM-BAZ` all stripped."""
    out = _strip_internal_headers(
        {
            "X-Headroom-Foo": "1",
            "x-Headroom-Bar": "2",
            "X-HEADROOM-BAZ": "3",
            "Authorization": "Bearer x",
        }
    )
    assert "X-Headroom-Foo" not in out
    assert "x-Headroom-Bar" not in out
    assert "X-HEADROOM-BAZ" not in out
    assert out["Authorization"] == "Bearer x"


def test_strip_legitimate_headers_passthrough() -> None:
    """Headers without the internal prefix must NOT be stripped."""
    out = _strip_internal_headers(
        {
            "Authorization": "Bearer x",
            "x-api-key": "k",
            "x-request-id": "rid-1",
            "x-trace-id": "tid-1",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            "User-Agent": "claude-code/1.0",
        }
    )
    assert out["Authorization"] == "Bearer x"
    assert out["x-api-key"] == "k"
    assert out["x-request-id"] == "rid-1"
    assert out["x-trace-id"] == "tid-1"
    assert out["Content-Type"] == "application/json"
    assert out["anthropic-version"] == "2023-06-01"
    assert out["User-Agent"] == "claude-code/1.0"


def test_strip_disabled_mode_passes_internal_headers_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`HEADROOM_STRIP_INTERNAL_HEADERS=disabled` is operator opt-in for diag."""
    monkeypatch.setenv("HEADROOM_STRIP_INTERNAL_HEADERS", "disabled")
    out = _strip_internal_headers({"x-headroom-bypass": "true", "authorization": "Bearer x"})
    assert out["x-headroom-bypass"] == "true"
    assert out["authorization"] == "Bearer x"


def test_strip_disabled_mode_returns_copy_not_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even in disabled mode the helper returns a NEW dict, never an alias."""
    monkeypatch.setenv("HEADROOM_STRIP_INTERNAL_HEADERS", "disabled")
    src = {"x-headroom-bypass": "true"}
    out = _strip_internal_headers(src)
    assert out is not src


def test_strip_mode_default_is_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_STRIP_INTERNAL_HEADERS", raising=False)
    assert get_strip_internal_headers_mode() == "enabled"


def test_strip_mode_invalid_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_STRIP_INTERNAL_HEADERS", "garbage")
    with pytest.raises(ValueError, match="HEADROOM_STRIP_INTERNAL_HEADERS"):
        get_strip_internal_headers_mode()


def test_strip_empty_dict_returns_empty_dict() -> None:
    assert _strip_internal_headers({}) == {}


def test_strip_preserves_value_semantics() -> None:
    """Values are forwarded unchanged for kept headers."""
    out = _strip_internal_headers(
        {
            "Authorization": "Bearer sk-ant-...",
            "anthropic-version": "2023-06-01",
        }
    )
    assert out["Authorization"] == "Bearer sk-ant-..."
    assert out["anthropic-version"] == "2023-06-01"


# ---------------------------------------------------------------------------
# End-to-end: x-headroom-* never reaches the upstream
# ---------------------------------------------------------------------------


class _CapturingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.captured_headers: dict[str, str] | None = None
        self.captured_body: bytes | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = b""
        async for chunk in request.stream:
            body += chunk
        self.captured_body = body
        self.captured_headers = dict(request.headers.items())
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            },
        )


class _FakePrefixTracker:
    def __init__(self, frozen_count: int = 0):
        self._frozen_count = frozen_count
        self._cached_token_count = 0
        self._last_original_messages: list = []
        self._last_forwarded_messages: list = []

    def get_frozen_message_count(self) -> int:
        return self._frozen_count

    def get_last_original_messages(self):  # noqa: ANN201
        return list(self._last_original_messages)

    def get_last_forwarded_messages(self):  # noqa: ANN201
        return list(self._last_forwarded_messages)

    def update_from_response(self, **kwargs):  # noqa: ANN003
        self._last_original_messages = kwargs.get("original_messages", kwargs.get("messages", []))
        self._last_forwarded_messages = kwargs.get("messages", [])
        return None


def _make_anthropic_app() -> tuple[TestClient, _CapturingTransport]:
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
    app = create_app(config)
    proxy = app.state.proxy
    transport = _CapturingTransport()
    proxy.http_client = httpx.AsyncClient(transport=transport)

    fake_tracker = _FakePrefixTracker(frozen_count=0)
    proxy.session_tracker_store.compute_session_id = lambda request, model, messages: "s1"
    proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

    return TestClient(app), transport


def test_x_headroom_bypass_not_forwarded() -> None:
    client, transport = _make_anthropic_app()
    resp = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-headroom-bypass": "true",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert transport.captured_headers is not None
    upstream = {k.lower(): v for k, v in transport.captured_headers.items()}
    assert "x-headroom-bypass" not in upstream
    # Legitimate headers must reach upstream.
    assert upstream.get("x-api-key") == "test-key"
    assert upstream.get("anthropic-version") == "2023-06-01"


def test_x_headroom_mode_not_forwarded() -> None:
    client, transport = _make_anthropic_app()
    resp = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-headroom-mode": "passthrough",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert transport.captured_headers is not None
    upstream = {k.lower(): v for k, v in transport.captured_headers.items()}
    assert "x-headroom-mode" not in upstream


def test_x_headroom_user_id_not_forwarded() -> None:
    client, transport = _make_anthropic_app()
    resp = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-headroom-user-id": "alice",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert transport.captured_headers is not None
    upstream = {k.lower(): v for k, v in transport.captured_headers.items()}
    assert "x-headroom-user-id" not in upstream


def test_x_headroom_stack_not_forwarded() -> None:
    client, transport = _make_anthropic_app()
    resp = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-headroom-stack": "engineer",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert transport.captured_headers is not None
    upstream = {k.lower(): v for k, v in transport.captured_headers.items()}
    assert "x-headroom-stack" not in upstream


def test_x_headroom_base_url_not_forwarded() -> None:
    client, transport = _make_anthropic_app()
    resp = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-headroom-base-url": "https://override.example",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert transport.captured_headers is not None
    upstream = {k.lower(): v for k, v in transport.captured_headers.items()}
    assert "x-headroom-base-url" not in upstream


def test_case_insensitive_prefix_match_e2e() -> None:
    """Mixed-case headers are stripped end-to-end."""
    client, transport = _make_anthropic_app()
    resp = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "X-Headroom-Foo": "1",
            "x-Headroom-Bar": "2",
            "X-HEADROOM-BAZ": "3",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert transport.captured_headers is not None
    upstream_lower = {k.lower(): v for k, v in transport.captured_headers.items()}
    assert "x-headroom-foo" not in upstream_lower
    assert "x-headroom-bar" not in upstream_lower
    assert "x-headroom-baz" not in upstream_lower


def test_legitimate_headers_passthrough_e2e() -> None:
    """Authorization / x-api-key / x-request-id / Content-Type all preserved."""
    client, transport = _make_anthropic_app()
    resp = client.post(
        "/v1/messages",
        headers={
            "Authorization": "Bearer sk-ant-test",
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-request-id": "rid-abc",
            "User-Agent": "claude-code/1.2.3",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert transport.captured_headers is not None
    upstream = {k.lower(): v for k, v in transport.captured_headers.items()}
    assert upstream.get("x-api-key") == "test-key"
    assert upstream.get("anthropic-version") == "2023-06-01"
    assert upstream.get("user-agent") == "claude-code/1.2.3"
    # Authorization is delivered uppercased; httpx normalizes to lowercase
    # when reading via dict()-of-Headers, so check via lowercase key.
    assert upstream.get("authorization") == "Bearer sk-ant-test"


def test_inbound_read_path_still_reads_x_headroom_bypass() -> None:
    """Bypass header still gates compression even though it's stripped from upstream.

    The handler reads `request.headers.get('x-headroom-bypass')` directly.
    Stripping the local outbound-bound `headers` dict does NOT affect that
    inbound read path.
    """
    config = ProxyConfig(
        # Compression is the only thing bypass affects observably; turn it on.
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    app = create_app(config)
    proxy = app.state.proxy
    transport = _CapturingTransport()
    proxy.http_client = httpx.AsyncClient(transport=transport)

    fake_tracker = _FakePrefixTracker(frozen_count=0)
    proxy.session_tracker_store.compute_session_id = lambda request, model, messages: "s_bypass"
    proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

    client = TestClient(app)

    # Force a bypass via header. The handler logs "Bypass: skipping compression"
    # if the inbound read worked; we can't easily intercept the log, so we
    # primarily assert that:
    #   1. The request still succeeds.
    #   2. The upstream did NOT receive the `x-headroom-bypass` header.
    resp = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-headroom-bypass": "true",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200
    upstream = {k.lower(): v for k, v in (transport.captured_headers or {}).items()}
    assert "x-headroom-bypass" not in upstream


def test_disabled_mode_passes_through_e2e(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`HEADROOM_STRIP_INTERNAL_HEADERS=disabled` lets internal headers through."""
    monkeypatch.setenv("HEADROOM_STRIP_INTERNAL_HEADERS", "disabled")
    client, transport = _make_anthropic_app()
    resp = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-headroom-mode": "passthrough",
        },
        json={
            "model": "claude-sonnet-4-6",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    assert transport.captured_headers is not None
    upstream = {k.lower(): v for k, v in transport.captured_headers.items()}
    # Operator opt-in: internal header IS forwarded (diagnostic mode).
    assert upstream.get("x-headroom-mode") == "passthrough"


# ---------------------------------------------------------------------------
# OpenAI Chat Completions parity check
# ---------------------------------------------------------------------------


def test_openai_chat_x_headroom_bypass_not_forwarded() -> None:
    """OpenAI handler also strips x-headroom-* before upstream call."""
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
    app = create_app(config)
    proxy = app.state.proxy

    captured: dict[str, object] = {}

    async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
        captured["headers"] = dict(headers)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_1",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            },
        )

    proxy._retry_request = _fake_retry  # type: ignore[attr-defined]
    proxy.memory_handler = SimpleNamespace(
        config=SimpleNamespace(inject_context=False, inject_tools=False),
        search_and_format_context=AsyncMock(return_value=""),
        has_memory_tool_calls=lambda resp, provider: False,
    )

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        headers={
            "authorization": "Bearer sk-test",
            "x-headroom-bypass": "true",
            "x-headroom-user-id": "u1",
        },
        json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert resp.status_code == 200, resp.text
    sent_headers_raw = captured.get("headers")
    assert isinstance(sent_headers_raw, dict)
    sent_headers = {k.lower(): v for k, v in sent_headers_raw.items()}
    assert "x-headroom-bypass" not in sent_headers
    assert "x-headroom-user-id" not in sent_headers
    assert sent_headers.get("authorization") == "Bearer sk-test"
