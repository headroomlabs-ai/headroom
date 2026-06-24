"""Tests for agy/antigravity path in handle_google_cloudcode_stream.

Scope: compression behaviour, routing, stealth, SSE pass-through,
accept-encoding stripping, single-upstream-origination, gzip request body,
auth-redaction on the default log path, and fail-open observability.

All tests use TestClient(create_app(…)) — in-process, no real port bind.
ALL upstream/network calls are stubbed via monkeypatch on HeadroomProxy._stream_response
or HeadroomProxy.openai_pipeline (the compression pipeline).
Never contacts 8787 or any real network destination.
"""

from __future__ import annotations

import gzip
import json
import logging
from typing import Any

import pytest
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app

# ---------------------------------------------------------------------------
# Shared fixture body — large enough that CompressionDecision.should_compress
# is True when optimize=True (default).  Repeated text triggers compression.
# ---------------------------------------------------------------------------

_REPEAT_UNIT = "The quick brown fox jumps over the lazy dog. " * 60  # ~2 700 chars

_LARGE_AGY_BODY: dict[str, Any] = {
    "project": "test-project-123",
    "model": "gemini-3-flash-agent",
    "request": {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": _REPEAT_UNIT}],
            }
        ]
    },
}

# Minimal SSE payload the handler's _stream_response would return.
_SSE_PAYLOAD = (
    b'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}\r\n\r\ndata: [DONE]\r\n\r\n'
)

# ---------------------------------------------------------------------------
# Helper: build a minimal SSE StreamingResponse suitable for the stub
# ---------------------------------------------------------------------------


def _make_sse_streaming_response() -> StreamingResponse:
    async def _body():  # type: ignore[return]
        yield _SSE_PAYLOAD

    return StreamingResponse(_body(), status_code=200, media_type="text/event-stream")


# ---------------------------------------------------------------------------
# 1. COMPRESSION DELTA — optimization runs on the cloudcode/antigravity path
# ---------------------------------------------------------------------------


def test_compression_delta_on_antigravity_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sufficiently large/redundant body triggers the compression code path.

    We spy on openai_pipeline.apply to confirm it is called at least once,
    confirming the CloudCode/antigravity handler enters the compression branch
    when should_compress=True. The spy wraps the real apply so the return
    value is genuine (no fake result required).

    Note: when monkeypatching an *instance* attribute, the function receives
    no implicit self — use *args/**kwargs to capture the call faithfully.
    """
    call_log: list[dict[str, Any]] = []

    async def _fake_stream(
        proxy_self: Any, url: str, headers: dict, body: dict, *args: Any, **kwargs: Any
    ) -> StreamingResponse:
        return _make_sse_streaming_response()

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    with TestClient(create_app(ProxyConfig(optimize=True))) as client:
        proxy: HeadroomProxy = client.app.state.proxy  # type: ignore[attr-defined]
        real_apply = proxy.openai_pipeline.apply

        # Instance-level patch: function is called without implicit self.
        def _spy_apply(*args: Any, **kwargs: Any) -> Any:
            result = real_apply(*args, **kwargs)
            call_log.append(
                {
                    "tokens_before": result.tokens_before,
                    "tokens_after": result.tokens_after,
                    "transforms": result.transforms_applied,
                }
            )
            return result

        proxy.openai_pipeline.apply = _spy_apply  # type: ignore[method-assign]

        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            headers={"User-Agent": "antigravity/1.0.5"},
            json=_LARGE_AGY_BODY,
        )

    assert response.status_code == 200
    # Pipeline must have been called at least once.
    assert len(call_log) >= 1, "openai_pipeline.apply was never called — compression path not taken"


# ---------------------------------------------------------------------------
# 2. CORRECT HOST — antigravity traffic routes to ANTIGRAVITY_DAILY_API_URL
# ---------------------------------------------------------------------------


def test_antigravity_routes_to_daily_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """antigravity UA → https://daily-cloudcode-pa.googleapis.com target URL."""
    captured: list[str] = []

    async def _fake_stream(proxy_self: Any, url: str, *args: Any, **kwargs: Any) -> JSONResponse:
        captured.append(url)
        return JSONResponse({"url": url})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    with TestClient(create_app(ProxyConfig(optimize=False))) as client:
        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            headers={"User-Agent": "antigravity/1.0.5"},
            json=_LARGE_AGY_BODY,
        )

    assert response.status_code == 200
    assert len(captured) == 1
    assert captured[0].startswith("https://daily-cloudcode-pa.googleapis.com"), (
        f"Expected daily endpoint, got: {captured[0]}"
    )


# ---------------------------------------------------------------------------
# 3. SSE PRESERVED — response Content-Type text/event-stream passes through
# ---------------------------------------------------------------------------


def test_sse_response_content_type_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    """StreamingResponse with text/event-stream is forwarded unchanged."""

    async def _fake_stream(
        proxy_self: Any, url: str, *args: Any, **kwargs: Any
    ) -> StreamingResponse:
        return _make_sse_streaming_response()

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    with TestClient(create_app(ProxyConfig(optimize=False))) as client:
        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            headers={"User-Agent": "antigravity/1.0.5"},
            json=_LARGE_AGY_BODY,
        )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", ""), (
        f"Expected text/event-stream content-type, got: {response.headers.get('content-type')}"
    )


# ---------------------------------------------------------------------------
# 4a. STEALTH — no x-headroom-* headers reach upstream
# ---------------------------------------------------------------------------


def test_stealth_no_x_headroom_headers_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """x-headroom-* headers are stripped before the upstream call (gemini.py:826)."""
    captured_headers: dict[str, str] = {}

    async def _fake_stream(
        proxy_self: Any, url: str, headers: dict, *args: Any, **kwargs: Any
    ) -> JSONResponse:
        captured_headers.update(headers)
        return JSONResponse({"ok": True})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    with TestClient(create_app(ProxyConfig(optimize=False))) as client:
        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            headers={
                "User-Agent": "antigravity/1.0.5",
                "x-headroom-bypass": "true",
                "x-headroom-user-id": "tester",
                "x-headroom-mode": "passthrough",
            },
            json=_LARGE_AGY_BODY,
        )

    assert response.status_code == 200
    x_headroom_keys = [k for k in captured_headers if k.lower().startswith("x-headroom-")]
    assert x_headroom_keys == [], f"x-headroom-* headers leaked to upstream: {x_headroom_keys}"


# ---------------------------------------------------------------------------
# 4b. STEALTH — agy User-Agent is passed through unchanged
# ---------------------------------------------------------------------------


def test_stealth_agy_user_agent_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The agy UA is not rewritten by the handler."""
    captured_headers: dict[str, str] = {}
    _AGY_UA = "antigravity/1.0.5 linux/x86_64"

    async def _fake_stream(
        proxy_self: Any, url: str, headers: dict, *args: Any, **kwargs: Any
    ) -> JSONResponse:
        captured_headers.update(headers)
        return JSONResponse({"ok": True})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    with TestClient(create_app(ProxyConfig(optimize=False))) as client:
        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            headers={"User-Agent": _AGY_UA},
            json=_LARGE_AGY_BODY,
        )

    assert response.status_code == 200
    sent_ua = captured_headers.get("user-agent", "")
    assert sent_ua == _AGY_UA, f"UA was rewritten: expected {_AGY_UA!r}, got {sent_ua!r}"


# ---------------------------------------------------------------------------
# 5. ACCEPT-ENCODING STRIPPED — handler removes it before upstream (gemini.py:817)
# ---------------------------------------------------------------------------


def test_accept_encoding_stripped_before_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Handler pops accept-encoding from headers before calling _stream_response."""
    captured_headers: dict[str, str] = {}

    async def _fake_stream(
        proxy_self: Any, url: str, headers: dict, *args: Any, **kwargs: Any
    ) -> JSONResponse:
        captured_headers.update(headers)
        return JSONResponse({"ok": True})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    with TestClient(create_app(ProxyConfig(optimize=False))) as client:
        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            headers={
                "User-Agent": "antigravity/1.0.5",
                "Accept-Encoding": "gzip, deflate, br",
            },
            json=_LARGE_AGY_BODY,
        )

    assert response.status_code == 200
    assert "accept-encoding" not in {k.lower() for k in captured_headers}, (
        f"accept-encoding reached upstream: {captured_headers}"
    )


# ---------------------------------------------------------------------------
# 6. SINGLE-UPSTREAM-ORIGINATION — _stream_response called exactly once
# ---------------------------------------------------------------------------


def test_single_upstream_origination(monkeypatch: pytest.MonkeyPatch) -> None:
    """_stream_response is called EXACTLY once per request (no duplicate origination)."""
    call_count = 0

    async def _fake_stream(proxy_self: Any, url: str, *args: Any, **kwargs: Any) -> JSONResponse:
        nonlocal call_count
        call_count += 1
        return JSONResponse({"ok": True})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    with TestClient(create_app(ProxyConfig(optimize=False))) as client:
        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            headers={"User-Agent": "antigravity/1.0.5"},
            json=_LARGE_AGY_BODY,
        )

    assert response.status_code == 200
    assert call_count == 1, f"_stream_response called {call_count} times (expected exactly 1)"


# ---------------------------------------------------------------------------
# 7. GZIP REQUEST BODY — handler decodes gzip-encoded JSON body correctly
# ---------------------------------------------------------------------------


def test_gzip_request_body_decoded_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the client sends a gzip-encoded request body, _read_request_json decompresses it.

    _read_request_body_bytes (helpers.py:2689) handles Content-Encoding: gzip.
    We confirm that handle_google_cloudcode_stream successfully parses the body
    (returns 200, not 400) and forwards the correct model to _stream_response.
    """
    captured_body: dict[str, Any] = {}

    async def _fake_stream(
        proxy_self: Any, url: str, headers: dict, body: dict, *args: Any, **kwargs: Any
    ) -> JSONResponse:
        captured_body.update(body)
        return JSONResponse({"ok": True})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    raw_json = json.dumps(_LARGE_AGY_BODY).encode()
    compressed = gzip.compress(raw_json)

    with TestClient(create_app(ProxyConfig(optimize=False))) as client:
        response = client.post(
            "/v1internal:streamGenerateContent",
            params={"alt": "sse"},
            headers={
                "User-Agent": "antigravity/1.0.5",
                "Content-Encoding": "gzip",
                "Content-Type": "application/json",
            },
            content=compressed,
        )

    assert response.status_code == 200, (
        f"Expected 200 for gzip body, got {response.status_code}: {response.text}"
    )
    assert captured_body.get("model") == "gemini-3-flash-agent", (
        f"Body not correctly decoded: model={captured_body.get('model')!r}"
    )


# ---------------------------------------------------------------------------
# 8. AUTH REDACTION on default log path — Bearer + x-goog-api-key must NOT
#    appear in plaintext in default-level logs (caplog).
# ---------------------------------------------------------------------------


def test_auth_not_leaked_in_default_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Authorization and x-goog-api-key values must not appear in headroom logs.

    The handler uses log_outbound_headers (gemini.py:828-832) which only logs
    stripped_count, never header values. This asserts that the default log path
    does not leak secrets for the cloudcode/antigravity handler.
    """
    SECRET_BEARER = "supersecret-bearer-token-xyz789"
    SECRET_API_KEY = "AIzaSyFakeSecret1234567890"

    async def _fake_stream(proxy_self: Any, url: str, *args: Any, **kwargs: Any) -> JSONResponse:
        return JSONResponse({"ok": True})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    with caplog.at_level(logging.DEBUG, logger="headroom"):
        with TestClient(create_app(ProxyConfig(optimize=False))) as client:
            response = client.post(
                "/v1internal:streamGenerateContent",
                params={"alt": "sse"},
                headers={
                    "User-Agent": "antigravity/1.0.5",
                    "Authorization": f"Bearer {SECRET_BEARER}",
                    "x-goog-api-key": SECRET_API_KEY,
                },
                json=_LARGE_AGY_BODY,
            )

    assert response.status_code == 200
    log_text = "\n".join(r.getMessage() for r in caplog.records)
    assert SECRET_BEARER not in log_text, "Bearer token leaked into headroom logs"
    assert SECRET_API_KEY not in log_text, "x-goog-api-key leaked into headroom logs"


# ---------------------------------------------------------------------------
# 9. FAIL-OPEN OBSERVABILITY — pipeline raises → original bytes forwarded,
#    warning logged.
# ---------------------------------------------------------------------------


def test_fail_open_on_compression_pipeline_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If openai_pipeline.apply raises, handler falls through (fail-open) with
    original messages and emits a warning log. _stream_response is still called
    exactly once (original body forwarded, not dropped).

    The production code at gemini.py:882-883:
        except Exception as e:
            logger.warning(f"[{request_id}] Cloud Code Assist optimization failed: {e}")
    ensures the outer _stream_response call still proceeds with original messages.

    We capture the warning via a direct logging.Handler installed on the
    headroom.proxy logger to avoid scope-ordering issues between TestClient's
    event-loop dispatch and pytest caplog's propagation-reset fixture.
    """
    call_count = 0
    upstream_body_received: dict[str, Any] = {}

    async def _fake_stream(
        proxy_self: Any, url: str, headers: dict, body: dict, *args: Any, **kwargs: Any
    ) -> JSONResponse:
        nonlocal call_count
        call_count += 1
        upstream_body_received.update(body)
        return JSONResponse({"ok": True})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    def _exploding_apply(*_args: Any, **_kw: Any) -> None:
        raise RuntimeError("Simulated compression pipeline failure")

    # Direct handler on headroom.proxy so we capture regardless of propagation state.
    warning_messages: list[str] = []

    class _CapturingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                warning_messages.append(record.getMessage())

    proxy_logger = logging.getLogger("headroom.proxy")
    cap_handler = _CapturingHandler()
    proxy_logger.addHandler(cap_handler)
    # Pin the emit logger's own level so WARNING records are enabled regardless
    # of any ancestor level another test left raised (isEnabledFor walks parents).
    prev_level = proxy_logger.level
    proxy_logger.setLevel(logging.WARNING)

    try:
        with TestClient(create_app(ProxyConfig(optimize=True))) as client:
            proxy: HeadroomProxy = client.app.state.proxy  # type: ignore[attr-defined]
            # Direct instance assignment (not monkeypatch.setattr) so the function
            # is stored exactly as given — no implicit self when called.
            proxy.openai_pipeline.apply = _exploding_apply  # type: ignore[method-assign]

            response = client.post(
                "/v1internal:streamGenerateContent",
                params={"alt": "sse"},
                headers={"User-Agent": "antigravity/1.0.5"},
                json=_LARGE_AGY_BODY,
            )
    finally:
        proxy_logger.removeHandler(cap_handler)
        proxy_logger.setLevel(prev_level)

    # Fail-open: must not 500/502; upstream call must proceed.
    assert response.status_code == 200, (
        f"Expected fail-open 200, got {response.status_code}: {response.text}"
    )

    # Upstream called exactly once.
    assert call_count == 1, f"_stream_response called {call_count} times (expected 1)"

    # Original body forwarded (model unchanged).
    assert upstream_body_received.get("model") == "gemini-3-flash-agent", (
        f"Body not forwarded correctly: {upstream_body_received.get('model')!r}"
    )

    # Warning was emitted on the headroom.proxy logger.
    assert any(
        "optimization failed" in msg.lower() or "cloud code assist" in msg.lower()
        for msg in warning_messages
    ), "Expected a warning about compression failure. Got: " + "\n".join(warning_messages)


# ---------------------------------------------------------------------------
# 10. FAIL-OPEN BODY IDENTITY — compression raises → original body forwarded
#     byte-for-byte (no mutation, no gzip, no truncation).
# ---------------------------------------------------------------------------


def test_fail_open_compression_degrades_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compression pipeline raises → request completes successfully (degrade open).

    Complementary to test_fail_open_on_compression_pipeline_exception which
    verifies status-200 + exactly-one upstream call + warning log.  This test
    pins the BODY IDENTITY guarantee: the body forwarded upstream when
    compression explodes is identical to the original request body — no
    tokens modified, no gzip wrapping, no partial writes.

    Also verifies the agy session is not crashed: a second request in the
    same session after the first fail-open also completes with status 200.
    """
    received_bodies: list[dict[str, Any]] = []

    async def _fake_stream(
        proxy_self: Any, url: str, headers: dict, body: dict, *args: Any, **kwargs: Any
    ) -> JSONResponse:
        received_bodies.append(dict(body))
        return JSONResponse({"ok": True})

    monkeypatch.setattr(HeadroomProxy, "_stream_response", _fake_stream)

    def _exploding_apply(*_args: Any, **_kw: Any) -> None:
        raise RuntimeError("Simulated compression pipeline failure — body identity check")

    warning_messages: list[str] = []

    class _CapturingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if record.levelno >= logging.WARNING:
                warning_messages.append(record.getMessage())

    proxy_logger = logging.getLogger("headroom.proxy")
    cap_handler = _CapturingHandler()
    proxy_logger.addHandler(cap_handler)
    # Pin the emit logger's own level so WARNING records are enabled regardless
    # of any ancestor level another test left raised (isEnabledFor walks parents).
    prev_level = proxy_logger.level
    proxy_logger.setLevel(logging.WARNING)

    try:
        with TestClient(create_app(ProxyConfig(optimize=True))) as client:
            proxy: HeadroomProxy = client.app.state.proxy  # type: ignore[attr-defined]
            proxy.openai_pipeline.apply = _exploding_apply  # type: ignore[method-assign]

            # First request — fails compression, must degrade open.
            response1 = client.post(
                "/v1internal:streamGenerateContent",
                params={"alt": "sse"},
                headers={"User-Agent": "antigravity/1.0.5"},
                json=_LARGE_AGY_BODY,
            )

            # Second request in same session — session must still be alive.
            response2 = client.post(
                "/v1internal:streamGenerateContent",
                params={"alt": "sse"},
                headers={"User-Agent": "antigravity/1.0.5"},
                json=_LARGE_AGY_BODY,
            )
    finally:
        proxy_logger.removeHandler(cap_handler)
        proxy_logger.setLevel(prev_level)

    # Both requests must degrade open (200).
    assert response1.status_code == 200, (
        f"First fail-open request must return 200, got {response1.status_code}: {response1.text}"
    )
    assert response2.status_code == 200, (
        f"Second request proves session not crashed; got {response2.status_code}: {response2.text}"
    )

    # Both requests must have reached upstream — session not aborted.
    assert len(received_bodies) == 2, (
        f"Expected 2 upstream calls (one per request); got {len(received_bodies)}"
    )

    # Body identity: every upstream call received the original (uncompressed) body.
    for i, body in enumerate(received_bodies):
        assert body.get("model") == _LARGE_AGY_BODY["model"], (
            f"Request {i + 1}: model field mutated — body identity broken: {body.get('model')!r}"
        )
        assert body.get("project") == _LARGE_AGY_BODY["project"], (
            f"Request {i + 1}: project field mutated — body identity broken: {body.get('project')!r}"
        )
        contents = body.get("request", {}).get("contents", [])
        assert len(contents) == 1, (
            f"Request {i + 1}: contents list mutated — expected 1 item, got {len(contents)}"
        )
        text = contents[0].get("parts", [{}])[0].get("text", "")
        assert text == _REPEAT_UNIT, (
            f"Request {i + 1}: text body mutated or truncated — body identity broken"
        )

    # Fail-open is observable: at least one warning logged per fail.
    assert len(warning_messages) >= 2, (
        f"Expected at least 2 warnings (one per fail-open); got {len(warning_messages)}: "
        + "\n".join(warning_messages)
    )
    for msg in warning_messages:
        assert "optimization failed" in msg.lower() or "cloud code assist" in msg.lower(), (
            f"Warning message does not mention compression failure: {msg!r}"
        )


# ---------------------------------------------------------------------------
# CROSS-AGENT REGRESSION: aider wrap-env byte-identity
#
# test_cli/test_wrap_aider.py already covers the aider env builder:
#   test_wrap_aider_sets_provider_envs asserts OPENAI_API_BASE + ANTHROPIC_BASE_URL
#   and agent_type == "aider".
#
# test_wrap_agy.py covers _inject_ssl_bypass byte-identity for claude:
#   TestInjectSslBypassAgentAware.test_claude_sets_node_tls_reject_unauthorized_0 etc.
#   The aider path uses the same _inject_ssl_bypass code path; adding a separate
#   aider assertion here would duplicate test_cli/test_wrap_aider.py coverage.
#   Recorded as: covered: tests/test_cli/test_wrap_aider.py::test_wrap_aider_sets_provider_envs
# ---------------------------------------------------------------------------
