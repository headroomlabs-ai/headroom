"""Tests that ``proxy_inbound_response`` logs ``response_content_length``.

The inbound response log line previously recorded only timing (``duration_ms``)
and status, making it impossible to correlate slow responses with payload size.
The fix adds ``response_content_length`` read from the Starlette response's
``Content-Length`` header — the same approach the inbound request line uses for
``content_length``.

Contract pinned here:
- response with Content-Length → value appears in the log line
- response without Content-Length (streaming/chunked) → empty string, no crash
"""

from __future__ import annotations

import logging
import re

import pytest

fastapi = pytest.importorskip("fastapi")

from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


def _make_config(**overrides) -> ProxyConfig:
    base = {
        "optimize": False,
        "cache_enabled": False,
        "rate_limit_enabled": False,
        "mode": "token",
    }
    base.update(overrides)
    return ProxyConfig(**base)


_RESPONSE_LOG_RE = re.compile(r"event=proxy_inbound_response\b")


def _get_inbound_response_logs(records: list[logging.LogRecord]) -> list[str]:
    return [r.getMessage() for r in records if _RESPONSE_LOG_RE.search(r.getMessage())]


@pytest.fixture
def capture_logs():
    """Capture headroom.proxy log records via a dedicated handler."""
    records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Handler()
    handler.setLevel(logging.DEBUG)
    lg = logging.getLogger("headroom.proxy")
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        lg.removeHandler(handler)


async def _json_handler(*a, **kw):
    return JSONResponse({"ok": True, "data": "x" * 200})


async def _stream_handler(*a, **kw):
    from starlette.responses import StreamingResponse

    async def _stream():
        yield b'{"ok": true}'
        yield b""

    return StreamingResponse(_stream())


def test_response_content_length_logged(capture_logs):
    """A JSON response sets Content-Length; the value must appear in the log."""
    app = create_app(_make_config())
    with TestClient(app) as client:
        client.app.state.proxy.handle_anthropic_messages = _json_handler
        resp = client.post("/v1/messages", json={"model": "glm-5.2", "messages": []})
        assert resp.status_code == 200

    lines = _get_inbound_response_logs(capture_logs)
    assert lines, "expected at least one proxy_inbound_response log line"
    line = lines[-1]
    assert "response_content_length=" in line
    m = re.search(r"response_content_length=(\d+)", line)
    assert m, f"expected numeric content_length in: {line}"
    assert int(m.group(1)) > 0


def test_streaming_response_no_content_length(capture_logs):
    """A streaming response has no Content-Length header.

    The log line must still appear with an empty ``response_content_length=``
    value — not crash or omit the field.
    """
    app = create_app(_make_config())
    with TestClient(app) as client:
        client.app.state.proxy.handle_anthropic_messages = _stream_handler
        resp = client.post("/v1/messages", json={"model": "glm-5.2", "messages": []})
        assert resp.status_code == 200

    lines = _get_inbound_response_logs(capture_logs)
    assert lines, "expected at least one proxy_inbound_response log line"
    line = lines[-1]
    assert "response_content_length=" in line
    m = re.search(r"response_content_length=(\S*)", line)
    assert m, f"expected response_content_length field in: {line}"
    assert m.group(1) in ("", '""'), f"expected empty for streaming, got: {m.group(1)}"
