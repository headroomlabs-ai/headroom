"""F3 handler-integration tests: tenant_key plumbing through the proxy.

Pins that the OpenAI / Anthropic handlers call ``resolve_tenant_key``
once at request entry, populate ``request.state.tenant_key`` /
``request.state.tenant_key_source``, AND set the request-scoped
ContextVar so downstream TOIN ``record_compression`` calls observe the
right tenant slice.

Uses the same ``_DummyOpenAIHandler`` pattern as
``tests/test_openai_codex_routing.py`` so we don't need the Rust core
or a live upstream HTTP server.
"""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace

import anyio
import pytest
from fastapi import Request

from headroom.proxy.handlers.openai import OpenAIHandlerMixin
from headroom.proxy.tenant_key import (
    GLOBAL_TENANT_KEY,
    SOURCE_GLOBAL,
    SOURCE_HEADER,
    get_current_tenant_key,
    set_request_tenant_key,
)


def _jwt(payload: dict) -> str:
    """Build a 3-segment JWT for OAuth classification."""
    header = {"alg": "none", "typ": "JWT"}

    def encode(part: dict) -> str:
        raw = json.dumps(part, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode(header)}.{encode(payload)}."


# ── Test fixtures (Dummy handler) ─────────────────────────────────────────


class _DummyMetrics:
    async def record_request(self, **kwargs):  # noqa: ANN003
        return None

    async def record_failed(self, **kwargs):  # noqa: ANN003
        return None


class _DummyTokenizer:
    def count_messages(self, messages):
        return len(messages)


class _ResponseStub:
    status_code = 200
    headers = {"content-type": "application/json", "content-length": "42"}
    content = b'{"id":"resp_123","output":[{"type":"message"}]}'

    def json(self):
        return {"usage": {"input_tokens": 2, "output_tokens": 1}}


class _DummyOpenAIHandler(OpenAIHandlerMixin):
    """Minimal handler that captures the request that flows through F3."""

    OPENAI_API_URL = "https://api.openai.com"

    def __init__(self) -> None:
        self.rate_limiter = None
        self.metrics = _DummyMetrics()
        self.config = SimpleNamespace(
            optimize=False,
            retry_max_attempts=3,
            retry_base_delay_ms=10,
            retry_max_delay_ms=50,
            connect_timeout_seconds=10,
            openai_extra_headers=None,
        )
        self.usage_reporter = None
        self.openai_provider = SimpleNamespace(get_context_limit=lambda model: 128_000)
        self.openai_pipeline = SimpleNamespace(apply=lambda **kwargs: None)
        self.anthropic_backend = None
        self.cost_tracker = None
        self.memory_handler = None
        self.session_tracker_store = SimpleNamespace(
            compute_session_id=lambda *a, **k: "sess-openai-1",
        )
        self.captured_request: Request | None = None

    async def _next_request_id(self) -> str:
        return "req-1"

    def _extract_tags(self, headers: dict[str, str]) -> dict[str, str]:
        return {}

    async def _retry_request(self, method: str, url: str, headers: dict, body: dict):
        return _ResponseStub()

    async def _run_compression_in_executor(self, fn, *, timeout: float):
        return fn()

    async def _stream_response(self, **kwargs):
        return SimpleNamespace(status_code=200)


def _build_request(headers: dict[str, str], body: dict) -> Request:
    payload = json.dumps(body).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
        "query_string": b"",
        "headers": [
            (key.lower().encode("utf-8"), value.encode("utf-8")) for key, value in headers.items()
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 443),
    }
    return Request(scope, receive)


@pytest.fixture(autouse=True)
def _reset_tenant_key_contextvar():
    """Clean ContextVar after every test so they don't leak."""
    set_request_tenant_key(None)
    yield
    set_request_tenant_key(None)


# ── F3 plumbing pins ──────────────────────────────────────────────────────


def test_handler_populates_request_state_tenant_key_for_header_path(monkeypatch):
    """Header-set tenant_id ⇒ request.state.tenant_key + source="header"."""
    handler = _DummyOpenAIHandler()
    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    request = _build_request(
        headers={
            "Authorization": "Bearer sk-test-key",
            "X-Headroom-Tenant-ID": "cust_acme",
        },
        body={"model": "gpt-5.4", "input": "hi"},
    )

    anyio.run(handler.handle_openai_responses, request)

    assert getattr(request.state, "tenant_key", None) == "cust_acme"
    assert getattr(request.state, "tenant_key_source", None) == SOURCE_HEADER


def test_handler_populates_request_state_tenant_key_for_global_fallback(monkeypatch):
    """No tenant header, even with bearer auth, ⇒ tenant_key is global."""
    handler = _DummyOpenAIHandler()
    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    # PAYG bearer (sk-...). F1 will classify auth_mode=PAYG, but F3
    # requires an explicit tenant header instead of deriving from secrets.
    request = _build_request(
        headers={"Authorization": "Bearer sk-ant-api03-aaaabbbbccccdddd"},
        body={"model": "gpt-5.4", "input": "hi"},
    )

    anyio.run(handler.handle_openai_responses, request)

    assert getattr(request.state, "tenant_key_source", None) == SOURCE_GLOBAL
    tenant_key = getattr(request.state, "tenant_key", None)
    assert tenant_key == GLOBAL_TENANT_KEY


def test_handler_populates_global_when_no_signals(monkeypatch):
    """No header, no bearer ⇒ tenant_key="global" with source="global"."""
    handler = _DummyOpenAIHandler()
    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    # Neither auth nor explicit tenant header.
    request = _build_request(
        headers={},
        body={"model": "gpt-5.4", "input": "hi"},
    )

    anyio.run(handler.handle_openai_responses, request)

    assert getattr(request.state, "tenant_key", None) == GLOBAL_TENANT_KEY
    assert getattr(request.state, "tenant_key_source", None) == SOURCE_GLOBAL


def test_handler_populates_contextvar_so_toin_records_under_tenant(monkeypatch):
    """Handler must call set_request_tenant_key — TOIN reads the ContextVar.

    We verify by asserting the contextvar is populated AFTER the
    handler returns (the handler doesn't reset it on success — that
    would defeat the whole point, since downstream record_compression
    happens deep inside the pipeline). The contextvar is per-task; the
    next request's handler will overwrite it.
    """
    handler = _DummyOpenAIHandler()
    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    request = _build_request(
        headers={"X-Headroom-Tenant-ID": "tenant_for_toin"},
        body={"model": "gpt-5.4", "input": "hi"},
    )

    # Run inside an explicit context so the test can read back the
    # ContextVar the handler set. Without `anyio.run`'s default
    # context isolation, we'd see a stale value across tests.
    async def _drive() -> str:
        await handler.handle_openai_responses(request)
        return get_current_tenant_key()

    seen = anyio.run(_drive)
    assert seen == "tenant_for_toin"
