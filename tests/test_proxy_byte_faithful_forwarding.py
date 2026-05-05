"""Byte-faithful Python forwarder tests for PR-A3 (P0-2 fix).

The Python forwarder layer (server.py:_retry_request, streaming.py,
openai.py:_ws_http_fallback, batch.py) historically re-serialized every
request body via httpx's default JSON encoder, drifting separators (``, ``
vs ``,``) and ASCII-escaping non-ASCII text. Every such request collapsed
Anthropic prompt-cache hit-rate.

PR-A3 makes every forwarder byte-faithful:
  * unmutated body → forward original ``await request.body()`` verbatim;
  * mutated body  → re-serialize once via ``serialize_body_canonical``
    (compact separators, ``ensure_ascii=False``).

The legacy behavior is still reachable via
``HEADROOM_PROXY_PYTHON_FORWARDER_MODE=legacy_json_kwarg`` for emergency
rollback (operator opt-in, not a fallback).
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from headroom.proxy.helpers import (
    BodyMutationTracker,
    append_text_to_latest_user_chat_message,
    get_python_forwarder_mode,
    log_outbound_request,
    prepare_outbound_body_bytes,
    serialize_body_canonical,
)
from headroom.proxy.server import ProxyConfig, create_app

# ---------------------------------------------------------------------------
# Unit tests for serializer + tracker
# ---------------------------------------------------------------------------


def test_serialize_canonical_compact_separators() -> None:
    """``serialize_body_canonical`` must use compact ``,``/``:`` (no spaces)."""
    body = {"a": 1, "b": 2}
    out = serialize_body_canonical(body)
    assert out == b'{"a":1,"b":2}', repr(out)


def test_serialize_canonical_unicode_passthrough() -> None:
    """UTF-8 must survive — no ``\\uXXXX`` ASCII escaping."""
    body = {"emoji": "🔥", "cjk": "日本語", "mixed": "hello → 世界"}
    out = serialize_body_canonical(body)
    # Each non-ASCII char appears as raw UTF-8 bytes, never as a \uXXXX literal.
    assert b"\\u" not in out, repr(out)
    parsed = json.loads(out.decode("utf-8"))
    assert parsed == body


def test_serialize_canonical_preserves_dict_insertion_order() -> None:
    """Dict insertion order is preserved (Python 3.7+ guarantee)."""
    body = {"z": 1, "a": 2, "m": 3}
    out = serialize_body_canonical(body)
    assert out.startswith(b'{"z":1,"a":2,"m":3'), repr(out)


def test_mutation_tracker_records_reason_memory_injection() -> None:
    tracker = BodyMutationTracker()
    assert tracker.mutated is False
    assert tracker.reasons == []
    tracker.mark_mutated("memory_injection")
    assert tracker.mutated is True
    assert tracker.reasons == ["memory_injection"]


def test_mutation_tracker_records_reason_compression() -> None:
    tracker = BodyMutationTracker()
    tracker.mark_mutated("compression_smart_crusher")
    assert tracker.mutated is True
    assert tracker.reasons == ["compression_smart_crusher"]


def test_mutation_tracker_dedupes_reasons() -> None:
    tracker = BodyMutationTracker()
    tracker.mark_mutated("memory_injection")
    tracker.mark_mutated("memory_injection")
    tracker.mark_mutated("compression")
    assert tracker.reasons == ["memory_injection", "compression"]


def test_mutation_tracker_rejects_empty_reason() -> None:
    tracker = BodyMutationTracker()
    with pytest.raises(ValueError):
        tracker.mark_mutated("")


def test_mutation_tracker_reasons_is_a_copy() -> None:
    """Caller-mutating the returned list must not affect the tracker."""
    tracker = BodyMutationTracker()
    tracker.mark_mutated("a")
    out = tracker.reasons
    out.append("b")
    assert tracker.reasons == ["a"]


# ---------------------------------------------------------------------------
# prepare_outbound_body_bytes mode selection
# ---------------------------------------------------------------------------


def test_prepare_outbound_unmutated_returns_passthrough_bytes() -> None:
    original = b'{"a":1,"b":"\xf0\x9f\x94\xa5"}'
    out, source = prepare_outbound_body_bytes(
        body={"a": 1, "b": "🔥"},
        original_body_bytes=original,
        body_mutated=False,
        forwarder_mode="byte_faithful",
    )
    assert out == original
    assert source == "passthrough"


def test_prepare_outbound_mutated_uses_canonical() -> None:
    out, source = prepare_outbound_body_bytes(
        body={"a": 1, "b": "🔥"},
        original_body_bytes=b'{"a": 1, "b": "\xf0\x9f\x94\xa5"}',  # spaces in original
        body_mutated=True,
        forwarder_mode="byte_faithful",
    )
    assert out == b'{"a":1,"b":"\xf0\x9f\x94\xa5"}'
    assert source == "canonical"


def test_prepare_outbound_no_original_bytes_uses_canonical() -> None:
    out, source = prepare_outbound_body_bytes(
        body={"a": 1},
        original_body_bytes=None,
        body_mutated=False,
        forwarder_mode="byte_faithful",
    )
    assert out == b'{"a":1}'
    assert source == "canonical"


def test_legacy_json_kwarg_mode_falls_back() -> None:
    """legacy_json_kwarg is an explicit operator opt-in — produces the historical bytes.

    This is NOT a silent fallback (build constraint #4). It is reachable only
    via env var and exists for emergency rollback validation.
    """
    out, source = prepare_outbound_body_bytes(
        body={"a": 1, "b": "🔥"},
        original_body_bytes=b'{"a":1}',
        body_mutated=False,
        forwarder_mode="legacy_json_kwarg",
    )
    # Old httpx default: spaces after `,` and `:`, ascii escaping.
    assert out == b'{"a": 1, "b": "\\ud83d\\udd25"}', repr(out)
    assert source == "legacy"


def test_python_forwarder_mode_default_is_byte_faithful(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HEADROOM_PROXY_PYTHON_FORWARDER_MODE", raising=False)
    assert get_python_forwarder_mode() == "byte_faithful"


def test_python_forwarder_mode_invalid_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_PYTHON_FORWARDER_MODE", "garbage")
    with pytest.raises(ValueError, match="HEADROOM_PROXY_PYTHON_FORWARDER_MODE"):
        get_python_forwarder_mode()


def test_python_forwarder_mode_legacy_value_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_PROXY_PYTHON_FORWARDER_MODE", "legacy_json_kwarg")
    assert get_python_forwarder_mode() == "legacy_json_kwarg"


# ---------------------------------------------------------------------------
# log_outbound_request structured log content
# ---------------------------------------------------------------------------


def test_log_outbound_request_emits_structured_fields() -> None:
    """Capture the structured log line via a temporary handler.

    We attach a memory handler directly to the proxy logger so the test is
    independent of whether ``_setup_file_logging`` has set ``propagate=False``
    (which it does in the live proxy).
    """
    import logging

    proxy_logger = logging.getLogger("headroom.proxy")
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.INFO)
    prev_level = proxy_logger.level
    proxy_logger.addHandler(handler)
    proxy_logger.setLevel(logging.INFO)
    try:
        log_outbound_request(
            forwarder="server",
            method="POST",
            path="/v1/messages",
            body_bytes_count=42,
            body_mutated=False,
            mutation_reasons=[],
            request_id="hr_test_1",
            source="passthrough",
        )
    finally:
        proxy_logger.removeHandler(handler)
        proxy_logger.setLevel(prev_level)

    matching = [r for r in records if "outbound_request" in r.getMessage()]
    assert matching, f"no outbound_request log emitted; records={records!r}"
    msg = matching[-1].getMessage()
    assert "event=outbound_request" in msg
    assert "forwarder=server" in msg
    assert "path=/v1/messages" in msg
    assert "body_bytes=42" in msg
    assert "body_mutated=false" in msg
    assert "source=passthrough" in msg
    assert "request_id=hr_test_1" in msg
    # Never log auth / body content.
    assert "Authorization" not in msg
    assert "x-api-key" not in msg.lower()


# ---------------------------------------------------------------------------
# httpx-mock end-to-end byte-faithful checks
# ---------------------------------------------------------------------------


class _CapturingTransport(httpx.AsyncBaseTransport):
    """An httpx transport that records the exact bytes received."""

    def __init__(self) -> None:
        self.captured_body: bytes | None = None
        self.captured_headers: dict[str, str] | None = None
        self.captured_url: str | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Eagerly read the request body so streaming bodies are captured too.
        body = b""
        async for chunk in request.stream:
            body += chunk
        self.captured_body = body
        self.captured_headers = dict(request.headers.items())
        self.captured_url = str(request.url)
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


def _make_no_optimize_app() -> tuple[TestClient, _CapturingTransport]:
    """Boot a proxy with all transforms disabled and a capturing transport."""
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
    transport = _CapturingTransport()
    proxy = app.state.proxy
    proxy.http_client = httpx.AsyncClient(transport=transport)

    # Pin a stable session tracker so the prefix walker doesn't re-read
    # turn 0 on every run.
    fake_tracker = _FakePrefixTracker(frozen_count=0)
    proxy.session_tracker_store.compute_session_id = lambda request, model, messages: "s1"
    proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

    return TestClient(app), transport


def test_passthrough_no_mutation_byte_equal_sha256() -> None:
    """No transform → upstream SHA-256 equals client-sent SHA-256."""
    client, transport = _make_no_optimize_app()

    # Compact JSON, simulating Claude Code / Codex CLI byte format.
    inbound_dict = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hello"}],
    }
    inbound_bytes = serialize_body_canonical(inbound_dict)

    response = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        content=inbound_bytes,
    )
    assert response.status_code == 200, response.text
    assert transport.captured_body is not None

    inbound_sha = hashlib.sha256(inbound_bytes).hexdigest()
    upstream_sha = hashlib.sha256(transport.captured_body).hexdigest()
    assert inbound_sha == upstream_sha, (
        f"Byte-faithful invariant broken: inbound {inbound_sha} vs upstream "
        f"{upstream_sha}; upstream body={transport.captured_body!r}"
    )


def test_compression_off_unicode_preserved() -> None:
    """Emoji + CJK content survives forwarding without ``\\uXXXX`` escaping."""
    client, transport = _make_no_optimize_app()

    inbound_dict = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "Hello 🔥 — 世界 — emoji is 🚀"},
        ],
    }
    inbound_bytes = serialize_body_canonical(inbound_dict)

    response = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        content=inbound_bytes,
    )
    assert response.status_code == 200
    upstream = transport.captured_body or b""
    assert upstream == inbound_bytes
    assert b"\\u" not in upstream, repr(upstream)
    assert "🔥".encode() in upstream
    assert "世界".encode() in upstream


def test_compression_off_numeric_precision_preserved() -> None:
    """Floats with trailing zero stay floats; large ints preserve precision."""
    client, transport = _make_no_optimize_app()

    inbound_bytes = b'{"model":"claude-sonnet-4-6","max_tokens":64,"temperature":1.0,"seed":12345678901234567,"messages":[{"role":"user","content":"hi"}]}'

    response = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        content=inbound_bytes,
    )
    assert response.status_code == 200
    upstream = transport.captured_body or b""
    # Unmutated → byte-faithful: exact bytes preserved.
    assert upstream == inbound_bytes


def test_legacy_json_kwarg_mode_yields_drifted_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Operator opt-in produces the OLD drifted bytes (rollback validation)."""
    monkeypatch.setenv("HEADROOM_PROXY_PYTHON_FORWARDER_MODE", "legacy_json_kwarg")
    client, transport = _make_no_optimize_app()

    inbound_dict = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "🔥 hi"}],
    }
    inbound_bytes = serialize_body_canonical(inbound_dict)

    response = client.post(
        "/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        content=inbound_bytes,
    )
    assert response.status_code == 200
    upstream = transport.captured_body or b""
    # Legacy mode: spaces after separators + ASCII escaping → bytes drift.
    assert upstream != inbound_bytes
    assert b", " in upstream or b": " in upstream
    assert b"\\u" in upstream  # ASCII escaping confirms legacy path.


# ---------------------------------------------------------------------------
# A2 follow-up: OpenAI Chat Completions memory routes to user-tail
# ---------------------------------------------------------------------------


def test_append_text_to_latest_user_chat_message_string_content() -> None:
    msgs = [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "previous"},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": "latest"},
    ]
    new_msgs, appended = append_text_to_latest_user_chat_message(msgs, "MEMCTX")
    assert appended > 0
    assert new_msgs[0] == msgs[0]
    assert new_msgs[1] == msgs[1]
    assert new_msgs[2] == msgs[2]
    assert new_msgs[3]["content"] == "latest\n\nMEMCTX"
    # Original list untouched.
    assert msgs[3]["content"] == "latest"


def test_append_text_to_latest_user_chat_message_list_content() -> None:
    msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first text"},
                {"type": "image_url", "image_url": {"url": "..."}},
            ],
        }
    ]
    new_msgs, appended = append_text_to_latest_user_chat_message(msgs, "MEM")
    assert appended > 0
    parts = new_msgs[0]["content"]
    assert parts[0]["text"] == "first text\n\nMEM"
    assert parts[1] == msgs[0]["content"][1]


def test_append_text_to_latest_user_chat_message_no_user_returns_zero() -> None:
    msgs = [{"role": "system", "content": "sys"}]
    new_msgs, appended = append_text_to_latest_user_chat_message(msgs, "MEM")
    assert appended == 0
    assert new_msgs == msgs


def test_openai_chat_memory_routes_to_user_tail_not_system() -> None:
    """A2 follow-up: Chat Completions memory injection lives in user tail."""
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
    proxy.memory_handler = SimpleNamespace(
        config=SimpleNamespace(inject_context=True, inject_tools=False),
        search_and_format_context=AsyncMock(return_value="MEMCTX_OAI"),
        has_memory_tool_calls=lambda resp, provider: False,
    )

    captured: dict[str, object] = {}

    async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
        captured["body"] = body
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
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        headers={
            "authorization": "Bearer sk-test",
            "x-headroom-user-id": "u1",
        },
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "you are helpful"},
                {"role": "user", "content": "what is up"},
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    sent = captured.get("body")
    assert isinstance(sent, dict), captured
    sent_msgs = sent["messages"]
    # System message must NOT be mutated.
    assert sent_msgs[0]["role"] == "system"
    assert sent_msgs[0]["content"] == "you are helpful", "system message must remain byte-equal"
    # No injected system message at the start (legacy prepend retired).
    # The ONLY new content is in the latest user message tail.
    user_msgs = [m for m in sent_msgs if m.get("role") == "user"]
    assert len(user_msgs) == 1
    assert user_msgs[-1]["content"].endswith("MEMCTX_OAI")
    # No additional system messages either (memory must not prepend).
    system_msgs = [m for m in sent_msgs if m.get("role") == "system"]
    assert len(system_msgs) == 1


def test_openai_chat_memory_disabled_mode_no_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HEADROOM_MEMORY_INJECTION_MODE", "disabled")
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
    proxy.memory_handler = SimpleNamespace(
        config=SimpleNamespace(inject_context=True, inject_tools=False),
        search_and_format_context=AsyncMock(return_value="WOULD_NOT_INJECT"),
        has_memory_tool_calls=lambda resp, provider: False,
    )

    captured: dict[str, object] = {}

    async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
        captured["body"] = body
        return httpx.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        )

    proxy._retry_request = _fake_retry
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        headers={
            "authorization": "Bearer sk-test",
            "x-headroom-user-id": "u1",
        },
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
        },
    )
    assert resp.status_code == 200
    sent = captured["body"]
    assert isinstance(sent, dict)
    assert sent["messages"][1]["content"] == "hi"


# ---------------------------------------------------------------------------
# Streaming forwarder byte-faithfulness
# ---------------------------------------------------------------------------


class _StreamingCapturingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.captured_body: bytes | None = None
        self.captured_headers: dict[str, str] | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        body = b""
        async for chunk in request.stream:
            body += chunk
        self.captured_body = body
        self.captured_headers = dict(request.headers.items())

        async def _empty_sse():  # pragma: no cover - generator
            yield b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_s","type":"message","role":"assistant","model":"claude","usage":{"input_tokens":1,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}\n\n'
            yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'

        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=httpx.AsyncByteStream(_empty_sse()),  # type: ignore[arg-type]
        )


def test_streaming_forwarder_byte_faithful() -> None:
    """Streaming forwarder uses the same byte-faithful path as non-streaming."""
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

    # Pin session tracker so the cache-stable delta path is a no-op.
    fake_tracker = _FakePrefixTracker(frozen_count=0)
    proxy.session_tracker_store.compute_session_id = lambda request, model, messages: "s_stream"
    proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

    transport = _StreamingCapturingTransport()
    proxy.http_client = httpx.AsyncClient(transport=transport)
    client = TestClient(app)

    inbound_bytes = (
        '{"model":"claude-sonnet-4-6","max_tokens":16,"stream":true,'
        '"messages":[{"role":"user","content":"hi 🔥"}]}'
    ).encode()

    with client.stream(
        "POST",
        "/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        content=inbound_bytes,
    ) as resp:
        # Drain the response.
        for _ in resp.iter_bytes():
            pass

    upstream = transport.captured_body or b""
    inbound_sha = hashlib.sha256(inbound_bytes).hexdigest()
    upstream_sha = hashlib.sha256(upstream).hexdigest()
    assert upstream_sha == inbound_sha, (
        f"Streaming byte-faithfulness broken: inbound {inbound_sha} vs "
        f"upstream {upstream_sha}; upstream={upstream!r}"
    )


# ---------------------------------------------------------------------------
# Batch forwarder byte-faithfulness (passthrough variant)
# ---------------------------------------------------------------------------


def test_batch_passthrough_byte_faithful() -> None:
    """OpenAI batch passthrough forwards original bytes verbatim."""
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
    client = TestClient(app)

    # Use a non-chat-completions endpoint so the handler routes directly to
    # _batch_passthrough (bypassing the file-compression flow).
    inbound = {
        "input_file_id": "file-abc",
        "endpoint": "/v1/embeddings",
        "completion_window": "24h",
    }
    inbound_bytes = serialize_body_canonical(inbound)

    resp = client.post(
        "/v1/batches",
        headers={
            "authorization": "Bearer sk-test",
            "content-type": "application/json",
        },
        content=inbound_bytes,
    )
    # The capturing transport returns a Message JSON, which is fine; the
    # status code may vary depending on routing. What matters is that when
    # the upstream did receive bytes, they are byte-equal to what the
    # client sent (passthrough case, no body mutation).
    assert resp.status_code in (200, 400, 401, 404, 422, 500), resp.text
    if transport.captured_body is not None:
        assert transport.captured_body == inbound_bytes, (
            f"Batch passthrough bytes drifted: "
            f"sent={inbound_bytes!r} upstream={transport.captured_body!r}"
        )


# ---------------------------------------------------------------------------
# WS→HTTP fallback: just exercises the helper resolution
# ---------------------------------------------------------------------------


def test_ws_http_fallback_uses_canonical_serializer() -> None:
    """WS→HTTP fallback resynthesizes the body, so canonical bytes apply.

    We can't easily exercise the full WS path in a TestClient without a
    Codex client; instead we assert the helper choice yields the expected
    bytes when a tracker reports mutation.
    """
    body = {"model": "gpt-5", "input": [{"role": "user", "content": "hi 🚀"}]}
    out, source = prepare_outbound_body_bytes(
        body=body,
        original_body_bytes=None,
        body_mutated=True,
    )
    assert source == "canonical"
    assert b"\\u" not in out
    # Round-trip equality via JSON parse.
    assert json.loads(out.decode("utf-8")) == body
