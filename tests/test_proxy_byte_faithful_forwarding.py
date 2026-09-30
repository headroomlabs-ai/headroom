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

import gzip
import hashlib
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.pipeline import PipelineStage
from headroom.proxy.body_forwarding import (
    BodyMutationTracker,
    OutboundBody,
    get_python_forwarder_mode,
    outbound_body_is_client_bytes,
    prepare_outbound_body_bytes,
    select_outbound_body,
    serialize_body_canonical,
    thinking_blocks_survived_mutation,
)
from headroom.proxy.helpers import (
    _reset_session_beta_tracker_for_test,
    append_text_to_latest_user_chat_message,
    get_session_beta_tracker,
    log_outbound_request,
)
from headroom.proxy.server import ProxyConfig, create_app

pytest.importorskip("fastapi")


@pytest.fixture(autouse=True)
def _disable_output_shaper(monkeypatch: pytest.MonkeyPatch) -> None:
    # Isolate this suite from the opt-in HEADROOM_OUTPUT_SHAPER a developer shell
    # may export, which otherwise perturbs the byte-faithful assertions.
    monkeypatch.delenv("HEADROOM_OUTPUT_SHAPER", raising=False)


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


def test_select_outbound_body_returns_value_object() -> None:
    original = b'{"a":1}'
    outbound = select_outbound_body(
        body={"a": 1},
        original_body_bytes=original,
        body_mutated=False,
        forwarder_mode="byte_faithful",
    )
    assert outbound == OutboundBody(content=original, source="passthrough")


def test_helpers_preserve_body_forwarding_compatibility_exports() -> None:
    from headroom.proxy import helpers

    assert helpers.BodyMutationTracker is BodyMutationTracker
    assert helpers.get_python_forwarder_mode is get_python_forwarder_mode
    assert helpers.prepare_outbound_body_bytes is prepare_outbound_body_bytes
    assert helpers.serialize_body_canonical is serialize_body_canonical


def test_prepare_outbound_mutated_uses_canonical() -> None:
    out, source = prepare_outbound_body_bytes(
        body={"a": 1, "b": "🔥"},
        original_body_bytes=b'{"a": 1, "b": "\xf0\x9f\x94\xa5"}',  # spaces in original
        body_mutated=True,
        forwarder_mode="byte_faithful",
    )
    assert out == b'{"a":1,"b":"\xf0\x9f\x94\xa5"}'
    assert source == "canonical"


@pytest.mark.parametrize("block_type", ["thinking", "redacted_thinking"])
def test_signed_thinking_history_with_original_bytes_uses_passthrough(
    block_type: str,
) -> None:
    body = {
        "model": "claude-sonnet-4-5",
        "messages": [
            {"role": "user", "content": "Solve this"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": block_type,
                        "thinking": "private reasoning",
                        "signature": "sig123",
                    },
                    {"type": "text", "text": "The answer is 42."},
                ],
            },
            {"role": "user", "content": "Continue"},
        ],
    }
    original = json.dumps(body, indent=2).encode("utf-8")
    # The lock triggers on a MODIFIED thinking block, not merely a present one:
    # the signature covers the block's own content, so a body whose blocks are
    # untouched has no seal to break. Tamper with the block so this test still
    # exercises the guard it was written for.
    body["messages"][1]["content"][0]["thinking"] = "rewritten by a transform"

    outbound = select_outbound_body(
        body=body,
        original_body_bytes=original,
        body_mutated=True,
        forwarder_mode="byte_faithful",
    )

    assert outbound.source == "passthrough"
    assert outbound.content == original


def test_signed_thinking_history_without_original_bytes_uses_canonical() -> None:
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "private reasoning",
                        "signature": "sig123",
                    }
                ],
            }
        ]
    }

    outbound = select_outbound_body(
        body=body,
        original_body_bytes=None,
        body_mutated=True,
        forwarder_mode="byte_faithful",
    )

    assert outbound.source == "canonical"
    assert outbound.content == serialize_body_canonical(body)


def test_signed_thinking_history_overrides_legacy_encoder() -> None:
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "signature": "sig123"}],
            }
        ]
    }
    original = json.dumps(body, indent=2).encode("utf-8")
    # Modified block => the lock engages and still outranks the legacy encoder.
    # (An UNmodified block no longer overrides legacy mode: with every seal
    # provably intact there is nothing for the override to protect, and honoring
    # the operator's explicit rollback request is the more useful behaviour.)
    body["messages"][0]["content"][0]["signature"] = "tampered"

    outbound = select_outbound_body(
        body=body,
        original_body_bytes=original,
        body_mutated=True,
        forwarder_mode="legacy_json_kwarg",
    )

    assert outbound == OutboundBody(content=original, source="passthrough", dropped_mutations=True)


def test_signed_thinking_passthrough_reports_the_mutations_it_discarded() -> None:
    """Passthrough silently winning over a mutated body is what hid #2952."""
    body = {
        "stream": False,
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "signature": "sig123"}],
            }
        ],
    }
    original = json.dumps({**body, "stream": True}).encode("utf-8")
    # A thinking block the transforms did NOT touch no longer forces
    # passthrough, so the CCR stream flip in this very scenario now reaches
    # upstream — which is the outcome #2952 wanted in the first place. The
    # discard-reporting path still has to work when a block really was edited,
    # so tamper with it here and keep guarding that.
    body["messages"][0]["content"][0]["signature"] = "rewritten"

    outbound = select_outbound_body(
        body=body,
        original_body_bytes=original,
        body_mutated=True,
        forwarder_mode="byte_faithful",
        mutation_reasons=["ccr_streaming_retrieve_buffered_non_stream"],
    )

    assert outbound.source == "passthrough"
    assert outbound.dropped_mutations is True
    assert outbound.dropped_mutation_reasons == ("ccr_streaming_retrieve_buffered_non_stream",)


def test_original_signed_thinking_still_locks_when_mutation_removed_the_block() -> None:
    original_body = {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "signature": "sig123"}],
            }
        ]
    }
    mutated_body = {"messages": [{"role": "assistant", "content": "rewritten"}]}
    original = json.dumps(original_body, indent=2).encode()

    outbound = select_outbound_body(
        body=mutated_body,
        original_body_bytes=original,
        body_mutated=True,
        forwarder_mode="byte_faithful",
        mutation_reasons=["compression"],
    )

    assert outbound.content == original
    assert outbound.source == "passthrough"
    assert outbound.dropped_mutation_reasons == ("compression",)


def test_signed_thinking_passthrough_reports_nothing_when_body_unmutated() -> None:
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "signature": "sig123"}],
            }
        ]
    }
    original = json.dumps(body).encode("utf-8")

    outbound = select_outbound_body(
        body=body,
        original_body_bytes=original,
        body_mutated=False,
        forwarder_mode="byte_faithful",
        mutation_reasons=["irrelevant"],
    )

    assert outbound.source == "passthrough"
    assert outbound.dropped_mutations is False
    assert outbound.dropped_mutation_reasons == ()


def test_canonical_path_reports_no_dropped_mutations() -> None:
    body = {"messages": [{"role": "user", "content": "hi"}]}

    outbound = select_outbound_body(
        body=body,
        original_body_bytes=b'{"messages": []}',
        body_mutated=True,
        forwarder_mode="byte_faithful",
        mutation_reasons=["compression"],
    )

    assert outbound.source == "canonical"
    assert outbound.dropped_mutations is False
    assert outbound.dropped_mutation_reasons == ()


@pytest.mark.parametrize(
    ("original_body_bytes", "expected"),
    [(b'{"messages": []}', True), (None, False)],
)
def test_outbound_body_is_client_bytes_matches_selection(
    original_body_bytes: bytes | None, expected: bool
) -> None:
    """Handlers gate on this before mutating a body for their own upstream call."""
    body = {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "thinking", "signature": "sig123"}],
            }
        ]
    }

    assert (
        outbound_body_is_client_bytes(body=body, original_body_bytes=original_body_bytes)
        is expected
    )
    outbound = select_outbound_body(
        body=body,
        original_body_bytes=original_body_bytes,
        body_mutated=True,
        forwarder_mode="byte_faithful",
    )
    assert (outbound.source == "passthrough") is expected


def test_outbound_body_is_client_bytes_false_without_thinking_blocks() -> None:
    assert (
        outbound_body_is_client_bytes(
            body={"messages": [{"role": "user", "content": "hi"}]},
            original_body_bytes=b'{"messages": []}',
        )
        is False
    )


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


class _SortedEmptyToolsPreSendExtension:
    def on_pipeline_event(self, event):  # noqa: ANN001
        if event.stage is PipelineStage.PRE_SEND:
            event.tools = []
        return None


def _make_anthropic_app(*, optimize: bool) -> tuple[TestClient, _CapturingTransport]:
    """Boot an Anthropic proxy with a capturing transport."""
    config = ProxyConfig(
        optimize=optimize,
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


def _make_no_optimize_app() -> tuple[TestClient, _CapturingTransport]:
    """Boot a proxy with all transforms disabled and a capturing transport."""
    return _make_anthropic_app(optimize=False)


def test_signed_thinking_discarded_mutation_uses_wire_truth_for_all_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exercised with the relaxation DISABLED, which is both the documented
    # rollback and the pre-existing behaviour. Keeping #3015's wire-truth
    # assertions under the kill switch proves two things at once: the accounting
    # neutralisation still works whenever the lock does engage, and the env-var
    # rollback really is a complete restoration rather than a partial one.
    monkeypatch.setenv("HEADROOM_THINKING_PRESERVING_MUTATIONS", "0")
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
    proxy._record_request_outcome = AsyncMock(wraps=proxy._record_request_outcome)

    tracker = _FakePrefixTracker(frozen_count=0)
    proxy.session_tracker_store.compute_session_id = lambda request, model, messages: "signed"
    proxy.session_tracker_store.get_or_create = lambda session_id, provider: tracker

    inbound = {
        "model": "claude-opus-5",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "Solve this."},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "private",
                        "signature": "sig123",
                    },
                    {"type": "text", "text": "Working."},
                ],
            },
            {"role": "user", "content": "Continue."},
        ],
        "tools": [
            {
                "name": "lookup",
                "description": "  Look up a value.  ",
                "input_schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                },
            }
        ],
    }
    inbound_bytes = json.dumps(inbound, indent=2).encode()

    response = TestClient(app).post(
        "/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        content=inbound_bytes,
    )

    assert response.status_code == 200
    assert transport.captured_body == inbound_bytes
    assert response.headers["x-headroom-tokens-saved"] == "0"
    assert "x-headroom-transforms" not in response.headers

    outcome = proxy._record_request_outcome.await_args.args[0]
    assert outcome.tokens_saved == 0
    assert outcome.optimized_tokens == outcome.original_tokens
    assert outcome.transforms_applied == ()
    assert outcome.tags["wire_mutations_discarded"] > 0
    assert "anthropic:tool_schema_compaction" not in outcome.transforms_applied
    assert "tool_search_deferred_tokens" not in outcome.tags
    assert outcome.tags.get("_headroom_savings_attribution") == []
    assert proxy.metrics.tokens_saved_total == 0
    assert proxy.metrics.tool_search_saved_total == 0
    assert tracker._last_forwarded_messages[: len(inbound["messages"])] == inbound["messages"]


def test_untouched_thinking_lets_tool_compaction_reach_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end counterpart: the same request, with the relaxation active.

    This is the whole point of the change. Tool schemas are a top-level field --
    not inside ``messages`` at all, so no per-block thinking signature can
    possibly cover them -- yet the blanket lock discarded their compaction on
    every turn of a thinking-bearing session. Here the compaction must reach
    upstream AND be credited, while the thinking block goes out untouched.
    """
    monkeypatch.delenv("HEADROOM_THINKING_PRESERVING_MUTATIONS", raising=False)
    config = ProxyConfig(
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
    proxy._record_request_outcome = AsyncMock(wraps=proxy._record_request_outcome)

    tracker = _FakePrefixTracker(frozen_count=0)
    proxy.session_tracker_store.compute_session_id = lambda request, model, messages: "signed"
    proxy.session_tracker_store.get_or_create = lambda session_id, provider: tracker

    signed_block = {"type": "thinking", "thinking": "private", "signature": "sig123"}
    inbound = {
        "model": "claude-opus-5",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "Solve this."},
            {
                "role": "assistant",
                "content": [dict(signed_block), {"type": "text", "text": "Working."}],
            },
            {"role": "user", "content": "Continue."},
        ],
        "tools": [
            {
                "name": "lookup",
                "description": "  Look up a value.  ",
                "input_schema": {
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "title": "LookupArgs",
                    "type": "object",
                    "properties": {"q": {"type": "string", "title": "Q"}},
                },
            }
        ],
    }
    inbound_bytes = json.dumps(inbound, indent=2).encode()

    response = TestClient(app).post(
        "/v1/messages",
        headers={
            "x-api-key": "test-key",
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        content=inbound_bytes,
    )

    assert response.status_code == 200
    # The edit shipped: the annotation keys the compaction strips are gone.
    assert transport.captured_body != inbound_bytes
    wire = json.loads(transport.captured_body)
    assert "$schema" not in wire["tools"][0]["input_schema"]
    assert "title" not in wire["tools"][0]["input_schema"]
    # ...and the seal went out byte-identical, which is what makes it safe.
    assert wire["messages"][1]["content"][0] == signed_block

    outcome = proxy._record_request_outcome.await_args.args[0]
    assert "wire_mutations_discarded" not in outcome.tags


def _openai_responses_body_bytes(*, stream: bool) -> bytes:
    payload = {
        "model": "gpt-5.5",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "hello 🔥 with spaces preserved",
                    }
                ],
            }
        ],
        "stream": stream,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _openai_responses_codex_headers(content_encoding: str) -> dict[str, str]:
    return {
        "authorization": "Bearer test-token",
        "chatgpt-account-id": "acct_test",
        "originator": "Codex Desktop",
        "content-type": "application/json",
        "content-encoding": content_encoding,
        "accept": "text/event-stream",
    }


def _start_proxy_log_capture() -> tuple[
    logging.Logger,
    logging.Handler,
    int,
    list[logging.LogRecord],
]:
    proxy_logger = logging.getLogger("headroom.proxy")
    records: list[logging.LogRecord] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _ListHandler(level=logging.INFO)
    prev_level = proxy_logger.level
    proxy_logger.addHandler(handler)
    proxy_logger.setLevel(logging.INFO)
    return proxy_logger, handler, prev_level, records


def _stop_proxy_log_capture(
    proxy_logger: logging.Logger,
    handler: logging.Handler,
    prev_level: int,
) -> None:
    proxy_logger.removeHandler(handler)
    proxy_logger.setLevel(prev_level)


def _assert_openai_responses_encoded_passthrough(
    transport: _CapturingTransport,
    decoded_body: bytes,
) -> None:
    assert transport.captured_body == decoded_body
    assert transport.captured_headers is not None
    captured_headers = {key.lower(): value for key, value in transport.captured_headers.items()}
    assert "content-encoding" not in captured_headers
    assert captured_headers.get("content-length") == str(len(decoded_body))


def _assert_outbound_passthrough_log(
    records: list[logging.LogRecord],
    *,
    forwarder: str,
) -> None:
    messages = [record.getMessage() for record in records]
    assert any(
        "event=outbound_request" in message
        and f"forwarder={forwarder}" in message
        and "body_mutated=false" in message
        and "source=passthrough" in message
        for message in messages
    ), messages


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


# Forward coverage only; the PRE_SEND case below is the base-fails proof for this fix.
def test_anthropic_tools_canonical_order_preserves_byte_faithful_request() -> None:
    client, transport = _make_no_optimize_app()
    inbound_dict = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "plan test"}],
        "tools": [
            {"name": "alpha"},
            {"name": "zeta", "description": "later"},
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
    assert response.status_code == 200, response.text
    upstream = transport.captured_body or b""
    assert upstream == inbound_bytes, (
        f"Expected byte-faithful passthrough for canonical tools; upstream={upstream!r}"
    )


def test_anthropic_tools_unsorted_order_preserves_byte_faithful_request() -> None:
    client, transport = _make_no_optimize_app()
    inbound_dict = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "plan test"}],
        "tools": [
            {"name": "zeta", "description": "later"},
            {"name": "alpha"},
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
    assert response.status_code == 200, response.text
    upstream = transport.captured_body or b""
    assert upstream == inbound_bytes
    forwarded = json.loads(upstream.decode("utf-8"))
    assert [tool["name"] for tool in forwarded["tools"]] == ["zeta", "alpha"]


def test_anthropic_tools_unsorted_reordered_and_canonicalized_when_optimized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This asserts an exact upstream body to pin *tool* canonicalization, so
    # isolate it from output shaping, which is on by default for an optimized
    # proxy and appends a steering block to the system tail. Shaping is
    # covered by its own suites; folding its text into this expectation would
    # make a tool-ordering test fail every time the steering copy is edited.
    monkeypatch.setenv("HEADROOM_OUTPUT_SHAPER", "0")
    client, transport = _make_anthropic_app(optimize=True)
    proxy = client.app.state.proxy
    proxy.config.mode = "token"

    def _fake_apply(**kwargs):
        return SimpleNamespace(
            messages=kwargs["messages"],
            transforms_applied=[],
            timing={},
            tokens_before=100,
            tokens_after=100,
            waste_signals=None,
        )

    proxy.anthropic_pipeline.apply = _fake_apply
    inbound_dict = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "plan test"}],
        "tools": [
            {"name": "zeta", "description": "later"},
            {"name": "alpha"},
        ],
    }
    expected_dict = {
        **inbound_dict,
        "tools": [
            inbound_dict["tools"][1],
            inbound_dict["tools"][0],
        ],
    }
    inbound_bytes = serialize_body_canonical(inbound_dict)
    expected_bytes = serialize_body_canonical(expected_dict)

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
    upstream = transport.captured_body or b""
    assert upstream == expected_bytes
    assert upstream != inbound_bytes


def test_anthropic_presend_sorted_empty_tools_keeps_body_unmutated() -> None:
    inbound_dict = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "plan test"}],
    }
    inbound_bytes = serialize_body_canonical(inbound_dict)

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
        pipeline_extensions=[_SortedEmptyToolsPreSendExtension()],
        discover_pipeline_extensions=False,
    )
    app = create_app(config)
    client = TestClient(app)

    captured: dict[str, object] = {}

    async def _fake_retry(
        method: str,  # noqa: ARG001
        url: str,  # noqa: ARG001
        headers: dict[str, str],  # noqa: ARG001
        body: dict[str, object],  # noqa: ARG001
        body_mutated: bool,
        mutation_reasons: list[str],
        **kwargs: object,  # noqa: ANN003
    ) -> httpx.Response:  # noqa: ANN201
        captured["body_mutated"] = body_mutated
        captured["mutation_reasons"] = mutation_reasons
        captured["body"] = body
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

    app.state.proxy._retry_request = _fake_retry  # type: ignore[assignment]
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
    assert captured["body_mutated"] is False
    assert captured["mutation_reasons"] == []
    forwarded = captured["body"]
    assert isinstance(forwarded, dict)
    assert "tools" not in forwarded


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

        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_SSEByteStream(),
        )


class _SSEByteStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_s","type":"message","role":"assistant","model":"claude","usage":{"input_tokens":1,"output_tokens":0,"cache_read_input_tokens":0,"cache_creation_input_tokens":0}}}\n\n'
        yield b'event: message_stop\ndata: {"type":"message_stop"}\n\n'


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


def test_vertex_stream_rawpredict_preserves_client_beta_header_on_passthrough() -> None:
    _reset_session_beta_tracker_for_test()
    try:
        client, transport = _make_anthropic_app(optimize=False)
        get_session_beta_tracker().record_and_get_sticky_betas(
            provider="anthropic",
            session_id="s1",
            client_value="sticky-beta-2024-01-01",
        )

        inbound_bytes = (
            b'{"model":"claude-sonnet-4-6","stream":true,'
            b'"messages":[{"role":"user","content":"hi"}]}'
        )
        client_beta = "claude-code-20250219"

        with client.stream(
            "POST",
            "/projects/p/locations/us-central1/publishers/anthropic/models/"
            "claude-sonnet-4-6:streamRawPredict",
            headers={
                "x-api-key": "test-key",
                "x-headroom-session-id": "vertex-stream-beta-1",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": client_beta,
                "content-type": "application/json",
            },
            content=inbound_bytes,
        ) as resp:
            response_body = b"".join(resp.iter_bytes())
            assert resp.status_code == 200, response_body

        assert transport.captured_body == inbound_bytes
        assert transport.captured_headers is not None
        captured_headers = {key.lower(): value for key, value in transport.captured_headers.items()}
        assert captured_headers["anthropic-beta"] == client_beta
    finally:
        _reset_session_beta_tracker_for_test()


def test_messages_custom_upstream_stream_preserves_client_beta_header() -> None:
    _reset_session_beta_tracker_for_test()
    old_anthropic_url = None
    try:
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
        old_anthropic_url = type(proxy).ANTHROPIC_API_URL
        type(proxy).ANTHROPIC_API_URL = "https://custom.example"
        fake_tracker = _FakePrefixTracker(frozen_count=0)
        proxy.session_tracker_store.compute_session_id = lambda request, model, messages: (
            "custom-stream-beta-1"
        )
        proxy.session_tracker_store.get_or_create = lambda session_id, provider: fake_tracker

        transport = _StreamingCapturingTransport()
        proxy.http_client = httpx.AsyncClient(transport=transport)
        client = TestClient(app)

        get_session_beta_tracker().record_and_get_sticky_betas(
            provider="anthropic",
            session_id="custom-stream-beta-1",
            client_value="sticky-beta-2024-01-01",
        )

        inbound_bytes = (
            b'{"model":"claude-sonnet-4-6","max_tokens":16,"stream":true,'
            b'"messages":[{"role":"user","content":"hi"}]}'
        )
        client_beta = "claude-code-20250219"

        with client.stream(
            "POST",
            "/v1/messages",
            headers={
                "x-api-key": "test-key",
                "x-headroom-session-id": "custom-stream-beta-1",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": client_beta,
                "content-type": "application/json",
            },
            content=inbound_bytes,
        ) as resp:
            response_body = b"".join(resp.iter_bytes())
            assert resp.status_code == 200, response_body

        assert transport.captured_body == inbound_bytes
        assert transport.captured_headers is not None
        captured_headers = {key.lower(): value for key, value in transport.captured_headers.items()}
        assert captured_headers["anthropic-beta"] == client_beta
    finally:
        if old_anthropic_url is not None:
            type(proxy).ANTHROPIC_API_URL = old_anthropic_url
        _reset_session_beta_tracker_for_test()


def test_vertex_rawpredict_keeps_sticky_beta_union_on_non_stream_passthrough() -> None:
    _reset_session_beta_tracker_for_test()
    try:
        client, transport = _make_anthropic_app(optimize=False)
        get_session_beta_tracker().record_and_get_sticky_betas(
            provider="anthropic",
            session_id="s1",
            client_value="sticky-beta-2024-01-01",
        )

        inbound_bytes = b'{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"hi"}]}'
        client_beta = "claude-code-20250219"

        response = client.post(
            "/projects/p/locations/us-central1/publishers/anthropic/models/"
            "claude-sonnet-4-6:rawPredict",
            headers={
                "x-api-key": "test-key",
                "x-headroom-session-id": "vertex-raw-beta-1",
                "anthropic-version": "2023-06-01",
                "anthropic-beta": client_beta,
                "content-type": "application/json",
            },
            content=inbound_bytes,
        )

        assert response.status_code == 200
        assert transport.captured_body == inbound_bytes
        assert transport.captured_headers is not None
        captured_headers = {key.lower(): value for key, value in transport.captured_headers.items()}
        assert captured_headers["anthropic-beta"] == "sticky-beta-2024-01-01,claude-code-20250219"
    finally:
        _reset_session_beta_tracker_for_test()


def test_openai_responses_gzip_nonstream_passthrough_strips_content_encoding() -> None:
    client, transport = _make_no_optimize_app()
    decoded_body = _openai_responses_body_bytes(stream=False)
    encoded_body = gzip.compress(decoded_body)
    proxy_logger, handler, prev_level, records = _start_proxy_log_capture()

    try:
        response = client.post(
            "/v1/responses",
            headers=_openai_responses_codex_headers("gzip"),
            content=encoded_body,
        )
    finally:
        _stop_proxy_log_capture(proxy_logger, handler, prev_level)

    assert response.status_code == 200, response.text
    _assert_openai_responses_encoded_passthrough(transport, decoded_body)
    _assert_outbound_passthrough_log(records, forwarder="openai_responses")


def test_openai_responses_gzip_stream_passthrough_strips_content_encoding() -> None:
    client, transport = _make_no_optimize_app()
    decoded_body = _openai_responses_body_bytes(stream=True)
    encoded_body = gzip.compress(decoded_body)
    proxy_logger, handler, prev_level, records = _start_proxy_log_capture()

    try:
        with client.stream(
            "POST",
            "/v1/responses",
            headers=_openai_responses_codex_headers("gzip"),
            content=encoded_body,
        ) as response:
            assert response.status_code == 200
            for _ in response.iter_bytes():
                pass
    finally:
        _stop_proxy_log_capture(proxy_logger, handler, prev_level)

    _assert_openai_responses_encoded_passthrough(transport, decoded_body)
    _assert_outbound_passthrough_log(records, forwarder="streaming")


def test_openai_responses_codex_desktop_zstd_stream_passthrough_strips_content_encoding() -> None:
    zstandard = pytest.importorskip("zstandard")
    client, transport = _make_no_optimize_app()
    decoded_body = _openai_responses_body_bytes(stream=True)
    encoded_body = zstandard.ZstdCompressor().compress(decoded_body)
    proxy_logger, handler, prev_level, records = _start_proxy_log_capture()

    try:
        with client.stream(
            "POST",
            "/v1/responses",
            headers=_openai_responses_codex_headers("zstd"),
            content=encoded_body,
        ) as response:
            assert response.status_code == 200
            for _ in response.iter_bytes():
                pass
    finally:
        _stop_proxy_log_capture(proxy_logger, handler, prev_level)

    _assert_openai_responses_encoded_passthrough(transport, decoded_body)
    _assert_outbound_passthrough_log(records, forwarder="streaming")


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


# ---------------------------------------------------------------------------
# Thinking-preserving mutations (relaxing the blanket signed-thinking byte-lock)
#
# Anthropic signs the thinking BLOCK, not the request. The signature covers that
# block's own content, so edits elsewhere -- a compressed ``tool_result`` twenty
# turns back, a compacted ``tools`` array that is not even inside ``messages``
# -- cannot invalidate it. The original lock (#2254) froze the entire body
# whenever any thinking block was present, which on Claude Code traffic meant
# every computed compression was discarded from turn 2 of a session onward.
#
# The relaxation ships dark behind ``HEADROOM_THINKING_PRESERVING_MUTATIONS``
# and only engages when every thinking block is provably byte-equal to the one
# the client sent. These tests pin both directions: what must now ship, and what
# must still lock.
# ---------------------------------------------------------------------------

_TB_SIGNED_BLOCK = {
    "type": "thinking",
    "thinking": "Let me think about this… café",
    "signature": "EuYBCkQYBCKMAQ==",
}


def _tb_body(tool_text: str = "LOG LINE\n" * 500) -> dict:
    """Claude-Code-shaped turn: signed thinking in history + a fat tool_result."""
    return {
        "model": "claude-sonnet-5",
        "tools": [{"name": "Bash", "description": "x" * 400, "input_schema": {"type": "object"}}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "content": [dict(_TB_SIGNED_BLOCK), {"type": "text", "text": "ok"}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "t1",
                        "content": [{"type": "text", "text": tool_text}],
                    }
                ],
            },
        ],
    }


def _tb_select(mutated: dict, original_bytes: bytes):
    return select_outbound_body(
        body=mutated,
        original_body_bytes=original_bytes,
        body_mutated=True,
        forwarder_mode="byte_faithful",
        mutation_reasons=["content_router", "anthropic:tool_schema_compaction"],
    )


def test_thinking_preserving_mutation_ships_compression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untouched thinking block => the rest of the body may be re-serialized."""
    monkeypatch.setenv("HEADROOM_THINKING_PRESERVING_MUTATIONS", "1")
    original = json.dumps(_tb_body()).encode()
    mutated = json.loads(original)
    mutated["messages"][2]["content"][0]["content"][0]["text"] = "[compressed]"

    outbound = _tb_select(mutated, original)

    assert outbound.source == "canonical"
    assert outbound.dropped_mutations is False
    assert b"[compressed]" in outbound.content
    # The seal itself must survive verbatim, or we have merely moved the bug.
    assert _TB_SIGNED_BLOCK["signature"].encode() in outbound.content
    assert json.loads(outbound.content)["messages"][1]["content"][0] == _TB_SIGNED_BLOCK


@pytest.mark.parametrize(
    ("label", "tamper"),
    [
        ("edited_text", lambda b: b["messages"][1]["content"][0].__setitem__("thinking", "x")),
        (
            "edited_signature",
            lambda b: b["messages"][1]["content"][0].__setitem__("signature", "x"),
        ),
        ("dropped_block", lambda b: b["messages"][1]["content"].__delitem__(0)),
        ("reordered_blocks", lambda b: b["messages"][1]["content"].reverse()),
    ],
)
def test_touching_a_thinking_block_still_locks(
    monkeypatch: pytest.MonkeyPatch, label: str, tamper
) -> None:
    """Any detectable change to a thinking block keeps today's verbatim passthrough."""
    monkeypatch.setenv("HEADROOM_THINKING_PRESERVING_MUTATIONS", "1")
    original = json.dumps(_tb_body()).encode()
    mutated = json.loads(original)
    tamper(mutated)

    outbound = _tb_select(mutated, original)

    assert outbound.source == "passthrough", label
    assert outbound.dropped_mutations is True
    assert outbound.content == original


def test_thinking_relaxation_is_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset flag => relaxation active (maintainer chose on-by-default)."""
    monkeypatch.delenv("HEADROOM_THINKING_PRESERVING_MUTATIONS", raising=False)
    original = json.dumps(_tb_body()).encode()
    mutated = json.loads(original)
    mutated["messages"][2]["content"][0]["content"][0]["text"] = "[compressed]"

    outbound = _tb_select(mutated, original)

    assert outbound.source == "canonical"
    assert outbound.dropped_mutations is False


@pytest.mark.parametrize("off_value", ["0", "false", "no", "off", "OFF"])
def test_kill_switch_restores_the_blanket_lock(
    monkeypatch: pytest.MonkeyPatch, off_value: str
) -> None:
    """The documented rollback must work without a deploy, on every falsey spelling."""
    monkeypatch.setenv("HEADROOM_THINKING_PRESERVING_MUTATIONS", off_value)
    original = json.dumps(_tb_body()).encode()
    mutated = json.loads(original)
    mutated["messages"][2]["content"][0]["content"][0]["text"] = "[compressed]"

    outbound = _tb_select(mutated, original)

    assert outbound.source == "passthrough", off_value
    assert outbound.dropped_mutations is True
    assert outbound.content == original


def test_thinking_block_key_reorder_is_not_an_edit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The contract is over parsed values, so dict key order must not matter."""
    monkeypatch.setenv("HEADROOM_THINKING_PRESERVING_MUTATIONS", "1")
    original = json.dumps(_tb_body()).encode()
    mutated = json.loads(original)
    block = mutated["messages"][1]["content"][0]
    mutated["messages"][1]["content"][0] = {
        "signature": block["signature"],
        "thinking": block["thinking"],
        "type": block["type"],
    }

    assert thinking_blocks_survived_mutation(mutated, original) is True


def test_is_client_bytes_agrees_with_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CCR buffering probe must never disagree with the forwarder (#2952)."""
    monkeypatch.setenv("HEADROOM_THINKING_PRESERVING_MUTATIONS", "1")
    original = json.dumps(_tb_body()).encode()

    preserved = json.loads(original)
    preserved["messages"][2]["content"][0]["content"][0]["text"] = "[compressed]"
    tampered = json.loads(original)
    tampered["messages"][1]["content"][0]["thinking"] = "tampered"

    for mutated in (preserved, tampered):
        probe_says_locked = outbound_body_is_client_bytes(
            body=mutated, original_body_bytes=original
        )
        forwarder_locked = _tb_select(mutated, original).source == "passthrough"
        assert probe_says_locked is forwarder_locked


def test_unparseable_original_cannot_prove_preservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No proof => no relaxation. Fail closed."""
    monkeypatch.setenv("HEADROOM_THINKING_PRESERVING_MUTATIONS", "1")
    assert thinking_blocks_survived_mutation(_tb_body(), b"{not json") is False
    assert thinking_blocks_survived_mutation(_tb_body(), None) is False


def test_lone_surrogate_in_thinking_body_serializes_instead_of_raising():
    """A lone surrogate must not turn a mutated thinking body into a 500.

    ``"\\ud800"`` is valid JSON, so ``json.loads`` accepts it and a tool result
    carrying truncated UTF-16 produces one. Before #3124 a mutated
    thinking-bearing body returned the client's bytes verbatim and never reached
    canonical serialization; now it does, and both forwarders resolve outbound
    bytes outside their retry loop, so a raise here escapes as an unretried 500.
    """
    import json

    from headroom.proxy.body_forwarding import select_outbound_body

    lone_surrogate = chr(0xD800)
    original = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": f"reasoning {lone_surrogate}",
                        "signature": "sig",
                    }
                ],
            }
        ]
    }
    original_bytes = json.dumps(original, ensure_ascii=True).encode("utf-8")
    mutated = json.loads(original_bytes)
    mutated["messages"].append({"role": "user", "content": "compressed"})

    outbound = select_outbound_body(
        body=mutated,
        original_body_bytes=original_bytes,
        body_mutated=True,
        forwarder_mode="byte_faithful",
    )

    # The relaxation still applies (the thinking block is untouched) and the
    # mutation reaches the wire rather than being discarded or crashing.
    assert outbound.source == "canonical"
    assert not outbound.dropped_mutations
    reparsed = json.loads(outbound.content)
    assert reparsed == mutated
    # The signed block round-trips to exactly the values the client sent.
    assert reparsed["messages"][0] == original["messages"][0]
