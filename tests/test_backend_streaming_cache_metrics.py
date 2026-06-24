"""Cache-metric coverage for backend-routed streaming.

Two regressions in main as of 2026-05-14 (issue #327):

* ``StreamingMixin._stream_openai_via_backend`` (Azure/LiteLLM/AnyLLM OpenAI
  streaming) never inspects ``usage.prompt_tokens_details.cached_tokens`` from
  the upstream SSE chunks. Cache reads/writes are absent from
  ``cost_tracker.record_tokens``, ``SavingsTracker.record_request``, the
  ``RequestLog``, *and* the ``PERF`` log line — the latter is missing entirely
  for this path, so ``headroom perf`` shows 0 cache writes for every
  Azure-GPT/Codex backend-routed request.

* ``StreamingMixin._stream_response_bedrock`` (Bedrock-native streaming) hard-
  codes ``cache_read=0 cache_write=0 cache_hit_pct=0`` in its PERF log line
  regardless of what ``message_start.usage`` reported.

Both surface to the user as "Cache write: 0 tokens" in ``headroom perf``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.backends.base import StreamEvent  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

PERF_RE = re.compile(
    r"\bcache_read=(?P<cr>\d+)\s+cache_write=(?P<cw>\d+)\s+cache_hit_pct=(?P<chp>\d+)"
)


def _find_perf_record(records: list[logging.LogRecord]) -> tuple[int, int, int]:
    """Find the structured PERF log line and return (cache_read, cache_write, hit_pct)."""
    for record in records:
        msg = record.getMessage()
        if " PERF " not in msg:
            continue
        m = PERF_RE.search(msg)
        if m:
            return int(m["cr"]), int(m["cw"]), int(m["chp"])
    raise AssertionError(
        "No PERF log line with cache_read/cache_write/cache_hit_pct found. "
        f"Captured {len(records)} records.\n" + "\n".join(r.getMessage() for r in records[-15:])
    )


class _ListHandler(logging.Handler):
    """Tiny direct handler that survives the proxy disabling propagation.

    ``caplog`` attaches to root; ``headroom.proxy.helpers._setup_file_logging``
    flips ``logging.getLogger("headroom").propagate = False`` once a proxy
    instance is constructed in the test, after which root-attached handlers
    stop receiving headroom-namespaced records. Attaching directly to
    ``headroom.proxy`` sidesteps that.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
        self.records.append(record)


def _attach_proxy_log_capture():
    handler = _ListHandler()
    target = logging.getLogger("headroom.proxy")
    target.addHandler(handler)
    prior_level = target.level
    target.setLevel(logging.INFO)
    return handler, target, prior_level


def _detach_proxy_log_capture(handler, target, prior_level) -> None:
    target.removeHandler(handler)
    target.setLevel(prior_level)


def _make_openai_backend(chunks: list[str]) -> MagicMock:
    """Build a mock backend that yields OpenAI-format SSE chunks."""

    async def fake_stream(body: dict, headers: dict) -> AsyncIterator[str]:
        for chunk in chunks:
            yield chunk

    mock = MagicMock()
    mock.name = "anyllm-openai"
    mock.stream_openai_message = fake_stream
    return mock


def _make_bedrock_backend(events: list[StreamEvent]) -> MagicMock:
    """Build a mock backend that yields Anthropic StreamEvent objects."""

    async def fake_stream(body: dict, headers: dict) -> AsyncIterator[StreamEvent]:
        for evt in events:
            yield evt

    mock = MagicMock()
    mock.name = "bedrock"
    mock.stream_message = fake_stream
    mock.map_model_id = MagicMock(return_value="claude-3-5-sonnet-20241022")
    mock.supports_model = MagicMock(return_value=True)
    return mock


# =============================================================================
# Bug A — _stream_openai_via_backend (Azure/LiteLLM/AnyLLM OpenAI streaming)
# =============================================================================


def test_openai_backend_streaming_emits_perf_with_cache_read_and_inferred_write() -> None:
    """OpenAI backend streaming must surface cache reads + inferred writes.

    Real upstream (OpenAI Chat Completions w/ ``stream_options.include_usage=true``,
    or Azure GPT-5.5 through LiteLLM) emits a final chunk carrying::

        usage: {
          prompt_tokens: 1000,
          completion_tokens: 50,
          prompt_tokens_details: { cached_tokens: 700 }
        }

    OpenAI never reports a separate write counter, so we infer it as
    ``max(prompt_tokens - cached_tokens, 0)`` (see
    ``_infer_openai_cache_write_tokens``). The PERF log line consumed by
    ``headroom perf`` must report both.
    """
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        backend="anyllm",
        anyllm_provider="openai",
    )

    chunks = [
        'data: {"id":"c1","object":"chat.completion.chunk",'
        '"choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}\n\n',
        'data: {"id":"c1","object":"chat.completion.chunk",'
        '"choices":[{"index":0,"delta":{"content":" there"},"finish_reason":"stop"}]}\n\n',
        'data: {"id":"c1","object":"chat.completion.chunk","choices":[],'
        '"usage":{"prompt_tokens":1000,"completion_tokens":50,"total_tokens":1050,'
        '"prompt_tokens_details":{"cached_tokens":700}}}\n\n',
        "data: [DONE]\n\n",
    ]
    backend = _make_openai_backend(chunks)

    log_handle = _attach_proxy_log_capture()
    try:
        with patch("headroom.proxy.server.AnyLLMBackend", return_value=backend):
            app = create_app(config)
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-5.5",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                    headers={"Authorization": "Bearer test-key"},
                )

                assert resp.status_code == 200, resp.text[:200]
                body = resp.text
                assert "[DONE]" in body, body[:300]
    finally:
        _detach_proxy_log_capture(*log_handle)

    handler = log_handle[0]
    cr, cw, chp = _find_perf_record(handler.records)
    assert cr == 700, f"expected cache_read=700, got {cr}"
    assert cw == 300, f"expected inferred cache_write=300 (=1000-700), got {cw}"
    assert chp == 70, f"expected cache_hit_pct=70, got {chp}"


def test_openai_backend_streaming_perf_zeros_when_upstream_omits_usage() -> None:
    """When the upstream omits a usage chunk, cache values must be zero — not absent.

    Without ``stream_options.include_usage=true`` (or when upstream drops the
    final usage chunk) the PERF line still has to emit so ``headroom perf``
    counts the request.
    """
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        backend="anyllm",
        anyllm_provider="openai",
    )
    chunks = [
        'data: {"id":"c1","object":"chat.completion.chunk",'
        '"choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}\n\n',
        "data: [DONE]\n\n",
    ]
    backend = _make_openai_backend(chunks)

    log_handle = _attach_proxy_log_capture()
    try:
        with patch("headroom.proxy.server.AnyLLMBackend", return_value=backend):
            app = create_app(config)
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-5.5",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": True,
                    },
                    headers={"Authorization": "Bearer test-key"},
                )
                assert resp.status_code == 200
                assert "[DONE]" in resp.text
    finally:
        _detach_proxy_log_capture(*log_handle)

    handler = log_handle[0]
    cr, cw, chp = _find_perf_record(handler.records)
    assert (cr, cw, chp) == (0, 0, 0)


# =============================================================================
# Bug B — _stream_response_bedrock (Bedrock-native Anthropic streaming)
# =============================================================================


def _sse_data(event_type: str, data: dict[str, Any]) -> str:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def test_bedrock_streaming_emits_perf_with_message_start_cache_usage() -> None:
    """Bedrock streaming must surface cache_read + cache_write from message_start.

    Anthropic streaming reports cache usage on ``message_start.message.usage``
    (cache_read_input_tokens + cache_creation_input_tokens). The Bedrock streamer
    currently captures only ``input_tokens`` and ``output_tokens`` from the same
    event and hardcodes ``cache_read=0 cache_write=0`` into the PERF log.
    """
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        backend="anyllm",
        anyllm_provider="anthropic",
    )

    message_start = {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "model": "claude-3-5-sonnet-20241022",
            "role": "assistant",
            "type": "message",
            "content": [],
            "usage": {
                "input_tokens": 1000,
                "cache_read_input_tokens": 500,
                "cache_creation_input_tokens": 200,
            },
        },
    }
    block_start = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    }
    block_delta = {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "hi"},
    }
    block_stop = {"type": "content_block_stop", "index": 0}
    message_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 50},
    }
    message_stop = {"type": "message_stop"}

    events = [
        StreamEvent(
            event_type=e["type"],
            data=e,
            raw_sse=_sse_data(e["type"], e),
        )
        for e in [
            message_start,
            block_start,
            block_delta,
            block_stop,
            message_delta,
            message_stop,
        ]
    ]
    backend = _make_bedrock_backend(events)

    log_handle = _attach_proxy_log_capture()
    try:
        with patch("headroom.proxy.server.AnyLLMBackend", return_value=backend):
            app = create_app(config)
            with TestClient(app) as client:
                resp = client.post(
                    "/v1/messages",
                    json={
                        "model": "claude-3-5-sonnet-20241022",
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 64,
                        "stream": True,
                    },
                    headers={
                        "x-api-key": "sk-ant-test",
                        "anthropic-version": "2023-06-01",
                    },
                )
                assert resp.status_code == 200, resp.text[:200]
                assert "message_stop" in resp.text
    finally:
        _detach_proxy_log_capture(*log_handle)

    handler = log_handle[0]
    cr, cw, chp = _find_perf_record(handler.records)
    assert cr == 500, f"expected cache_read=500, got {cr}"
    assert cw == 200, f"expected cache_write=200, got {cw}"
    # round(500 / (500 + 200) * 100) = round(71.43) = 71
    assert chp == 71, f"expected cache_hit_pct=71, got {chp}"


# =============================================================================
# Regression guard
# =============================================================================


def test_streaming_perf_log_has_no_hardcoded_cache_zeros() -> None:
    """Catch any future re-introduction of ``cache_read=0 cache_write=0`` literal."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "headroom" / "proxy" / "handlers" / "streaming.py"
    text = src.read_text()
    assert "cache_read=0 cache_write=0" not in text, (
        "streaming.py contains a hardcoded `cache_read=0 cache_write=0` PERF log fragment. "
        "Wire the real cache_read/cache_write values into the PERF line instead."
    )
