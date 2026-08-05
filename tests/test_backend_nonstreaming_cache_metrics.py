"""Cache-metric coverage for backend-routed **non-streaming** requests.

Sibling of ``tests/test_backend_streaming_cache_metrics.py`` (issue #327).
That file fixed the *streaming* backend paths so cache reads/writes reach the
``PERF`` log line consumed by ``headroom perf``. The **non-streaming** backend
paths were left behind — the same bug class on the parallel code path:

* ``AnthropicHandlerMixin`` non-streaming backend branch
  (``anthropic.py`` ``send_message`` path): reads ``usage`` from the backend
  response body but extracts only ``output_tokens``. The accompanying comment
  admits "Cache metrics aren't extracted from the backend response here yet —
  that's a follow-up." So Bedrock / Vertex non-streaming traffic reported
  ``cache_read=0 cache_write=0`` even though the response carried
  ``cache_read_input_tokens`` / ``cache_creation_input_tokens``.

* ``OpenAIHandlerMixin`` non-streaming backend branch
  (``openai.py`` ``send_openai_message`` path): worse — cache fields ARE
  extracted and fed to ``openai_prefix_tracker``, but never threaded into the
  ``RequestOutcome``, so the funnel (Prometheus / cost tracker / RequestLog /
  PERF) all see zeros.

Both surface to the user as "Cache write: 0 tokens" in ``headroom perf``,
identical to the streaming regression that motivated issue #327.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.backends.base import BackendResponse  # noqa: E402
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


def _make_anthropic_backend(body: dict[str, Any]) -> MagicMock:
    """Build a mock backend whose ``send_message`` returns ``body`` (Anthropic shape).

    The body is the Anthropic Messages non-streaming response, including a
    ``usage`` block that carries cache counters — exactly what Bedrock /
    Vertex / LiteLLM(anthropic) return for a cached turn.
    """

    async def fake_send(body_: dict, headers: dict) -> BackendResponse:
        return BackendResponse(body=body, status_code=200)

    # A streaming coroutine is never exercised on the non-streaming path, but
    # the server's backend-factory calls ``map_model_id`` / ``supports_model``
    # during wiring, so provide no-op mocks for those too.
    mock = MagicMock()
    mock.name = "anyllm-anthropic"
    mock.send_message = fake_send
    mock.map_model_id = MagicMock(return_value="claude-3-5-sonnet-20241022")
    mock.supports_model = MagicMock(return_value=True)
    return mock


def _make_openai_backend(body: dict[str, Any]) -> MagicMock:
    """Build a mock backend whose ``send_openai_message`` returns ``body`` (OpenAI shape).

    The body carries a ``usage`` block with ``prompt_tokens_details.cached_tokens``
    (the OpenAI / Azure-GPT-via-LiteLLM non-streaming shape). Bedrock-style
    top-level ``cache_*_input_tokens`` keys are also honored by the handler.
    """

    async def fake_send(body_: dict, headers: dict) -> BackendResponse:
        return BackendResponse(body=body, status_code=200)

    mock = MagicMock()
    mock.name = "anyllm-openai"
    mock.send_openai_message = fake_send
    mock.map_model_id = MagicMock(return_value="gpt-5.5")
    mock.supports_model = MagicMock(return_value=True)
    return mock


# =============================================================================
# Bug A — OpenAI backend non-streaming (Azure/LiteLLM/AnyLLM OpenAI, stream=False)
# =============================================================================


def test_openai_backend_nonstreaming_emits_perf_with_cache_read_and_inferred_write() -> None:
    """OpenAI backend non-streaming must surface cache reads + inferred writes.

    OpenAI Chat Completions non-streaming carries::

        usage: {
          prompt_tokens: 1000,
          completion_tokens: 50,
          prompt_tokens_details: { cached_tokens: 700 }
        }

    OpenAI never reports a separate write counter, so it is inferred as
    ``max(prompt_tokens - cached_tokens, 0)``. The handler already computes
    both and feeds them to ``openai_prefix_tracker`` — this test pins that they
    also reach the PERF log line (previously computed-then-dropped).
    """
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        backend="anyllm",
        anyllm_provider="openai",
    )

    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
        "usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 50,
            "total_tokens": 1050,
            "prompt_tokens_details": {"cached_tokens": 700},
        },
    }
    backend = _make_openai_backend(body)

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
                    },
                    headers={"Authorization": "Bearer test-key"},
                )
                assert resp.status_code == 200, resp.text[:200]
    finally:
        _detach_proxy_log_capture(*log_handle)

    handler = log_handle[0]
    cr, cw, chp = _find_perf_record(handler.records)
    assert cr == 700, f"expected cache_read=700, got {cr}"
    assert cw == 300, f"expected inferred cache_write=300 (=1000-700), got {cw}"
    assert chp == 70, f"expected cache_hit_pct=70, got {chp}"


def test_openai_backend_nonstreaming_perf_zeros_when_upstream_omits_cache_usage() -> None:
    """When the upstream omits usage entirely, cache values must be zero — not absent.

    Mirrors the streaming twin: no ``usage`` block at all means no
    ``prompt_tokens`` to infer a write from, so all cache counters stay 0.
    (When ``usage`` IS present but lacks cache details, the inferred write is
    ``prompt_tokens - 0``, which is non-zero — that is covered by the positive
    test above.)
    """
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        backend="anyllm",
        anyllm_provider="openai",
    )

    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}
        ],
    }
    backend = _make_openai_backend(body)

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
                    },
                    headers={"Authorization": "Bearer test-key"},
                )
                assert resp.status_code == 200
    finally:
        _detach_proxy_log_capture(*log_handle)

    handler = log_handle[0]
    cr, cw, chp = _find_perf_record(handler.records)
    assert (cr, cw, chp) == (0, 0, 0)


# =============================================================================
# Bug B — Anthropic backend non-streaming (Bedrock / Vertex / LiteLLM, stream=False)
# =============================================================================


def test_anthropic_backend_nonstreaming_emits_perf_with_cache_read_and_write() -> None:
    """Anthropic backend non-streaming must surface cache_read + cache_write.

    Bedrock / Vertex / LiteLLM(anthropic) non-streaming returns an Anthropic
    Messages body whose ``usage`` carries ``cache_read_input_tokens`` and
    ``cache_creation_input_tokens``. The handler previously extracted only
    ``output_tokens`` from this same dict — the cache counters were right
    there, unread. Mirror of the streaming test
    ``test_bedrock_streaming_emits_perf_with_message_start_cache_usage``.
    """
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        backend="anyllm",
        anyllm_provider="anthropic",
    )

    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 50,
            "cache_read_input_tokens": 500,
            "cache_creation_input_tokens": 200,
        },
    }
    backend = _make_anthropic_backend(body)

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
                    },
                    headers={
                        "x-api-key": "sk-ant-test",
                        "anthropic-version": "2023-06-01",
                    },
                )
                assert resp.status_code == 200, resp.text[:200]
    finally:
        _detach_proxy_log_capture(*log_handle)

    handler = log_handle[0]
    cr, cw, chp = _find_perf_record(handler.records)
    assert cr == 500, f"expected cache_read=500, got {cr}"
    assert cw == 200, f"expected cache_write=200, got {cw}"
    # round(500 / (500 + 200) * 100) = round(71.43) = 71
    assert chp == 71, f"expected cache_hit_pct=71, got {chp}"


def test_anthropic_backend_nonstreaming_perf_zeros_when_upstream_omits_cache_usage() -> None:
    """When the upstream omits cache counters, cache values must be zero."""
    config = ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        backend="anyllm",
        anyllm_provider="anthropic",
    )

    body = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1000, "output_tokens": 50},
    }
    backend = _make_anthropic_backend(body)

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
                    },
                    headers={
                        "x-api-key": "sk-ant-test",
                        "anthropic-version": "2023-06-01",
                    },
                )
                assert resp.status_code == 200
    finally:
        _detach_proxy_log_capture(*log_handle)

    handler = log_handle[0]
    cr, cw, chp = _find_perf_record(handler.records)
    assert (cr, cw, chp) == (0, 0, 0)
