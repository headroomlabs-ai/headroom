"""Issue #2357: cache mode must not silently forward everything uncompressed.

Cache mode compresses only the inter-turn delta against the previously
forwarded prefix. Before this fix, both delta-miss cases fell through to a
silent passthrough:

- session cold start (no previous turn recorded) — including a resumed
  100k-token transcript, so a fresh proxy never compressed anything while the
  dashboard kept reporting "optimization enabled";
- mid-session prefix mismatch (client rewrote history) — an intentional
  passthrough, but with no tag or log explaining the 0 savings.

Now a cold start runs the same full-message compression as non-cache modes
(there is no provider cache prefix to protect yet), and the mismatch
passthrough is tagged with ``passthrough_reason=cache_mode_prefix_mismatch``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import anyio
from fastapi import Request

from headroom.config import TransformResult
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.models import ProxyConfig

_COMPRESSED_TEXT = "compressed tool output"


class _DummyTokenizer:
    def count_messages(self, messages) -> int:
        return json.dumps(messages).count(" ") + 1

    def count_text(self, text: str) -> int:
        return max(1, text.count(" ") + 1)


class _DummyMetrics:
    async def record_request(self, **kwargs):
        return None

    async def record_stage_timings(self, path, timings):
        return None

    async def record_failed(self, **kwargs):
        return None

    def record_compression_failed(self, reason: str) -> None:
        return None

    async def record_rate_limited(self, **kwargs):
        return None


class _ResponseStub:
    status_code = 200
    headers: dict[str, str] = {}
    content = b'{"id":"msg_1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":1,"output_tokens":1}}'

    def json(self):
        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }


def _fake_pipeline_apply(messages, model, **kwargs):
    compressed = []
    for msg in messages:
        new = dict(msg)
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            new["content"] = [
                {**part, "content": _COMPRESSED_TEXT}
                if isinstance(part, dict) and part.get("type") == "tool_result"
                else part
                for part in msg["content"]
            ]
        compressed.append(new)
    return TransformResult(
        messages=compressed,
        tokens_before=1000,
        tokens_after=100,
        transforms_applied=["read_lifecycle:stale:test.py"],
    )


class _DummyAnthropicHandler(AnthropicHandlerMixin):
    ANTHROPIC_API_URL = "https://api.anthropic.com"

    def __init__(self, previous_messages: list | None = None) -> None:
        self.rate_limiter = None
        self.metrics = _DummyMetrics()
        self.config = ProxyConfig(
            optimize=True,
            image_optimize=False,
            retry_max_attempts=1,
            retry_base_delay_ms=1,
            retry_max_delay_ms=1,
            connect_timeout_seconds=10,
            mode="cache",
            cache_enabled=False,
            rate_limit_enabled=False,
            fallback_enabled=False,
            fallback_provider=None,
            prefix_freeze_enabled=False,
            memory_enabled=False,
        )
        self.usage_reporter = None
        self.anthropic_provider = SimpleNamespace(get_context_limit=lambda model: 200_000)
        self.anthropic_pipeline = SimpleNamespace(apply=MagicMock(side_effect=_fake_pipeline_apply))
        self.anthropic_backend = None
        self.cost_tracker = None
        self.memory_handler = None
        self.cache = None
        self.security = None
        self.ccr_context_tracker = None
        self.ccr_injector = None
        self.ccr_response_handler = None
        self.ccr_feedback = None
        self.ccr_batch_processor = None
        self.ccr_mcp_server = None
        self.traffic_learner = None
        self.tool_injector = None
        self.read_lifecycle_manager = None
        self.logger = SimpleNamespace(log=lambda *a, **k: None)
        self.request_logger = self.logger
        self.usage_observer = None
        self.image_compressor = None
        prev = previous_messages or []
        tracker = MagicMock()
        tracker.get_frozen_message_count.return_value = 0
        tracker.get_last_original_messages.return_value = prev
        tracker.get_last_forwarded_messages.return_value = prev
        tracker._cached_token_count = 0
        tracker.classify_cache_miss.return_value = SimpleNamespace(is_miss=False)
        self.session_tracker_store = SimpleNamespace(
            compute_session_id=lambda *a, **k: "sess-1",
            get_or_create=lambda *a, **k: tracker,
            resolve_tracker=lambda *a, **k: tracker,
        )
        self._background_compression_enabled = False
        self.recorded_tags: dict = {}

    async def _record_request_outcome(self, outcome) -> None:
        self.recorded_tags = dict(outcome.tags or {})

    async def _run_compression_in_executor(self, fn, timeout):
        return fn()

    async def _next_request_id(self) -> str:
        return "req-cache-cold-start-test"

    async def _retry_request(self, method, url, headers, body, **_kwargs):
        self.captured_body = body
        return _ResponseStub()


def _build_request(body: dict) -> Request:
    payload = json.dumps(body).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/messages",
        "raw_path": b"/v1/messages",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer sk-ant-api-test")],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 443),
    }
    return Request(scope, receive)


_TOOL_RESULT_MESSAGE = {
    "role": "user",
    "content": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "verbose stale tool output " * 200,
        }
    ],
}


def test_cache_mode_cold_start_compresses_full_request(monkeypatch):
    import headroom.tokenizers as _tk

    monkeypatch.setattr(_tk, "get_tokenizer", lambda model: _DummyTokenizer())

    handler = _DummyAnthropicHandler(previous_messages=[])
    request = _build_request(
        {"model": "claude-3-5-sonnet-latest", "messages": [_TOOL_RESULT_MESSAGE]}
    )

    anyio.run(handler.handle_anthropic_messages, request)

    # The full pipeline ran and its output was forwarded upstream.
    assert handler.anthropic_pipeline.apply.call_count == 1
    forwarded = handler.captured_body["messages"]
    assert forwarded[0]["content"][0]["content"] == _COMPRESSED_TEXT
    # No passthrough tag: compression genuinely ran.
    assert "passthrough_reason" not in handler.recorded_tags


def test_cache_mode_prefix_mismatch_passes_through_with_tag(monkeypatch):
    import headroom.tokenizers as _tk

    monkeypatch.setattr(_tk, "get_tokenizer", lambda model: _DummyTokenizer())

    # A previous turn exists but is NOT a prefix of the current request.
    previous = [{"role": "user", "content": [{"type": "text", "text": "totally different"}]}]
    handler = _DummyAnthropicHandler(previous_messages=previous)
    original_text = "verbose stale tool output " * 200
    request = _build_request(
        {"model": "claude-3-5-sonnet-latest", "messages": [_TOOL_RESULT_MESSAGE]}
    )

    anyio.run(handler.handle_anthropic_messages, request)

    # Conservative passthrough preserved: bytes forwarded unmodified...
    assert handler.anthropic_pipeline.apply.call_count == 0
    forwarded = handler.captured_body["messages"]
    assert forwarded[0]["content"][0]["content"] == original_text
    # ...but now visibly tagged instead of silent.
    assert handler.recorded_tags["passthrough_reason"] == "cache_mode_prefix_mismatch"
