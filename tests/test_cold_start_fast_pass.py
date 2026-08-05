"""Cold-start fast pass: when background compression defers a cold-start-large
request, the handler still runs the pipeline synchronously with
skip_kompress=True so the FORWARDED (and therefore provider-cached,
byte-identically frozen) form carries the cheap savings. Only the Kompress ML
stage stays deferred to the background job."""

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


class _RecordingBackgroundCompressor:
    def __init__(self) -> None:
        self.enqueued: list[tuple[str, object, object]] = []

    def enqueue(self, key, compress, store) -> bool:
        self.enqueued.append((key, compress, store))
        return True


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

    def __init__(self) -> None:
        self.rate_limiter = None
        self.metrics = _DummyMetrics()
        self.config = ProxyConfig(
            optimize=True,
            image_optimize=False,
            retry_max_attempts=1,
            retry_base_delay_ms=1,
            retry_max_delay_ms=1,
            connect_timeout_seconds=10,
            mode="token",
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
        self.session_tracker_store = SimpleNamespace(
            compute_session_id=lambda *a, **k: "sess-1",
            get_or_create=lambda *a, **k: SimpleNamespace(
                get_frozen_message_count=lambda: 0,
                get_last_original_messages=lambda: [],
                get_last_forwarded_messages=lambda: [],
                record_request=lambda *a, **k: None,
            ),
            resolve_tracker=lambda *a, **k: SimpleNamespace(
                get_frozen_message_count=lambda: 0,
                get_last_original_messages=lambda: [],
                get_last_forwarded_messages=lambda: [],
                record_request=lambda *a, **k: None,
            ),
        )
        # Cold-start deferral wiring under test.
        self._background_compression_enabled = True
        self._background_compression_min_tokens = 1
        self._background_compressor = _RecordingBackgroundCompressor()
        self.executor_calls: list[float] = []

    async def _run_compression_in_executor(self, fn, timeout):
        self.executor_calls.append(timeout)
        return fn()

    async def _next_request_id(self) -> str:
        return "req-fastpass-test"

    def _extract_tags(self, headers):
        return {}

    async def _retry_request(self, method, url, headers, body, **_kwargs):
        self.captured_body = body
        return _ResponseStub()

    def _get_compression_cache(self, session_id):
        self.comp_cache_updates: list[tuple] = getattr(self, "comp_cache_updates", [])
        return SimpleNamespace(
            apply_cached=lambda m: m,
            compute_frozen_count=lambda m: 0,
            mark_stable_from_messages=lambda *a, **k: None,
            should_defer_compression=lambda h: False,
            mark_stable=lambda h: None,
            content_hash=lambda c: "h",
            update_from_result=lambda *a: self.comp_cache_updates.append(a),
            _cache={},
            _stable_hashes=set(),
        )


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


def test_cold_start_runs_fast_pass_and_defers_only_kompress(monkeypatch):
    import headroom.tokenizers as _tk

    monkeypatch.setattr(_tk, "get_tokenizer", lambda model: _DummyTokenizer())

    handler = _DummyAnthropicHandler()
    request = _build_request(
        {
            "model": "claude-3-5-sonnet-latest",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "verbose stale tool output " * 200,
                        }
                    ],
                },
            ],
        }
    )

    anyio.run(handler.handle_anthropic_messages, request)

    # The fast pass ran synchronously with the ML stage disabled.
    assert handler.executor_calls, "fast pass never ran through the executor"
    sync_calls = [
        c for c in handler.anthropic_pipeline.apply.call_args_list if c.kwargs.get("skip_kompress")
    ]
    assert len(sync_calls) == 1, "expected exactly one synchronous skip_kompress pass"

    # The full pipeline (kompress included) went to the background queue,
    # keyed against the ORIGINAL messages for content-hash reuse.
    assert len(handler._background_compressor.enqueued) == 1
    _key, bg_compress, _store = handler._background_compressor.enqueued[0]
    bg_compress()
    bg_calls = [
        c
        for c in handler.anthropic_pipeline.apply.call_args_list
        if not c.kwargs.get("skip_kompress")
    ]
    assert len(bg_calls) == 1, "background job must run the full pipeline"

    # The FORWARDED body carries the fast-pass form — that is what the
    # provider caches and the byte-identical freeze locks in.
    forwarded = handler.captured_body["messages"]
    assert forwarded[0]["content"][0]["content"] == _COMPRESSED_TEXT

    # Fast-pass results were stored in the compression cache.
    assert handler.comp_cache_updates


def test_fast_pass_failure_falls_back_to_full_deferral(monkeypatch):
    import headroom.tokenizers as _tk

    monkeypatch.setattr(_tk, "get_tokenizer", lambda model: _DummyTokenizer())

    handler = _DummyAnthropicHandler()

    async def _boom(fn, timeout):
        raise TimeoutError("fast pass exceeded budget")

    handler._run_compression_in_executor = _boom  # type: ignore[method-assign]

    original_text = "verbose stale tool output " * 200
    request = _build_request(
        {
            "model": "claude-3-5-sonnet-latest",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": original_text,
                        }
                    ],
                },
            ],
        }
    )

    anyio.run(handler.handle_anthropic_messages, request)

    # Fail-open: original messages forwarded, background job still queued.
    forwarded = handler.captured_body["messages"]
    assert forwarded[0]["content"][0]["content"] == original_text
    assert len(handler._background_compressor.enqueued) == 1
