"""Unit 2: stage-timing instrumentation on the Anthropic HTTP path."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import anyio
import pytest
from fastapi import Request

from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.models import ProxyConfig


class _DummyTokenizer:
    def count_messages(self, messages) -> int:
        return 1


class _DummyMetrics:
    def __init__(self) -> None:
        self.stage_timings: list[tuple[str, dict[str, float]]] = []

    async def record_request(self, **kwargs):
        return None

    async def record_stage_timings(self, path: str, timings: dict[str, float]) -> None:
        self.stage_timings.append((path, dict(timings)))

    async def record_failed(self, **kwargs):
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


class _DummyAnthropicHandler(AnthropicHandlerMixin):
    ANTHROPIC_API_URL = "https://api.anthropic.com"

    def __init__(self) -> None:
        self.rate_limiter = None
        self.metrics = _DummyMetrics()
        self.config = ProxyConfig(
            optimize=False,
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
        self.anthropic_pipeline = SimpleNamespace(apply=MagicMock())
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
                update_from_response=lambda *a, **k: None,
            ),
            resolve_tracker=lambda *a, **k: SimpleNamespace(
                _cached_token_count=0,
                get_frozen_message_count=lambda: 0,
                get_last_original_messages=lambda: [],
                get_last_forwarded_messages=lambda: [],
                record_request=lambda *a, **k: None,
                update_from_response=lambda *a, **k: None,
            ),
        )

    async def _next_request_id(self) -> str:
        return "req-anth-test"

    async def _record_request_outcome(self, outcome) -> None:
        return None

    def _extract_tags(self, headers):
        return {}

    async def _retry_request(
        self,
        method: str,
        url: str,
        headers: dict,
        body: dict,
        **_kwargs,
    ):
        self.captured = (method, url, headers, body)
        return _ResponseStub()

    def _get_compression_cache(self, session_id):
        return SimpleNamespace(
            apply_cached=lambda m: m,
            compute_frozen_count=lambda m: 0,
            mark_stable_from_messages=lambda *a, **k: None,
            should_defer_compression=lambda h: False,
            mark_stable=lambda h: None,
            content_hash=lambda c: "h",
            update_from_result=lambda *a, **k: None,
            _cache={},
            _stable_hashes=set(),
        )


def _build_request(body: dict, headers: dict[str, str]) -> Request:
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
        "headers": [
            (key.lower().encode("utf-8"), value.encode("utf-8")) for key, value in headers.items()
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 443),
    }
    return Request(scope, receive)


class _CapturingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def stage_log_capture():
    target = logging.getLogger("headroom.proxy")
    handler = _CapturingHandler()
    previous_level = target.level
    target.addHandler(handler)
    target.setLevel(logging.INFO)
    try:
        yield handler
    finally:
        target.removeHandler(handler)
        target.setLevel(previous_level)


def _parse_stage_log(handler: _CapturingHandler) -> dict:
    for record in handler.records:
        msg = record.getMessage()
        if "STAGE_TIMINGS" in msg:
            payload_start = msg.index("STAGE_TIMINGS ") + len("STAGE_TIMINGS ")
            return json.loads(msg[payload_start:])
    raise AssertionError("no STAGE_TIMINGS log line captured")


def test_anthropic_http_happy_path_emits_stage_timings(stage_log_capture):
    request = _build_request(
        {
            "model": "claude-3-5-sonnet-latest",
            "messages": [{"role": "user", "content": "hello"}],
        },
        {"authorization": "Bearer sk-ant-api-test"},
    )
    handler = _DummyAnthropicHandler()

    # Force tokenizer to a stub.
    import headroom.tokenizers as _tk

    orig_get = _tk.get_tokenizer
    _tk.get_tokenizer = lambda model: _DummyTokenizer()
    try:
        anyio.run(handler.handle_anthropic_messages, request)
    finally:
        _tk.get_tokenizer = orig_get

    payload = _parse_stage_log(stage_log_capture)
    assert payload["event"] == "stage_timings"
    assert payload["path"] == "anthropic_messages"
    assert payload["request_id"] == "req-anth-test"
    assert payload["session_id"]
    stages = payload["stages"]

    for key in (
        "read_request_json",
        "deep_copy",
        "compression_first_stage",
        "memory_context",
        "upstream_connect",
        "upstream_first_byte",
        "total_pre_upstream",
    ):
        assert key in stages, f"missing stage: {key}"

    assert stages["read_request_json"] is not None
    assert stages["deep_copy"] is not None
    assert stages["upstream_connect"] is not None
    assert stages["upstream_first_byte"] is not None
    assert stages["total_pre_upstream"] is not None
    # compression + memory_context were skipped (optimize=False, no memory)
    assert stages["compression_first_stage"] is None
    assert stages["memory_context"] is None

    # Metrics sink got the observation too.
    assert handler.metrics.stage_timings
    path, emitted = handler.metrics.stage_timings[-1]
    assert path == "anthropic_messages"
    assert "total_pre_upstream" in emitted


def test_subscription_contribution_receives_request_savings_usd(monkeypatch):
    request = _build_request(
        {
            "model": "claude-3-5-sonnet-latest",
            "messages": [{"role": "user", "content": "hello"}],
        },
        {"authorization": "Bearer oauth-test"},
    )
    handler = _DummyAnthropicHandler()
    captured: dict[str, object] = {}
    tracker = SimpleNamespace(
        notify_active=lambda token: None,
        update_contribution=lambda **kwargs: captured.update(kwargs),
    )

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr(
        "headroom.subscription.tracker.get_subscription_tracker",
        lambda: tracker,
    )
    monkeypatch.setattr(
        "headroom.proxy.savings_tracker.estimate_request_savings_usd",
        lambda model, **kwargs: {"compression": 1.25, "provider_cache": 2.5},
    )

    anyio.run(handler.handle_anthropic_messages, request)

    assert captured["compression_savings_usd"] == 1.25
    assert captured["cache_savings_usd"] == 2.5


def test_anthropic_no_optimize_preserves_client_tool_order():
    tools = [
        {
            "name": "Read",
            "description": "Read a file",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "Bash",
            "description": "Run a shell command",
            "input_schema": {"type": "object", "properties": {}},
        },
    ]
    request = _build_request(
        {
            "model": "claude-3-5-sonnet-latest",
            "messages": [{"role": "user", "content": "use a tool"}],
            "tools": tools,
        },
        {"authorization": "Bearer sk-ant-api-test"},
    )
    handler = _DummyAnthropicHandler()

    import headroom.tokenizers as _tk

    orig_get = _tk.get_tokenizer
    _tk.get_tokenizer = lambda model: _DummyTokenizer()
    try:
        anyio.run(handler.handle_anthropic_messages, request)
    finally:
        _tk.get_tokenizer = orig_get

    _, _, _, forwarded_body = handler.captured
    assert [tool["name"] for tool in forwarded_body["tools"]] == ["Read", "Bash"]


def test_anthropic_third_party_upstream_strips_tool_search_tools():
    tools = [
        {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
        {
            "name": "Bash",
            "description": "run a command",
            "input_schema": {"type": "object", "properties": {}},
        },
        {"type": "web_search_20250305", "name": "web_search"},
    ]
    request = _build_request(
        {
            "model": "claude-3-5-sonnet-latest",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "use a tool"}],
            "tools": tools,
        },
        {"authorization": "Bearer sk-ant-api-test"},
    )
    handler = _DummyAnthropicHandler()

    import headroom.tokenizers as _tk

    orig_get = _tk.get_tokenizer
    _tk.get_tokenizer = lambda model: _DummyTokenizer()
    try:
        response = anyio.run(
            handler.handle_anthropic_messages,
            request,
            "https://api.deepseek.com/anthropic",
        )
    finally:
        _tk.get_tokenizer = orig_get

    assert response.status_code == 200
    _, forwarded_url, _, forwarded_body = handler.captured
    assert forwarded_url == "https://api.deepseek.com/anthropic/v1/messages"
    assert forwarded_body["tools"] == [tools[1], tools[2]]
    assert not any(
        str(tool.get("type", "")).startswith("tool_search_tool_")
        for tool in forwarded_body["tools"]
    )


def test_anthropic_direct_path_repairs_typeless_tool_search_regression():
    """Do not double-inject a typeless search tool; heal its stale history."""
    tools = [
        {
            "name": f"mcp_tool_{index}",
            "description": f"tool {index}",
            "input_schema": {"type": "object", "properties": {}},
        }
        for index in range(20)
    ]
    tools.append(
        {
            "name": "tool_search_tool_regex",
            "description": "client-provided tool search",
            "input_schema": {"type": "object", "properties": {}},
        }
    )
    messages = [
        {"role": "user", "content": "search for a tool"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "srvtoolu_regex",
                    "name": "tool_search_tool_regex",
                    "input": {"pattern": "regex"},
                },
                {
                    "type": "tool_search_tool_result",
                    "tool_use_id": "srvtoolu_regex",
                    "content": {
                        "type": "tool_search_tool_search_result",
                        "tool_references": [
                            {
                                "type": "tool_reference",
                                "tool_name": "tool_search_tool_regex",
                            }
                        ],
                    },
                },
            ],
        },
        {"role": "user", "content": "continue"},
    ]
    request = _build_request(
        {
            "model": "claude-sonnet-4-6",
            "max_tokens": 100,
            "messages": messages,
            "tools": tools,
        },
        {"authorization": "Bearer sk-ant-api-test"},
    )
    handler = _DummyAnthropicHandler()

    import headroom.tokenizers as _tk

    orig_get = _tk.get_tokenizer
    _tk.get_tokenizer = lambda model: _DummyTokenizer()
    try:
        response = anyio.run(handler.handle_anthropic_messages, request)
    finally:
        _tk.get_tokenizer = orig_get

    assert response.status_code == 200
    _, _, _, forwarded_body = handler.captured
    # The client-owned typeless entry suppresses Headroom's typed search-tool
    # injection, and the tools array remains byte-for-byte equivalent.
    assert forwarded_body["tools"] == tools
    assert not any(tool.get("type") for tool in forwarded_body["tools"])
    # The stale server-side round trip is removed before Anthropic validates it.
    block_types = [
        block.get("type")
        for message in forwarded_body["messages"]
        for block in message.get("content", [])
        if isinstance(message.get("content"), list) and isinstance(block, dict)
    ]
    assert "server_tool_use" not in block_types
    assert "tool_search_tool_result" not in block_types


def test_anthropic_http_invalid_body_still_emits_stage_timings(stage_log_capture):
    async def receive():
        # Invalid JSON — produces ``ValueError`` from ``_read_request_json``.
        return {"type": "http.request", "body": b"not-json", "more_body": False}

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
    request = Request(scope, receive)
    handler = _DummyAnthropicHandler()

    anyio.run(handler.handle_anthropic_messages, request)

    payload = _parse_stage_log(stage_log_capture)
    stages = payload["stages"]
    # Even on invalid body, ``read_request_json`` duration is recorded
    # (the measure wraps the call, including the raised exception path).
    assert stages["read_request_json"] is not None
    # Downstream stages never ran:
    assert stages["upstream_connect"] is None
    assert stages["upstream_first_byte"] is None


def test_anthropic_http_request_and_session_ids_present(stage_log_capture):
    request = _build_request(
        {
            "model": "claude-3-5-sonnet-latest",
            "messages": [{"role": "user", "content": "hi"}],
        },
        {"authorization": "Bearer sk-ant-api-test"},
    )
    handler = _DummyAnthropicHandler()

    import headroom.tokenizers as _tk

    orig_get = _tk.get_tokenizer
    _tk.get_tokenizer = lambda model: _DummyTokenizer()
    try:
        anyio.run(handler.handle_anthropic_messages, request)
    finally:
        _tk.get_tokenizer = orig_get

    payload = _parse_stage_log(stage_log_capture)
    assert payload["request_id"] == "req-anth-test"
    assert isinstance(payload["session_id"], str)
    assert len(payload["session_id"]) >= 16
