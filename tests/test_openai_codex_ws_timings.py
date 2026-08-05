"""Unit 2: stage-timing instrumentation on the Codex WS path."""

from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anyio
import pytest

import headroom.proxy.handlers.openai as openai_handler
from headroom.proxy.handlers.openai import OpenAIHandlerMixin


class _DummyMetrics:
    def __init__(self) -> None:
        self.stage_timings: list[tuple[str, dict[str, float]]] = []

    async def record_request(self, **kwargs):  # pragma: no cover - unused here
        return None

    async def record_stage_timings(self, path: str, timings: dict[str, float]) -> None:
        self.stage_timings.append((path, dict(timings)))


class _DummyOpenAIHandler(OpenAIHandlerMixin):
    OPENAI_API_URL = "https://api.openai.com"

    def __init__(self) -> None:
        self.rate_limiter = None
        self.metrics = _DummyMetrics()
        self.config = SimpleNamespace(
            optimize=False,
            retry_max_attempts=1,
            retry_base_delay_ms=1,
            retry_max_delay_ms=1,
            connect_timeout_seconds=10,
            openai_extra_headers=None,
        )
        self.usage_reporter = None
        self.openai_provider = SimpleNamespace(get_context_limit=lambda model: 128_000)
        self.openai_pipeline = SimpleNamespace(apply=MagicMock())
        self.anthropic_backend = None
        self.cost_tracker = None
        self.memory_handler = None
        self.traffic_learner = None

    async def _next_request_id(self) -> str:
        return "req-ws-test"


class _MemoryToolsOnlyHandler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            inject_context=False,
            inject_tools=True,
            project_root_override="",
        )
        self.compute_calls = 0

    def compute_memory_tool_definitions(self, provider: str) -> list[dict]:
        self.compute_calls += 1
        assert provider == "openai"
        return [
            {
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "description": "Search memory.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]


class _FakeWebSocket:
    """Minimal async WebSocket stub that delivers a scripted frame list."""

    def __init__(self, frames: list[str] | None = None, headers: dict | None = None) -> None:
        self.headers = headers or {"authorization": "Bearer test"}
        self._frames = list(frames or [])
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.accepted_subprotocol = None
        self.accepted_headers: list[tuple[bytes, bytes]] | None = None
        self.closed = False
        self.close_code: int | None = None

    async def accept(self, subprotocol=None, headers=None) -> None:
        self.accepted_subprotocol = subprotocol
        self.accepted_headers = list(headers) if headers is not None else None

    async def receive_text(self) -> str:
        if not self._frames:
            # Simulate client disconnect: raise a WebSocketDisconnect-like error.
            raise RuntimeError("WebSocketDisconnect: no more frames")
        return self._frames.pop(0)

    async def send_text(self, text: str) -> None:
        self.sent_text.append(text)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def close(self, code: int | None = None, reason: str | None = None) -> None:
        self.closed = True
        self.close_code = code


class _FakeUpstream:
    """Async context manager mirroring the websockets.connect API."""

    def __init__(self, events: list[str]) -> None:
        self._events = list(events)
        self.sent: list[str] = []
        self.closed = False

    async def __aenter__(self) -> _FakeUpstream:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.closed = True

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for ev in self._events:
            yield ev


def _make_fake_websockets_module(upstream: _FakeUpstream):
    module = MagicMock()

    # Production now does ``upstream = await websockets.connect(...)`` then
    # ``async with upstream`` — so connect must return an awaitable.
    async def _connect(*args, **kwargs):
        return upstream

    module.connect = _connect
    module.Subprotocol = str  # the handler wraps client subprotocols if present
    return module


class _CapturingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def stage_log_capture():
    """Attach a ``Handler`` directly to the ``headroom.proxy`` logger.

    Using a direct handler is more robust than ``caplog`` for this
    logger because upstream configuration may set ``propagate=False``
    during module import, which bypasses pytest's root-logger capture.
    """
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
            # msg format: "[req-id] STAGE_TIMINGS {json}"
            payload_start = msg.index("STAGE_TIMINGS ") + len("STAGE_TIMINGS ")
            return json.loads(msg[payload_start:])
    raise AssertionError("no STAGE_TIMINGS log line captured")


def test_codex_ws_happy_path_emits_all_stage_timings(stage_log_capture):
    upstream_events = [
        json.dumps({"type": "response.created", "response": {"id": "resp_1"}}),
        json.dumps({"type": "response.completed", "response": {"id": "resp_1"}}),
    ]
    upstream = _FakeUpstream(upstream_events)
    fake_ws_mod = _make_fake_websockets_module(upstream)

    first_frame = json.dumps(
        {
            "type": "response.create",
            "response": {"model": "gpt-5.4", "input": "hello"},
        }
    )
    client_ws = _FakeWebSocket(frames=[first_frame])
    handler = _DummyOpenAIHandler()

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        anyio.run(handler.handle_openai_responses_ws, client_ws)

    # Upstream received the compressed (or unmodified) first frame
    assert len(upstream.sent) == 1

    # Structured log emitted with all expected stages
    payload = _parse_stage_log(stage_log_capture)
    assert payload["event"] == "stage_timings"
    assert payload["path"] == "openai_responses_ws"
    assert payload["request_id"] == "req-ws-test"
    assert payload["session_id"]  # non-empty UUID

    stages = payload["stages"]
    # Every expected stage key appears in the dict (may be None when not run)
    for key in (
        "accept",
        "first_client_frame",
        "upstream_connect",
        "upstream_first_event",
        "memory_context",
        "compression",
        "total_session",
    ):
        assert key in stages, f"missing stage: {key}"

    # Stages that actually ran are positive floats
    assert stages["accept"] is not None and stages["accept"] >= 0.0
    assert stages["first_client_frame"] is not None
    assert stages["upstream_connect"] is not None
    assert stages["upstream_first_event"] is not None
    assert stages["total_session"] > 0.0

    # Stages that were skipped (no memory handler, optimize=False) are None.
    assert stages["memory_context"] is None
    assert stages["compression"] is None

    # Prometheus metric sink captured the same path + timings.
    assert handler.metrics.stage_timings
    path, emitted = handler.metrics.stage_timings[-1]
    assert path == "openai_responses_ws"
    assert "total_session" in emitted


def test_codex_ws_chatgpt_auth_skips_memory_tools(stage_log_capture):
    upstream_events = [
        json.dumps({"type": "response.created", "response": {"id": "resp_1"}}),
        json.dumps({"type": "response.completed", "response": {"id": "resp_1"}}),
    ]
    upstream = _FakeUpstream(upstream_events)
    fake_ws_mod = _make_fake_websockets_module(upstream)

    first_frame = json.dumps(
        {
            "type": "response.create",
            "response": {"model": "gpt-5.4", "input": "hello", "store": True},
        }
    )
    client_ws = _FakeWebSocket(
        frames=[first_frame],
        headers={
            "authorization": "Bearer chatgpt-session-token",
            "chatgpt-account-id": "acct_123",
            "x-headroom-user-id": "user-1",
        },
    )
    handler = _DummyOpenAIHandler()
    memory_handler = _MemoryToolsOnlyHandler()
    handler.memory_handler = memory_handler

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        anyio.run(handler.handle_openai_responses_ws, client_ws)

    assert len(upstream.sent) == 1
    sent = json.loads(upstream.sent[0])
    response_body = sent["response"]
    assert response_body["store"] is False
    assert "tools" not in response_body
    assert "## Memory" not in response_body.get("instructions", "")
    assert memory_handler.compute_calls == 0


def test_codex_ws_upstream_connect_failure_still_logs_timings(stage_log_capture):
    """A session that never connects upstream still logs a timing line
    with ``upstream_first_event`` absent (null)."""

    fake_ws_mod = MagicMock()

    async def _boom_connect(*args, **kwargs):
        raise RuntimeError("upstream refused")

    fake_ws_mod.connect = _boom_connect
    fake_ws_mod.Subprotocol = str

    first_frame = json.dumps(
        {"type": "response.create", "response": {"model": "gpt-5.4", "input": "hi"}}
    )
    client_ws = _FakeWebSocket(frames=[first_frame])
    handler = _DummyOpenAIHandler()
    # With retry_max_attempts=1 we do not retry; fallback path attempts HTTP.

    # Stub the HTTP fallback so we don't need a network mock.
    async def _fallback(*args, **kwargs):
        return None

    handler._ws_http_fallback = _fallback  # type: ignore[assignment]

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        anyio.run(handler.handle_openai_responses_ws, client_ws)

    payload = _parse_stage_log(stage_log_capture)
    stages = payload["stages"]

    # upstream_first_event never fired because connect failed.
    assert stages.get("upstream_first_event") is None
    # upstream_connect is also None because we record it only after a
    # successful ``await websockets.connect(...)``.
    assert stages.get("upstream_connect") is None
    # But the envelope is still complete: the client is accepted and its
    # first frame is read before falling back to HTTP, even on connect
    # failure.
    assert stages["accept"] is not None
    assert stages["first_client_frame"] is not None
    assert stages["total_session"] > 0.0


def test_codex_ws_request_id_and_session_id_present_in_log(stage_log_capture):
    upstream = _FakeUpstream([])
    fake_ws_mod = _make_fake_websockets_module(upstream)

    first_frame = json.dumps(
        {"type": "response.create", "response": {"model": "gpt-5.4", "input": "hi"}}
    )
    client_ws = _FakeWebSocket(frames=[first_frame])
    handler = _DummyOpenAIHandler()

    with patch.dict(sys.modules, {"websockets": fake_ws_mod}):
        anyio.run(handler.handle_openai_responses_ws, client_ws)

    payload = _parse_stage_log(stage_log_capture)
    assert payload["request_id"] == "req-ws-test"
    assert isinstance(payload["session_id"], str)
    assert len(payload["session_id"]) >= 16


def test_codex_compression_debug_noop_skips_expensive_payload_debug(monkeypatch):
    handler = _DummyOpenAIHandler()

    def _fail_context_budget(_payload):
        raise AssertionError("debug context budget should not be built")

    monkeypatch.setattr(openai_handler, "_openai_responses_context_budget", _fail_context_budget)

    result = handler._compress_openai_responses_payload(
        {"model": "gpt-5.4", "input": "hello"},
        model="gpt-5.4",
        request_id="req-ws-test",
    )

    assert result[1] is False
    assert result[4] == "router_no_compression"
