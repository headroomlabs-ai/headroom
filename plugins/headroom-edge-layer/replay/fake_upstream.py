"""A fake Anthropic-shaped upstream for the offline replay harness.

Records every request body it receives (so the harness can check what the
proxy actually forwarded — including whether an edge's edits survived the
signed-thinking-block passthrough in `body_forwarding.py`, per the
correction that the extension running is not the same as its edits
arriving on the wire) and replays canned responses in call order, one per
POST to `/v1/messages`.

Deliberately not async/production-grade — this stands in for the real
Anthropic API only inside the replay harness, on loopback, for one test run
at a time.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class RecordedRequest:
    path: str
    headers: dict[str, str]
    body: bytes

    @property
    def body_json(self) -> Any:
        return json.loads(self.body)


@dataclass
class FakeUpstreamState:
    responses: list[dict[str, Any]] = field(default_factory=list)
    received: list[RecordedRequest] = field(default_factory=list)
    call_index: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class _Handler(BaseHTTPRequestHandler):
    state: FakeUpstreamState  # set per-server via a subclass in start_server()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        pass  # silence default request logging

    def do_POST(self) -> None:  # noqa: N802 - stdlib method name
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length else b""
        with self.state.lock:
            self.state.received.append(
                RecordedRequest(path=self.path, headers=dict(self.headers), body=body)
            )
            index = self.state.call_index
            self.state.call_index += 1
            if index < len(self.state.responses):
                response_body = self.state.responses[index]
            else:
                response_body = _default_response()

        payload = json.dumps(response_body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _default_response() -> dict[str, Any]:
    return {
        "id": "msg_fallback",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [{"type": "text", "text": "[fake-upstream: no more canned responses]"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


class FakeUpstream:
    """Context manager: `with FakeUpstream(responses) as upstream: ...`"""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self.state = FakeUpstreamState(responses=list(responses or []))
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "FakeUpstream":
        handler_cls = type("_BoundHandler", (_Handler,), {"state": self.state})
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        self._server = None
        self._thread = None

    @property
    def port(self) -> int:
        assert self._server is not None, "FakeUpstream must be used as a context manager"
        return self._server.server_address[1]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def set_responses(self, responses: list[dict[str, Any]]) -> None:
        with self.state.lock:
            self.state.responses = list(responses)
            self.state.call_index = 0
            self.state.received.clear()

    def received_bodies(self) -> list[bytes]:
        with self.state.lock:
            return [r.body for r in self.state.received]

    def received_requests(self) -> list[RecordedRequest]:
        with self.state.lock:
            return list(self.state.received)


def anthropic_response_from_replay_turn(turn: Any) -> dict[str, Any]:
    """Builds a canned `/v1/messages` response from a `ReplayTurn` (see
    `benchmarks/claude_session_mode_benchmark.py`), so the fake upstream
    replays the *actual* logged assistant reply for that turn rather than a
    synthetic stand-in.
    """
    content = turn.assistant_message.get("content")
    if isinstance(content, str):
        content = [{"type": "text", "text": content}] if content else []
    return {
        "id": f"msg_replay_{turn.request_id}",
        "type": "message",
        "role": "assistant",
        "model": turn.model,
        "content": content,
        "stop_reason": "tool_use" if any(
            isinstance(b, dict) and b.get("type") == "tool_use" for b in (content or [])
        ) else "end_turn",
        "usage": {
            "input_tokens": turn.observed_input_tokens,
            "output_tokens": turn.output_tokens,
            "cache_read_input_tokens": turn.observed_cache_read_tokens,
            "cache_creation_input_tokens": turn.observed_cache_write_tokens,
        },
    }
