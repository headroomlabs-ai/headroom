"""A scriptable fake LLM provider for the gateway contract tests.

Public API (kept deliberately small - the docker/Kong driver imports it too)::

    provider = FakeProvider()
    provider.script([openai_text_response("hi"), openai_tool_call_response("f", {})])
    provider.calls            # list[ProviderCall]: every request, in order
    provider.app              # the FastAPI app (mount in a TestClient / ASGITransport)
    provider.handle(path, body, headers) -> ProviderReply   # pure-Python dispatch
    base_url = provider.start(port)   # uvicorn in a daemon thread; port 0 = ephemeral
    provider.stop()
    provider.reset()          # clears the queue and the call log

Endpoints:

* ``POST /v1/chat/completions`` - OpenAI chat. JSON, or SSE when ``stream`` is
  true. The final SSE chunk carries ``usage`` when
  ``stream_options.include_usage`` is set (exactly what OpenAI does).
* ``POST /v1/messages`` - Anthropic messages. JSON, or SSE with ``message_start``
  (input usage) and ``message_delta`` (output usage).

Scripting: each queued item is either a response *body* (dict, status 200) or a
:class:`ScriptedResponse` (status/headers/body), or a callable taking the
:class:`ProviderCall` and returning one of those. When the queue is empty the
provider answers with ``default`` (a shape-appropriate one-line text response
unless the constructor was given something else), so tests that only care about
the request side need not script anything.

Builders (all deterministic given their arguments):
``openai_text_response``, ``openai_tool_call_response``,
``anthropic_text_response``, ``anthropic_tool_use_response``,
``openai_usage``, ``anthropic_usage``.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:  # fastapi is an optional extra; the pure-Python ``handle`` path needs none of it
    from fastapi import FastAPI, Request
    from fastapi.responses import Response
except ImportError:  # pragma: no cover - exercised only in lean environments
    FastAPI = Request = Response = None  # type: ignore[assignment,misc]

OPENAI_CHAT_PATH = "/v1/chat/completions"
ANTHROPIC_MESSAGES_PATH = "/v1/messages"


# --------------------------------------------------------------------------- #
# Records                                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class ProviderCall:
    """One request the fake provider received."""

    path: str
    body: dict[str, Any]
    headers: dict[str, str]  # lower-cased keys
    received_at: float = field(default_factory=time.time)

    @property
    def stream(self) -> bool:
        return bool(self.body.get("stream"))

    @property
    def messages(self) -> list[dict[str, Any]]:
        return list(self.body.get("messages") or [])

    @property
    def tools(self) -> Any:
        return self.body.get("tools")


@dataclass
class ScriptedResponse:
    """A fully specified provider answer."""

    body: dict[str, Any]
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class ProviderReply:
    """What the provider sent back, in wire form plus the parsed JSON (when JSON)."""

    status: int
    headers: dict[str, str]
    content: bytes
    json: dict[str, Any] | None

    @property
    def is_sse(self) -> bool:
        return self.headers.get("content-type", "").startswith("text/event-stream")


Script = dict[str, Any] | ScriptedResponse | Callable[[ProviderCall], Any]


# --------------------------------------------------------------------------- #
# Builders                                                                     #
# --------------------------------------------------------------------------- #


def openai_usage(
    prompt: int, completion: int, cached: int | None = None, **extra: Any
) -> dict[str, Any]:
    usage: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    usage.update(extra)
    return usage


def anthropic_usage(
    input_tokens: int,
    output_tokens: int,
    cache_read: int | None = None,
    cache_write: int | None = None,
) -> dict[str, Any]:
    usage: dict[str, Any] = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    if cache_read is not None:
        usage["cache_read_input_tokens"] = cache_read
    if cache_write is not None:
        usage["cache_creation_input_tokens"] = cache_write
    return usage


def openai_text_response(
    text: str,
    usage: dict[str, Any] | None = None,
    *,
    model: str = "gpt-4o",
    response_id: str = "chatcmpl-fake",
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": usage if usage is not None else openai_usage(10, 5),
    }


def openai_tool_call_response(
    name: str,
    arguments: dict[str, Any] | str,
    usage: dict[str, Any] | None = None,
    *,
    call_id: str = "call_fake_1",
    model: str = "gpt-4o",
    response_id: str = "chatcmpl-fake-tool",
) -> dict[str, Any]:
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": usage if usage is not None else openai_usage(10, 5),
    }


def anthropic_text_response(
    text: str,
    usage: dict[str, Any] | None = None,
    *,
    model: str = "claude-sonnet-4-5",
    response_id: str = "msg_fake",
) -> dict[str, Any]:
    return {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage if usage is not None else anthropic_usage(10, 5),
    }


def anthropic_tool_use_response(
    name: str,
    tool_input: dict[str, Any],
    usage: dict[str, Any] | None = None,
    *,
    tool_use_id: str = "toolu_fake_1",
    model: str = "claude-sonnet-4-5",
    response_id: str = "msg_fake_tool",
) -> dict[str, Any]:
    return {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "tool_use", "id": tool_use_id, "name": name, "input": tool_input}],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": usage if usage is not None else anthropic_usage(10, 5),
    }


# --------------------------------------------------------------------------- #
# SSE rendering                                                                #
# --------------------------------------------------------------------------- #


def _sse(event: str | None, data: Any) -> bytes:
    head = f"event: {event}\n" if event else ""
    return (head + "data: " + json.dumps(data, separators=(",", ":")) + "\n\n").encode()


def render_openai_sse(response: dict[str, Any], include_usage: bool) -> bytes:
    """Chunk a chat-completion JSON body the way OpenAI streams it."""
    base = {
        "id": response.get("id", "chatcmpl-fake"),
        "object": "chat.completion.chunk",
        "created": response.get("created", 0),
        "model": response.get("model", ""),
    }
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    out = bytearray()
    out += _sse(
        None,
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]},
    )
    if message.get("content"):
        out += _sse(
            None,
            {
                **base,
                "choices": [
                    {"index": 0, "delta": {"content": message["content"]}, "finish_reason": None}
                ],
            },
        )
    for i, tc in enumerate(message.get("tool_calls") or []):
        out += _sse(
            None,
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"tool_calls": [{"index": i, **tc}]},
                        "finish_reason": None,
                    }
                ],
            },
        )
    out += _sse(
        None,
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": choice.get("finish_reason")}],
        },
    )
    if include_usage and response.get("usage") is not None:
        out += _sse(None, {**base, "choices": [], "usage": response["usage"]})
    out += b"data: [DONE]\n\n"
    return bytes(out)


def render_anthropic_sse(response: dict[str, Any]) -> bytes:
    """Stream a Messages JSON body the way Anthropic does."""
    usage = dict(response.get("usage") or {})
    start_usage = {k: v for k, v in usage.items() if k != "output_tokens"}
    start_usage.setdefault("output_tokens", 1)
    head = {k: v for k, v in response.items() if k not in ("content", "usage")}
    out = bytearray()
    out += _sse(
        "message_start",
        {"type": "message_start", "message": {**head, "content": [], "usage": start_usage}},
    )
    for i, block in enumerate(response.get("content") or []):
        if block.get("type") == "text":
            out += _sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": i,
                    "content_block": {"type": "text", "text": ""},
                },
            )
            out += _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "text_delta", "text": block.get("text", "")},
                },
            )
        else:
            start_block = {k: v for k, v in block.items() if k != "input"}
            start_block["input"] = {}
            out += _sse(
                "content_block_start",
                {"type": "content_block_start", "index": i, "content_block": start_block},
            )
            out += _sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(block.get("input") or {}),
                    },
                },
            )
        out += _sse("content_block_stop", {"type": "content_block_stop", "index": i})
    out += _sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": response.get("stop_reason"), "stop_sequence": None},
            "usage": {"output_tokens": usage.get("output_tokens", 0)},
        },
    )
    out += _sse("message_stop", {"type": "message_stop"})
    return bytes(out)


# --------------------------------------------------------------------------- #
# The provider                                                                 #
# --------------------------------------------------------------------------- #


def _default_body(call: ProviderCall) -> dict[str, Any]:
    if call.path.endswith(ANTHROPIC_MESSAGES_PATH):
        return anthropic_text_response("ok", model=str(call.body.get("model", "")))
    return openai_text_response("ok", model=str(call.body.get("model", "")))


class FakeProvider:
    """See the module docstring for the API summary."""

    def __init__(self, default: Script | None = None) -> None:
        self.calls: list[ProviderCall] = []
        self._queue: deque[Script] = deque()
        self._default: Script = default if default is not None else _default_body
        self._lock = threading.Lock()
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._app: Any = None

    # --- scripting ---------------------------------------------------------

    def script(self, responses: list[Script]) -> FakeProvider:
        """Replace the queue with ``responses`` (consumed in order)."""
        with self._lock:
            self._queue = deque(responses)
        return self

    def push(self, *responses: Script) -> FakeProvider:
        with self._lock:
            self._queue.extend(responses)
        return self

    def reset(self) -> None:
        with self._lock:
            self._queue.clear()
            self.calls.clear()

    @property
    def pending(self) -> int:
        return len(self._queue)

    # --- dispatch ----------------------------------------------------------

    def _next(self, call: ProviderCall) -> ScriptedResponse:
        with self._lock:
            item: Script = self._queue.popleft() if self._queue else self._default
        if callable(item) and not isinstance(item, dict):
            item = item(call)
        if isinstance(item, ScriptedResponse):
            return item
        if isinstance(item, dict):
            return ScriptedResponse(body=item)
        raise TypeError(f"unsupported scripted item: {type(item).__name__}")

    def handle(self, path: str, body: dict[str, Any], headers: dict[str, str]) -> ProviderReply:
        """Pure-Python request dispatch: record the call, pick the next scripted
        answer and render it (JSON or SSE). Used by the ASGI app, and directly by
        tests that bridge the proxy's httpx upstream into this provider."""
        call = ProviderCall(
            path=path, body=body, headers={k.lower(): v for k, v in headers.items()}
        )
        self.calls.append(call)
        scripted = self._next(call)
        response_headers = {k.lower(): v for k, v in scripted.headers.items()}
        if call.stream and scripted.status == 200:
            if path.endswith(ANTHROPIC_MESSAGES_PATH):
                content = render_anthropic_sse(scripted.body)
            else:
                opts = body.get("stream_options") or {}
                content = render_openai_sse(scripted.body, bool(opts.get("include_usage")))
            response_headers.setdefault("content-type", "text/event-stream")
            return ProviderReply(scripted.status, response_headers, content, None)
        response_headers.setdefault("content-type", "application/json")
        return ProviderReply(
            scripted.status,
            response_headers,
            json.dumps(scripted.body).encode(),
            scripted.body,
        )

    # --- ASGI ----------------------------------------------------------------

    @property
    def app(self) -> Any:
        if self._app is None:
            self._app = self._build_app()
        return self._app

    def _build_app(self) -> Any:
        if FastAPI is None:
            raise RuntimeError("fastapi is required for FakeProvider.app")
        app = FastAPI(title="fake-provider")
        provider = self

        async def _serve(request: Request) -> Response:
            raw = await request.body()
            try:
                body = json.loads(raw) if raw else {}
            except ValueError:
                return Response(
                    content=json.dumps({"error": {"message": "invalid json"}}),
                    status_code=400,
                    media_type="application/json",
                )
            reply = provider.handle(request.url.path, body, dict(request.headers))
            return Response(
                content=reply.content,
                status_code=reply.status,
                headers={k: v for k, v in reply.headers.items() if k != "content-type"},
                media_type=reply.headers.get("content-type", "application/json"),
            )

        # Registered under both the bare path and a ``/openai`` / ``/anthropic``
        # prefix so a Kong route can point at either without rewriting.
        for prefix in ("", "/openai", "/anthropic"):
            app.add_api_route(f"{prefix}{OPENAI_CHAT_PATH}", _serve, methods=["POST"])
            app.add_api_route(f"{prefix}{ANTHROPIC_MESSAGES_PATH}", _serve, methods=["POST"])

        @app.get("/health")
        async def _health() -> dict[str, Any]:
            return {"ok": True, "calls": len(provider.calls), "pending": provider.pending}

        @app.get("/calls")
        async def _calls() -> list[dict[str, Any]]:
            return [{"path": c.path, "body": c.body, "headers": c.headers} for c in provider.calls]

        return app

    # --- uvicorn thread ------------------------------------------------------

    def start(self, port: int = 0, host: str = "127.0.0.1", timeout: float = 10.0) -> str:
        """Serve ``app`` with uvicorn in a daemon thread; returns the base URL."""
        import uvicorn

        if self._server is not None:
            raise RuntimeError("fake provider already started")
        if port == 0:
            with socket.socket() as s:
                s.bind((host, 0))
                port = s.getsockname()[1]
        config = uvicorn.Config(self.app, host=host, port=port, log_level="warning", lifespan="off")
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, name="fake-provider", daemon=True)
        self._thread.start()
        deadline = time.time() + timeout
        while not self._server.started:
            if time.time() > deadline:
                self.stop()
                raise RuntimeError("fake provider did not start in time")
            if not self._thread.is_alive():
                raise RuntimeError("fake provider thread exited during startup")
            time.sleep(0.02)
        return f"http://{host}:{port}"

    def stop(self, timeout: float = 5.0) -> None:
        if self._server is None:
            return
        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout)
        self._server = None
        self._thread = None


# --------------------------------------------------------------------------- #
# Usage extraction from a wire response (what the Kong body_filter does)       #
# --------------------------------------------------------------------------- #


def extract_usage(content: bytes, content_type: str) -> dict[str, Any] | None:
    """Pull the usage object out of a JSON or SSE provider response.

    JSON: ``body["usage"]``. OpenAI SSE: the last ``data:`` frame carrying a
    ``usage`` object. Anthropic SSE: ``message_start.message.usage`` merged with
    ``message_delta.usage`` (input side from the start, output side from the
    delta) - exactly the merge the Lua ``body_filter`` performs.
    """
    if not content_type.startswith("text/event-stream"):
        try:
            parsed = json.loads(content)
        except ValueError:
            return None
        usage = parsed.get("usage") if isinstance(parsed, dict) else None
        return usage if isinstance(usage, dict) else None
    merged: dict[str, Any] = {}
    last_openai: dict[str, Any] | None = None
    for raw_line in content.decode("utf-8", errors="replace").splitlines():
        if not raw_line.startswith("data:"):
            continue
        payload = raw_line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            frame = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(frame, dict):
            continue
        kind = frame.get("type")
        if kind == "message_start":
            msg_usage = (frame.get("message") or {}).get("usage")
            if isinstance(msg_usage, dict):
                merged.update(msg_usage)
        elif kind == "message_delta":
            delta_usage = frame.get("usage")
            if isinstance(delta_usage, dict):
                merged.update(delta_usage)
        elif isinstance(frame.get("usage"), dict):
            last_openai = frame["usage"]
    if last_openai is not None:
        return last_openai
    return merged or None
