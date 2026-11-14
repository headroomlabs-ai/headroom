"""The proxy runs only the memory tools it injected on the request.

A memory-named call the proxy did not offer (``--no-memory-tools``) or that
the client declared itself belongs to the client: it reaches the client
unchanged, the memory backend is not touched, and no continuation is sent
upstream. Calls to tools the proxy did inject still run on the proxy.
"""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from headroom.proxy.server import ProxyConfig, create_app


class _FakeBackend:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def list_memories(self, *, user_id: str, limit: int) -> list[Any]:
        self.calls.append("list")
        return [SimpleNamespace(id="mem-1", content="stored", created_at=None)]

    async def delete_memory(self, memory_id: str, *args: Any, **kwargs: Any) -> bool:
        self.calls.append("delete")
        return True


def _make_app(tmp_path, *, inject_tools: bool) -> tuple[Any, _FakeBackend]:
    config = ProxyConfig(
        openai_api_url="https://openai.invalid",
        anthropic_api_url="https://anthropic.invalid",
        optimize=False,
        image_optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        discover_pipeline_extensions=False,
        compression_max_workers=1,
        memory_enabled=True,
        memory_storage_mode="global",
        memory_inject_tools=inject_tools,
        memory_inject_context=False,
        memory_db_path=str(tmp_path / "memory.db"),
    )
    app = create_app(config)
    handler = app.state.proxy.memory_handler
    backend = _FakeBackend()
    handler._backend = backend
    handler._initialized = True
    router = handler._router
    if router is not None:
        router.backend_for = lambda ctx: (backend, router.scope_for(ctx))
    return app, backend


# --- upstream turns ---------------------------------------------------------


def _chat_turn(name: str | None) -> dict[str, Any]:
    if name is None:
        message: dict[str, Any] = {"role": "assistant", "content": "done"}
    else:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": name, "arguments": '{"memory_id": "mem-1"}'},
                }
            ],
        }
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _responses_item(name: str) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": name,
        "arguments": '{"memory_id": "mem-1"}',
        "status": "completed",
    }


def _responses_turn(name: str | None) -> dict[str, Any]:
    item = (
        _responses_item(name)
        if name
        else {
            "type": "message",
            "id": "msg_1",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "done", "annotations": []}],
        }
    )
    return {
        "id": "resp_1",
        "object": "response",
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [item],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


def _anthropic_turn(name: str | None) -> dict[str, Any]:
    if name is None:
        content: list[dict[str, Any]] = [{"type": "text", "text": "done"}]
    else:
        content = [
            {"type": "tool_use", "id": "toolu_1", "name": name, "input": {"memory_id": "mem-1"}}
        ]
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": content,
        "stop_reason": "tool_use" if name else "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _anthropic_sse(name: str) -> bytes:
    turn = _anthropic_turn(name)
    events = [
        ("message_start", {"type": "message_start", "message": {**turn, "content": []}}),
        (
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": name, "input": {}},
            },
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"memory_id": "mem-1"}'},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events).encode()


# --- HTTP paths -------------------------------------------------------------

_OPENAI_TOOL = {
    "type": "function",
    "function": {"name": "shell", "parameters": {"type": "object", "properties": {}}},
}
_OPENAI_CLIENT_MEMORY_DELETE = {
    "type": "function",
    "function": {"name": "memory_delete", "parameters": {"type": "object", "properties": {}}},
}
_RESPONSES_TOOL = {"type": "function", "name": "shell", "parameters": {"type": "object"}}
_RESPONSES_CLIENT_MEMORY_DELETE = {
    "type": "function",
    "name": "memory_delete",
    "parameters": {"type": "object"},
}
_ANTHROPIC_TOOL = {"name": "Bash", "input_schema": {"type": "object"}}
_ANTHROPIC_CLIENT_MEMORY_DELETE = {"name": "memory_delete", "input_schema": {"type": "object"}}

_PATHS: dict[str, dict[str, Any]] = {
    "chat": {
        "url": "/v1/chat/completions",
        "headers": {"authorization": "Bearer FAKE-KEY"},
        "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
        "tool": _OPENAI_TOOL,
        "client_memory_delete": _OPENAI_CLIENT_MEMORY_DELETE,
        "turn": _chat_turn,
    },
    "responses": {
        "url": "/v1/responses",
        "headers": {"authorization": "Bearer FAKE-KEY"},
        "body": {"model": "gpt-4o-mini", "input": "hi"},
        "tool": _RESPONSES_TOOL,
        "client_memory_delete": _RESPONSES_CLIENT_MEMORY_DELETE,
        "turn": _responses_turn,
    },
    "anthropic": {
        "url": "/v1/messages",
        "headers": {"x-api-key": "FAKE-KEY", "anthropic-version": "2023-06-01"},
        "body": {
            "model": "claude-sonnet-4-5",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}],
        },
        "tool": _ANTHROPIC_TOOL,
        "client_memory_delete": _ANTHROPIC_CLIENT_MEMORY_DELETE,
        "turn": _anthropic_turn,
    },
    "anthropic_stream": {
        "url": "/v1/messages",
        "headers": {"x-api-key": "FAKE-KEY", "anthropic-version": "2023-06-01"},
        "body": {
            "model": "claude-sonnet-4-5",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        "tool": _ANTHROPIC_TOOL,
        "client_memory_delete": _ANTHROPIC_CLIENT_MEMORY_DELETE,
        "turn": _anthropic_turn,
    },
}


async def _post(app: Any, kind: str, tools: list[dict[str, Any]], call: str) -> tuple:
    spec = _PATHS[kind]
    upstream_bodies: list[dict[str, Any]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        upstream_bodies.append(json.loads(request.content))
        first = len(upstream_bodies) == 1
        if kind == "anthropic_stream" and first:
            return httpx.Response(
                200, content=_anthropic_sse(call), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(200, json=spec["turn"](call if first else None))

    mock = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    app.state.proxy.http_client = mock
    app.state.proxy.http_client_h1 = mock
    transport = httpx.ASGITransport(app=app, client=("127.0.0.1", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post(
            spec["url"],
            json={**spec["body"], "tools": tools},
            headers={**spec["headers"], "x-headroom-user-id": "user-1"},
            timeout=30,
        )
    await mock.aclose()
    return response, upstream_bodies


@pytest.mark.parametrize("kind", list(_PATHS))
@pytest.mark.parametrize("case", ["not_injected", "client_declared"])
async def test_memory_call_the_proxy_did_not_inject_reaches_the_client(
    tmp_path, monkeypatch, kind: str, case: str
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    spec = _PATHS[kind]
    app, backend = _make_app(tmp_path, inject_tools=case == "client_declared")
    tools = [spec["tool"]]
    if case == "client_declared":
        tools.append(spec["client_memory_delete"])

    response, upstream_bodies = await _post(app, kind, tools, "memory_delete")

    assert response.status_code == 200
    assert backend.calls == []
    assert len(upstream_bodies) == 1
    assert "memory_delete" in response.text
    assert ("toolu_1" if kind.startswith("anthropic") else "call_1") in response.text


@pytest.mark.parametrize("kind", ["chat", "responses", "anthropic"])
async def test_memory_call_the_proxy_injected_still_runs(tmp_path, monkeypatch, kind: str) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    app, backend = _make_app(tmp_path, inject_tools=True)

    response, upstream_bodies = await _post(app, kind, [_PATHS[kind]["tool"]], "memory_list")

    assert response.status_code == 200
    assert backend.calls == ["list"]
    assert len(upstream_bodies) == 2
    assert "memory_list" not in response.text


# --- Responses over WebSocket ---------------------------------------------


class _Upstream:
    """Upstream WS that answers the first response.create with one tool call."""

    def __init__(self, call: str) -> None:
        self.call = call
        self.sent: list[dict[str, Any]] = []
        self._events: asyncio.Queue[str] = asyncio.Queue()

    async def __aenter__(self) -> _Upstream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))
        rid = f"resp_{len(self.sent)}"
        if len(self.sent) == 1:
            item = _responses_item(self.call)
        else:
            item = _responses_turn(None)["output"][0]
        for event in (
            {"type": "response.created", "response": {"id": rid}},
            {"type": "response.output_item.added", "item": item},
            {"type": "response.output_item.done", "item": item},
            {"type": "response.completed", "response": {"id": rid}},
        ):
            await self._events.put(json.dumps(event))

    async def close(self) -> None:
        return None

    def __aiter__(self) -> _Upstream:
        return self

    async def __anext__(self) -> str:
        return await self._events.get()


def _ws_exchange(
    app: Any, tools: list[dict[str, Any]], call: str
) -> tuple[list[dict[str, Any]], _Upstream]:
    upstream = _Upstream(call)

    async def connect(*_args: Any, **_kwargs: Any) -> _Upstream:
        return upstream

    websockets_module = MagicMock()
    websockets_module.connect = connect
    websockets_module.Subprotocol = str
    frame = {
        "type": "response.create",
        "response": {"model": "gpt-4o-mini", "input": "hi", "tools": tools},
    }
    received: list[dict[str, Any]] = []
    with patch.dict(sys.modules, {"websockets": websockets_module}):
        client = TestClient(app, client=("127.0.0.1", 12345))
        with client.websocket_connect(
            "/v1/responses",
            headers={"authorization": "Bearer FAKE-KEY", "x-headroom-user-id": "user-1"},
        ) as websocket:
            websocket.send_text(json.dumps(frame))
            while True:
                event = json.loads(websocket.receive_text())
                received.append(event)
                if event.get("type") == "response.completed":
                    break
    return received, upstream


@pytest.mark.parametrize("case", ["not_injected", "client_declared"])
def test_ws_memory_call_the_proxy_did_not_inject_reaches_the_client(
    tmp_path, monkeypatch, case: str
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    app, backend = _make_app(tmp_path, inject_tools=case == "client_declared")
    tools = [_RESPONSES_TOOL]
    if case == "client_declared":
        tools.append(_RESPONSES_CLIENT_MEMORY_DELETE)

    received, upstream = _ws_exchange(app, tools, "memory_delete")

    assert backend.calls == []
    assert len(upstream.sent) == 1
    assert any(
        event.get("type") == "response.output_item.done"
        and event["item"].get("name") == "memory_delete"
        for event in received
    )


def test_ws_memory_call_the_proxy_injected_still_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    app, backend = _make_app(tmp_path, inject_tools=True)

    received, upstream = _ws_exchange(app, [_RESPONSES_TOOL], "memory_list")

    assert backend.calls == ["list"]
    assert len(upstream.sent) == 2
    assert not any(event.get("item", {}).get("name") == "memory_list" for event in received)
