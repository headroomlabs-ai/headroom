"""``on_responses_create``: one routing seam for OpenAI Responses on both transports.

Codex sends ``/v1/responses`` over a WebSocket when wrapped, and an HTTP
middleware never sees those frames, so a router installed as middleware did
nothing for wrapped Codex. A turn hook's ``on_responses_create`` runs on the
HTTP body and on every WebSocket ``response.create`` frame, before the proxy's
own stages, so a model/effort rewrite reaches the upstream and the accounting.
"""

from __future__ import annotations

import json
import sys
from typing import Any
from unittest.mock import patch

import pytest

from headroom.proxy.turn_hooks import (
    ResponsesCreate,
    clear_turn_hooks,
    register_turn_hook,
)
from tests.test_openai_codex_ws_lifecycle import (
    _DummyOpenAIHandler,
    _FakeUpstream,
    _FakeWebSocket,
    _make_fake_websockets_module,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    clear_turn_hooks()
    yield
    clear_turn_hooks()


class _Router:
    name = "test-router"

    def __init__(self) -> None:
        self.seen: list[tuple[str, str, str | None]] = []
        self.connections: list[str | None] = []

    async def on_responses_create(self, request: ResponsesCreate) -> dict:
        self.seen.append(
            (request.transport, str(request.body.get("model")), request.headers.get("user-agent"))
        )
        self.connections.append(request.connection_id)
        return {**request.body, "model": "gpt-5.4-mini", "reasoning": {"effort": "low"}}


def _create(text: str, **extra: Any) -> str:
    return json.dumps(
        {"type": "response.create", "response": {"model": "gpt-5.5", "input": text, **extra}}
    )


def _upstream() -> _FakeUpstream:
    return _FakeUpstream(
        [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
        ],
        hold_after_events=True,
    )


async def _run_ws(frames: list[str], headers: dict[str, str] | None = None) -> list[dict]:
    upstream = _upstream()
    client_ws = _FakeWebSocket(frames=frames, headers=headers)
    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        await _DummyOpenAIHandler().handle_openai_responses_ws(client_ws)
    return [json.loads(frame) for frame in upstream.sent]


@pytest.mark.asyncio
async def test_ws_create_frames_carry_the_hooks_model_and_effort_upstream() -> None:
    router = _Router()
    register_turn_hook(router)

    # previous_response_id chains are left to the client and OpenAI: a model
    # switch mid-chain is allowed, so the hook's rewrite rides alongside it.
    sent = await _run_ws(
        [
            _create("first"),
            json.dumps({"type": "response.cancel"}),
            _create("next", previous_response_id="r_1"),
        ]
    )

    creates = [frame["response"] for frame in sent if frame.get("type") == "response.create"]
    assert [c["model"] for c in creates] == ["gpt-5.4-mini", "gpt-5.4-mini"]
    assert [c["reasoning"] for c in creates] == [{"effort": "low"}] * 2
    assert creates[1]["previous_response_id"] == "r_1"
    assert {"type": "response.cancel"} in sent
    # Compression is off in the dummy handler: routing must not depend on it.
    assert router.seen == [
        ("websocket", "gpt-5.5", "codex-cli/0.5"),
        ("websocket", "gpt-5.5", "codex-cli/0.5"),
    ]
    # One connection, one id: what a hook keys a previous_response_id chain on.
    assert router.connections[0] and len(set(router.connections)) == 1


@pytest.mark.asyncio
async def test_ws_bypass_header_skips_the_hook() -> None:
    router = _Router()
    register_turn_hook(router)
    frames = [_create("first"), _create("next")]

    sent = await _run_ws(frames, headers={"authorization": "Bearer t", "x-headroom-bypass": "true"})

    assert router.seen == []
    assert [frame["response"]["model"] for frame in sent] == ["gpt-5.5", "gpt-5.5"]


@pytest.mark.asyncio
async def test_a_hook_that_raises_does_not_break_the_session() -> None:
    class _Broken:
        name = "broken"

        async def on_responses_create(self, request: ResponsesCreate) -> None:
            # The body is read-only: a hook cannot half-apply an edit and then fail.
            request.body["model"] = "gpt-5.4-mini"  # type: ignore[index]

    register_turn_hook(_Broken())
    frames = [_create("first")]

    sent = await _run_ws(frames)

    assert sent == [json.loads(frames[0])]


class _NoOp:
    name = "no-op"

    async def on_responses_create(self, request: ResponsesCreate) -> None:
        return None


def _post_http(raw: bytes, headers: dict[str, str] | None = None) -> tuple[Any, list, list]:
    """POST raw bytes to /v1/responses.

    Returns the response, what reached the upstream call as
    ``(body, body_mutated)`` pairs, and the recorded outcomes.
    """
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from headroom.proxy.loopback_guard import require_loopback
    from headroom.proxy.server import ProxyConfig, create_app

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
    )
    app.dependency_overrides[require_loopback] = lambda: None
    sent: list[Any] = []
    outcomes: list[Any] = []

    with TestClient(app) as client:
        server = app.state.proxy

        async def fake_retry(method, url, headers, body, stream=False, **kwargs):
            sent.append((_as_json(body), kwargs.get("body_mutated")))
            return httpx.Response(
                200,
                json={
                    "id": "resp_1",
                    "object": "response",
                    "status": "completed",
                    "model": "gpt-5.4-mini",
                    "output": [],
                    "usage": {"input_tokens": 10, "output_tokens": 2},
                },
                request=httpx.Request("POST", url),
            )

        record = server._record_request_outcome

        async def capture(outcome):
            outcomes.append(outcome)
            await record(outcome)

        server._retry_request = fake_retry
        server._record_request_outcome = capture
        resp = client.post(
            "/v1/responses",
            content=raw,
            headers={
                "Authorization": "Bearer sk-test",
                "Content-Type": "application/json",
                "User-Agent": "codex_exec/0.162.1",
                **(headers or {}),
            },
        )
    return resp, sent, outcomes


def _as_json(body: Any) -> dict:
    return json.loads(body) if isinstance(body, (bytes, str)) else body


_RAW = json.dumps({"model": "gpt-5.5", "input": "hi", "stream": False}).encode()


def test_http_body_carries_the_hooks_model_and_the_outcome_prices_it() -> None:
    router = _Router()
    register_turn_hook(router)

    resp, sent, outcomes = _post_http(_RAW)

    assert resp.status_code == 200, resp.text
    assert router.seen == [("http", "gpt-5.5", "codex_exec/0.162.1")]
    assert router.connections == [None]
    body, mutated = sent[0]
    assert body["model"] == "gpt-5.4-mini"
    assert body["reasoning"] == {"effort": "low"}
    assert mutated is True
    assert outcomes and all(o.model == "gpt-5.4-mini" for o in outcomes)


def test_http_hook_that_changes_nothing_keeps_the_clients_bytes() -> None:
    register_turn_hook(_NoOp())

    resp, sent, _ = _post_http(_RAW)

    assert resp.status_code == 200, resp.text
    # Unmutated is what sends the client's original bytes (#1542).
    assert sent == [(json.loads(_RAW), False)]


def test_http_bypass_header_skips_the_hook() -> None:
    router = _Router()
    register_turn_hook(router)

    resp, sent, _ = _post_http(_RAW, headers={"x-headroom-bypass": "true"})

    assert resp.status_code == 200, resp.text
    assert router.seen == []
    assert sent == [(json.loads(_RAW), False)]


def test_http_hook_that_turns_on_streaming_takes_the_streaming_path() -> None:
    class _Streamer:
        name = "streamer"

        async def on_responses_create(self, request: ResponsesCreate) -> dict:
            return {**request.body, "stream": True}

    register_turn_hook(_Streamer())
    pytest.importorskip("fastapi")
    from fastapi.responses import JSONResponse
    from fastapi.testclient import TestClient

    from headroom.proxy.loopback_guard import require_loopback
    from headroom.proxy.server import ProxyConfig, create_app

    app = create_app(
        ProxyConfig(
            optimize=False,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
    )
    app.dependency_overrides[require_loopback] = lambda: None
    streamed: list[dict] = []

    with TestClient(app) as client:
        server = app.state.proxy

        async def fake_stream(url, headers, body, *args, **kwargs):
            streamed.append(body)
            return JSONResponse({"ok": True})

        async def no_buffered(*args, **kwargs):
            raise AssertionError("a streaming upstream request took the buffered path")

        server._stream_response = fake_stream
        server._retry_request = no_buffered
        resp = client.post(
            "/v1/responses",
            content=_RAW,
            headers={"Authorization": "Bearer sk-test", "Content-Type": "application/json"},
        )

    assert resp.status_code == 200, resp.text
    assert [b["stream"] for b in streamed] == [True]
