"""Round-one independent-review regressions at the serving/observer boundary."""

import asyncio
import json
import time

import httpx
import pytest
from starlette.requests import Request

from headroom.proxy.gateway.config import LimitsConfig
from headroom.proxy.gateway.dispatch import dispatch_native_http
from headroom.proxy.gateway.errors import GatewayPublicError
from headroom.proxy.gateway.streaming import StreamObserver
from tests.unified_gateway.test_http_execution import application


async def chunks(*parts):
    for part in parts:
        yield part


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", [b"", b'data: {"type":"error","error":{"message":"SECRET"}}\n\n'])
async def test_anthropic_intermediate_counters_retain_liability(tail):
    observer = StreamObserver("anthropic-messages", LimitsConfig(), time.monotonic() + 1)
    initial = b'data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":1}}}\n\n'
    with pytest.raises(GatewayPublicError):
        _ = [part async for part in observer.observe(chunks(initial, tail))]
    assert observer.usage.input_tokens == 10
    assert observer.usage.output_tokens == 1
    assert observer.usage.availability == "partial"


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["gemini-generate", "vertex-generate"])
async def test_all_gemini_candidates_and_final_usage_survive(protocol):
    frames = [
        b'data: {"candidates":[{"index":0,"content":{"parts":[{"text":"a"}]}},{"index":1,"content":{"parts":[{"text":"b"}]}}]}\n\n',
        b'data: {"candidates":[{"index":0,"finishReason":"STOP"}]}\n\n',
        b'data: {"candidates":[{"index":1,"content":{"parts":[{"text":"tail"}]} ,"finishReason":"STOP"}]}\n\n',
        b'data: {"usageMetadata":{"promptTokenCount":3,"candidatesTokenCount":7}}\n\n',
    ]
    observer = StreamObserver(protocol, LimitsConfig(), time.monotonic() + 1)
    assert [part async for part in observer.observe(chunks(*frames))] == frames
    assert observer.terminal == "success"
    assert observer.usage.output_tokens == 7


@pytest.mark.asyncio
async def test_responses_refusal_preserves_terminal_usage_without_success():
    frames = [
        b'data: {"type":"response.refusal.delta","delta":"Cannot comply"}\n\n',
        b'data: {"type":"response.refusal.done","refusal":"Cannot comply"}\n\n',
        b'data: {"type":"response.completed","response":{"status":"completed","output":[{"type":"message","content":[{"type":"refusal","refusal":"Cannot comply"}]}],"usage":{"input_tokens":3,"output_tokens":2}}}\n\n',
    ]
    observer = StreamObserver("openai-responses", LimitsConfig(), time.monotonic() + 1)
    with pytest.raises(GatewayPublicError):
        _ = [part async for part in observer.observe(chunks(*frames))]
    assert observer.terminal == "failed"
    assert observer.usage.output_tokens == 2


def test_nonstream_responses_refusal_is_failed_with_accounting(monkeypatch):
    from fastapi.testclient import TestClient

    async def handler(request):
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "refusal", "refusal": "Cannot comply"}],
                    }
                ],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        )

    app, model = application(monkeypatch, handler)
    response = TestClient(app).post(
        "/v1/responses",
        headers={"host": "127.0.0.1:8787", "authorization": "Bearer client-secret"},
        json={"model": model, "input": "hello"},
    )
    assert response.status_code == 502
    assert app.state.gateway_runtime.admission.snapshot().known_micro_usd == 7


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage,availability",
    [({"prompt_tokens": 3, "completion_tokens": 2}, "complete"), ({"prompt_tokens": 3}, "partial")],
)
async def test_error_event_extracts_usage_before_sanitizing(usage, availability):
    observer = StreamObserver("openai-chat", LimitsConfig(), time.monotonic() + 1)
    frame = (
        b"event: error\r\ndata: "
        + json.dumps({"error": {"message": "SECRET"}, "usage": usage}).encode()
        + b"\r\n\r\n"
    )
    with pytest.raises(GatewayPublicError) as error:
        _ = [part async for part in observer.observe(chunks(frame))]
    assert "SECRET" not in str(error.value)
    assert observer.usage.input_tokens == 3
    assert observer.usage.availability == availability


@pytest.mark.asyncio
async def test_buffered_frames_cannot_cross_absolute_deadline(monkeypatch):
    observer = StreamObserver("openai-chat", LimitsConfig(), time.monotonic() + 60)
    initial = b'data: {"choices":[{"index":0,"finish_reason":"stop"}]}\n\n'
    iterator = observer.observe(chunks(initial + b"data: [DONE]\n\n"))
    assert await anext(iterator) == initial
    # Only the observer clock is replaced; the event loop retains its real clock.
    monkeypatch.setattr(
        "headroom.proxy.gateway.streaming.time",
        type("Clock", (), {"monotonic": staticmethod(lambda: observer.deadline + 1)}),
    )
    with pytest.raises(GatewayPublicError):
        await anext(iterator)
    assert observer.terminal != "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["disconnect", "deadline-before-start", "deadline-after-body"])
async def test_starlette_disconnect_before_iterator_start_closes_owner(monkeypatch, mode):
    closed, iterated, response_start = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Upstream(httpx.AsyncByteStream):
        async def __aiter__(self):
            iterated.set()
            yield b'data: {"choices":[{"index":0,"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'

        async def aclose(self):
            closed.set()

    async def handler(request):
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Upstream())

    app, model = application(monkeypatch, handler)
    body = json.dumps({"model": model, "messages": [], "stream": True}).encode()

    async def body_receive():
        return {"type": "http.request", "body": body}

    scope = {
        "type": "http",
        "asgi": {"spec_version": "2.3"},
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": [],
        "server": ("127.0.0.1", 8787),
        "app": app,
    }
    request = Request(scope, body_receive)
    runtime = app.state.gateway_runtime
    request.state.gateway_generation = runtime.capture()
    request.state.gateway_principal = request.state.gateway_generation.authenticator.authenticate(
        {"authorization": "Bearer client-secret"}
    )
    response = await dispatch_native_http(request, app.state.proxy, "openai-chat")
    assert runtime.admission.active_count == 1
    if mode.startswith("deadline"):
        response.operation.deadline = time.monotonic() + 0.05

    async def send(message):
        if message["type"] == (
            "http.response.body" if mode == "deadline-after-body" else "http.response.start"
        ):
            response_start.set()
            await asyncio.Event().wait()

    async def receive():
        await response_start.wait()
        if mode != "disconnect":
            await asyncio.Event().wait()
        return {"type": "http.disconnect"}

    if mode == "disconnect":
        await asyncio.wait_for(response(scope, receive, send), 1)
    else:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(response(scope, receive, send), 1)
        assert response.operation.terminal != "success"
    assert iterated.is_set() == (mode == "deadline-after-body")
    assert closed.is_set()
    assert runtime.admission.active_count == 0
    assert not runtime.active_work
    assert runtime.admission.snapshot().unknown_charge_count == 1
