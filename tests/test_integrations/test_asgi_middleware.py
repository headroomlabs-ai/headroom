"""Request-body forwarding through the ASGI compression middleware."""

import json

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from headroom.integrations.asgi import CompressionMiddleware


@pytest.mark.parametrize("compress", [True, False])
def test_forwarded_length_matches_request_body(monkeypatch, compress):
    async def downstream(request: Request):
        body = await request.body()
        return JSONResponse(
            {
                "declared_length": int(request.headers["content-length"]),
                "received_length": len(body),
                "messages": json.loads(body)["messages"],
            }
        )

    original = [{"role": "tool", "content": "Repeated build output\n" * 200}]
    compressed = [{"role": "tool", "content": "Build output"}]
    monkeypatch.setattr(
        CompressionMiddleware,
        "_local_compress",
        lambda *_: {"messages": compressed, "tokens_saved": 100} if compress else None,
    )
    app = Starlette(routes=[Route("/v1/chat/completions", downstream, methods=["POST"])])
    app.add_middleware(CompressionMiddleware)
    request_body = json.dumps({"model": "gpt-4o", "messages": original}).encode()

    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", content=request_body)

    response.raise_for_status()
    observed = response.json()
    assert observed["declared_length"] == observed["received_length"]
    assert observed["messages"] == (compressed if compress else original)
    if compress:
        assert observed["received_length"] < len(request_body)
        assert response.headers["x-headroom-compressed"] == "true"
    else:
        assert observed["received_length"] == len(request_body)


@pytest.mark.asyncio
async def test_streamed_request_updates_mixed_case_content_length(monkeypatch):
    original = [{"role": "tool", "content": "Repeated build output\n" * 200}]
    compressed = [{"role": "tool", "content": "Build output"}]
    monkeypatch.setattr(
        CompressionMiddleware,
        "_local_compress",
        lambda *_: {"messages": compressed, "tokens_saved": 100},
    )
    request_body = json.dumps({"model": "gpt-4o", "messages": original}).encode()
    split = len(request_body) // 2
    chunks = iter(
        [
            {"type": "http.request", "body": request_body[:split], "more_body": True},
            {"type": "http.request", "body": request_body[split:], "more_body": False},
        ]
    )
    headers = [(b"Content-Length", str(len(request_body)).encode()), (b"x-request-id", b"123")]
    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": headers}
    sent = []

    async def receive():
        return next(chunks)

    async def send(message):
        sent.append(message)

    async def downstream(forwarded_scope, forwarded_receive, forwarded_send):
        request = await forwarded_receive()
        body = request["body"]
        assert request["more_body"] is False
        assert json.loads(body)["messages"] == compressed
        assert len(body) < len(request_body)
        assert forwarded_scope["headers"] == [
            (b"Content-Length", str(len(body)).encode()),
            (b"x-request-id", b"123"),
        ]
        await forwarded_send({"type": "http.response.start", "status": 204})
        await forwarded_send({"type": "http.response.body", "body": b""})

    await CompressionMiddleware(downstream)(scope, receive, send)

    assert headers[0] == (b"Content-Length", str(len(request_body)).encode())
    assert (b"x-headroom-compressed", b"true") in sent[0]["headers"]
