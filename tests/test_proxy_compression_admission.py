"""Large histories must be refused before their bodies enter preprocessing."""

from __future__ import annotations

import json

import pytest

from headroom.proxy.request_body_limit import RequestBodyLimitMiddleware


@pytest.mark.asyncio
@pytest.mark.parametrize("declared", [True, False])
async def test_image_heavy_history_is_refused_before_json_preprocessing(declared: bool) -> None:
    image = "data:image/png;base64," + "YQ==" * (80 * 1024)
    body = json.dumps(
        {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": image}}],
                }
                for _ in range(64)
            ],
        }
    ).encode()
    sent = []
    offset = 0
    buffered = bytearray()
    chunk_size = 64 * 1024

    async def receive():
        nonlocal offset
        chunk = body[offset : offset + chunk_size]
        offset += len(chunk)
        return {"type": "http.request", "body": chunk, "more_body": offset < len(body)}

    async def send(message):
        sent.append(message)

    async def preprocessing(scope, receive, send):
        while True:
            message = await receive()
            buffered.extend(message["body"])
            if not message["more_body"]:
                break
        json.loads(buffered)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"accepted"})

    headers = [(b"content-type", b"application/json")]
    if declared:
        headers.append((b"content-length", str(len(body)).encode()))
    scope = {"type": "http", "method": "POST", "path": "/v1/compress", "headers": headers}
    await RequestBodyLimitMiddleware(preprocessing)(scope, receive, send)

    assert [message["status"] for message in sent if message["type"] == "http.response.start"] == [
        413
    ]
    error = json.loads(sent[-1]["body"])
    assert error["error"]["code"] == "request_too_large"
    if declared:
        assert offset == 0
        assert buffered == b""
    else:
        assert offset < len(body)
        assert len(buffered) < offset


@pytest.mark.parametrize("raw", ["0", "-1", "not-int", "1.5"])
def test_invalid_request_body_ceiling_is_rejected(raw: str) -> None:
    from headroom.proxy.request_limit_policy import resolve_request_body_max_bytes

    with pytest.raises(ValueError):
        resolve_request_body_max_bytes(raw)
