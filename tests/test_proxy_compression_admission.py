"""Large histories must be refused before their bodies enter preprocessing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

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


def test_startup_override_changes_http_admission_after_restart() -> None:
    program = textwrap.dedent(
        """
        import json
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Route
        from starlette.testclient import TestClient
        from headroom.proxy.request_body_limit import RequestBodyLimitMiddleware

        async def preprocess(request):
            await request.json()
            return JSONResponse({"preprocessed": True})

        app = Starlette(routes=[Route("/v1/compress", preprocess, methods=["POST"])])
        app.add_middleware(RequestBodyLimitMiddleware)
        with TestClient(app) as client:
            response = client.post(
                "/v1/compress",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "a" * 1500}]},
            )
            print(json.dumps({"status": response.status_code, "body": response.json()}))
        """
    )
    results = []
    for ceiling in (1024, 2048):
        process = subprocess.run(
            [sys.executable, "-c", program],
            env={
                **os.environ,
                "HEADROOM_REQUEST_BODY_MAX_BYTES": str(ceiling),
                "DO_NOT_TRACK": "1",
                "HEADROOM_TELEMETRY": "off",
            },
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        results.append(json.loads(process.stdout))

    assert results[0]["status"] == 413
    assert results[0]["body"]["error"]["code"] == "request_too_large"
    assert results[1] == {"status": 200, "body": {"preprocessed": True}}
