"""The Cloud Code stream and :countTokens Gemini routes honour the request rate limiter."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from headroom.proxy.handlers.gemini import GeminiHandlerMixin
from headroom.proxy.rate_limiter import TokenBucketRateLimiter


def _request(path: str, payload: dict) -> Request:
    """Real Starlette Request whose JSON body arrives in two ASGI chunks."""
    body = json.dumps(payload).encode()
    chunks = [body[: len(body) // 2], body[len(body) // 2 :]]
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 50000),
    }

    async def receive() -> dict:
        if chunks:
            chunk = chunks.pop(0)
            return {"type": "http.request", "body": chunk, "more_body": bool(chunks)}
        return {"type": "http.disconnect"}

    return Request(scope, receive)


class _Response:
    status_code = 200
    content = b'{"totalTokens": 1}'
    headers = {"content-type": "application/json"}

    def json(self) -> object:
        return {"totalTokens": 1}


class _FakeMetrics:
    def __init__(self) -> None:
        self.rate_limited: list[str] = []

    async def record_rate_limited(self, *, provider: str, **_: object) -> None:
        self.rate_limited.append(provider)

    async def record_failed(self, *, provider: str, model: str = "") -> None:
        pass


class _Handler(GeminiHandlerMixin):
    GEMINI_API_URL = "https://gemini.example"

    def __init__(self) -> None:
        self.memory_handler = None
        self.usage_reporter = None
        self.rate_limiter = TokenBucketRateLimiter(requests_per_minute=1)
        self.metrics = _FakeMetrics()
        from headroom.cache.prefix_tracker import SessionTrackerStore

        self.session_tracker_store = SessionTrackerStore()
        self.config = SimpleNamespace(optimize=False)
        self.upstream_calls = 0
        self.streamed = 0

    async def _next_request_id(self) -> str:
        return "req-1"

    async def _retry_request(self, method, url, headers, body):  # noqa: ANN001, ANN201
        self.upstream_calls += 1
        return _Response()

    async def _stream_response(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.streamed += 1
        return SimpleNamespace(status_code=200)


async def _statuses(call, n: int) -> list[int]:  # noqa: ANN001
    from fastapi import HTTPException

    out: list[int] = []
    for _ in range(n):
        try:
            out.append((await call()).status_code)
        except HTTPException as exc:
            out.append(exc.status_code)
    return out


@pytest.mark.asyncio
async def test_count_tokens_is_rate_limited() -> None:
    # inlineData-only body takes the early forward path (no tokenizer needed).
    payload = {"contents": [{"role": "user", "parts": [{"inlineData": {"data": "x"}}]}]}
    handler = _Handler()

    statuses = await _statuses(
        lambda: handler.handle_gemini_count_tokens(
            _request("/v1beta/models/gemini-pro:countTokens", payload), "gemini-pro"
        ),
        4,
    )

    assert statuses == [200, 429, 429, 429]
    assert handler.upstream_calls == 1
    assert handler.metrics.rate_limited == ["gemini"] * 3


@pytest.mark.asyncio
async def test_cloudcode_stream_is_rate_limited() -> None:
    payload = {
        "model": "gemini-pro",
        "request": {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]},
    }
    handler = _Handler()

    statuses = await _statuses(
        lambda: handler.handle_google_cloudcode_stream(
            _request("/v1internal:streamGenerateContent", payload)
        ),
        4,
    )

    assert statuses == [200, 429, 429, 429]
    assert handler.streamed == 1
    assert handler.metrics.rate_limited == ["gemini"] * 3
