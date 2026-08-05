"""A non-JSON Gemini upstream body must not be masked as a synthetic 502."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from headroom.proxy.handlers.gemini import GeminiHandlerMixin


class _FakeRequest:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.query_params: dict[str, str] = {}
        self.url = SimpleNamespace(path="/v1beta/models/gemini-pro:generateContent", query="")


class _NonJsonResponse:
    status_code = 503
    content = b"<html>temporarily unavailable</html>"
    headers = {"content-type": "text/html", "content-length": str(len(content))}

    def json(self) -> object:
        raise json.JSONDecodeError("not json", self.content.decode("utf-8"), 0)


class _FakeMetrics:
    def __init__(self) -> None:
        self.failed: list[str] = []

    async def record_failed(self, *, provider: str, model: str = "") -> None:
        self.failed.append(f"{provider}:{model}")


class _Handler(GeminiHandlerMixin):
    GEMINI_API_URL = "https://gemini.example"

    def __init__(self) -> None:
        self.memory_handler = None
        self.rate_limiter = None
        self.usage_reporter = None
        self.config = SimpleNamespace(
            optimize=False,
            anthropic_pre_upstream_memory_context_timeout_seconds=0.1,
        )
        self.metrics = _FakeMetrics()
        self.outcomes = []

    async def _next_request_id(self) -> str:
        return "req-1"

    async def _retry_request(self, method, url, headers, body):  # noqa: ANN001, ANN201
        return _NonJsonResponse()

    async def _record_request_outcome(self, outcome) -> None:  # noqa: ANN001
        self.outcomes.append(outcome)

    async def _count_tokens_offloaded(self, model, messages):  # noqa: ANN001, ANN201
        # Test stub for HeadroomProxy._count_tokens_offloaded: resolve the
        # tokenizer and count inline (the real method offloads to the executor).
        from headroom.tokenizers import get_tokenizer

        tokenizer = get_tokenizer(model)
        return tokenizer, tokenizer.count_messages(messages)


@pytest.mark.asyncio
async def test_generate_content_forwards_non_json_upstream_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def payload(request):  # noqa: ANN001, ANN201
        return {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}

    class _Tokenizer:
        def count_messages(self, messages):  # noqa: ANN001, ANN201
            return 7

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", payload)
    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _Tokenizer())

    handler = _Handler()
    response = await handler.handle_gemini_generate_content(_FakeRequest(), "gemini-pro")

    assert response.status_code == 503
    assert response.body == _NonJsonResponse.content
    assert response.headers["content-type"] == "text/html"
    assert response.headers["x-headroom-tokens-before"] == "7"
    assert response.headers["x-headroom-tokens-after"] == "7"
    assert handler.metrics.failed == []
    assert handler.outcomes[0].status_code == 503
