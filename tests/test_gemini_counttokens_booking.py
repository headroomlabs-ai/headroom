"""Gemini :countTokens is a sizing call: not booked as a billed request, errors still recorded."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from headroom.proxy.handlers.gemini import GeminiHandlerMixin


class _FakeRequest:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.query_params: dict[str, str] = {}
        self.url = SimpleNamespace(path="/v1beta/models/gemini-pro:countTokens", query="")
        self.scope: dict = {"type": "http", "method": "POST"}


class _CountResponse:
    content = b'{"totalTokens": 40}'
    headers = {"content-type": "application/json", "content-length": str(len(content))}

    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code

    def json(self) -> object:
        return json.loads(self.content)


class _FakeMetrics:
    async def record_failed(self, *, provider: str, model: str = "") -> None:
        pass


class _Tokenizer:
    def count_messages(self, messages):  # noqa: ANN001, ANN201
        return 100


class _Handler(GeminiHandlerMixin):
    GEMINI_API_URL = "https://gemini.example"

    def __init__(self) -> None:
        self.memory_handler = None
        self.rate_limiter = None
        self.usage_reporter = None
        from headroom.cache.prefix_tracker import SessionTrackerStore

        self.session_tracker_store = SessionTrackerStore()
        self.config = SimpleNamespace(
            optimize=False,
            anthropic_pre_upstream_memory_context_timeout_seconds=0.1,
        )
        self.metrics = _FakeMetrics()
        self.outcomes: list = []
        self.status = 200

    async def _next_request_id(self) -> str:
        return "req-1"

    async def _retry_request(self, method, url, headers, body):  # noqa: ANN001, ANN201
        return _CountResponse(self.status)

    async def _record_request_outcome(self, outcome) -> None:  # noqa: ANN001
        self.outcomes.append(outcome)

    async def _count_tokens_offloaded(self, model, messages):  # noqa: ANN001, ANN201
        return _Tokenizer(), _Tokenizer().count_messages(messages)


async def _call(handler: _Handler, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN202
    async def payload(request):  # noqa: ANN001, ANN201
        return {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", payload)
    return await handler.handle_gemini_count_tokens(_FakeRequest(), "gemini-pro")


@pytest.mark.asyncio
async def test_count_tokens_is_not_recorded_as_a_request_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def payload(request):  # noqa: ANN001, ANN201
        return {"contents": [{"role": "user", "parts": [{"text": "hello"}]}]}

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", payload)

    handler = _Handler()
    response = await handler.handle_gemini_count_tokens(_FakeRequest(), "gemini-pro")

    # The upstream count is still returned to the client unchanged.
    assert response.status_code == 200
    assert json.loads(response.body) == {"totalTokens": 40}
    # Nothing was generated or billed, so no outcome (spend / savings) is booked.
    assert handler.outcomes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500, 529])
async def test_count_tokens_upstream_error_is_still_recorded(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    handler = _Handler()
    handler.status = status
    response = await _call(handler, monkeypatch)

    assert response.status_code == status
    # The outcome funnel turns this into a rate-limit / failure count.
    assert [o.status_code for o in handler.outcomes] == [status]
