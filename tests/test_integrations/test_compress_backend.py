"""Cloud path of the compression backend shared by the ASGI and LiteLLM integrations."""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest

from headroom.integrations.asgi import CompressionMiddleware
from headroom.integrations.litellm_callback import HeadroomCallback

_MESSAGES = [{"role": "user", "content": "secret prompt text"}]


@pytest.fixture(params=["asgi", "litellm"])
def backend(request, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
    if request.param == "asgi":
        return CompressionMiddleware(app=None, api_key="hdr_test", api_url="https://cloud.test/")
    return HeadroomCallback(api_key="hdr_test", api_url="https://cloud.test/")


def _serve(backend, handler) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    backend._client = httpx.AsyncClient(transport=httpx.MockTransport(record))
    return seen


def test_cloud_success_returns_the_service_result(backend) -> None:
    payload = {"messages": [{"role": "user", "content": "short"}], "tokens_saved": 3}
    seen = _serve(backend, lambda request: httpx.Response(200, json=payload))

    result = asyncio.run(backend._cloud_compress(_MESSAGES, "gpt-4o"))

    assert result == payload
    (request,) = seen
    assert str(request.url) == "https://cloud.test/v1/saas/compress"
    assert request.headers["X-Headroom-Key"] == "hdr_test"
    assert json.loads(request.content) == {
        "messages": _MESSAGES,
        "model": "gpt-4o",
        "model_limit": backend._model_limit,
    }


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_cloud_error_warns_with_status_only(backend, monkeypatch: pytest.MonkeyPatch) -> None:
    _serve(backend, lambda request: httpx.Response(503, text="echo: secret prompt text"))
    # Attach to the module logger itself: proxy startup elsewhere in the suite can
    # stop headroom.* records from reaching pytest's root-level caplog handler.
    handler = _Records()
    monkeypatch.setattr(backend._log, "level", logging.DEBUG)
    monkeypatch.setattr(backend._log, "disabled", False)
    backend._log.addHandler(handler)
    try:
        assert asyncio.run(backend._cloud_compress(_MESSAGES, "gpt-4o")) is None
    finally:
        backend._log.removeHandler(handler)

    by_level = {r.levelno: r.getMessage() for r in handler.records}
    assert by_level[logging.WARNING] == (
        "Headroom Cloud API error: HTTP 503 from https://cloud.test; request sent uncompressed"
    )
    assert "secret prompt text" not in by_level[logging.WARNING]
    assert by_level[logging.DEBUG] == "Headroom Cloud API error body: echo: secret prompt text"
