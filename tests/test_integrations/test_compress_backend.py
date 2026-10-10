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


_PROMPT_CANARY = "PROMPT-CANARY-7f3e"
_KEY_CANARY = "hdr_KEY-CANARY-91ab"
_URL_CANARY = "URL-CANARY-c0de"


@pytest.mark.parametrize("kind", ["asgi", "litellm"])
@pytest.mark.parametrize("level", [logging.DEBUG, logging.WARNING])
def test_cloud_error_logs_no_echoed_content_at_any_level(
    kind: str, level: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error response echoing the prompt must not put it, the key or URL
    credentials in any formatted record, with DEBUG on or off."""
    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    api_url = f"https://cloud.test/?token={_URL_CANARY}"
    if kind == "asgi":
        backend = CompressionMiddleware(app=None, api_key=_KEY_CANARY, api_url=api_url)
    else:
        backend = HeadroomCallback(api_key=_KEY_CANARY, api_url=api_url)
    messages = [{"role": "user", "content": f"my secret is {_PROMPT_CANARY}"}]
    echo = f"bad request: {json.dumps(messages)} key={_KEY_CANARY}"
    _serve(backend, lambda request: httpx.Response(503, text=echo))
    # Attach to the module logger itself: proxy startup elsewhere in the suite can
    # stop headroom.* records from reaching pytest's root-level caplog handler.
    handler = _Records()
    monkeypatch.setattr(backend._log, "level", level)
    monkeypatch.setattr(backend._log, "disabled", False)
    backend._log.addHandler(handler)
    try:
        assert asyncio.run(backend._cloud_compress(messages, "gpt-4o")) is None
    finally:
        backend._log.removeHandler(handler)

    formatted = [logging.Formatter().format(r) for r in handler.records]
    warnings = [r.getMessage() for r in handler.records if r.levelno == logging.WARNING]
    assert warnings == [
        "Headroom Cloud API error: HTTP 503 from https://cloud.test/?<redacted>; "
        "request sent uncompressed"
    ]
    if level == logging.DEBUG:
        assert any("content-type=" in line for line in formatted)
    for line in formatted:
        for canary in (_PROMPT_CANARY, _KEY_CANARY, _URL_CANARY):
            assert canary not in line
