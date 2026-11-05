"""Cloud path of the compression backend shared by the ASGI and LiteLLM integrations."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

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
_PATH_CANARY = "PRIVATE_PATH_CANARY"

# (api_url, how the WARNING shows it); every canary must stay out of every record.
_API_URLS = {
    "query": (f"https://cloud.test/?token={_URL_CANARY}", "https://cloud.test/?<redacted>"),
    "path": (f"https://cloud.test/bot{_PATH_CANARY}", "https://cloud.test/<path>"),
    # %50 is "P": the encoded form must not slip through either.
    "encoded-path": ("https://cloud.test/bot%50RIVATE_PATH_CANARY", "https://cloud.test/<path>"),
}


@pytest.mark.parametrize("kind", ["asgi", "litellm"])
def test_real_cloud_error_media_type_cannot_echo_credentials(kind, monkeypatch) -> None:
    canary = "PRIVATE_CREDENTIAL_CANARY"
    body = b"service error"
    received = []

    class ErrorServer(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(503)
            self.send_header("Content-Type", f"text/{canary}")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    with HTTPServer(("127.0.0.1", 0), ErrorServer) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        api_url = f"http://127.0.0.1:{server.server_port}"
        integration = CompressionMiddleware if kind == "asgi" else HeadroomCallback
        kwargs = {"app": None} if kind == "asgi" else {}
        instance = integration(api_key=_KEY_CANARY, api_url=api_url, **kwargs)
        handler = _Records()
        monkeypatch.setattr(instance._log, "level", logging.DEBUG)
        monkeypatch.setattr(instance._log, "disabled", False)
        instance._log.addHandler(handler)

        async def request():
            try:
                return await instance._cloud_compress(_MESSAGES, "gpt-4o")
            finally:
                if instance._client is not None:
                    await instance._client.aclose()

        try:
            assert asyncio.run(request()) is None
        finally:
            instance._log.removeHandler(handler)
            server.shutdown()
            worker.join(timeout=5)
    assert received[0]["messages"] == _MESSAGES
    records = [logging.Formatter().format(record) for record in handler.records]
    assert any("HTTP 503" in record for record in records)
    assert any(f"{len(body)} bytes" in record for record in records)
    assert all(canary.lower() not in record.lower() for record in records)


@pytest.mark.parametrize("kind", ["asgi", "litellm"])
@pytest.mark.parametrize("level", [logging.DEBUG, logging.WARNING])
@pytest.mark.parametrize("url_case", sorted(_API_URLS))
def test_cloud_error_logs_no_echoed_content_at_any_level(
    kind: str, level: int, url_case: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An error response echoing the prompt must not put it, the key, or URL
    credentials (query or path) in any formatted record, with DEBUG on or off."""
    monkeypatch.delenv("HEADROOM_OFFLINE", raising=False)
    monkeypatch.delenv("HEADROOM_DEBUG_DUMP", raising=False)
    api_url, shown_url = _API_URLS[url_case]
    if kind == "asgi":
        backend = CompressionMiddleware(app=None, api_key=_KEY_CANARY, api_url=api_url)
    else:
        backend = HeadroomCallback(api_key=_KEY_CANARY, api_url=api_url)
    messages = [{"role": "user", "content": f"my secret is {_PROMPT_CANARY}"}]
    echo = f"bad request: {json.dumps(messages)} key={_KEY_CANARY}"
    _serve(
        backend,
        lambda request: httpx.Response(
            503,
            text=echo,
            # A header can echo content too: only the bare media type may be logged.
            headers={"content-type": f"text/plain; note={_PROMPT_CANARY}"},
        ),
    )
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
        f"Headroom Cloud API error: HTTP 503 from {shown_url}; request sent uncompressed"
    ]
    if level == logging.DEBUG:
        assert any("content-type=text/plain," in line for line in formatted)
    for line in formatted:
        for canary in (_PROMPT_CANARY, _KEY_CANARY, _URL_CANARY, _PATH_CANARY, "%50RIVATE"):
            assert canary not in line
