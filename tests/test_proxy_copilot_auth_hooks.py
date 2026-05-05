from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
_ISOLATED_MODULE_NAMES = (
    "headroom.proxy",
    "headroom.proxy.handlers",
    "httpx",
    "fastapi.responses",
    "tests.headroom_proxy_handlers_openai",
    "tests.headroom_proxy_handlers_streaming",
)


@pytest.fixture(autouse=True)
def restore_isolated_modules() -> None:
    saved_modules = {name: sys.modules.get(name) for name in _ISOLATED_MODULE_NAMES}
    try:
        yield
    finally:
        for name in _ISOLATED_MODULE_NAMES:
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module


def _load_handler_module(monkeypatch: pytest.MonkeyPatch, module_name: str, relative_path: str):
    proxy_pkg = types.ModuleType("headroom.proxy")
    proxy_pkg.__path__ = [str(ROOT / "headroom" / "proxy")]
    monkeypatch.setitem(sys.modules, "headroom.proxy", proxy_pkg)

    handlers_pkg = types.ModuleType("headroom.proxy.handlers")
    handlers_pkg.__path__ = [str(ROOT / "headroom" / "proxy" / "handlers")]
    monkeypatch.setitem(sys.modules, "headroom.proxy.handlers", handlers_pkg)

    httpx_mod = types.ModuleType("httpx")
    httpx_mod.ConnectError = type("ConnectError", (Exception,), {})
    httpx_mod.ConnectTimeout = type("ConnectTimeout", (Exception,), {})
    httpx_mod.PoolTimeout = type("PoolTimeout", (Exception,), {})
    monkeypatch.setitem(sys.modules, "httpx", httpx_mod)

    responses_mod = types.ModuleType("fastapi.responses")

    class Response:
        def __init__(self, content=None, status_code: int = 200, headers=None, media_type=None):
            self.content = content
            self.status_code = status_code
            self.headers = headers or {}
            self.media_type = media_type

    class StreamingResponse(Response):
        pass

    class JSONResponse(Response):
        pass

    responses_mod.Response = Response
    responses_mod.StreamingResponse = StreamingResponse
    responses_mod.JSONResponse = JSONResponse
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses_mod)

    spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_openai_passthrough_applies_copilot_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    openai_mod = _load_handler_module(
        monkeypatch,
        "tests.headroom_proxy_handlers_openai",
        "headroom/proxy/handlers/openai.py",
    )

    seen: dict[str, object] = {}

    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        seen["headers"] = dict(headers)
        seen["url"] = url
        return {"Authorization": "Bearer upstream-token"}

    monkeypatch.setattr(openai_mod, "apply_copilot_api_auth", fake_apply)

    class Dummy(openai_mod.OpenAIHandlerMixin):
        def __init__(self) -> None:
            self.metrics = SimpleNamespace(record_request=self._record_request)
            self.http_client = SimpleNamespace(request=self._request)

        async def _record_request(self, **kwargs) -> None:  # noqa: ANN003
            return None

        async def _request(self, **kwargs):  # noqa: ANN003
            seen["request_kwargs"] = kwargs
            return SimpleNamespace(headers={}, content=b"{}", status_code=200)

    request = SimpleNamespace(
        url=SimpleNamespace(path="/v1/models", query=""),
        headers={
            "authorization": "Bearer downstream",
            "host": "localhost",
            "accept-encoding": "gzip",
        },
        method="GET",
        body=lambda: None,
    )

    async def body() -> bytes:
        return b""

    request.body = body

    handler = Dummy()
    response = asyncio.run(
        handler.handle_passthrough(
            request,
            "https://api.githubcopilot.com",
            "models",
            "openai",
        )
    )

    assert seen["url"] == "https://api.githubcopilot.com/models"
    assert seen["request_kwargs"]["headers"] == {"Authorization": "Bearer upstream-token"}
    assert response.status_code == 200


def test_streaming_response_applies_copilot_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    streaming_mod = _load_handler_module(
        monkeypatch,
        "tests.headroom_proxy_handlers_streaming",
        "headroom/proxy/handlers/streaming.py",
    )

    seen: dict[str, object] = {}

    async def fake_apply(headers: dict[str, str], *, url: str) -> dict[str, str]:
        seen["headers"] = dict(headers)
        seen["url"] = url
        return {"Authorization": "Bearer upstream-token"}

    monkeypatch.setattr(streaming_mod, "apply_copilot_api_auth", fake_apply)

    class Dummy(streaming_mod.StreamingMixin):
        def __init__(self) -> None:
            self.memory_handler = None
            self.config = SimpleNamespace(
                retry_max_attempts=1,
                retry_base_delay_ms=1,
                retry_max_delay_ms=1,
            )
            self.http_client = SimpleNamespace(
                build_request=self._build_request,
                send=self._send,
            )

        def _build_request(self, method: str, url: str, **kwargs):  # noqa: ANN003
            # PR-A3: streaming forwarder is byte-faithful; it now passes
            # ``content=<bytes>`` instead of ``json=<dict>``.
            seen["request"] = {
                "method": method,
                "url": url,
                **kwargs,
            }
            return SimpleNamespace()

        async def _send(self, request, stream: bool):  # noqa: ANN001, ANN003
            return SimpleNamespace(headers={}, status_code=200)

    handler = Dummy()
    response = asyncio.run(
        handler._stream_response(
            url="https://api.githubcopilot.com/v1/responses",
            headers={"authorization": "Bearer downstream"},
            body={"model": "gpt-4o"},
            provider="openai",
            model="gpt-4o",
            request_id="req-test",
            original_tokens=0,
            optimized_tokens=0,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )
    )

    assert seen["url"] == "https://api.githubcopilot.com/v1/responses"
    # PR-A3: byte-faithful forwarder always sets ``content-type`` explicitly.
    sent_headers = seen["request"]["headers"]
    assert sent_headers["Authorization"] == "Bearer upstream-token"
    assert sent_headers["content-type"] == "application/json"
    assert response.status_code == 200
