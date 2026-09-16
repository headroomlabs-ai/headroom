"""Responses tool-search deferral is skipped for harnesses that cannot run it (GH #2660)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from starlette.datastructures import Headers

from headroom.proxy.auth_mode import classify_client
from headroom.proxy.handlers.openai import OpenAIHandlerMixin
from headroom.proxy.helpers import (
    inject_tool_search_deferral_openai,
    openai_tool_search_client_supported,
)

TOOL_SEARCH_MODEL = "gpt-5.5"


def _tool_payload() -> list[dict[str, object]]:
    """Six core coding tools plus ten non-core ones, over the injection minimum."""
    names = ["bash", "read", "write", "edit", "grep", "glob"]
    names += [f"slack_{index}" for index in range(10)]
    return [
        {"type": "function", "name": name, "parameters": {"type": "object", "properties": {}}}
        for name in names
    ]


@pytest.mark.parametrize(
    ("headers", "expected_client", "supported"),
    [
        ({"user-agent": "opencode/0.4.2"}, "opencode", False),
        ({"x-client": "opencode"}, "opencode", False),
        ({"user-agent": "codex-cli/1.2.3"}, "codex", False),
        ({"user-agent": "claude-code/2.0"}, "claude-code", True),
        ({"user-agent": "cursor/1.0"}, "cursor", True),
        ({}, None, True),
        ({"user-agent": "some-unknown-sdk/1.0"}, None, True),
        # CLIENT_UA_MAP matches by substring, so a wrapper that embeds the
        # opencode UA is classified as opencode and excluded with it.
        ({"user-agent": "acme-wrapper opencode/1.0"}, "opencode", False),
    ],
)
def test_only_the_reported_harness_is_excluded(
    headers: dict[str, str], expected_client: str | None, supported: bool
) -> None:
    """The exclusion keys on the client name the proxy already resolves."""
    assert classify_client(headers) == expected_client
    assert openai_tool_search_client_supported(classify_client(headers)) is supported


def test_a_similar_client_name_does_not_match() -> None:
    """The exclusion set is exact membership on the resolved client name.

    Substring matching happens upstream in ``CLIENT_UA_MAP``; this pins that the
    set itself does not widen a name that already classified.
    """
    assert openai_tool_search_client_supported("opencode-fork") is True
    assert openai_tool_search_client_supported("open") is True
    assert openai_tool_search_client_supported("opencode") is False


def test_request_headers_decide_the_outbound_tools_payload() -> None:
    """End of the route: real request headers in, final Responses tools out.

    This is the symptom the issue reports. An opencode request must not find an
    injected ``{"type": "tool_search"}`` tool it cannot execute, and every other
    client must still get the deferral it got before.
    """
    tools = _tool_payload()

    opencode = Headers({"user-agent": "opencode/0.4.2", "content-type": "application/json"})
    forwarded = inject_tool_search_deferral_openai(
        tools,
        TOOL_SEARCH_MODEL,
        client=classify_client(opencode),
    )
    assert forwarded is tools
    assert not any(tool.get("type") == "tool_search" for tool in forwarded)
    assert not any(tool.get("defer_loading") for tool in forwarded)

    codex = Headers({"user-agent": "codex-cli/1.2.3", "content-type": "application/json"})
    forwarded = inject_tool_search_deferral_openai(
        tools,
        TOOL_SEARCH_MODEL,
        client=classify_client(codex),
    )
    assert forwarded is tools
    assert not any(tool.get("type") == "tool_search" for tool in forwarded)
    assert not any(tool.get("defer_loading") for tool in tools)


def test_websocket_and_http_header_shapes_classify_alike() -> None:
    """The WebSocket path builds a plain dict from the same multidict."""
    multidict = Headers({"user-agent": "opencode/0.4.2"})

    assert classify_client(multidict) == "opencode"
    assert classify_client(dict(multidict)) == "opencode"


def test_native_responses_compressor_scopes_the_exclusion_per_call() -> None:
    """The flag rides one request; a later request is unaffected by an earlier one."""
    seen: list[dict[str, object]] = []
    handler = object.__new__(OpenAIHandlerMixin)

    async def _run_compression(fn, *, timeout):  # noqa: ANN001, ANN202
        return fn()

    def _compress(payload, *, model, request_id, **kwargs):  # noqa: ANN001, ANN202
        seen.append(kwargs)
        return (payload, False, 0, [], "no-op", 0, 0, 0, {})

    handler._run_compression_in_executor = _run_compression
    handler._compress_openai_responses_payload = _compress

    async def _run() -> None:
        await handler._compress_openai_responses_payload_in_executor(
            {"input": "hello"},
            model=TOOL_SEARCH_MODEL,
            request_id="req-opencode",
            client="opencode",
        )
        await handler._compress_openai_responses_payload_in_executor(
            {"input": "hello"},
            model=TOOL_SEARCH_MODEL,
            request_id="req-codex",
        )

    asyncio.run(_run())

    assert [{key: value for key, value in call.items() if key != "timing"} for call in seen] == [
        {"client": "opencode"},
        {"client": None},
    ]


def test_supported_clients_send_no_extra_compressor_argument() -> None:
    """A compressor override written before this change keeps its exact signature."""
    calls: list[str] = []
    handler = object.__new__(OpenAIHandlerMixin)

    async def _run_compression(fn, *, timeout):  # noqa: ANN001, ANN202
        return fn()

    def _narrow_compress(payload, *, model, request_id, timing=None):  # noqa: ANN001, ANN202
        calls.append(request_id)
        return (payload, False, 0, [], "no-op", 0, 0, 0, {})

    handler._run_compression_in_executor = _run_compression
    handler._compress_openai_responses_payload = _narrow_compress

    asyncio.run(
        handler._compress_openai_responses_payload_in_executor(
            {"input": "hello"},
            model=TOOL_SEARCH_MODEL,
            request_id="req-codex",
        )
    )

    assert calls == ["req-codex"]


def test_a_narrow_compressor_override_still_works_for_an_excluded_client() -> None:
    """The retry drops the optional keywords rather than failing the request."""
    calls: list[str] = []
    handler = object.__new__(OpenAIHandlerMixin)

    async def _run_compression(fn, *, timeout):  # noqa: ANN001, ANN202
        return fn()

    def _narrow_compress(payload, *, model, request_id):  # noqa: ANN001, ANN202
        calls.append(request_id)
        return (payload, False, 0, [], "no-op", 0, 0, 0, {})

    handler._run_compression_in_executor = _run_compression
    handler._compress_openai_responses_payload = _narrow_compress

    asyncio.run(
        handler._compress_openai_responses_payload_in_executor(
            {"input": "hello"},
            model=TOOL_SEARCH_MODEL,
            request_id="req-opencode",
            client="opencode",
        )
    )

    assert calls == ["req-opencode"]


def test_native_responses_compressor_reraises_internal_type_error() -> None:
    """An internal compressor TypeError is propagated without a signature retry."""
    calls = 0
    handler = object.__new__(OpenAIHandlerMixin)
    sentinel = TypeError("internal compressor failure")

    async def _run_compression(fn, *, timeout):  # noqa: ANN001, ANN202
        return fn()

    def _compress(payload, *, model, request_id, client, timing=None):  # noqa: ANN001, ANN202
        nonlocal calls
        calls += 1
        raise sentinel

    handler._run_compression_in_executor = _run_compression
    handler._compress_openai_responses_payload = _compress

    with pytest.raises(TypeError) as exc_info:
        asyncio.run(
            handler._compress_openai_responses_payload_in_executor(
                {"input": "hello"},
                model=TOOL_SEARCH_MODEL,
                request_id="req-opencode",
                client="opencode",
            )
        )

    assert exc_info.value is sentinel
    assert calls == 1


class _ResponsesRequest:
    method = "POST"
    url = SimpleNamespace(path="/custom/v1/responses", query="")

    def __init__(self, headers: dict[str, str]) -> None:
        self.headers = headers

    async def body(self) -> bytes:
        return json.dumps({"model": TOOL_SEARCH_MODEL, "input": "hello"}).encode()


class _UpstreamClient:
    async def request(self, **kwargs):  # noqa: ANN001, ANN201
        request = httpx.Request(kwargs["method"], kwargs["url"])
        return httpx.Response(200, request=request, json={"ok": True})


def _passthrough_handler(seen: list[dict[str, object]]) -> OpenAIHandlerMixin:
    handler = object.__new__(OpenAIHandlerMixin)
    handler.config = SimpleNamespace(
        optimize=True,
        compress_passthrough=True,
        openai_extra_headers=None,
    )
    handler.http_client = _UpstreamClient()
    handler.http_client_h1 = None

    async def _next_request_id() -> str:
        return "req-test"

    async def _compress(payload, *, model, request_id, **kwargs):  # noqa: ANN001, ANN202
        seen.append(kwargs)
        return (payload, False, 0, [], "no-op", 0, len(json.dumps(payload)), 0, {})

    handler._next_request_id = _next_request_id
    handler._compress_openai_responses_payload_in_executor = _compress
    return handler


def test_custom_base_path_excludes_the_reported_harness() -> None:
    seen: list[dict[str, object]] = []

    asyncio.run(
        _passthrough_handler(seen).handle_passthrough(
            _ResponsesRequest({"user-agent": "opencode/0.4.2", "content-type": "application/json"}),
            "https://api.example.com",
        )
    )

    assert seen == [{"client": "opencode"}]


def test_custom_base_path_leaves_other_clients_alone() -> None:
    seen: list[dict[str, object]] = []

    asyncio.run(
        _passthrough_handler(seen).handle_passthrough(
            _ResponsesRequest(
                {"user-agent": "codex-cli/1.2.3", "content-type": "application/json"}
            ),
            "https://api.example.com",
        )
    )

    assert seen == [{"client": "codex"}]


# --- production route: the native /v1/responses handler -----------------------

_OPENAI_OK_RESPONSE = {
    "id": "resp_test",
    "object": "response",
    "status": "completed",
    "model": TOOL_SEARCH_MODEL,
    "output": [],
    "usage": {"input_tokens": 10, "output_tokens": 2},
}


class _NativeCapturingTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.call_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.call_count += 1
        async for _ in request.stream:
            pass
        return httpx.Response(200, json=_OPENAI_OK_RESPONSE)


def _native_responses_client():  # noqa: ANN202
    """Boot the real app and observe what the Responses compressor is handed.

    The transport and the spy are installed after the lifespan runs, because
    startup builds the proxy's HTTP clients.
    """
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from headroom.proxy.server import ProxyConfig, create_app

    config = ProxyConfig(
        optimize=True,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )
    app = create_app(config)
    seen: list[dict[str, object]] = []

    # Loopback client so the proxy-token middleware treats this as a local call.
    with TestClient(app, client=("127.0.0.1", 50000)) as client:
        proxy = app.state.proxy
        transport = _NativeCapturingTransport()
        proxy.http_client = httpx.AsyncClient(transport=transport)
        proxy.http_client_h1 = httpx.AsyncClient(transport=transport)

        original = proxy._compress_openai_responses_payload_in_executor

        async def _spy(payload, **kwargs):  # noqa: ANN001, ANN202
            seen.append({k: v for k, v in kwargs.items() if k not in {"model", "request_id"}})
            return await original(payload, **kwargs)

        proxy._compress_openai_responses_payload_in_executor = _spy
        yield client, seen, transport


def _responses_body() -> dict[str, object]:
    return {"model": TOOL_SEARCH_MODEL, "input": "hello", "tools": _tool_payload()}


@pytest.mark.parametrize(
    ("user_agent", "expected"),
    [
        ("opencode/0.4.2", {"client": "opencode"}),
        ("codex-cli/1.2.3", {"client": "codex"}),
    ],
)
def test_native_responses_route_carries_the_client_decision(
    user_agent: str, expected: dict[str, object]
) -> None:
    """Drives POST /v1/responses on the real app, not a handler method in isolation.

    Deleting the kwargs splat at the native call site leaves every other test in
    this file green; this one fails.
    """
    for client, seen, transport in _native_responses_client():
        response = client.post(
            "/v1/responses",
            headers={
                "content-type": "application/json",
                "authorization": "Bearer sk-test-0000000000000000000000000000000000000000000",
                "user-agent": user_agent,
            },
            json=_responses_body(),
        )

        assert transport.call_count == 1, response.text
        assert response.status_code == 200, response.text
        assert seen, "the Responses compressor was never reached"
        assert {k: v for k, v in seen[0].items() if k not in {"timing", "savings_tags"}} == expected


# --- production route: the Codex WebSocket handler ---------------------------


@pytest.mark.parametrize(
    ("user_agent", "expected"),
    [
        ("opencode/0.4.2", {"client": "opencode"}),
        ("codex-cli/1.2.3", {"client": "codex"}),
    ],
)
def test_websocket_route_carries_the_client_decision(
    user_agent: str, expected: dict[str, object]
) -> None:
    """The WS frame path resolves the client the same way the HTTP path does.

    Reuses the repo's existing Codex WS harness so the real
    ``handle_openai_responses_ws`` ingress runs, rather than asserting the
    wiring structurally.
    """
    import sys
    from unittest.mock import patch

    from tests.test_openai_codex_ws_lifecycle import (
        _DummyOpenAIHandler,
        _FakeUpstream,
        _FakeWebSocket,
        _first_frame,
        _make_fake_websockets_module,
    )

    upstream = _FakeUpstream(
        [
            json.dumps({"type": "response.created", "response": {"id": "r_1"}}),
            json.dumps({"type": "response.completed", "response": {"id": "r_1"}}),
        ]
    )
    client_ws = _FakeWebSocket(
        frames=[_first_frame()],
        headers={"authorization": "Bearer test", "user-agent": user_agent},
    )
    handler = _DummyOpenAIHandler()
    handler.config.optimize = True

    seen: list[dict[str, object]] = []

    def _compress(payload, *, model, request_id, **kwargs):  # noqa: ANN001, ANN202
        seen.append(kwargs)
        return (payload, False, 0, [], "router_no_compression", 10, 10)

    handler._compress_openai_responses_payload = _compress

    with patch.dict(sys.modules, {"websockets": _make_fake_websockets_module(upstream)}):
        asyncio.run(handler.handle_openai_responses_ws(client_ws))

    assert seen, "the Responses compressor was never reached on the WS path"
    assert {k: v for k, v in seen[0].items() if k != "timing"} == expected
