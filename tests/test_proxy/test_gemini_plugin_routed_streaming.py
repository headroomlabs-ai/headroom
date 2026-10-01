"""Production-route regressions for native Gemini model requests."""

from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("fastapi")

from fastapi import Request  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from headroom.compress import CompressResult  # noqa: E402
from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402

# The OpenCode transport sets x-headroom-base-url to the upstream *origin* and
# forwards the upstream pathname verbatim, so a custom Google endpoint arrives
# as origin + /v1beta/models/{model}:... on the proxy.
PLUGIN_BASE_URL = "https://gateway.example"
DIRECT_STREAM_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:streamGenerateContent?alt=sse"
)
PLUGIN_STREAM_URL = (
    "https://gateway.example/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
)
DIRECT_KEY_STREAM_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:streamGenerateContent?key=secret&alt=sse"
)
PLUGIN_QUERY_STREAM_URL = "https://gateway.example/v1beta/models/gemini-2.5-flash:streamGenerateContent?trace=1&foo=bar&alt=sse"
DIRECT_ENCODED_QUERY_STREAM_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:streamGenerateContent?trace=hello%20world&encoded=%7E&alt=sse"
)


@pytest.fixture(autouse=True)
def _allow_fake_plugin_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", "gateway.example")


def _config() -> ProxyConfig:
    return ProxyConfig(
        optimize=False,
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
        log_requests=False,
        ccr_inject_tool=False,
        ccr_handle_responses=False,
        ccr_context_tracking=False,
        image_optimize=False,
    )


def _streaming_optimize_config() -> ProxyConfig:
    return ProxyConfig(
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


def test_plugin_generate_content_uses_native_gemini_route_and_preserves_body() -> None:
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "Keep this prompt."},
                    {"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}},
                    {"functionCall": {"name": "lookup", "args": {"city": "Paris"}}},
                ],
            }
        ],
        "generationConfig": {"temperature": 0.2},
    }
    upstream = {
        "candidates": [{"content": {"parts": [{"text": "answer"}]}}],
        "usageMetadata": {"promptTokenCount": 42, "candidatesTokenCount": 7},
    }
    captured: dict[str, object] = {}

    async def fake_retry(
        method: str,
        url: str,
        headers: dict[str, str],
        request_body: dict,
        **_: object,
    ) -> httpx.Response:
        captured.update(method=method, url=url, headers=headers, body=request_body)
        return httpx.Response(200, json=upstream)

    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy._retry_request = fake_retry
        response = client.post(
            "/v1beta/models/gemini-2.5-flash:generateContent",
            headers={
                "x-headroom-base-url": PLUGIN_BASE_URL,
            },
            json=body,
        )

    assert response.status_code == 200, response.text
    assert response.json() == upstream
    assert captured["method"] == "POST"
    assert captured["url"] == (
        "https://gateway.example/v1beta/models/gemini-2.5-flash:generateContent"
    )
    assert captured["body"] == body
    outbound_headers = captured["headers"]
    assert isinstance(outbound_headers, dict)
    assert not any(key.lower().startswith("x-headroom-") for key in outbound_headers)


@pytest.mark.parametrize(
    ("headers", "expected_url"),
    [
        ({}, DIRECT_STREAM_URL),
        ({"x-headroom-base-url": PLUGIN_BASE_URL}, PLUGIN_STREAM_URL),
    ],
    ids=["direct", "plugin"],
)
def test_stream_generate_content_preserves_sse_and_native_parts(
    headers: dict[str, str], expected_url: str
) -> None:
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": "Keep this prompt."}]},
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "lookup", "args": {"city": "Paris"}}},
                    {"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}},
                ],
            },
        ]
    }
    sse_bytes = (
        b'data: {"candidates":[{"content":{"parts":[{"text":"answer"}]}}],'
        b'"usageMetadata":{"promptTokenCount":42,"candidatesTokenCount":7}}\n\n'
    )
    captured: dict[str, object] = {}

    async def fake_stream_response(
        url: str,
        headers: dict[str, str],
        request_body: dict,
        *_: object,
        **__: object,
    ) -> StreamingResponse:
        captured.update(url=url, headers=headers, body=request_body)
        return StreamingResponse(iter([sse_bytes]), media_type="text/event-stream")

    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy._stream_response = fake_stream_response
        response = client.post(
            "/v1beta/models/gemini-2.5-flash:streamGenerateContent",
            headers=headers,
            json=body,
        )

    assert response.status_code == 200, response.text
    assert response.content == sse_bytes
    assert captured["url"] == expected_url
    assert captured["body"] == body
    outbound_headers = captured["headers"]
    assert isinstance(outbound_headers, dict)
    assert not any(key.lower().startswith("x-headroom-") for key in outbound_headers)


def test_adjacent_custom_base_path_stays_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_ALLOWED_BASE_URLS", PLUGIN_BASE_URL)
    captured: dict[str, object] = {}

    async def fake_passthrough(
        request: Request,
        base_url: str,
        sub_path: str = "",
        provider_name: str = "",
    ) -> dict[str, object]:
        captured.update(
            path=request.url.path,
            base_url=base_url,
            sub_path=sub_path,
            provider_name=provider_name,
        )
        return {"handler": "passthrough"}

    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.handle_passthrough = fake_passthrough
        response = client.post(
            "/v1/adjacent-provider-path",
            headers={"x-headroom-base-url": PLUGIN_BASE_URL},
            content=b"adjacent",
        )

    assert response.status_code == 200, response.text
    assert captured == {
        "path": "/v1/adjacent-provider-path",
        "base_url": PLUGIN_BASE_URL,
        "sub_path": "",
        "provider_name": "",
    }


@pytest.mark.parametrize(
    ("headers", "expected_url"),
    [
        ({}, DIRECT_STREAM_URL),
        ({"x-headroom-base-url": PLUGIN_BASE_URL}, PLUGIN_STREAM_URL),
    ],
    ids=["direct", "plugin"],
)
def test_native_streaming_reaches_the_transform_pipeline(
    headers: dict[str, str], expected_url: str
) -> None:
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": "compress me " * 200}]},
        ]
    }
    captured: dict[str, object] = {}

    def fake_apply(**kwargs: object) -> CompressResult:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        captured["pipeline_messages"] = messages
        return CompressResult(
            messages=[{"role": "user", "content": "compressed"}],
            tokens_before=1000,
            tokens_after=10,
            tokens_saved=990,
            compression_ratio=0.99,
            transforms_applied=["smart_crusher"],
        )

    async def fake_stream_response(*args: object, **__: object) -> StreamingResponse:
        captured["url"] = args[0]
        captured["body"] = args[2]
        captured["transforms_applied"] = args[9]
        return StreamingResponse(iter([b"data: {}\n\n"]), media_type="text/event-stream")

    app = create_app(_streaming_optimize_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.openai_pipeline.apply = fake_apply
        proxy._stream_response = fake_stream_response
        response = client.post(
            "/v1beta/models/gemini-2.5-flash:streamGenerateContent",
            headers=headers,
            json=body,
        )

    assert response.status_code == 200, response.text
    assert captured["transforms_applied"] == ["smart_crusher"]
    assert captured["url"] == expected_url
    outbound = captured["body"]
    assert isinstance(outbound, dict)
    assert outbound["contents"] == [{"role": "user", "parts": [{"text": "compressed"}]}]
    assert captured["pipeline_messages"]


@pytest.mark.parametrize(
    ("headers", "expected_url"),
    [
        ({}, DIRECT_STREAM_URL),
        ({"x-headroom-base-url": PLUGIN_BASE_URL}, PLUGIN_STREAM_URL),
    ],
    ids=["direct", "plugin"],
)
def test_native_streaming_transforms_text_and_preserves_native_parts(
    headers: dict[str, str], expected_url: str
) -> None:
    body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "compress me " * 200}],
            },
            {
                "role": "model",
                "parts": [
                    {"functionCall": {"name": "lookup", "args": {"city": "Paris"}}},
                    {"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}},
                ],
            },
        ]
    }
    captured: dict[str, object] = {}

    def fake_apply(**kwargs: object) -> CompressResult:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        captured["pipeline_messages"] = messages
        return CompressResult(
            messages=[{"role": "user", "content": "compressed"}],
            tokens_before=1000,
            tokens_after=10,
            tokens_saved=990,
            compression_ratio=0.99,
            transforms_applied=["smart_crusher"],
        )

    async def fake_stream_response(*args: object, **__: object) -> StreamingResponse:
        captured["url"] = args[0]
        captured["body"] = args[2]
        captured["transforms_applied"] = args[9]
        return StreamingResponse(iter([b"data: mixed\n\n"]), media_type="text/event-stream")

    app = create_app(_streaming_optimize_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.openai_pipeline.apply = fake_apply
        proxy._stream_response = fake_stream_response
        response = client.post(
            "/v1beta/models/gemini-2.5-flash:streamGenerateContent",
            headers=headers,
            json=body,
        )

    assert response.status_code == 200, response.text
    assert response.content == b"data: mixed\n\n"
    assert captured["url"] == expected_url
    assert captured["transforms_applied"] == ["smart_crusher"]
    outbound = captured["body"]
    assert isinstance(outbound, dict)
    assert outbound["contents"] == [
        {"role": "user", "parts": [{"text": "compressed"}]},
        {
            "role": "model",
            "parts": [
                {"functionCall": {"name": "lookup", "args": {"city": "Paris"}}},
                {"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}},
            ],
        },
    ]


@pytest.mark.parametrize(
    ("headers", "query", "all_non_text", "expected_url"),
    [
        ({}, "key=secret&alt=sse", False, DIRECT_KEY_STREAM_URL),
        (
            {"x-headroom-base-url": PLUGIN_BASE_URL},
            "trace=1&foo=bar",
            False,
            PLUGIN_QUERY_STREAM_URL,
        ),
        ({}, "trace=hello%20world&encoded=%7E&alt=raw", False, DIRECT_ENCODED_QUERY_STREAM_URL),
        ({}, "key=secret&alt=sse", True, DIRECT_KEY_STREAM_URL),
        (
            {"x-headroom-base-url": PLUGIN_BASE_URL},
            "trace=1&foo=bar",
            True,
            PLUGIN_QUERY_STREAM_URL,
        ),
    ],
    ids=[
        "direct-normal",
        "plugin-normal",
        "direct-encoded-query",
        "direct-all-native",
        "plugin-all-native",
    ],
)
def test_native_streaming_preserves_query_and_adds_sse_once(
    headers: dict[str, str], query: str, all_non_text: bool, expected_url: str
) -> None:
    parts = (
        [{"functionCall": {"name": "lookup", "args": {"city": "Paris"}}}]
        if all_non_text
        else [{"text": "keep this"}]
    )
    body = {"contents": [{"role": "user", "parts": parts}]}
    captured: dict[str, object] = {}

    async def fake_stream_response(*args: object, **__: object) -> StreamingResponse:
        captured["url"] = args[0]
        return StreamingResponse(iter([b"data: query\n\n"]), media_type="text/event-stream")

    app = create_app(_streaming_optimize_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy._stream_response = fake_stream_response
        response = client.post(
            f"/v1beta/models/gemini-2.5-flash:streamGenerateContent?{query}",
            headers=headers,
            json=body,
        )

    assert response.status_code == 200, response.text
    assert response.content == b"data: query\n\n"
    assert captured["url"] == expected_url


@pytest.mark.parametrize(
    ("path", "handler_name"),
    [
        ("/v1beta/models/gemini-2.5-flash:generateContent", "handle_gemini_generate_content"),
        (
            "/v1beta/models/gemini-2.5-flash:streamGenerateContent",
            "handle_gemini_stream_generate_content",
        ),
        ("/v1beta/models/gemini-2.5-flash:countTokens", "handle_gemini_count_tokens"),
    ],
    ids=["generate", "stream", "count"],
)
def test_native_gemini_routes_reject_loopback_custom_base_before_handler(
    monkeypatch: pytest.MonkeyPatch, path: str, handler_name: str
) -> None:
    monkeypatch.delenv("HEADROOM_ALLOWED_BASE_URLS", raising=False)
    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        handler = AsyncMock(return_value=JSONResponse({"ok": True}))
        setattr(proxy, handler_name, handler)
        response = client.post(
            path,
            headers={"x-headroom-base-url": "http://127.0.0.1:12345"},
            json={"contents": []},
        )

    assert response.status_code == 400
    handler.assert_not_awaited()


def test_gemini_count_tokens_follows_the_same_tagged_upstream() -> None:
    """Counting and generating must agree on the upstream.

    Otherwise the client counts tokens at Google for a model the tagged gateway
    serves, and the gateway's key travels to Google.
    """
    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.handle_gemini_count_tokens = AsyncMock(return_value=JSONResponse({"ok": True}))
        response = client.post(
            "/v1beta/models/gemini-2.5-flash:countTokens",
            headers={"x-headroom-base-url": PLUGIN_BASE_URL},
            json={"contents": []},
        )

    assert response.status_code == 200, response.text
    proxy.handle_gemini_count_tokens.assert_awaited_once()
    call = proxy.handle_gemini_count_tokens.await_args
    assert call is not None
    assert call.args[1] == "gemini-2.5-flash"
    assert call.kwargs == {"upstream_base_url": PLUGIN_BASE_URL}


def test_gemini_count_tokens_without_the_header_stays_on_the_default_upstream() -> None:
    app = create_app(_config())
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        proxy.handle_gemini_count_tokens = AsyncMock(return_value=JSONResponse({"ok": True}))
        response = client.post(
            "/v1beta/models/gemini-2.5-flash:countTokens",
            json={"contents": []},
        )

    assert response.status_code == 200, response.text
    call = proxy.handle_gemini_count_tokens.await_args
    assert call is not None
    assert call.kwargs == {}
