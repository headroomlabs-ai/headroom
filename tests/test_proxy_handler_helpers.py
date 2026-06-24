from __future__ import annotations

import asyncio
import base64
import builtins
import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from fastapi.responses import StreamingResponse

from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.handlers.openai import (
    OpenAIHandlerMixin,
    _decode_openai_bearer_payload,
    _passthrough_usage_from_json,
    _prefers_http1_passthrough,
)
from headroom.proxy.helpers import _headroom_bypass_enabled
from headroom.proxy.server import HeadroomProxy


def _jwt(payload: object) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def encode(part: object) -> str:
        raw = json.dumps(part, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode(header)}.{encode(payload)}."


class _ImageCompressor:
    def __init__(self, compressed_message):
        self._compressed_message = compressed_message

    def compress(self, messages, provider):  # noqa: ANN001, ANN201
        assert provider == "anthropic"
        return [self._compressed_message]


class _FreshCompressor:
    instances = 0

    def __init__(self):
        type(self).instances += 1


class _TimeoutHttpClient:
    async def request(self, **kwargs):  # noqa: ANN001, ANN201
        raise httpx.ConnectTimeout("connect timed out")


class _RecordingHttpClient:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls = 0

    async def request(self, **kwargs):  # noqa: ANN001, ANN201
        self.calls += 1
        request = httpx.Request(kwargs["method"], kwargs["url"])
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "application/json"},
            json={"client": self.label},
        )


class _ChatGPTAccountRequest:
    method = "GET"
    headers = {}
    url = SimpleNamespace(path="/backend-api/me", query="")

    async def body(self) -> bytes:
        return b""


class _PassthroughRequest:
    method = "GET"
    headers = {}
    url = SimpleNamespace(path="/favicon.ico", query="")

    async def body(self) -> bytes:
        return b""


class _VertexPassthroughRequest:
    method = "POST"
    headers = {}
    url = SimpleNamespace(
        path="/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent",
        query="",
    )

    async def body(self) -> bytes:
        return b'{"contents":[]}'


class _VertexStreamPassthroughRequest:
    method = "POST"
    headers = {}
    url = SimpleNamespace(
        path="/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.0-flash:streamGenerateContent",
        query="alt=sse",
    )

    async def body(self) -> bytes:
        return b'{"contents":[]}'


class _VertexGeminiImageRequest:
    method = "POST"
    headers = {}
    query_params = {}
    url = SimpleNamespace(
        path="/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent",
        query="",
    )

    async def body(self) -> bytes:
        return json.dumps(
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "image/png",
                                    "data": "aW1hZ2U=",
                                }
                            }
                        ],
                    }
                ]
            }
        ).encode("utf-8")


class _VertexUsageClient:
    async def request(self, **kwargs):  # noqa: ANN001, ANN201
        request = httpx.Request(kwargs["method"], kwargs["url"], content=kwargs["content"])
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "application/json"},
            json={
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 11,
                    "candidatesTokenCount": 7,
                    "cachedContentTokenCount": 3,
                },
            },
        )


class _AsyncChunks(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):  # noqa: ANN204
        for chunk in self._chunks:
            yield chunk


class _VertexStreamClient:
    def __init__(self) -> None:
        self.sent_url = ""

    def build_request(self, method, url, headers, content):  # noqa: ANN001, ANN201
        self.sent_url = str(url)
        return httpx.Request(method, url, headers=headers, content=content)

    async def send(self, request, stream=False):  # noqa: ANN001, ANN201
        assert stream is True
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "text/event-stream"},
            stream=_AsyncChunks(
                [
                    b'data: {"candidates":[{"content":{"parts":[{"text":"hello"}]}}]}\n\n',
                    b'data: {"usageMetadata":{"promptTokenCount":13,'
                    b'"candidatesTokenCount":5,"cachedContentTokenCount":2}}\n\n',
                ]
            ),
        )


class _RetryThenSuccessClient:
    def __init__(self) -> None:
        self.attempts = 0

    async def post(self, url, content, headers, timeout=None):  # noqa: ANN001, ANN201
        self.attempts += 1
        if self.attempts == 1:
            raise httpx.ConnectTimeout("connect timed out")
        del timeout
        request = httpx.Request("POST", url, headers=headers, content=content)
        return httpx.Response(200, request=request, content=b"{}")


def test_decode_openai_bearer_payload_handles_missing_and_non_mapping_payloads() -> None:
    assert _decode_openai_bearer_payload({}) is None
    assert _decode_openai_bearer_payload({"authorization": "Basic abc"}) is None
    assert (
        _decode_openai_bearer_payload({"authorization": f"Bearer {_jwt(['not', 'a', 'dict'])}"})
        is None
    )


def test_openai_handler_prefix_helpers_cover_edge_cases() -> None:
    assert OpenAIHandlerMixin._strict_previous_turn_frozen_count([], 2) == 2
    assert (
        OpenAIHandlerMixin._strict_previous_turn_frozen_count(
            [{"role": "assistant"}, {"role": "user"}],
            0,
        )
        == 1
    )
    assert (
        OpenAIHandlerMixin._strict_previous_turn_frozen_count(
            [{"role": "user"}, {"role": "assistant"}],
            0,
        )
        == 2
    )

    original = [{"role": "system", "content": "keep"}, {"role": "user", "content": "hello"}]
    restored, changed = OpenAIHandlerMixin._restore_frozen_prefix(
        original,
        [],
        frozen_message_count=1,
    )
    assert restored == [{"role": "system", "content": "keep"}]
    assert changed == 1

    restored, changed = OpenAIHandlerMixin._restore_frozen_prefix(
        original,
        [{"role": "system", "content": "changed"}, {"role": "user", "content": "hello"}],
        frozen_message_count=1,
    )
    assert restored == original
    assert changed == 1


def test_headroom_bypass_helper_is_transport_neutral() -> None:
    assert _headroom_bypass_enabled({"x-headroom-bypass": "true"}) is True
    assert _headroom_bypass_enabled({"x-headroom-bypass": " TRUE "}) is True
    assert _headroom_bypass_enabled({"x-headroom-mode": "passthrough"}) is True
    assert _headroom_bypass_enabled({"x-headroom-mode": " PASSTHROUGH "}) is True
    assert _headroom_bypass_enabled({"x-headroom-bypass": "false"}) is False
    assert _headroom_bypass_enabled({}) is False
    assert _headroom_bypass_enabled(None) is False
    assert OpenAIHandlerMixin._headroom_bypass_enabled({"x-headroom-bypass": "true"}) is True


def test_openai_passthrough_connect_timeout_returns_502() -> None:
    handler = object.__new__(OpenAIHandlerMixin)
    handler.http_client = _TimeoutHttpClient()

    async def run():
        return await handler.handle_passthrough(
            _PassthroughRequest(),
            "https://api.openai.com",
        )

    response = asyncio.run(run())

    assert response.status_code == 502
    payload = json.loads(response.body)
    assert payload["error"]["type"] == "connection_error"
    assert "Failed to connect to upstream API" in payload["error"]["message"]


def test_prefers_http1_passthrough_matches_chatgpt_hosts_only() -> None:
    assert _prefers_http1_passthrough("https://chatgpt.com") is True
    assert _prefers_http1_passthrough("https://chatgpt.com/backend-api/me") is True
    assert _prefers_http1_passthrough("https://api.chatgpt.com") is True
    assert _prefers_http1_passthrough("https://CHATGPT.COM/backend-api/me") is True
    assert _prefers_http1_passthrough("https://api.openai.com") is False
    assert _prefers_http1_passthrough("https://notchatgpt.com") is False
    assert _prefers_http1_passthrough("https://chatgpt.com.evil.com") is False
    assert _prefers_http1_passthrough("") is False


def test_chatgpt_passthrough_uses_http1_client() -> None:
    handler = object.__new__(OpenAIHandlerMixin)
    handler.http_client = _RecordingHttpClient("h2")
    handler.http_client_h1 = _RecordingHttpClient("h1")

    response = asyncio.run(
        handler.handle_passthrough(_ChatGPTAccountRequest(), "https://chatgpt.com")
    )

    assert response.status_code == 200
    assert json.loads(response.body)["client"] == "h1"
    assert handler.http_client.calls == 0
    assert handler.http_client_h1.calls == 1


def test_non_chatgpt_passthrough_uses_default_client() -> None:
    handler = object.__new__(OpenAIHandlerMixin)
    handler.http_client = _RecordingHttpClient("h2")
    handler.http_client_h1 = _RecordingHttpClient("h1")

    response = asyncio.run(
        handler.handle_passthrough(_PassthroughRequest(), "https://api.openai.com")
    )

    assert response.status_code == 200
    assert json.loads(response.body)["client"] == "h2"
    assert handler.http_client.calls == 1
    assert handler.http_client_h1.calls == 0


def test_chatgpt_passthrough_falls_back_when_h1_client_missing() -> None:
    handler = object.__new__(OpenAIHandlerMixin)
    handler.http_client = _RecordingHttpClient("h2")
    handler.http_client_h1 = None

    response = asyncio.run(
        handler.handle_passthrough(_ChatGPTAccountRequest(), "https://chatgpt.com")
    )

    assert response.status_code == 200
    assert json.loads(response.body)["client"] == "h2"
    assert handler.http_client.calls == 1


def test_passthrough_usage_normalizes_vertex_usage_metadata() -> None:
    usage = _passthrough_usage_from_json(
        {
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 7,
                "cachedContentTokenCount": 3,
            }
        }
    )

    assert usage == {
        "input_tokens": 11,
        "output_tokens": 7,
        "cache_read_input_tokens": 3,
    }


def test_vertex_passthrough_records_usage_metadata_for_dashboard() -> None:
    handler = object.__new__(HeadroomProxy)
    handler.http_client = _VertexUsageClient()
    outcomes = []

    async def next_request_id():  # noqa: ANN202
        return "req_vertex"

    async def record(outcome):  # noqa: ANN001, ANN202
        outcomes.append(outcome)

    handler._next_request_id = next_request_id
    handler._record_request_outcome = record

    response = asyncio.run(
        handler.handle_passthrough(
            _VertexPassthroughRequest(),
            "https://vertex.test",
            "generateContent",
            "vertex:google",
        )
    )

    assert response.status_code == 200
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.provider == "vertex:google"
    assert outcome.model == "gemini-2.0-flash"
    assert outcome.optimized_tokens == 11
    assert outcome.output_tokens == 7
    assert outcome.cache_read_tokens == 3


def test_vertex_stream_passthrough_preserves_chunks_and_records_usage() -> None:
    handler = object.__new__(HeadroomProxy)
    handler.http_client = _VertexStreamClient()
    outcomes = []

    async def next_request_id():  # noqa: ANN202
        return "req_vertex_stream"

    async def record(outcome):  # noqa: ANN001, ANN202
        outcomes.append(outcome)

    handler._next_request_id = next_request_id
    handler._record_request_outcome = record

    response = asyncio.run(
        handler.handle_passthrough(
            _VertexStreamPassthroughRequest(),
            "https://vertex.test",
            "streamGenerateContent",
            "vertex:google",
        )
    )

    assert isinstance(response, StreamingResponse)

    async def collect():  # noqa: ANN202
        return [chunk async for chunk in response.body_iterator]

    chunks = asyncio.run(collect())

    assert len(chunks) == 2
    assert chunks[0].startswith(b'data: {"candidates"')
    assert b'"usageMetadata"' in chunks[1]
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.provider == "vertex:google"
    assert outcome.model == "gemini-2.0-flash"
    assert outcome.optimized_tokens == 13
    assert outcome.output_tokens == 5
    assert outcome.cache_read_tokens == 2


def test_stream_finalizer_records_vertex_provider_for_dashboard() -> None:
    handler = object.__new__(HeadroomProxy)
    handler.config = SimpleNamespace(log_full_messages=False)
    outcomes = []

    async def record(outcome):  # noqa: ANN001, ANN202
        outcomes.append(outcome)

    handler._record_request_outcome = record

    asyncio.run(
        handler._finalize_stream_response(
            body={"contents": [{"role": "user", "parts": [{"text": "hello"}]}]},
            provider="gemini",
            outcome_provider="vertex:google",
            model="gemini-2.0-flash",
            request_id="req_vertex_stream_final",
            original_tokens=20,
            optimized_tokens=12,
            tokens_saved=8,
            transforms_applied=["test-transform"],
            optimization_latency=3.0,
            stream_state={
                "input_tokens": 12,
                "output_tokens": 5,
                "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 0,
                "cache_creation_ephemeral_5m_input_tokens": 0,
                "cache_creation_ephemeral_1h_input_tokens": 0,
                "total_bytes": 100,
                "sse_buffer": bytearray(),
                "ttfb_ms": 4.0,
            },
            start_time=0.0,
            tags={"route": "vertex"},
        )
    )

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.provider == "vertex:google"
    assert outcome.model == "gemini-2.0-flash"
    assert outcome.optimized_tokens == 12
    assert outcome.output_tokens == 5
    assert outcome.tokens_saved == 8
    assert outcome.cache_read_tokens == 2


def test_vertex_gemini_non_text_generate_records_dashboard_outcome() -> None:
    handler = object.__new__(HeadroomProxy)
    handler.memory_handler = None
    handler.rate_limiter = None
    outcomes = []
    upstream_urls = []

    async def next_request_id():  # noqa: ANN202
        return "req_vertex_image"

    async def record(outcome):  # noqa: ANN001, ANN202
        outcomes.append(outcome)

    async def retry_request(method, url, headers, body):  # noqa: ANN001, ANN202
        upstream_urls.append(url)
        request = httpx.Request(method, url, headers=headers)
        return httpx.Response(
            200,
            request=request,
            headers={"content-type": "application/json"},
            json={
                "usageMetadata": {
                    "promptTokenCount": 31,
                    "candidatesTokenCount": 4,
                    "cachedContentTokenCount": 6,
                }
            },
        )

    handler._next_request_id = next_request_id
    handler._record_request_outcome = record
    handler._retry_request = retry_request

    response = asyncio.run(
        handler.handle_gemini_generate_content(
            _VertexGeminiImageRequest(),
            "gemini-2.0-flash",
            "https://vertex.test",
            "vertex:google",
        )
    )

    assert response.status_code == 200
    assert upstream_urls == [
        "https://vertex.test/v1/projects/p/locations/us-central1/publishers/google/models/gemini-2.0-flash:generateContent"
    ]
    assert response.headers["x-headroom-tokens-before"] == "31"
    assert response.headers["x-headroom-tokens-after"] == "31"
    assert response.headers["x-headroom-tokens-saved"] == "0"
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.provider == "vertex:google"
    assert outcome.model == "gemini-2.0-flash"
    assert outcome.original_tokens == 31
    assert outcome.optimized_tokens == 31
    assert outcome.output_tokens == 4
    assert outcome.cache_read_tokens == 6
    assert outcome.num_messages == 1


def test_retry_request_retries_connect_timeout() -> None:
    proxy = object.__new__(HeadroomProxy)
    proxy.http_client = _RetryThenSuccessClient()
    proxy.config = SimpleNamespace(
        retry_enabled=True,
        retry_max_attempts=2,
        retry_base_delay_ms=0,
        retry_max_delay_ms=0,
    )

    response = asyncio.run(
        proxy._retry_request(
            "POST",
            "https://api.openai.com/v1/responses",
            {},
            {"model": "gpt-5"},
        )
    )

    assert response.status_code == 200
    assert proxy.http_client.attempts == 2


def test_anthropic_tool_sort_and_context_append_helpers() -> None:
    tools = [
        {"type": "function", "function": {"name": "beta"}},
        {"name": "alpha"},
        {"type": "tool"},
    ]

    sorted_tools = AnthropicHandlerMixin._sort_tools_deterministically(tools)

    assert [AnthropicHandlerMixin._tool_sort_key(tool)[0] for tool in sorted_tools] == [
        "alpha",
        "beta",
        "tool",
    ]
    assert AnthropicHandlerMixin._sort_tools_deterministically(None) is None
    assert (
        AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
            [], "ctx", frozen_message_count=0
        )
        == []
    )
    assert AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
        [{"role": "user", "content": "hello"}],
        "ctx",
        frozen_message_count=0,
    ) == [{"role": "user", "content": "hello\n\nctx"}]
    # PR-A2 semantics: list-content user messages get the context appended
    # to the first text block (live-zone-tail injection).
    assert AnthropicHandlerMixin._append_context_to_latest_non_frozen_user_turn(
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "ctx",
        frozen_message_count=0,
    ) == [{"role": "user", "content": [{"type": "text", "text": "hello\n\nctx"}]}]


def test_anthropic_image_compression_helper_only_rewrites_latest_eligible_turn() -> None:
    image_message = {
        "role": "user",
        "content": [{"type": "image", "source": {"type": "base64", "data": "abc"}}],
    }
    compressed = {
        "role": "user",
        "content": [{"type": "image", "source": {"type": "base64", "data": "xyz"}}],
    }

    assert (
        AnthropicHandlerMixin._compress_latest_user_turn_images_cache_safe(
            [],
            frozen_message_count=0,
            compressor=_ImageCompressor(compressed),
        )
        == []
    )
    assert AnthropicHandlerMixin._compress_latest_user_turn_images_cache_safe(
        [image_message],
        frozen_message_count=1,
        compressor=_ImageCompressor(compressed),
    ) == [image_message]
    assert AnthropicHandlerMixin._compress_latest_user_turn_images_cache_safe(
        [{"role": "assistant", "content": image_message["content"]}],
        frozen_message_count=0,
        compressor=_ImageCompressor(compressed),
    ) == [{"role": "assistant", "content": image_message["content"]}]
    assert AnthropicHandlerMixin._compress_latest_user_turn_images_cache_safe(
        [{"role": "user", "content": "no-image"}],
        frozen_message_count=0,
        compressor=_ImageCompressor(compressed),
    ) == [{"role": "user", "content": "no-image"}]
    assert AnthropicHandlerMixin._compress_latest_user_turn_images_cache_safe(
        [image_message],
        frozen_message_count=0,
        compressor=_ImageCompressor(image_message),
    ) == [image_message]
    assert AnthropicHandlerMixin._compress_latest_user_turn_images_cache_safe(
        [image_message],
        frozen_message_count=0,
        compressor=_ImageCompressor(compressed),
    ) == [compressed]


def test_proxy_helper_creates_fresh_image_compressors(monkeypatch) -> None:
    from headroom.proxy import helpers

    monkeypatch.setattr(helpers, "_image_compressor_available", None)
    _FreshCompressor.instances = 0

    with patch("headroom.image.ImageCompressor", _FreshCompressor):
        first = helpers._get_image_compressor()
        second = helpers._get_image_compressor()

    assert isinstance(first, _FreshCompressor)
    assert isinstance(second, _FreshCompressor)
    assert first is not second
    assert _FreshCompressor.instances == 2


def test_proxy_helper_caches_image_stack_import_failure(monkeypatch) -> None:
    from headroom.proxy import helpers

    real_import = builtins.__import__
    calls = 0

    def fake_import(name, *args, **kwargs):  # noqa: ANN001, ANN202
        nonlocal calls
        if name == "headroom.image":
            calls += 1
            raise ImportError("image extras unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(helpers, "_image_compressor_available", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert helpers._get_image_compressor() is None
    assert helpers._get_image_compressor() is None
    assert calls == 1
    assert helpers._image_compressor_available is False


def test_anthropic_cache_delta_helpers_cover_string_list_and_role_mismatch() -> None:
    previous_original = [{"role": "user", "content": "hello"}]
    previous_forwarded = [{"role": "user", "content": "HELLO"}]

    assert AnthropicHandlerMixin._extract_cache_stable_delta(
        [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "next"}],
        previous_original,
        previous_forwarded,
    ) == (previous_forwarded, [{"role": "assistant", "content": "next"}])
    assert (
        AnthropicHandlerMixin._extract_cache_stable_delta(
            [{"role": "assistant", "content": "hello"}],
            previous_original,
            previous_forwarded,
        )
        is None
    )

    string_suffix = AnthropicHandlerMixin._extract_cache_stable_last_message_suffix(
        [{"role": "user", "content": "hello world"}],
        previous_original,
        previous_forwarded,
    )
    assert string_suffix == ([], previous_forwarded[0], [{"role": "user", "content": " world"}])

    list_suffix = AnthropicHandlerMixin._extract_cache_stable_last_message_suffix(
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            }
        ],
        [{"role": "user", "content": [{"type": "text", "text": "a"}]}],
        [{"role": "user", "content": [{"type": "text", "text": "A"}]}],
    )
    assert list_suffix == (
        [],
        {"role": "user", "content": [{"type": "text", "text": "A"}]},
        [{"role": "user", "content": [{"type": "text", "text": "b"}]}],
    )

    assert AnthropicHandlerMixin._merge_appended_message_delta(
        {"role": "user", "content": "HELLO"},
        {"role": "user", "content": " world"},
    ) == {"role": "user", "content": "HELLO world"}
    assert AnthropicHandlerMixin._merge_appended_message_delta(
        {"role": "user", "content": [{"type": "text", "text": "A"}]},
        {"role": "user", "content": [{"type": "text", "text": "b"}]},
    ) == {"role": "user", "content": [{"type": "text", "text": "A"}, {"type": "text", "text": "b"}]}
    assert (
        AnthropicHandlerMixin._merge_appended_message_delta(
            {"role": "user", "content": "A"},
            {"role": "assistant", "content": "B"},
        )
        is None
    )


def test_anthropic_assistant_message_helper_requires_assistant_role() -> None:
    assert AnthropicHandlerMixin._assistant_message_from_response_json(None) is None
    assert AnthropicHandlerMixin._assistant_message_from_response_json({"role": "user"}) is None
    assert AnthropicHandlerMixin._assistant_message_from_response_json(
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}
    ) == {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}


# ============================================================================
# CCR workspace resolution (cross-project leak fix, 2026-05-26).
#
# These tests pin the `_resolve_ccr_workspace` static helper that the
# anthropic handler uses to scope the proactive-expansion cache by
# project identity. The resolver shares its tier order with the memory
# subsystem's ProjectResolver: x-headroom-project-id → x-headroom-cwd →
# system-prompt `cwd:` line. Returns `("", None)` on no signal — the
# fail-closed signal that callers gate on.
# ============================================================================


def _fake_request(headers: dict[str, str]) -> SimpleNamespace:
    """Minimal Starlette/FastAPI-shaped request object for resolver tests."""
    return SimpleNamespace(headers=headers)


def test_resolve_ccr_workspace_explicit_project_id_wins() -> None:
    """x-headroom-project-id is the highest-priority signal."""
    request = _fake_request({"x-headroom-project-id": "my-cool-project"})
    body = {}
    key, label = AnthropicHandlerMixin._resolve_ccr_workspace(request, body)
    assert key == "my-cool-project"
    assert label == "my-cool-project"


def test_resolve_ccr_workspace_cwd_header() -> None:
    """x-headroom-cwd produces a stable per-cwd key + basename label."""
    request = _fake_request({"x-headroom-cwd": "/home/user/code/daphni-rails"})
    body = {}
    key, label = AnthropicHandlerMixin._resolve_ccr_workspace(request, body)
    # Key format: "{basename}-{sha256[:16]}" — stable per absolute cwd.
    assert key.startswith("daphni-rails-")
    assert len(key) >= len("daphni-rails-") + 16
    assert label == "daphni-rails"


def test_resolve_ccr_workspace_two_cwds_get_distinct_keys() -> None:
    """Two different cwds produce different workspace keys (cross-leak prevention)."""
    key_a, _ = AnthropicHandlerMixin._resolve_ccr_workspace(
        _fake_request({"x-headroom-cwd": "/home/user/code/daphni-rails"}), {}
    )
    key_b, _ = AnthropicHandlerMixin._resolve_ccr_workspace(
        _fake_request({"x-headroom-cwd": "/home/user/code/tamag0"}), {}
    )
    assert key_a != key_b, "different cwds must yield different workspace keys"


def test_resolve_ccr_workspace_no_signal_returns_empty() -> None:
    """No project-id, no cwd header, no system prompt → fail-closed signal."""
    request = _fake_request({})
    body = {}
    key, label = AnthropicHandlerMixin._resolve_ccr_workspace(request, body)
    assert key == ""
    assert label is None


def test_resolve_ccr_workspace_system_prompt_cwd_fallback() -> None:
    """System prompt with `cwd:` line is the lowest-tier fallback."""
    request = _fake_request({})
    body = {
        "system": [{"type": "text", "text": "You are helpful.\ncwd: /home/u/code/my-project\nGo."}]
    }
    key, label = AnthropicHandlerMixin._resolve_ccr_workspace(request, body)
    # The label is the basename of the cwd extracted from the prompt.
    assert label == "my-project"
    assert key.startswith("my-project-")


def test_resolve_ccr_workspace_malformed_request_returns_empty() -> None:
    """A request whose headers attribute can't be dict()-ed fails closed, not crashes."""

    class _BrokenHeaders:
        def __iter__(self):
            raise RuntimeError("boom")

    request = SimpleNamespace(headers=_BrokenHeaders())
    body = {}
    # The helper catches the exception, logs it, and returns the fail-
    # closed sentinel ("", None). Critically, it does NOT raise — the
    # proxy must continue serving the request even if CCR scoping fails.
    key, label = AnthropicHandlerMixin._resolve_ccr_workspace(request, body)
    assert key == ""
    assert label is None
