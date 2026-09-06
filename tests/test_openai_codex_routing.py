import asyncio
import base64
import json
import sys
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anyio
import httpx
import pytest
from fastapi import Request

from headroom.proxy.handlers.openai import (
    OpenAIHandlerMixin,
    _is_allowed_websocket_origin,
    _openai_responses_unit_cache_key,
    _resolve_codex_routing_headers,
    _responses_stateless_output_items,
)


def _jwt(payload: dict) -> str:
    header = {"alg": "none", "typ": "JWT"}

    def encode(part: dict) -> str:
        raw = json.dumps(part, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    return f"{encode(header)}.{encode(payload)}."


def test_resolve_codex_routing_prefers_explicit_header():
    headers, is_chatgpt = _resolve_codex_routing_headers(
        {
            "Authorization": "Bearer sk-test",
            "ChatGPT-Account-ID": "acct-explicit",
        }
    )

    assert is_chatgpt is True
    assert headers["ChatGPT-Account-ID"] == "acct-explicit"


def test_resolve_codex_routing_derives_account_id_from_oauth_jwt():
    token = _jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-from-jwt",
            }
        }
    )

    headers, is_chatgpt = _resolve_codex_routing_headers(
        {
            "authorization": f"Bearer {token}",
        }
    )

    assert is_chatgpt is True
    assert headers["ChatGPT-Account-ID"] == "acct-from-jwt"


def test_resolve_codex_routing_leaves_regular_openai_bearer_tokens_unchanged():
    token = _jwt({"aud": ["https://api.openai.com/v1"]})

    headers, is_chatgpt = _resolve_codex_routing_headers(
        {
            "authorization": f"Bearer {token}",
        }
    )

    assert is_chatgpt is False
    assert "ChatGPT-Account-ID" not in headers


def test_resolve_codex_routing_returns_none_without_bearer_auth():
    headers, is_chatgpt = _resolve_codex_routing_headers({})

    assert is_chatgpt is False
    assert headers == {}


def test_resolve_codex_routing_ignores_non_jwt_bearer_tokens():
    headers, is_chatgpt = _resolve_codex_routing_headers(
        {
            "authorization": "Bearer not-a-jwt",
        }
    )

    assert is_chatgpt is False
    assert headers["authorization"] == "Bearer not-a-jwt"


def test_resolve_codex_routing_ignores_invalid_jwt_payloads():
    invalid_payload = base64.urlsafe_b64encode(b"not-json").decode("ascii").rstrip("=")
    token = f"test-header.{invalid_payload}.signature"

    headers, is_chatgpt = _resolve_codex_routing_headers(
        {
            "authorization": f"Bearer {token}",
        }
    )

    assert is_chatgpt is False
    assert headers["authorization"] == f"Bearer {token}"


def test_openai_responses_unit_cache_key_includes_target_ratio() -> None:
    unit = SimpleNamespace(
        text="large tool output",
        provider="openai",
        endpoint="responses",
        role="tool",
        item_type="function_call_output",
        cache_zone="live",
        mutable=True,
        min_bytes=100,
        context=None,
        question=None,
        bias=None,
        metadata={},
    )

    default_key = _openai_responses_unit_cache_key(unit, model="gpt-5.4")
    aggressive_key = _openai_responses_unit_cache_key(
        unit,
        model="gpt-5.4",
        target_ratio=0.10,
    )
    balanced_key = _openai_responses_unit_cache_key(
        unit,
        model="gpt-5.4",
        target_ratio=0.50,
    )

    assert aggressive_key != default_key
    assert aggressive_key != balanced_key


def test_responses_stateless_output_items_drop_unencrypted_reasoning() -> None:
    assert _responses_stateless_output_items(None) == []
    assert _responses_stateless_output_items(
        [
            {"type": "reasoning", "id": "rs-unusable", "summary": []},
            {
                "type": "reasoning",
                "id": "rs-reusable",
                "summary": [],
                "encrypted_content": "encrypted",
            },
            {"type": "function_call", "call_id": "call-1"},
        ]
    ) == [
        {
            "type": "reasoning",
            "id": "rs-reusable",
            "summary": [],
            "encrypted_content": "encrypted",
        },
        {"type": "function_call", "call_id": "call-1"},
    ]


class _DummyMetrics:
    async def record_request(self, **kwargs):  # noqa: ANN003
        return None

    async def record_failed(self, **kwargs):  # noqa: ANN003
        return None


class _DummyTokenizer:
    def count_messages(self, messages):
        return len(messages)


class _ResponseStub:
    status_code = 200
    headers = {"content-type": "application/json", "content-length": "42"}
    content = b'{"id":"resp_123","output":[{"type":"message"}]}'

    def json(self):
        return {"usage": {"input_tokens": 2, "output_tokens": 1}}


class _DummyOpenAIHandler(OpenAIHandlerMixin):
    OPENAI_API_URL = "https://api.openai.com"

    def __init__(self) -> None:
        self.rate_limiter = None
        self.metrics = _DummyMetrics()
        self.config = SimpleNamespace(
            optimize=False,
            retry_max_attempts=3,
            retry_base_delay_ms=10,
            retry_max_delay_ms=50,
            connect_timeout_seconds=10,
            openai_extra_headers=None,
        )
        self.usage_reporter = None
        self.openai_provider = SimpleNamespace(get_context_limit=lambda model: 128_000)
        self.openai_pipeline = SimpleNamespace(apply=MagicMock())
        self.anthropic_backend = None
        self.cost_tracker = None
        self.memory_handler = None
        self.traffic_learner = None
        # PR-A6 wires session-sticky `OpenAI-Beta` merging into the
        # responses HTTP handler — it reads `compute_session_id` to key
        # the SessionBetaTracker. The routing tests don't exercise the
        # tracker semantics themselves, so a fixed-id stub is enough.
        self.session_tracker_store = SimpleNamespace(
            compute_session_id=lambda *a, **k: "sess-openai-1",
        )
        self.captured_request: tuple[str, str, dict, dict] | None = None
        self.captured_stream_request: tuple[str, dict, dict] | None = None

    async def _next_request_id(self) -> str:
        return "req-1"

    def _extract_tags(self, headers: dict[str, str]) -> dict[str, str]:
        return {}

    async def _retry_request(self, method: str, url: str, headers: dict, body: dict, **kwargs):
        self.captured_request = (method, url, headers, body)
        return _ResponseStub()

    async def _run_compression_in_executor(self, fn, *, timeout: float):
        # Test stub for HeadroomProxy._run_compression_in_executor.
        # The real implementation runs `fn` on a bounded thread pool with
        # a wall-clock timeout; tests just need the callable invoked
        # synchronously so MagicMock call_count assertions fire.
        return fn()

    async def _count_tokens_offloaded(self, model, messages):  # noqa: ANN001, ANN201
        # Test stub for HeadroomProxy._count_tokens_offloaded: resolve the
        # tokenizer and count inline (the real method offloads to the executor).
        from headroom.tokenizers import get_tokenizer

        tokenizer = get_tokenizer(model)
        return tokenizer, tokenizer.count_messages(messages)

    async def _record_request_outcome(self, outcome) -> None:
        # Test stub: delegates to the production funnel so wire shape
        # matches HeadroomProxy._record_request_outcome.
        from headroom.proxy.outcome import emit_request_outcome

        await emit_request_outcome(self, outcome)

    async def _stream_response(
        self,
        url: str,
        headers: dict,
        body: dict,
        provider: str,
        model: str,
        request_id: str,
        original_tokens: int,
        optimized_tokens: int,
        tokens_saved: int,
        transforms_applied: list[str],
        tags: dict[str, str],
        optimization_latency: float,
        memory_user_id: str | None = None,
        **kwargs,
    ):
        self.captured_stream_request = (url, headers, body)
        return SimpleNamespace(
            status_code=200,
            url=url,
            headers=headers,
            body=body,
            memory_user_id=memory_user_id,
        )


class _MemoryToolsOnlyHandler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            inject_context=False,
            inject_tools=True,
            project_root_override="",
        )
        self.compute_calls = 0

    def compute_memory_tool_definitions(self, provider: str) -> list[dict]:
        self.compute_calls += 1
        assert provider == "openai"
        return [
            {
                "type": "function",
                "function": {
                    "name": "memory_search",
                    "description": "Search memory.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def has_memory_tool_calls(self, response: dict, provider: str) -> bool:
        return False


class _MemoryContinuationHandler(_MemoryToolsOnlyHandler):
    async def _ensure_initialized(self) -> None:
        self._backend = True

    async def _execute_memory_tool(
        self,
        name: str,
        args: dict,
        user_id: str,
        provider: str,
    ) -> str:
        assert (name, args, user_id, provider) == (
            "memory_search",
            {},
            "user-1",
            "openai",
        )
        return '{"memories": []}'

    def has_memory_tool_calls(self, response: dict, provider: str) -> bool:
        assert provider == "openai"
        return any(
            item.get("name") == "memory_search"
            for item in response.get("output", [])
            if isinstance(item, dict)
        )


class _ZdrResponsesHandler(_DummyOpenAIHandler):
    def __init__(self) -> None:
        super().__init__()
        self.memory_handler = _MemoryContinuationHandler()
        self.requests: list[dict] = []

    async def _retry_request(self, method: str, url: str, headers: dict, body: dict, **kwargs):
        assert (method, url) == ("POST", "https://api.openai.com/v1/responses")
        self.requests.append(deepcopy(body))
        request = httpx.Request(method, url)
        if len(self.requests) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "resp-initial",
                    "output": [
                        {
                            "type": "reasoning",
                            "id": "reasoning-1",
                            "summary": [],
                            "encrypted_content": "encrypted-1",
                        },
                        {
                            "type": "function_call",
                            "id": "fc-1",
                            "call_id": "call-1",
                            "name": "memory_search",
                            "arguments": "{}",
                        },
                    ],
                    "usage": {"input_tokens": 3, "output_tokens": 2},
                },
                request=request,
            )
        if any(
            isinstance(item, dict)
            and item.get("type") == "reasoning"
            and not item.get("encrypted_content")
            for item in body.get("input", [])
            if isinstance(body.get("input"), list)
        ):
            return httpx.Response(
                400,
                json={
                    "error": {"message": "Reasoning item is not reusable without encrypted_content"}
                },
                request=request,
            )
        if "previous_response_id" in body:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "message": "Unknown parameter: 'previous_response_id'.",
                        "type": "invalid_request_error",
                        "param": "previous_response_id",
                        "code": "unsupported_parameter",
                    }
                },
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "id": "resp-final",
                "output": [{"type": "message", "id": "message-1"}],
                "usage": {"input_tokens": 8, "output_tokens": 3},
            },
            request=request,
        )


def _build_request(body: dict, headers: dict[str, str], path: str = "/v1/responses") -> Request:
    payload = json.dumps(body).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": path,
        "raw_path": path.encode("utf-8"),
        "query_string": b"",
        "headers": [
            (key.lower().encode("utf-8"), value.encode("utf-8")) for key, value in headers.items()
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 443),
    }
    return Request(scope, receive)


def test_handle_openai_responses_routes_chatgpt_auth_to_backend_api(monkeypatch):
    token = _jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-from-jwt",
            }
        }
    )
    request = _build_request(
        {"model": "gpt-5.4", "input": "hello"},
        {"Authorization": f"Bearer {token}"},
    )
    handler = _DummyOpenAIHandler()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert handler.captured_request is not None
    method, url, headers, body = handler.captured_request
    assert method == "POST"
    assert url == "https://chatgpt.com/backend-api/codex/responses"
    assert headers["ChatGPT-Account-ID"] == "acct-from-jwt"
    assert body["input"] == "hello"
    assert body["store"] is False
    assert response.status_code == 200


def test_handle_openai_responses_strips_codex_lite_header_upstream(monkeypatch):
    # OpenAI rejects newer Codex models when the client-only lite header leaks
    # upstream. The HTTP POST path must drop it like the WS handler does, while
    # leaving adjacent headers intact.
    token = _jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-from-jwt",
            }
        }
    )
    request = _build_request(
        {"model": "gpt-5.4", "input": "hello"},
        {
            "Authorization": f"Bearer {token}",
            "X-OpenAI-Internal-Codex-Responses-Lite": "true",
            "X-OpenAI-Debug": "keep-me",
        },
    )
    handler = _DummyOpenAIHandler()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert handler.captured_request is not None
    _method, _url, headers, _body = handler.captured_request
    lowered = {k.lower(): v for k, v in headers.items()}
    assert "x-openai-internal-codex-responses-lite" not in lowered
    assert lowered.get("x-openai-debug") == "keep-me"


def test_handle_openai_responses_chatgpt_auth_skips_memory_tools(monkeypatch):
    token = _jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-from-jwt",
            }
        }
    )
    request = _build_request(
        {"model": "gpt-5.4", "input": "hello", "store": True},
        {"Authorization": f"Bearer {token}", "x-headroom-user-id": "user-1"},
    )
    handler = _DummyOpenAIHandler()
    memory_handler = _MemoryToolsOnlyHandler()
    handler.memory_handler = memory_handler
    handler.session_tracker_store = SimpleNamespace(
        compute_session_id=lambda *a, **k: "sess-chatgpt-no-memory-tools",
    )

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert handler.captured_request is not None
    _, url, _, body = handler.captured_request
    assert url == "https://chatgpt.com/backend-api/codex/responses"
    assert body["store"] is False
    assert "tools" not in body
    assert memory_handler.compute_calls == 0


def test_handle_openai_responses_chatgpt_codex_timeout_fails_open(monkeypatch):
    token = _jwt(
        {
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acct-from-jwt",
            }
        }
    )
    request = _build_request(
        {"model": "gpt-5.4", "input": "large context"},
        {"Authorization": f"Bearer {token}"},
    )
    handler = _DummyOpenAIHandler()
    handler.config.optimize = True

    async def timeout_compression(*args, **kwargs):  # noqa: ANN002, ANN003
        raise asyncio.TimeoutError()

    handler._compress_openai_responses_payload_in_executor = timeout_compression
    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert handler.captured_request is not None
    method, url, headers, body = handler.captured_request
    assert method == "POST"
    assert url == "https://chatgpt.com/backend-api/codex/responses"
    assert body["input"] == "large context"
    assert body["store"] is False


def test_handle_openai_responses_api_auth_store_false_injects_stateless_memory_tools(monkeypatch):
    request = _build_request(
        {"model": "gpt-4o-mini", "input": "hello", "store": False},
        {"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
    )
    handler = _DummyOpenAIHandler()
    memory_handler = _MemoryToolsOnlyHandler()
    handler.memory_handler = memory_handler
    handler.session_tracker_store = SimpleNamespace(
        compute_session_id=lambda *a, **k: "sess-api-memory-tools",
    )

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert handler.captured_request is not None
    _, url, _, body = handler.captured_request
    assert url == "https://api.openai.com/v1/responses"
    assert body["store"] is False
    assert [tool["name"] for tool in body["tools"]] == ["memory_search"]
    assert body["include"] == ["reasoning.encrypted_content"]
    assert memory_handler.compute_calls == 1


@pytest.mark.parametrize("store", [pytest.param(None, id="omitted"), True, False])
@pytest.mark.parametrize(
    "include",
    [
        pytest.param(None, id="omitted"),
        pytest.param(["response.output_text.done"], id="missing-marker"),
        pytest.param(
            ["response.output_text.done", "reasoning.encrypted_content"],
            id="existing-marker",
        ),
        pytest.param("not-a-list", id="non-list"),
    ],
)
def test_openai_responses_memory_continuation_is_zdr_safe(store, include, monkeypatch):
    body = {
        "model": "gpt-5.4",
        "previous_response_id": "resp-inherited",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
            {
                "type": "reasoning",
                "id": "prior-reasoning",
                "summary": [],
                "encrypted_content": "prior-encrypted",
            },
            {
                "type": "reasoning",
                "id": "prior-unencrypted-reasoning",
                "summary": [],
            },
        ],
    }
    if include is not None:
        body["include"] = include
    if store is not None:
        body["store"] = store
    request = _build_request(
        body,
        {"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
    )
    handler = _ZdrResponsesHandler()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert len(handler.requests) == 2
    first_body, continuation_body = handler.requests
    expected_include = (
        ["reasoning.encrypted_content"]
        if include is None
        else (
            include
            if not isinstance(include, list)
            else (
                include
                if "reasoning.encrypted_content" in include
                else [*include, "reasoning.encrypted_content"]
            )
        )
    )
    assert first_body["include"] == expected_include
    assert continuation_body["include"] == expected_include
    assert first_body["previous_response_id"] == "resp-inherited"
    assert ("store" in first_body) is (store is not None)
    if store is not None:
        assert first_body["store"] is store
    else:
        assert "store" not in first_body
    assert continuation_body["input"] == [
        body["input"][0],
        body["input"][1],
        {
            "type": "reasoning",
            "id": "reasoning-1",
            "summary": [],
            "encrypted_content": "encrypted-1",
        },
        {
            "type": "function_call",
            "id": "fc-1",
            "call_id": "call-1",
            "name": "memory_search",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"memories": []}',
        },
    ]
    assert "previous_response_id" not in continuation_body
    assert ("store" in continuation_body) is (store is not None)
    if store is not None:
        assert continuation_body["store"] is store


def test_handle_openai_responses_routes_api_key_auth_direct_to_openai(monkeypatch):
    request = _build_request(
        {"model": "gpt-4o-mini", "input": "hello"},
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert handler.captured_request is not None
    method, url, headers, body = handler.captured_request
    assert method == "POST"
    assert url == "https://api.openai.com/v1/responses"
    assert headers.get("ChatGPT-Account-ID") is None
    assert body["input"] == "hello"
    assert response.status_code == 200


def test_handle_openai_responses_non_stream_adapts_sse_upstream(monkeypatch):
    """A ``stream: false`` request whose upstream replies ``200
    text/event-stream`` must be adapted to the terminal response JSON, not
    converted into a 502 proxy_error (#2613)."""
    import httpx

    sse = (
        b"event: response.completed\n"
        b'data: {"type":"response.completed","response":{"id":"resp_sse_repro",'
        b'"output":[],"usage":{"input_tokens":2,"output_tokens":1}}}\n\n'
    )

    class _SSEUpstreamHandler(_DummyOpenAIHandler):
        async def _retry_request(self, method, url, headers, body, **kwargs):
            self.captured_request = (method, url, headers, body)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse,
            )

    request = _build_request(
        {"model": "gpt-5.4", "stream": False, "input": "hello"},
        {"Authorization": "Bearer sk-test"},
    )
    handler = _SSEUpstreamHandler()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200, response.body
    payload = json.loads(response.body)
    assert payload["id"] == "resp_sse_repro"
    assert response.headers["content-type"].startswith("application/json")


def test_handle_openai_responses_non_stream_passes_through_unparseable_sse(monkeypatch):
    """A 200 SSE upstream body with no recognizable terminal response event
    must be forwarded as-is — never converted into a 502 (#2613)."""
    import httpx

    sse = b"event: response.weird\ndata: not-json\n\n"

    class _SSEUpstreamHandler(_DummyOpenAIHandler):
        async def _retry_request(self, method, url, headers, body, **kwargs):
            self.captured_request = (method, url, headers, body)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse,
            )

    request = _build_request(
        {"model": "gpt-5.4", "stream": False, "input": "hello"},
        {"Authorization": "Bearer sk-test"},
    )
    handler = _SSEUpstreamHandler()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200, response.body
    assert response.body == sse
    assert response.headers["content-type"] == "text/event-stream"


def test_handle_openai_responses_stream_skips_python_compression(monkeypatch):
    """PR-C5: Python no longer compresses /v1/responses (Rust handles it
    natively). The streaming forward path must still fire — only the
    Python compression dispatch is retired."""
    request = _build_request(
        {
            "model": "gpt-5.4",
            "stream": True,
            "instructions": "Keep it short",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                }
            ],
        },
        {"Authorization": "Bearer sk-test"},
    )
    handler = _DummyOpenAIHandler()
    handler.config.optimize = True

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert handler.captured_stream_request is not None
    assert handler.openai_pipeline.apply.call_count == 0
    assert handler.captured_stream_request[2]["stream"] is True


def test_handle_openai_responses_memory_timeout_fails_open(monkeypatch):
    class _SlowMemoryHandler:
        def __init__(self):
            self.config = SimpleNamespace(inject_context=True, inject_tools=False)

        async def search_and_format_context(self, memory_user_id, messages, **_kwargs):
            return "should not be used"

        def has_memory_tool_calls(self, response, provider):
            return False

    async def _timeout_wait_for(awaitable, timeout):
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
        raise TimeoutError

    request = _build_request(
        {"model": "gpt-5.4", "input": "hello"},
        {"Authorization": "Bearer sk-test", "x-headroom-user-id": "user-1"},
    )
    handler = _DummyOpenAIHandler()
    handler.memory_handler = _SlowMemoryHandler()

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr("headroom.proxy.handlers.openai.asyncio.wait_for", _timeout_wait_for)

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert handler.captured_request is not None
    _, _, _, body = handler.captured_request
    assert body.get("instructions") is None


def test_codex_responses_timeout_fails_open_in_standalone_proxy(monkeypatch):
    """Codex users running only the proxy still get fail-open on timeout."""
    request = _build_request(
        {
            "model": "gpt-5.4",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "large tool output",
                }
            ],
        },
        {"Authorization": "Bearer sk-test", "x-client": "codex"},
    )
    handler = _DummyOpenAIHandler()
    handler.config.optimize = True

    monkeypatch.setattr("headroom.tokenizers.get_tokenizer", lambda model: _DummyTokenizer())
    monkeypatch.setattr(
        handler,
        "_compress_openai_responses_payload",
        lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError()),
    )

    response = anyio.run(handler.handle_openai_responses, request)

    assert response.status_code == 200
    assert handler.captured_request is not None
    _, url, _, body = handler.captured_request
    assert url == "https://api.openai.com/v1/responses"
    assert body["input"][0]["output"] == "large tool output"


class _DummyWebSocket:
    def __init__(self, headers: dict[str, str]):
        self.headers = headers
        self.accepted_subprotocol = None
        self.closed = False
        self.close_code = None
        self.close_reason = None

    async def accept(self, subprotocol=None, headers=None):
        self.accepted_subprotocol = subprotocol

    async def close(self, code=1000, reason=None):
        self.closed = True
        self.close_code = code
        self.close_reason = reason


def test_websocket_origin_policy_allows_native_clients_without_origin(monkeypatch):
    monkeypatch.delenv("HEADROOM_WS_ORIGINS", raising=False)
    monkeypatch.delenv("HEADROOM_CORS_ORIGINS", raising=False)

    assert _is_allowed_websocket_origin({"authorization": "Bearer token"}) is True


def test_websocket_origin_policy_allows_loopback_origins_by_default(monkeypatch):
    monkeypatch.delenv("HEADROOM_WS_ORIGINS", raising=False)
    monkeypatch.delenv("HEADROOM_CORS_ORIGINS", raising=False)

    assert _is_allowed_websocket_origin({"origin": "http://localhost:3000"}) is True
    assert _is_allowed_websocket_origin({"origin": "https://127.0.0.1:8787"}) is True


def test_websocket_origin_policy_requires_config_for_remote_origins(monkeypatch):
    monkeypatch.delenv("HEADROOM_WS_ORIGINS", raising=False)
    monkeypatch.delenv("HEADROOM_CORS_ORIGINS", raising=False)

    assert _is_allowed_websocket_origin({"origin": "https://remote.example"}) is False
    assert _is_allowed_websocket_origin({"origin": "http://"}) is False


def test_websocket_origin_policy_can_be_pinned_with_env(monkeypatch):
    monkeypatch.setenv("HEADROOM_WS_ORIGINS", "https://dash.example.com")
    monkeypatch.delenv("HEADROOM_CORS_ORIGINS", raising=False)

    assert _is_allowed_websocket_origin({"origin": "https://dash.example.com"}) is True
    assert _is_allowed_websocket_origin({"origin": "http://localhost:3000"}) is False


def test_handle_openai_responses_ws_resolves_codex_routing_headers():
    class SentinelError(RuntimeError):
        pass

    handler = _DummyOpenAIHandler()
    websocket = _DummyWebSocket({"authorization": "Bearer token"})

    with patch.dict(sys.modules, {"websockets": MagicMock()}):
        with patch(
            "headroom.proxy.handlers.openai._resolve_codex_routing_headers",
            side_effect=SentinelError("resolved"),
        ):
            with pytest.raises(SentinelError, match="resolved"):
                anyio.run(handler.handle_openai_responses_ws, websocket)


def test_handle_openai_responses_ws_closes_unconfigured_origin(monkeypatch):
    handler = _DummyOpenAIHandler()
    websocket = _DummyWebSocket({"origin": "https://remote.example"})

    monkeypatch.delenv("HEADROOM_WS_ORIGINS", raising=False)
    monkeypatch.delenv("HEADROOM_CORS_ORIGINS", raising=False)

    with patch.dict(sys.modules, {"websockets": MagicMock()}):
        with patch(
            "headroom.proxy.handlers.openai._resolve_codex_routing_headers",
            side_effect=AssertionError("routing should not run"),
        ):
            anyio.run(handler.handle_openai_responses_ws, websocket)

    assert websocket.closed is True
    assert websocket.close_code == 1008
    assert websocket.close_reason == "origin not allowed"
    assert websocket.accepted_subprotocol is None
