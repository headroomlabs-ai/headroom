import base64
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anyio
import pytest
from fastapi import Request

from headroom.proxy.handlers.openai import (
    OpenAIHandlerMixin,
    _resolve_codex_routing_headers,
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
        )
        self.usage_reporter = None
        self.openai_provider = SimpleNamespace(get_context_limit=lambda model: 128_000)
        self.openai_pipeline = SimpleNamespace(apply=MagicMock())
        self.anthropic_backend = None
        self.cost_tracker = None
        self.memory_handler = None
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

    async def _retry_request(self, method: str, url: str, headers: dict, body: dict):
        self.captured_request = (method, url, headers, body)
        return _ResponseStub()

    async def _run_compression_in_executor(self, fn, *, timeout: float):
        # Test stub for HeadroomProxy._run_compression_in_executor.
        # The real implementation runs `fn` on a bounded thread pool with
        # a wall-clock timeout; tests just need the callable invoked
        # synchronously so MagicMock call_count assertions fire.
        return fn()

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


def _build_request(body: dict, headers: dict[str, str]) -> Request:
    payload = json.dumps(body).encode("utf-8")

    async def receive():
        return {"type": "http.request", "body": payload, "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "https",
        "path": "/v1/responses",
        "raw_path": b"/v1/responses",
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
    assert response.status_code == 200


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

        async def search_and_format_context(self, memory_user_id, messages):
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


class _DummyWebSocket:
    def __init__(self, headers: dict[str, str]):
        self.headers = headers
        self.accepted_subprotocol = None

    async def accept(self, subprotocol=None):
        self.accepted_subprotocol = subprotocol


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
