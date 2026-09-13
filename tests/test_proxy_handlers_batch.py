from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from headroom.cache.compression_store import (
    CompressionEntry,
    get_compression_store,
    reset_compression_store,
)
from headroom.ccr import response_handler as response_handler_module
from headroom.proxy.handlers import batch as batch_module
from headroom.proxy.handlers import gemini as gemini_module
from headroom.proxy.handlers.gemini import GeminiHandlerMixin


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        content: bytes = b"{}",
        headers: dict[str, str] | None = None,
        text: str | None = None,
        json_data=None,  # noqa: ANN001
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self.text = text if text is not None else content.decode("utf-8", errors="ignore")
        self._json_data = json_data

    def json(self):  # noqa: ANN201
        if self._json_data is not None:
            return self._json_data
        return json.loads(self.text)


class FakeHttpClient:
    def __init__(self) -> None:
        self.posts: list[dict[str, object]] = []
        self.gets: list[dict[str, object]] = []
        self.requests: list[dict[str, object]] = []
        self.post_response = FakeResponse()
        self.get_response = FakeResponse()
        self.raise_post: Exception | None = None
        self.raise_get: Exception | None = None

    async def post(self, url: str, **kwargs):  # noqa: ANN003, ANN201
        self.posts.append({"url": url, **kwargs})
        if self.raise_post is not None:
            raise self.raise_post
        return self.post_response

    async def get(self, url: str, **kwargs):  # noqa: ANN003, ANN201
        self.gets.append({"url": url, **kwargs})
        if self.raise_get is not None:
            raise self.raise_get
        return self.get_response

    async def request(self, method: str, url: str, **kwargs):  # noqa: ANN003, ANN201
        self.requests.append({"method": method, "url": url, **kwargs})
        if self.raise_get is not None:
            raise self.raise_get
        return self.get_response


class FakeMetrics:
    def __init__(self) -> None:
        self.record_calls: list[dict[str, object]] = []
        self.failed_calls: list[dict[str, object]] = []

    async def record_request(self, **kwargs) -> None:  # noqa: ANN003
        self.record_calls.append(kwargs)

    async def record_failed(self, **kwargs) -> None:  # noqa: ANN003
        self.failed_calls.append(kwargs)


class DummyBatchHandler(batch_module.BatchHandlerMixin, GeminiHandlerMixin):
    # GeminiHandlerMixin supplies the real _rebuild_gemini_contents (and the
    # other content helpers); the two converter methods below intentionally
    # override the mixin's for the stub-based tests.
    OPENAI_API_URL = "https://openai.example"
    GEMINI_API_URL = "https://gemini.example"

    def __init__(self) -> None:
        self.http_client = FakeHttpClient()
        self.metrics = FakeMetrics()
        self.config = SimpleNamespace(
            optimize=False,
            ccr_inject_tool=False,
            ccr_inject_system_instructions=False,
        )
        self.openai_provider = SimpleNamespace(get_context_limit=lambda model: 8192)
        self.openai_pipeline = SimpleNamespace(apply=lambda **kwargs: None)
        # Mirror of HeadroomProxy.usage_reporter (server.py), which is always
        # set, None when no licensing system is configured.
        self.usage_reporter = None
        self._request_counter = 0
        self._retry_response = FakeResponse()

    async def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"req-{self._request_counter}"

    async def _record_request_outcome(self, outcome) -> None:  # noqa: ANN001
        # Mirror of HeadroomProxy._record_request_outcome for the batch
        # mixin tests. Delegates to the free funnel so the wire shape
        # matches production.
        from headroom.proxy.outcome import emit_request_outcome

        await emit_request_outcome(self, outcome)

    def _extract_tags(self, headers: dict) -> dict[str, str]:
        # Mirror of HeadroomProxy._extract_tags. Handlers now call this
        # at entry to capture x-headroom-* slicing tags into the outcome.
        return {
            k.lower().replace("x-headroom-", ""): v
            for k, v in headers.items()
            if k.lower().startswith("x-headroom-")
        }

    async def handle_passthrough(self, request, base_url):  # noqa: ANN001, ANN201
        return {"request": request, "base_url": base_url}

    async def _run_compression_in_executor(self, fn, *, timeout):  # noqa: ANN001, ANN201
        # Mirror of HeadroomProxy._run_compression_in_executor: batch handlers
        # offload pipeline.apply() off the event loop (#1701). Inline is fine
        # for tests — only the call contract matters here.
        return fn()

    async def _retry_request(self, method, url, headers, body, **kwargs):  # noqa: ANN001, ANN201
        return self._retry_response

    def _gemini_contents_to_messages(self, contents, system_instruction):  # noqa: ANN001, ANN201
        messages = [{"role": "user", "content": part["parts"][0]["text"]} for part in contents]
        return messages, []

    def _messages_to_gemini_contents(self, messages):  # noqa: ANN001, ANN201
        return ([{"parts": [{"text": message["content"]}]} for message in messages], None)


class FakeRequest:
    def __init__(
        self,
        body: bytes | str,
        *,
        headers: dict[str, str] | None = None,
        method: str = "POST",
        path: str = "/v1/batches",
        query: str = "",
    ) -> None:
        self._body = body.encode("utf-8") if isinstance(body, str) else body
        self.headers = headers or {}
        self.method = method
        self.url = SimpleNamespace(path=path, query=query)
        self.query_params = {}
        # Every real Starlette Request has one, and handlers now share a
        # per-request attribution ledger through it (savings_attribution).
        self.scope: dict = {"type": "http", "method": method}

    async def body(self) -> bytes:
        return self._body


class NativeGeminiHandler(DummyBatchHandler):
    def __init__(self, responses: list[FakeResponse]) -> None:
        super().__init__()
        self.config.optimize = True
        self.config.ccr_inject_tool = True
        self.config.ccr_inject_system_instructions = False
        self.memory_handler = None
        self.rate_limiter = None
        self.usage_reporter = None
        self.responses = iter(responses)
        self.sent_bodies: list[dict] = []
        from headroom.ccr.response_handler import CCRResponseHandler

        self.ccr_response_handler = CCRResponseHandler()
        self.openai_pipeline = SimpleNamespace(
            apply=lambda **kwargs: SimpleNamespace(
                messages=[
                    {
                        "role": "user",
                        "content": "compressed [100 items compressed to 1. Retrieve more: hash=aaaaaaaaaaaaaaaaaaaaaaaa]",
                    }
                ],
                timing={},
                tokens_before=10,
                tokens_after=5,
                transforms_applied=[],
                waste_signals=SimpleNamespace(to_dict=lambda: {}),
            )
        )

    def _gemini_contents_to_messages(
        self, contents, system_instruction=None, *, include_function_responses=False
    ):  # noqa: ANN001, ANN201
        return GeminiHandlerMixin._gemini_contents_to_messages(
            self,
            contents,
            system_instruction,
            include_function_responses=include_function_responses,
        )

    def _messages_to_gemini_contents(self, messages):  # noqa: ANN001, ANN201
        return GeminiHandlerMixin._messages_to_gemini_contents(self, messages)

    async def _retry_request(self, method, url, headers, body, **kwargs):  # noqa: ANN001, ANN201
        self.sent_bodies.append(body)
        return next(self.responses)

    async def _run_compression_in_executor(self, fn, *, timeout):  # noqa: ANN001, ANN201
        return fn()


def install_native_gemini_compression(monkeypatch: pytest.MonkeyPatch) -> None:
    class Decision:
        should_compress = True
        passthrough_reason = ""

        def apply_to_tags(self, tags) -> None:  # noqa: ANN001
            return None

    monkeypatch.setattr(gemini_module.CompressionDecision, "decide", lambda **kwargs: Decision())


def native_gemini_request(tools=None) -> dict:  # noqa: ANN001
    return {
        "contents": [{"role": "user", "parts": [{"text": "compressed input"}]}],
        "generationConfig": {"temperature": 0.2},
        **({"tools": tools} if tools is not None else {}),
    }


def native_ccr_response() -> FakeResponse:
    return FakeResponse(
        json_data={
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "headroom_retrieve",
                                    "id": "call-1",
                                    "args": {"hash": "aaaaaaaaaaaaaaaaaaaaaaaa"},
                                }
                            }
                        ],
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 5},
        }
    )


@pytest.mark.asyncio
async def test_gemini_native_ccr_continuation(monkeypatch: pytest.MonkeyPatch) -> None:
    install_native_gemini_compression(monkeypatch)
    from headroom.ccr.response_handler import CCRToolResult

    final = FakeResponse(
        json_data={
            "candidates": [{"content": {"role": "model", "parts": [{"text": "final answer"}]}}]
        }
    )
    handler = NativeGeminiHandler([native_ccr_response(), final])
    handler.ccr_response_handler._execute_retrieval = lambda call: CCRToolResult(
        call.tool_call_id,
        json.dumps({"hash": call.hash_key, "original_content": [{"type": "code"}]}),
        True,
        1,
        "headroom_retrieve",
    )

    response = await handler.handle_gemini_generate_content(
        FakeRequest(
            json.dumps(native_gemini_request()),
            headers={"content-type": "application/json", "x-goog-api-key": "secret"},
            path="/v1beta/models/gemini-2.5-flash:generateContent",
        ),
        "gemini-2.5-flash",
    )

    assert response.status_code == 200
    assert (
        json.loads(response.body)["candidates"][0]["content"]["parts"][0]["text"] == "final answer"
    ), response.body
    assert len(handler.sent_bodies) == 2
    continuation = handler.sent_bodies[1]["contents"]
    assert continuation[-2]["role"] == "model"
    assert continuation[-2]["parts"][0]["functionCall"]["name"] == "headroom_retrieve"
    assert continuation[-1]["role"] == "user"
    assert continuation[-1]["parts"][0]["functionResponse"]["name"] == "headroom_retrieve"
    assert continuation[-1]["parts"][0]["functionResponse"]["id"] == "call-1"


@pytest.mark.asyncio
async def test_gemini_native_ccr_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    install_native_gemini_compression(monkeypatch)
    # verify_ownership() (issue #2836) requires the marker's hash to be a
    # real store entry; NativeGeminiHandler's mocked pipeline hand-types
    # "hash=aaaa...aaaa" rather than compressing through the real store.
    reset_compression_store()
    get_compression_store().store(
        original="original content",
        compressed="compressed [100 items compressed to 1]",
        explicit_hash="aaaaaaaaaaaaaaaaaaaaaaaa",
    )
    handler = NativeGeminiHandler(
        [FakeResponse(json_data={"candidates": [{"content": {"parts": [{"text": "answer"}]}}]})]
    )
    tools = [
        {"functionDeclarations": [{"name": "client_tool"}]},
        {"functionDeclarations": [{"name": "second_tool"}]},
        {"googleSearch": {}},
        {"codeExecution": {}},
    ]

    await handler.handle_gemini_generate_content(
        FakeRequest(
            json.dumps(native_gemini_request(tools)),
            headers={"content-type": "application/json"},
            path="/v1beta/models/gemini-2.5-flash:generateContent",
        ),
        "gemini-2.5-flash",
    )

    forwarded_tools = handler.sent_bodies[0]["tools"]
    assert forwarded_tools[2:] == tools[2:]
    declarations = forwarded_tools[0]["functionDeclarations"]
    assert {item["name"] for item in declarations} == {"client_tool", "headroom_retrieve"}
    assert forwarded_tools[1]["functionDeclarations"] == [{"name": "second_tool"}]
    reset_compression_store()


@pytest.mark.asyncio
async def test_gemini_native_ccr_does_not_duplicate_existing_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_native_gemini_compression(monkeypatch)
    tools = [
        {"functionDeclarations": [{"name": "client_tool"}]},
        {"functionDeclarations": [{"name": "headroom_retrieve"}]},
    ]
    handler = NativeGeminiHandler(
        [FakeResponse(json_data={"candidates": [{"content": {"parts": [{"text": "answer"}]}}]})]
    )

    await handler.handle_gemini_generate_content(
        FakeRequest(
            json.dumps(native_gemini_request(tools)),
            headers={"content-type": "application/json"},
            path="/v1beta/models/gemini-2.5-flash:generateContent",
        ),
        "gemini-2.5-flash",
    )

    names = [
        declaration["name"]
        for tool in handler.sent_bodies[0]["tools"]
        for declaration in tool.get("functionDeclarations", [])
    ]
    assert names.count("headroom_retrieve") == 1


@pytest.mark.asyncio
async def test_gemini_native_ccr_does_not_inject_into_streaming_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_native_gemini_compression(monkeypatch)
    handler = NativeGeminiHandler([FakeResponse()])
    captured: dict[str, object] = {}

    async def fake_stream(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        captured["body"] = args[2]
        return FakeResponse()

    monkeypatch.setattr(handler, "_stream_response", fake_stream, raising=False)
    tools = [{"functionDeclarations": [{"name": "client_tool"}]}]
    await handler.handle_gemini_generate_content(
        FakeRequest(
            json.dumps(native_gemini_request(tools)),
            headers={"content-type": "application/json"},
            path="/v1beta/models/gemini-2.5-flash:streamGenerateContent",
        ),
        "gemini-2.5-flash",
    )

    streamed_tools = captured["body"]["tools"]  # type: ignore[index]
    names = [
        declaration["name"]
        for tool in streamed_tools
        for declaration in tool.get("functionDeclarations", [])
    ]
    assert names == ["client_tool"]


@pytest.mark.asyncio
async def test_gemini_native_ccr_mixed(monkeypatch: pytest.MonkeyPatch) -> None:
    install_native_gemini_compression(monkeypatch)
    response_json = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "headroom_retrieve",
                                "args": {"hash": "aaaaaaaaaaaaaaaaaaaaaaaa"},
                            }
                        },
                        {"functionCall": {"name": "client_tool", "args": {}}},
                    ]
                }
            }
        ]
    }
    handler = NativeGeminiHandler([FakeResponse(json_data=response_json)])

    response = await handler.handle_gemini_generate_content(
        FakeRequest(
            json.dumps(native_gemini_request()),
            headers={"content-type": "application/json"},
            path="/v1beta/models/gemini-2.5-flash:generateContent",
        ),
        "gemini-2.5-flash",
    )

    assert response.status_code == 200
    assert len(handler.sent_bodies) == 1
    assert json.loads(response.body) == response_json


@pytest.mark.asyncio
async def test_gemini_native_ccr_non_ccr_function_call_is_not_intercepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_native_gemini_compression(monkeypatch)
    response_json = {
        "candidates": [
            {"content": {"parts": [{"functionCall": {"name": "client_tool", "args": {}}}]}}
        ]
    }
    handler = NativeGeminiHandler([FakeResponse(json_data=response_json)])

    response = await handler.handle_gemini_generate_content(
        FakeRequest(
            json.dumps(native_gemini_request()),
            headers={"content-type": "application/json"},
            path="/v1beta/models/gemini-2.5-flash:generateContent",
        ),
        "gemini-2.5-flash",
    )

    assert response.status_code == 200
    assert len(handler.sent_bodies) == 1
    assert response.body == b"{}"


@pytest.mark.asyncio
async def test_gemini_native_ccr_continuation_error_preserves_upstream_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_native_gemini_compression(monkeypatch)
    handler = NativeGeminiHandler(
        [
            native_ccr_response(),
            FakeResponse(status_code=503, content=b"busy", headers={"retry-after": "2"}),
        ]
    )

    response = await handler.handle_gemini_generate_content(
        FakeRequest(
            json.dumps(native_gemini_request()),
            headers={"content-type": "application/json"},
            path="/v1beta/models/gemini-2.5-flash:generateContent",
        ),
        "gemini-2.5-flash",
    )

    assert response.status_code == 503
    assert response.body == b"busy"
    assert response.headers["retry-after"] == "2"


@pytest.mark.asyncio
async def test_gemini_native_ccr_continuation_non_json_preserves_upstream_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_native_gemini_compression(monkeypatch)
    handler = NativeGeminiHandler(
        [native_ccr_response(), FakeResponse(status_code=200, content=b"upstream")]
    )

    response = await handler.handle_gemini_generate_content(
        FakeRequest(
            json.dumps(native_gemini_request()),
            headers={"content-type": "application/json"},
            path="/v1beta/models/gemini-2.5-flash:generateContent",
        ),
        "gemini-2.5-flash",
    )

    assert response.status_code == 200
    assert response.body == b"upstream"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "original_content",
    [[{"type": "code", "text": "print('x')"}], "plain text", {"key": "value"}, 42],
    ids=["code-aware-array", "kompress-text", "mcp-object", "mcp-scalar"],
)
async def test_gemini_native_ccr_uses_real_retrieval_result_shape(
    monkeypatch: pytest.MonkeyPatch, original_content
) -> None:  # noqa: ANN001
    install_native_gemini_compression(monkeypatch)
    entry = CompressionEntry(
        hash="a" * 24,
        original_content=json.dumps(original_content),
        compressed_content="compressed",
        original_tokens=10,
        compressed_tokens=2,
        original_item_count=1,
        compressed_item_count=1,
        tool_name="headroom_retrieve",
        tool_call_id="headroom_retrieve",
        query_context=None,
        created_at=0,
    )

    class Store:
        def get_entry_status(self, hash_key, clean_expired=True):  # noqa: ANN001, ARG002
            return {"status": "available", "default_ttl_seconds": 1800}

        def retrieve(self, hash_key):  # noqa: ANN001, ARG002
            return entry

    monkeypatch.setattr(response_handler_module, "get_compression_store", lambda: Store())
    handler = NativeGeminiHandler(
        [
            native_ccr_response(),
            FakeResponse(json_data={"candidates": [{"content": {"parts": [{"text": "done"}]}}]}),
        ]
    )

    response = await handler.handle_gemini_generate_content(
        FakeRequest(
            json.dumps(native_gemini_request()),
            headers={"content-type": "application/json"},
            path="/v1beta/models/gemini-2.5-flash:generateContent",
        ),
        "gemini-2.5-flash",
    )

    assert response.status_code == 200
    function_response = handler.sent_bodies[1]["contents"][-1]["parts"][0]["functionResponse"]
    assert function_response["response"]["original_content"] == json.dumps(original_content)


@pytest.mark.asyncio
async def test_gemini_native_ccr_preserves_non_ccr_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_native_gemini_compression(monkeypatch)
    handler = NativeGeminiHandler([FakeResponse(status_code=503, content=b"busy")])

    response = await handler.handle_gemini_generate_content(
        FakeRequest(
            json.dumps(native_gemini_request()),
            headers={"content-type": "application/json"},
            path="/v1beta/models/gemini-2.5-flash:generateContent",
        ),
        "gemini-2.5-flash",
    )

    assert response.status_code == 503
    assert response.body == b"busy"


@pytest.mark.asyncio
async def test_gemini_native_ccr_residual(monkeypatch: pytest.MonkeyPatch) -> None:
    install_native_gemini_compression(monkeypatch)
    from headroom.ccr.response_handler import CCRToolResult

    handler = NativeGeminiHandler([native_ccr_response()] * 4)
    handler.ccr_response_handler._execute_retrieval = lambda call: CCRToolResult(
        "headroom_retrieve", "still unresolved", True, 0
    )

    response = await handler.handle_gemini_generate_content(
        FakeRequest(
            json.dumps(native_gemini_request()),
            headers={"content-type": "application/json"},
            path="/v1beta/models/gemini-2.5-flash:generateContent",
        ),
        "gemini-2.5-flash",
    )

    assert response.status_code == 502


def install_batch_support_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    injector_result=None,  # noqa: ANN001
    tokenizer_count: int = 10,
) -> None:
    class FakeInjector:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.kwargs = kwargs

        def process_request(self, messages, tools):  # noqa: ANN001, ANN201
            if injector_result is not None:
                return injector_result
            return messages, tools, False

    class FakeTokenizer:
        def count_messages(self, messages) -> int:  # noqa: ANN001
            return tokenizer_count

    monkeypatch.setitem(sys.modules, "headroom.ccr", SimpleNamespace(CCRToolInjector=FakeInjector))
    monkeypatch.setitem(
        sys.modules,
        "headroom.tokenizers",
        SimpleNamespace(get_tokenizer=lambda model: FakeTokenizer()),
    )
    monkeypatch.setitem(
        sys.modules,
        "headroom.utils",
        SimpleNamespace(extract_user_query=lambda messages: "query"),
    )


@pytest.mark.asyncio
async def test_compress_batch_jsonl_without_optimization_handles_invalid_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_batch_support_modules(monkeypatch, tokenizer_count=12)
    handler = DummyBatchHandler()
    content = "\n".join(
        [
            json.dumps(
                {"body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}}
            ),
            json.dumps({"body": {"model": "gpt-4o", "messages": []}}),
            "not-json",
        ]
    )

    lines, stats = await handler._compress_batch_jsonl(content, "req-1")

    assert len(lines) == 3
    assert json.loads(lines[0])["body"]["messages"][0]["content"] == "hi"
    assert lines[2] == "not-json"
    assert stats == {
        "total_requests": 3,
        "total_original_tokens": 12,
        "total_compressed_tokens": 12,
        "total_tokens_saved": 0,
        "savings_percent": 0.0,
        "errors": 1,
        # No reporter configured, so the license half fails open — no reason to tag.
        "passthrough_reason": None,
    }


@pytest.mark.asyncio
async def test_compress_batch_jsonl_handles_non_object_lines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A JSONL line that is valid JSON but not a request object (array/string/
    # null), or a request whose `body` isn't a dict, must pass through instead
    # of crashing the whole batch (`.get` on a non-dict raises AttributeError,
    # which the JSONDecodeError guard does not catch).
    install_batch_support_modules(monkeypatch, tokenizer_count=12)
    handler = DummyBatchHandler()
    content = "\n".join(
        [
            json.dumps(
                {"body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}}
            ),
            json.dumps([1, 2, 3]),
            json.dumps("hello"),
            "null",
            json.dumps({"body": "not-a-dict"}),
        ]
    )

    lines, stats = await handler._compress_batch_jsonl(content, "req-1")

    assert len(lines) == 5
    assert json.loads(lines[1]) == [1, 2, 3]
    assert json.loads(lines[2]) == "hello"
    assert json.loads(lines[3]) is None
    assert json.loads(lines[4]) == {"body": "not-a-dict"}
    assert stats["total_requests"] == 5
    # None of these are JSON decode errors, so the error counter stays at 0.
    assert stats["errors"] == 0


@pytest.mark.asyncio
async def test_compress_batch_jsonl_uses_pipeline_and_ccr_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_batch_support_modules(
        monkeypatch,
        injector_result=(
            [{"role": "system", "content": "compressed"}],
            [{"name": "retrieval"}],
            True,
        ),
    )
    handler = DummyBatchHandler()
    handler.config.optimize = True
    handler.config.ccr_inject_tool = True
    handler.openai_pipeline = SimpleNamespace(
        apply=lambda **kwargs: SimpleNamespace(
            messages=[{"role": "assistant", "content": "short"}],
            tokens_before=100,
            tokens_after=40,
        )
    )

    lines, stats = await handler._compress_batch_jsonl(
        json.dumps(
            {
                "body": {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [{"name": "existing"}],
                }
            }
        ),
        "req-2",
    )

    body = json.loads(lines[0])["body"]
    assert body["messages"] == [{"role": "system", "content": "compressed"}]
    assert body["tools"] == [{"name": "retrieval"}]
    assert stats["total_tokens_saved"] == 60
    assert stats["savings_percent"] == 60.0


@pytest.mark.asyncio
async def test_compress_batch_jsonl_falls_back_when_pipeline_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_batch_support_modules(monkeypatch, tokenizer_count=33)
    handler = DummyBatchHandler()
    handler.config.optimize = True
    handler.openai_pipeline = SimpleNamespace(
        apply=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    lines, stats = await handler._compress_batch_jsonl(
        json.dumps({"body": {"messages": [{"role": "user", "content": "hello"}]}}),
        "req-3",
    )

    assert json.loads(lines[0])["body"]["messages"][0]["content"] == "hello"
    assert stats["total_original_tokens"] == 33
    assert stats["total_compressed_tokens"] == 33


@pytest.mark.asyncio
async def test_batch_passthrough_forwards_request_and_strips_response_headers() -> None:
    handler = DummyBatchHandler()
    handler.http_client.post_response = FakeResponse(
        content=b'{"ok":true}',
        headers={"content-encoding": "gzip", "content-length": "20", "x-kept": "1"},
    )

    response = await handler._batch_passthrough(
        FakeRequest(
            '{"input_file_id":"file-1"}', headers={"host": "example", "content-length": "10"}
        ),
        {"input_file_id": "file-1"},
    )

    assert response.status_code == 200
    assert dict(response.headers)["x-kept"] == "1"
    assert "content-encoding" not in dict(response.headers)
    assert handler.http_client.posts[0]["url"] == "https://openai.example/v1/batches"


@pytest.mark.asyncio
async def test_handle_batch_create_validates_json_and_required_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = DummyBatchHandler()

    async def raise_bad_json(request):  # noqa: ANN001
        raise ValueError("bad json")

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", raise_bad_json)

    bad = await handler.handle_batch_create(FakeRequest("{}"))
    assert bad.status_code == 400
    assert bad.body.decode().find("invalid_json") > 0

    async def missing_file_payload(request):  # noqa: ANN001
        return {"endpoint": "/v1/chat/completions"}

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", missing_file_payload)
    missing_file = await handler.handle_batch_create(FakeRequest("{}"))
    assert missing_file.status_code == 400
    assert missing_file.body.decode().find("input_file_id is required") > 0

    async def missing_endpoint_payload(request):  # noqa: ANN001
        return {"input_file_id": "file-1"}

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", missing_endpoint_payload)
    missing_endpoint = await handler.handle_batch_create(FakeRequest("{}"))
    assert missing_endpoint.status_code == 400
    assert missing_endpoint.body.decode().find("endpoint is required") > 0


@pytest.mark.asyncio
async def test_handle_batch_create_passthrough_and_download_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = DummyBatchHandler()
    passthrough_response = SimpleNamespace(marker="passthrough")

    async def fake_passthrough(request, body):  # noqa: ANN001
        return passthrough_response

    monkeypatch.setattr(handler, "_batch_passthrough", fake_passthrough)

    async def passthrough_payload(request):  # noqa: ANN001
        return {"input_file_id": "file-1", "endpoint": "/v1/responses"}

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", passthrough_payload)
    assert await handler.handle_batch_create(FakeRequest("{}")) is passthrough_response

    async def download_missing_payload(request):  # noqa: ANN001
        return {"input_file_id": "file-1", "endpoint": "/v1/chat/completions"}

    async def missing_download(file_id, headers):  # noqa: ANN001
        return None

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", download_missing_payload)
    monkeypatch.setattr(handler, "_download_openai_file", missing_download)
    missing = await handler.handle_batch_create(FakeRequest("{}"))
    assert missing.status_code == 404
    assert missing.body.decode().find("file_not_found") > 0


@pytest.mark.asyncio
async def test_handle_batch_create_handles_empty_upload_failure_and_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = DummyBatchHandler()

    async def request_payload(request):  # noqa: ANN001
        return {
            "input_file_id": "file-1",
            "endpoint": "/v1/chat/completions",
            "completion_window": "12h",
            "metadata": {"source": "test"},
        }

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", request_payload)

    async def fake_download(file_id, headers):  # noqa: ANN001
        return "downloaded"

    monkeypatch.setattr(handler, "_download_openai_file", fake_download)

    async def empty_compress(content, request_id):  # noqa: ANN001
        return [], {
            "total_requests": 0,
            "total_original_tokens": 0,
            "total_compressed_tokens": 0,
            "total_tokens_saved": 0,
            "savings_percent": 0.0,
            "errors": 0,
        }

    monkeypatch.setattr(handler, "_compress_batch_jsonl", empty_compress)
    empty = await handler.handle_batch_create(FakeRequest("{}"))
    assert empty.status_code == 400
    assert empty.body.decode().find("empty_file") > 0

    async def compressed(content, request_id):  # noqa: ANN001
        return ['{"body":{}}'], {
            "total_requests": 1,
            "total_original_tokens": 20,
            "total_compressed_tokens": 10,
            "total_tokens_saved": 10,
            "savings_percent": 50.0,
            "errors": 0,
        }

    monkeypatch.setattr(handler, "_compress_batch_jsonl", compressed)

    async def upload_failed_file(content, filename, headers):  # noqa: ANN001
        return None

    monkeypatch.setattr(handler, "_upload_openai_file", upload_failed_file)
    upload_failed = await handler.handle_batch_create(FakeRequest("{}"))
    assert upload_failed.status_code == 500
    assert upload_failed.body.decode().find("upload_failed") > 0

    handler.http_client.post_response = FakeResponse(
        content=b'{"id":"batch_123","object":"batch"}',
        headers={"content-encoding": "gzip", "content-length": "12", "x-openai": "1"},
    )

    async def upload_success(content, filename, headers):  # noqa: ANN001
        return "file-compressed"

    monkeypatch.setattr(handler, "_upload_openai_file", upload_success)
    success = await handler.handle_batch_create(
        FakeRequest(
            "{}", headers={"host": "proxy", "content-length": "4", "authorization": "Bearer test"}
        )
    )

    assert success.status_code == 200
    success_headers = dict(success.headers)
    assert success_headers["x-headroom-tokens-saved"] == "10"
    assert success_headers["x-headroom-savings-percent"] == "50.0"
    assert success_headers["x-openai"] == "1"
    # PR-A3: byte-faithful forwarder writes ``content`` (raw bytes), not
    # ``json``. Round-trip the captured bytes back to a dict for assertion.
    last_post = handler.http_client.posts[-1]
    if "json" in last_post:
        sent_body = last_post["json"]
    else:
        sent_body = json.loads(last_post["content"].decode("utf-8"))
    assert sent_body["metadata"]["headroom_compressed"] == "true"
    assert sent_body["metadata"]["headroom_original_file_id"] == "file-1"
    assert handler.metrics.record_calls[-1]["provider"] == "openai"


@pytest.mark.asyncio
async def test_handle_batch_create_records_failure_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handler = DummyBatchHandler()

    async def request_payload(request):  # noqa: ANN001
        return {"input_file_id": "file-1", "endpoint": "/v1/chat/completions"}

    async def boom(file_id, headers):  # noqa: ANN001
        raise RuntimeError("boom")

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", request_payload)
    monkeypatch.setattr(handler, "_download_openai_file", boom)

    response = await handler.handle_batch_create(FakeRequest("{}"))

    assert response.status_code == 500
    assert handler.metrics.failed_calls == [{"provider": "batch"}]


@pytest.mark.asyncio
async def test_download_and_upload_openai_file_helpers() -> None:
    handler = DummyBatchHandler()
    handler.http_client.get_response = FakeResponse(status_code=200, text="jsonl-content")
    downloaded = await handler._download_openai_file("file-1", {"authorization": "Bearer token"})
    assert downloaded == "jsonl-content"
    assert handler.http_client.gets[0]["url"] == "https://openai.example/v1/files/file-1/content"

    handler.http_client.get_response = FakeResponse(status_code=404, text="missing")
    assert await handler._download_openai_file("file-2", {}) is None

    handler.http_client.post_response = FakeResponse(
        status_code=200,
        json_data={"id": "file-uploaded"},
        headers={"content-type": "application/json"},
    )
    file_id = await handler._upload_openai_file(
        '{"body":{}}',
        "compressed.jsonl",
        {"authorization": "Bearer token", "content-type": "application/json"},
    )
    assert file_id == "file-uploaded"
    post_call = handler.http_client.posts[-1]
    assert post_call["headers"] == {"authorization": "Bearer token"}
    assert post_call["files"]["file"][0] == "compressed.jsonl"

    handler.http_client.post_response = FakeResponse(status_code=500, text="fail")
    assert await handler._upload_openai_file("{}", "bad.jsonl", {}) is None
    handler.http_client.raise_post = RuntimeError("network")
    assert await handler._upload_openai_file("{}", "bad.jsonl", {}) is None


@pytest.mark.asyncio
async def test_store_google_batch_context_persists_transformed_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_contexts: list[object] = []

    class FakeBatchContext:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.kwargs = kwargs
            self.requests: list[object] = []

        def add_request(self, request) -> None:  # noqa: ANN001
            self.requests.append(request)

    class FakeBatchRequestContext:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.kwargs = kwargs

    class FakeStore:
        async def store(self, context) -> None:  # noqa: ANN001
            stored_contexts.append(context)

    monkeypatch.setitem(
        sys.modules,
        "headroom.ccr",
        SimpleNamespace(
            BatchContext=FakeBatchContext,
            BatchRequestContext=FakeBatchRequestContext,
            get_batch_context_store=lambda: FakeStore(),
        ),
    )

    handler = DummyBatchHandler()
    await handler._store_google_batch_context(
        "batches/123",
        [
            {
                "metadata": {"key": "req-1"},
                "request": {
                    "contents": [{"parts": [{"text": "hello"}]}],
                    "systemInstruction": {"parts": [{"text": "system"}]},
                    "tools": [{"name": "tool"}],
                },
            }
        ],
        "gemini-2.0",
        "api-key",
    )

    context = stored_contexts[0]
    assert context.kwargs["batch_id"] == "batches/123"
    assert context.requests[0].kwargs["custom_id"] == "req-1"
    assert context.requests[0].kwargs["messages"] == [{"role": "user", "content": "hello"}]
    assert context.requests[0].kwargs["system_instruction"] == "system"


@pytest.mark.asyncio
async def test_handle_google_batch_results_passes_through_early_exit_cases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStore:
        async def get(self, batch_name):  # noqa: ANN001
            return None

    monkeypatch.setitem(
        sys.modules,
        "headroom.ccr",
        SimpleNamespace(
            BatchResultProcessor=lambda http_client: None,
            get_batch_context_store=lambda: FakeStore(),
        ),
    )

    handler = DummyBatchHandler()
    request = FakeRequest(
        "{}", headers={"x-goog-api-key": "secret"}, method="GET", path="/v1beta/batches/b1"
    )

    handler.http_client.get_response = FakeResponse(
        status_code=500, content=b"bad", headers={"x-upstream": "1"}
    )
    error_response = await handler.handle_google_batch_results(request, "batches/b1")
    assert error_response.status_code == 500
    assert dict(error_response.headers)["x-upstream"] == "1"

    class BadJsonResponse(FakeResponse):
        def json(self):  # noqa: ANN201
            raise json.JSONDecodeError("bad", "x", 0)

    handler.http_client.get_response = BadJsonResponse(
        status_code=200, content=b"plain", headers={"x-upstream": "2"}
    )
    non_json = await handler.handle_google_batch_results(request, "batches/b1")
    assert non_json.status_code == 200
    assert dict(non_json.headers)["x-upstream"] == "2"

    handler.http_client.get_response = FakeResponse(
        status_code=200,
        content=b"{}",
        json_data={"metadata": {"state": "RUNNING"}},
    )
    running = await handler.handle_google_batch_results(request, "batches/b1")
    assert running.status_code == 200

    handler.http_client.get_response = FakeResponse(
        status_code=200,
        content=b"{}",
        json_data={"metadata": {"state": "SUCCEEDED"}, "response": {"responses": []}},
    )
    no_results = await handler.handle_google_batch_results(request, "batches/b1")
    assert no_results.status_code == 200

    handler.http_client.get_response = FakeResponse(
        status_code=200,
        content=b"{}",
        json_data={"metadata": {"state": "SUCCEEDED"}, "response": {"responses": [{"id": 1}]}},
    )
    handler.config.ccr_inject_tool = False
    no_ccr = await handler.handle_google_batch_results(request, "batches/b1")
    assert no_ccr.status_code == 200
    assert "key=secret" in handler.http_client.gets[-1]["url"]


@pytest.mark.asyncio
async def test_handle_google_batch_results_processes_completed_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processed_calls: list[tuple[str, list[object], str]] = []

    class FakeProcessed:
        def __init__(
            self, result, custom_id: str, was_processed: bool, continuation_rounds: int
        ) -> None:  # noqa: ANN001
            self.result = result
            self.custom_id = custom_id
            self.was_processed = was_processed
            self.continuation_rounds = continuation_rounds

    class FakeProcessor:
        def __init__(self, http_client) -> None:  # noqa: ANN001
            self.http_client = http_client

        async def process_results(self, batch_name, results, provider):  # noqa: ANN001
            processed_calls.append((batch_name, results, provider))
            return [
                FakeProcessed({"id": "processed"}, "req-1", True, 2),
                FakeProcessed({"id": "unchanged"}, "req-2", False, 0),
            ]

    class FakeStore:
        async def get(self, batch_name):  # noqa: ANN001
            return SimpleNamespace(batch_name=batch_name)

    monkeypatch.setitem(
        sys.modules,
        "headroom.ccr",
        SimpleNamespace(
            BatchResultProcessor=FakeProcessor,
            get_batch_context_store=lambda: FakeStore(),
        ),
    )

    handler = DummyBatchHandler()
    handler.config.ccr_inject_tool = True
    handler.http_client.get_response = FakeResponse(
        status_code=200,
        content=b"{}",
        json_data={
            "metadata": {"state": "SUCCEEDED"},
            "response": {"responses": [{"id": "raw-1"}, {"id": "raw-2"}]},
        },
    )

    response = await handler.handle_google_batch_results(
        FakeRequest("{}", method="GET", path="/v1beta/batches/b1"),
        "batches/b1",
    )

    payload = json.loads(response.body)
    assert payload["response"]["responses"] == [{"id": "processed"}, {"id": "unchanged"}]
    assert processed_calls == [("batches/b1", [{"id": "raw-1"}, {"id": "raw-2"}], "google")]
    assert handler.metrics.record_calls[-1]["model"] == "batch:ccr-processed"


@pytest.mark.asyncio
async def test_google_batch_passthrough_helpers_forward_and_track_metrics() -> None:
    handler = DummyBatchHandler()
    handler.http_client.post_response = FakeResponse(
        content=b'{"ok":true}',
        headers={"content-encoding": "gzip", "content-length": "10", "x-kept": "1"},
    )
    handler.http_client.post_response = FakeResponse(
        content=b'{"ok":true}',
        headers={"content-encoding": "gzip", "content-length": "10", "x-kept": "1"},
    )

    passthrough = await handler._google_batch_passthrough(
        FakeRequest(
            "body", headers={"host": "proxy", "content-length": "4", "x-goog-api-key": "secret"}
        ),
        "gemini-pro",
        {"batch": {}},
    )
    assert passthrough.status_code == 200
    assert dict(passthrough.headers)["x-kept"] == "1"
    assert "key=secret" in handler.http_client.posts[-1]["url"]
    assert handler.metrics.record_calls[-1]["model"] == "passthrough:batch:gemini-pro"

    handler.http_client.get_response = FakeResponse(
        content=b'{"state":"ok"}',
        headers={"content-encoding": "gzip", "content-length": "10", "x-kept": "2"},
    )
    response = await handler.handle_google_batch_passthrough(
        FakeRequest(
            "ping",
            headers={"host": "proxy", "x-goog-api-key": "secret"},
            method="DELETE",
            path="/v1beta/batches/b1",
            query="alt=json",
        ),
        "b1",
    )
    assert response.status_code == 200
    assert dict(response.headers)["x-kept"] == "2"
    get_call = handler.http_client.requests[-1]
    assert get_call["url"] == "https://gemini.example/v1beta/batches/b1?alt=json&key=secret"
    assert handler.metrics.record_calls[-1]["model"] == "passthrough:batches"


@pytest.mark.asyncio
async def test_handle_google_batch_create_validates_and_passthroughs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_batch_support_modules(monkeypatch)
    handler = DummyBatchHandler()

    too_large = await handler.handle_google_batch_create(
        FakeRequest("{}", headers={"content-length": str(200 * 1024 * 1024)}),
        "gemini-pro",
    )
    assert too_large.status_code == 413

    async def bad_json(request):  # noqa: ANN001
        raise ValueError("bad json")

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", bad_json)
    invalid = await handler.handle_google_batch_create(FakeRequest("{}"), "gemini-pro")
    assert invalid.status_code == 400

    passthrough_response = SimpleNamespace(kind="passthrough")

    async def fake_google_passthrough(request, model, body=None):  # noqa: ANN001
        return passthrough_response

    async def no_inline(request):  # noqa: ANN001
        return {"batch": {"input_config": {"requests": {"requests": []}}}}

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", no_inline)
    monkeypatch.setattr(handler, "_google_batch_passthrough", fake_google_passthrough)
    assert (
        await handler.handle_google_batch_create(FakeRequest("{}"), "gemini-pro")
        is passthrough_response
    )


@pytest.mark.asyncio
async def test_handle_google_batch_create_success_and_failure_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_batch_support_modules(monkeypatch)
    handler = DummyBatchHandler()
    handler.config.optimize = True
    handler.config.ccr_inject_tool = True
    handler.openai_pipeline = SimpleNamespace(
        apply=lambda **kwargs: SimpleNamespace(
            messages=[{"role": "user", "content": "compressed"}],
            timing={"compress": 1.2},
            tokens_before=100,
            tokens_after=40,
        )
    )

    class FakeInjector:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            pass

        def process_request(self, messages, tools):  # noqa: ANN001, ANN201
            return (
                messages + [{"role": "system", "content": "retrieval"}],
                [{"name": "retrieval"}],
                True,
            )

    monkeypatch.setitem(sys.modules, "headroom.ccr", SimpleNamespace(CCRToolInjector=FakeInjector))

    stored: list[tuple[str, list[dict[str, object]], str, str | None]] = []

    async def fake_store(batch_name, requests_list, model, api_key):  # noqa: ANN001
        stored.append((batch_name, requests_list, model, api_key))

    async def fake_retry(method, url, headers, body, **kwargs):  # noqa: ANN001
        return FakeResponse(
            status_code=200,
            content=b'{"name":"batches/123"}',
            headers={"content-encoding": "gzip", "content-length": "10", "x-upstream": "1"},
            json_data={"name": "batches/123"},
        )

    async def good_payload(request):  # noqa: ANN001
        return {
            "batch": {
                "input_config": {
                    "requests": {
                        "requests": [
                            {
                                "request": {
                                    "contents": [{"parts": [{"text": "hello"}]}],
                                    "tools": [{"functionDeclarations": [{"name": "existing"}]}],
                                },
                                "metadata": {"key": "req-1"},
                            }
                        ]
                    }
                }
            }
        }

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", good_payload)
    monkeypatch.setattr(handler, "_retry_request", fake_retry)
    monkeypatch.setattr(handler, "_store_google_batch_context", fake_store)

    response = await handler.handle_google_batch_create(
        FakeRequest("{}", headers={"x-goog-api-key": "secret"}),
        "gemini-pro",
    )
    assert response.status_code == 200
    assert dict(response.headers)["x-upstream"] == "1"
    assert handler.metrics.record_calls[-1]["provider"] == "google"
    assert handler.metrics.record_calls[-1]["tokens_saved"] == 60
    assert stored[0][0] == "batches/123"
    assert stored[0][2:] == ("gemini-pro", "secret")
    assert stored[0][1][0]["metadata"] == {"key": "req-1"}

    async def broken_retry(method, url, headers, body, **kwargs):  # noqa: ANN001
        raise RuntimeError("forward failed")

    monkeypatch.setattr(handler, "_retry_request", broken_retry)
    failed = await handler.handle_google_batch_create(FakeRequest("{}"), "gemini-pro")
    assert failed.status_code == 500


@pytest.mark.asyncio
async def test_handle_google_batch_create_covers_passthrough_revert_and_store_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_batch_support_modules(
        monkeypatch, injector_result=([{"role": "user", "content": "kept"}], None, False)
    )
    handler = DummyBatchHandler()
    handler.config.optimize = True
    handler.config.ccr_inject_tool = True

    pipeline_calls: list[dict[str, object]] = []
    handler.openai_pipeline = SimpleNamespace(
        apply=lambda **kwargs: (
            pipeline_calls.append(kwargs)
            or SimpleNamespace(
                messages=[{"role": "user", "content": "inflated"}],
                timing={},
                tokens_before=40,
                tokens_after=80,
            )
        )
    )

    def fake_to_messages(contents, system_instruction):  # noqa: ANN001, ANN201
        if contents and "inlineData" in contents[0]["parts"][0]:
            return ([{"role": "user", "content": "binary"}], [0])
        return ([{"role": "user", "content": "compress"}], [])

    def fake_to_gemini(messages):  # noqa: ANN001, ANN201
        return ([{"parts": [{"text": "new"}]}], {"parts": [{"text": "sys"}]})

    async def payload(request):  # noqa: ANN001
        return {
            "batch": {
                "input_config": {
                    "requests": {
                        "requests": [
                            {"request": {"contents": []}, "metadata": {"key": "empty"}},
                            {
                                "request": {"contents": [{"parts": [{"inlineData": "x"}]}]},
                                "metadata": {"key": "preserved"},
                            },
                            {
                                "request": {
                                    "contents": [{"parts": [{"text": "hello"}]}],
                                    "tools": [
                                        {"other": True},
                                        {"functionDeclarations": [{"name": "existing"}]},
                                    ],
                                },
                                "metadata": {"key": "optimized"},
                            },
                        ]
                    }
                }
            }
        }

    seen_bodies: list[dict[str, object]] = []

    async def retry(method, url, headers, body, **kwargs):  # noqa: ANN001
        seen_bodies.append(body)
        return FakeResponse(status_code=200, content=b"{}", json_data={"name": "batches/123"})

    async def broken_store(batch_name, requests_list, model, api_key):  # noqa: ANN001
        raise RuntimeError("store failed")

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", payload)
    monkeypatch.setattr(handler, "_gemini_contents_to_messages", fake_to_messages)
    monkeypatch.setattr(handler, "_messages_to_gemini_contents", fake_to_gemini)
    monkeypatch.setattr(handler, "_retry_request", retry)
    monkeypatch.setattr(handler, "_store_google_batch_context", broken_store)

    response = await handler.handle_google_batch_create(FakeRequest("{}"), "gemini-pro")
    assert response.status_code == 200
    assert len(pipeline_calls) == 1
    assert handler.metrics.record_calls[-1]["tokens_saved"] == 0
    assert (
        seen_bodies[0]["batch"]["input_config"]["requests"]["requests"][0]["metadata"]["key"]
        == "empty"
    )
    optimized = seen_bodies[0]["batch"]["input_config"]["requests"]["requests"][2]["request"]
    assert optimized["contents"][0] == {"parts": [{"text": "new"}]}
    assert optimized["systemInstruction"] == {"parts": [{"text": "sys"}]}


@pytest.mark.asyncio
async def test_handle_google_batch_create_preserves_functioncall_response_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch request that interleaves text turns with text-less
    functionCall/functionResponse entries must reach Google with all entries
    intact and in order. The old raw-index restore loop overwrote the model's
    answer with the functionCall and dropped the functionResponse."""

    class RealConvHandler(batch_module.BatchHandlerMixin, GeminiHandlerMixin):
        # Real Gemini converters + _rebuild_gemini_contents (no stubs), so the
        # actual index interleaving runs.
        GEMINI_API_URL = "https://gemini.example"

        def __init__(self) -> None:
            self.http_client = FakeHttpClient()
            self.metrics = FakeMetrics()
            self.config = SimpleNamespace(
                optimize=True, ccr_inject_tool=False, ccr_inject_system_instructions=False
            )
            self.openai_provider = SimpleNamespace(get_context_limit=lambda m: 8192)
            self.usage_reporter = None
            # No-op pipeline: return the messages unchanged, no token inflation.
            self.openai_pipeline = SimpleNamespace(
                apply=lambda **kw: SimpleNamespace(
                    messages=kw["messages"], timing={}, tokens_before=100, tokens_after=100
                )
            )
            self.captured_body: dict | None = None

        async def _next_request_id(self) -> str:
            return "req-1"

        async def _record_request_outcome(self, outcome) -> None:  # noqa: ANN001
            pass

        def _extract_tags(self, headers: dict) -> dict[str, str]:
            return {}

        async def _run_compression_in_executor(self, fn, *, timeout):  # noqa: ANN001, ANN201
            return fn()

        async def _store_google_batch_context(self, *a, **k) -> None:  # noqa: ANN002, ANN003
            pass

        async def _retry_request(self, method, url, headers, body, **kwargs):  # noqa: ANN001, ANN201
            # Capture the (in-place mutated) forwarded batch body for assertions.
            self.captured_body = body
            return FakeResponse(status_code=200, content=b"{}", json_data={"name": "batches/1"})

    handler = RealConvHandler()

    contents = [
        {"role": "user", "parts": [{"text": "What's the weather in Paris?"}]},
        {
            "role": "model",
            "parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}],
        },
        {
            "role": "user",
            "parts": [{"functionResponse": {"name": "get_weather", "response": {"temp_c": 18}}}],
        },
        {"role": "model", "parts": [{"text": "It's 18C and cloudy in Paris."}]},
    ]
    batch_body = {
        "batch": {
            "input_config": {
                "requests": {"requests": [{"request": {"contents": contents}, "metadata": {}}]}
            }
        }
    }

    async def payload(request):  # noqa: ANN001, ANN201
        return batch_body

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", payload)

    resp = await handler.handle_google_batch_create(FakeRequest("{}"), "gemini-pro")
    assert resp.status_code == 200

    out = handler.captured_body["batch"]["input_config"]["requests"]["requests"][0]["request"][
        "contents"
    ]
    # All four entries survive in order. The old loop produced only two, dropping
    # the functionResponse and overwriting the model answer with the functionCall.
    assert len(out) == 4
    assert "text" in out[0]["parts"][0]
    assert out[1]["parts"][0].get("functionCall", {}).get("name") == "get_weather"
    assert out[2]["parts"][0].get("functionResponse", {}).get("name") == "get_weather"
    assert "Paris" in out[3]["parts"][0]["text"]


@pytest.mark.asyncio
async def test_handle_google_batch_create_preserves_sibling_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch request whose tools array carries googleSearch / codeExecution
    alongside functionDeclarations must reach Google with those siblings intact.
    The old code collapsed the whole array to a single functionDeclarations
    entry, silently disabling Google Search and code execution."""

    class RealConvHandler(batch_module.BatchHandlerMixin, GeminiHandlerMixin):
        GEMINI_API_URL = "https://gemini.example"

        def __init__(self) -> None:
            self.http_client = FakeHttpClient()
            self.metrics = FakeMetrics()
            self.config = SimpleNamespace(
                optimize=True, ccr_inject_tool=False, ccr_inject_system_instructions=False
            )
            self.openai_provider = SimpleNamespace(get_context_limit=lambda m: 8192)
            self.usage_reporter = None
            self.openai_pipeline = SimpleNamespace(
                apply=lambda **kw: SimpleNamespace(
                    messages=kw["messages"], timing={}, tokens_before=100, tokens_after=100
                )
            )
            self.captured_body: dict | None = None

        async def _next_request_id(self) -> str:
            return "req-1"

        async def _record_request_outcome(self, outcome) -> None:  # noqa: ANN001
            pass

        def _extract_tags(self, headers: dict) -> dict[str, str]:
            return {}

        async def _run_compression_in_executor(self, fn, *, timeout):  # noqa: ANN001, ANN201
            return fn()

        async def _store_google_batch_context(self, *a, **k) -> None:  # noqa: ANN002, ANN003
            pass

        async def _retry_request(self, method, url, headers, body, **kwargs):  # noqa: ANN001, ANN201
            self.captured_body = body
            return FakeResponse(status_code=200, content=b"{}", json_data={"name": "batches/1"})

    handler = RealConvHandler()

    tools = [
        {"functionDeclarations": [{"name": "get_weather"}]},
        {"googleSearch": {}},
        {"codeExecution": {}},
    ]
    batch_body = {
        "batch": {
            "input_config": {
                "requests": {
                    "requests": [
                        {
                            "request": {
                                "contents": [{"role": "user", "parts": [{"text": "hello there"}]}],
                                "tools": tools,
                            },
                            "metadata": {},
                        }
                    ]
                }
            }
        }
    }

    async def payload(request):  # noqa: ANN001, ANN201
        return batch_body

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", payload)

    resp = await handler.handle_google_batch_create(FakeRequest("{}"), "gemini-pro")
    assert resp.status_code == 200

    out_tools = handler.captured_body["batch"]["input_config"]["requests"]["requests"][0][
        "request"
    ]["tools"]
    keys = [next(iter(entry)) for entry in out_tools]
    assert "googleSearch" in keys
    assert "codeExecution" in keys
    assert "functionDeclarations" in keys


@pytest.mark.asyncio
async def test_google_batch_passthrough_without_body_and_query_variants() -> None:
    handler = DummyBatchHandler()
    handler.http_client.post_response = FakeResponse(content=b"ok", headers={"x-upstream": "1"})

    response = await handler._google_batch_passthrough(
        FakeRequest("raw-body", headers={"host": "proxy"}, method="POST"),
        "gemini-pro",
    )
    assert response.status_code == 200
    assert handler.http_client.posts[-1]["content"] == b"raw-body"

    handler.http_client.get_response = FakeResponse(content=b"{}", headers={"x-upstream": "2"})
    passthrough = await handler.handle_google_batch_passthrough(
        FakeRequest(
            "{}",
            headers={"host": "proxy", "x-goog-api-key": "secret"},
            method="GET",
            path="/v1beta/batches/b1",
        ),
        "b1",
    )
    assert passthrough.status_code == 200
    assert (
        handler.http_client.requests[-1]["url"]
        == "https://gemini.example/v1beta/batches/b1?key=secret"
    )


@pytest.mark.asyncio
async def test_batch_helper_methods_and_openai_file_error_branches() -> None:
    handler = DummyBatchHandler()
    marker = object()

    async def fake_passthrough(request, base_url):  # noqa: ANN001
        return marker

    handler.handle_passthrough = fake_passthrough
    request = FakeRequest("{}")
    assert await handler.handle_batch_list(request) is marker
    assert await handler.handle_batch_get(request, "b1") is marker
    assert await handler.handle_batch_cancel(request, "b1") is marker

    handler.http_client.raise_get = RuntimeError("download boom")
    assert await handler._download_openai_file("file-1", {}) is None

    handler.http_client.raise_get = None
    handler.http_client.post_response = FakeResponse(status_code=200, json_data={})
    assert await handler._upload_openai_file("{}", "missing-id.jsonl", {}) is None


@pytest.mark.asyncio
async def test_store_google_batch_context_without_system_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored_contexts: list[object] = []

    class FakeBatchContext:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.kwargs = kwargs
            self.requests: list[object] = []

        def add_request(self, request) -> None:  # noqa: ANN001
            self.requests.append(request)

    class FakeBatchRequestContext:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.kwargs = kwargs

    class FakeStore:
        async def store(self, context) -> None:  # noqa: ANN001
            stored_contexts.append(context)

    handler = DummyBatchHandler()
    monkeypatch.setitem(
        sys.modules,
        "headroom.ccr",
        SimpleNamespace(
            BatchContext=FakeBatchContext,
            BatchRequestContext=FakeBatchRequestContext,
            get_batch_context_store=lambda: FakeStore(),
        ),
    )

    await handler._store_google_batch_context(
        "batches/456",
        [
            {
                "request": {
                    "contents": [{"parts": [{"text": "hello"}]}],
                    "systemInstruction": {"parts": ["bad"]},
                }
            }
        ],
        "gemini-2.0",
        None,
    )

    context = stored_contexts[0]
    assert context.kwargs["api_key"] is None
    assert context.requests[0].kwargs["custom_id"] == ""
    assert context.requests[0].kwargs["system_instruction"] is None


@pytest.mark.asyncio
async def test_compress_batch_jsonl_skips_blank_lines_and_preserves_tools_when_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_batch_support_modules(
        monkeypatch,
        injector_result=([{"role": "assistant", "content": "short"}], [{"name": "orig"}], False),
    )
    handler = DummyBatchHandler()
    handler.config.optimize = True
    handler.config.ccr_inject_tool = True
    handler.openai_pipeline = SimpleNamespace(
        apply=lambda **kwargs: SimpleNamespace(
            messages=[{"role": "assistant", "content": "short"}],
            tokens_before=50,
            tokens_after=10,
        )
    )

    lines, stats = await handler._compress_batch_jsonl(
        "\n"
        + json.dumps(
            {
                "body": {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hello"}],
                    "tools": [{"name": "orig"}],
                }
            }
        )
        + "\n",
        "req-extra",
    )

    assert len(lines) == 1
    body = json.loads(lines[0])["body"]
    assert body["tools"] == [{"name": "orig"}]
    assert stats["total_requests"] == 1
    assert stats["errors"] == 0


# ── x-headroom-bypass: the client's "don't touch my bytes" contract ──────
# batch.py was never migrated onto CompressionDecision (see that module's
# docstring: the same omission on the Gemini paths was "a real bug"), so
# these lock the contract in on both batch surfaces.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bypass_headers",
    [
        {"x-headroom-bypass": "true"},
        {"x-headroom-mode": "passthrough"},
    ],
)
async def test_handle_batch_create_bypass_skips_compression_entirely(
    monkeypatch: pytest.MonkeyPatch,
    bypass_headers: dict[str, str],
) -> None:
    """Bypass must reach the byte-faithful passthrough without rewriting the file.

    Compressing would re-serialize every JSONL line and upload a NEW file id,
    so "skip compression" is not enough — the download/upload pair must not run.
    """
    handler = DummyBatchHandler()
    handler.config.optimize = True

    async def request_payload(request):  # noqa: ANN001
        return {"input_file_id": "file-1", "endpoint": "/v1/chat/completions"}

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", request_payload)

    called: list[str] = []

    async def fail_download(file_id, headers):  # noqa: ANN001
        called.append("download")
        return "downloaded"

    async def fail_upload(content, filename, headers):  # noqa: ANN001
        called.append("upload")
        return "file-2"

    # Records rather than raising. handle_batch_create wraps its whole body in
    # `except Exception`, which swallows an AssertionError and books a 500 — so
    # that message never reached anyone, and `called` below is the real guard.
    # The zero-request stats short-circuit the reverted path at the
    # total_requests==0 check, before it can reach the upload.
    async def fail_compress(content, request_id):  # noqa: ANN001
        called.append("compress")
        return [], {"total_requests": 0}

    monkeypatch.setattr(handler, "_download_openai_file", fail_download)
    monkeypatch.setattr(handler, "_upload_openai_file", fail_upload)
    monkeypatch.setattr(handler, "_compress_batch_jsonl", fail_compress)

    passthrough_response = SimpleNamespace(marker="passthrough")

    # Accepts the reason but does not assert on it: handle_batch_create wraps its
    # body in `except Exception`, so a raise here would be swallowed. The
    # dedicated *_tags_the_outcome_with_its_reason tests cover the value.
    async def fake_passthrough(request, body, passthrough_reason=None):  # noqa: ANN001
        called.append("passthrough")
        return passthrough_response

    monkeypatch.setattr(handler, "_batch_passthrough", fake_passthrough)

    response = await handler.handle_batch_create(FakeRequest("{}", headers=bypass_headers))

    assert response is passthrough_response
    assert called == ["passthrough"]


@pytest.mark.asyncio
async def test_handle_google_batch_create_bypass_skips_compression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same contract on the Google inline-batch path (batch.py's other gate)."""
    handler = DummyBatchHandler()
    handler.config.optimize = True
    install_batch_support_modules(monkeypatch)

    async def request_payload(request):  # noqa: ANN001
        return {
            "batch": {
                "input_config": {
                    "requests": {
                        "requests": [
                            {
                                "request": {"contents": [{"parts": [{"text": "hello"}]}]},
                                "metadata": {"key": "request-1"},
                            }
                        ]
                    }
                }
            }
        }

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", request_payload)

    # Count invocations rather than raising: the Google compression loop
    # swallows exceptions per request, so a raise here would be masked.
    applied: list[dict] = []

    def apply(**kwargs):  # noqa: ANN003
        applied.append(kwargs)
        return SimpleNamespace(
            messages=[{"role": "user", "content": "hi"}],
            tokens_before=50,
            tokens_after=10,
            timing={},
        )

    handler.openai_pipeline = SimpleNamespace(apply=apply)

    passthrough_response = SimpleNamespace(marker="google-passthrough")
    called: list[str] = []

    async def fake_passthrough(request, model, body=None, passthrough_reason=None):  # noqa: ANN001
        called.append("passthrough")
        return passthrough_response

    monkeypatch.setattr(handler, "_google_batch_passthrough", fake_passthrough)

    response = await handler.handle_google_batch_create(
        FakeRequest("{}", headers={"x-headroom-bypass": "true"}),
        "gemini-2.0-flash",
    )

    assert response is passthrough_response
    assert called == ["passthrough"]
    assert applied == []


@pytest.mark.asyncio
async def test_handle_batch_create_without_bypass_still_compresses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bypass guard must not misfire on an ordinary request.

    Gating the handler on ``CompressionDecision.decide(messages=None)`` looks
    right and is wrong: ``None`` hits the ``no_messages`` precedence step and
    returns ``should_compress=False``, sending every batch to passthrough.
    This test turns red on that shortcut.

    Scope: the handler-level guard only. ``_compress_batch_jsonl`` is stubbed
    here, so the same mistake made *inside* that method would sail past this
    test. ``test_compress_batch_jsonl_*`` cover the per-line gates.
    """
    handler = DummyBatchHandler()
    handler.config.optimize = True

    async def request_payload(request):  # noqa: ANN001
        return {"input_file_id": "file-1", "endpoint": "/v1/chat/completions"}

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", request_payload)

    async def fake_download(file_id, headers):  # noqa: ANN001
        return "downloaded"

    compressed_calls: list[str] = []

    async def fake_compress(content, request_id):  # noqa: ANN001
        compressed_calls.append(content)
        return ['{"body":{}}'], {
            "total_requests": 1,
            "total_original_tokens": 20,
            "total_compressed_tokens": 10,
            "total_tokens_saved": 10,
            "savings_percent": 50.0,
            "errors": 0,
        }

    async def fake_upload(content, filename, headers):  # noqa: ANN001
        return "file-compressed"

    monkeypatch.setattr(handler, "_download_openai_file", fake_download)
    monkeypatch.setattr(handler, "_compress_batch_jsonl", fake_compress)
    monkeypatch.setattr(handler, "_upload_openai_file", fake_upload)

    handler.http_client.post_response = FakeResponse(content=b'{"id":"batch_123","object":"batch"}')

    response = await handler.handle_batch_create(
        FakeRequest("{}", headers={"authorization": "Bearer test"})
    )

    assert response.status_code == 200
    assert compressed_calls == ["downloaded"]
    assert dict(response.headers)["x-headroom-tokens-saved"] == "10"


@pytest.mark.asyncio
async def test_compress_batch_jsonl_skips_when_license_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """License denial gates compression, the other half of the conjunction.

    Unlike bypass this is not a byte-fidelity contract, so it gates the
    pipeline in place rather than rerouting to the passthrough forwarder.
    """
    handler = DummyBatchHandler()
    handler.config.optimize = True
    handler.usage_reporter = SimpleNamespace(should_compress=False)
    install_batch_support_modules(monkeypatch, tokenizer_count=7)

    # Count invocations rather than raising: _compress_batch_jsonl wraps the
    # pipeline call in `except Exception`, which would swallow an AssertionError
    # and let this pass via the fallback path even with the gate removed.
    applied: list[dict] = []

    def apply(**kwargs):  # noqa: ANN003
        applied.append(kwargs)
        return SimpleNamespace(
            messages=[{"role": "user", "content": "hi"}],
            tokens_before=50,
            tokens_after=10,
        )

    handler.openai_pipeline = SimpleNamespace(apply=apply)

    line = json.dumps(
        {"body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}}
    )
    lines, stats = await handler._compress_batch_jsonl(line, "req-license")

    assert applied == []
    assert stats["total_requests"] == 1
    assert stats["total_tokens_saved"] == 0
    assert json.loads(lines[0])["body"]["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_compress_batch_jsonl_compresses_when_no_usage_reporter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fails open: no reporter configured means compress, not passthrough."""
    handler = DummyBatchHandler()
    handler.config.optimize = True
    install_batch_support_modules(monkeypatch)

    applied: list[dict] = []

    def apply(**kwargs):  # noqa: ANN003
        applied.append(kwargs)
        return SimpleNamespace(
            messages=[{"role": "user", "content": "hi"}],
            tokens_before=50,
            tokens_after=10,
        )

    handler.openai_pipeline = SimpleNamespace(apply=apply)

    assert handler.usage_reporter is None

    line = json.dumps(
        {"body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}}
    )
    _lines, stats = await handler._compress_batch_jsonl(line, "req-nolicense")

    assert len(applied) == 1
    assert stats["total_tokens_saved"] == 40


@pytest.mark.asyncio
async def test_batch_passthrough_records_request_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bypassed batch create must still reach the funnel.

    Routing bypass to the passthrough forwarder would otherwise drop the
    request from dashboards entirely — the compressed path records at the
    end of handle_batch_create, and _google_batch_passthrough records too.
    """
    handler = DummyBatchHandler()
    handler.config.optimize = True

    async def request_payload(request):  # noqa: ANN001
        return {"input_file_id": "file-1", "endpoint": "/v1/chat/completions"}

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", request_payload)

    response = await handler.handle_batch_create(
        FakeRequest("{}", headers={"x-headroom-bypass": "true", "x-headroom-client": "test"})
    )

    assert response.status_code == 200
    assert len(handler.metrics.record_calls) == 1
    recorded = handler.metrics.record_calls[0]
    assert recorded["provider"] == "openai"
    assert recorded["tokens_saved"] == 0


@pytest.mark.asyncio
async def test_google_batch_bypass_forwards_original_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bypass on the Google path must forward the wire bytes untouched.

    The sibling bypass test stubs _google_batch_passthrough, so it proves
    routing but not the contract. This one lets the real helper run: handing
    it the parsed dict re-serializes canonically (dropping the client's
    whitespace) and logs body_mutated=True, which is what bypass forbids.
    """
    handler = DummyBatchHandler()
    handler.config.optimize = True
    handler.http_client.post_response = FakeResponse(content=b"ok")

    raw = (
        '{"batch": {"input_config": {"requests": {"requests": '
        '[{"request": {"contents": [{"parts": [{"text": "hi"}]}]}, '
        '"metadata": {"key": "r1"}}]}}},  "spaced":  true}'
    )

    async def request_payload(request):  # noqa: ANN001
        return json.loads(raw)

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", request_payload)

    await handler.handle_google_batch_create(
        FakeRequest(raw, headers={"x-headroom-bypass": "true"}),
        "gemini-2.0-flash",
    )

    assert handler.http_client.posts[-1]["content"] == raw.encode("utf-8")


@pytest.mark.asyncio
async def test_batch_passthrough_records_upstream_failure_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 5xx on a bypassed batch create must not land in the success funnel.

    RequestOutcome.status_code defaults to 200, and emit_request_outcome only
    diverts to record_failed at >= 500, so omitting it books a failed upstream
    call as a success and inflates the save-rate.
    """
    handler = DummyBatchHandler()
    handler.config.optimize = True
    handler.http_client.post_response = FakeResponse(status_code=503, content=b'{"error":"busy"}')

    async def request_payload(request):  # noqa: ANN001
        return {"input_file_id": "file-1", "endpoint": "/v1/chat/completions"}

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", request_payload)

    response = await handler.handle_batch_create(
        FakeRequest("{}", headers={"x-headroom-bypass": "true"})
    )

    assert response.status_code == 503
    assert handler.metrics.record_calls == []
    assert len(handler.metrics.failed_calls) == 1
    assert handler.metrics.failed_calls[0]["provider"] == "openai"


@pytest.mark.asyncio
async def test_google_batch_passthrough_records_upstream_failure_as_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same 5xx contract as the OpenAI passthrough, on the Google path.

    The bypass guard routes traffic into this helper, so an omitted
    status_code books a failed Gemini call as a served request.
    """
    handler = DummyBatchHandler()
    handler.config.optimize = True
    handler.http_client.post_response = FakeResponse(status_code=503, content=b'{"error":"busy"}')

    raw = (
        '{"batch": {"input_config": {"requests": {"requests": '
        '[{"request": {"contents": [{"parts": [{"text": "hi"}]}]}, '
        '"metadata": {"key": "r1"}}]}}}}'
    )

    async def request_payload(request):  # noqa: ANN001
        return json.loads(raw)

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", request_payload)

    response = await handler.handle_google_batch_create(
        FakeRequest(raw, headers={"x-headroom-bypass": "true"}),
        "gemini-2.0-flash",
    )

    assert response.status_code == 503
    assert handler.metrics.record_calls == []
    assert len(handler.metrics.failed_calls) == 1
    assert handler.metrics.failed_calls[0]["provider"] == "google"


@pytest.mark.asyncio
async def test_google_batch_create_skips_compression_when_license_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The license gate's third disjunct at the Google per-item guard.

    Arc coverage marks that line hit via the other two disjuncts, so
    without this the `not license_ok` branch never actually decides.
    """
    handler = DummyBatchHandler()
    handler.config.optimize = True
    handler.usage_reporter = SimpleNamespace(should_compress=False)
    install_batch_support_modules(monkeypatch)

    applied: list[dict] = []

    def apply(**kwargs):  # noqa: ANN003
        applied.append(kwargs)
        return SimpleNamespace(
            messages=[{"role": "user", "content": "hi"}],
            tokens_before=50,
            tokens_after=10,
            timing={},
        )

    handler.openai_pipeline = SimpleNamespace(apply=apply)
    handler.http_client.post_response = FakeResponse(content=b'{"name":"batches/b1"}')

    raw = (
        '{"batch": {"input_config": {"requests": {"requests": '
        '[{"request": {"contents": [{"parts": [{"text": "hello"}]}]}, '
        '"metadata": {"key": "r1"}}]}}}}'
    )

    async def request_payload(request):  # noqa: ANN001
        return json.loads(raw)

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", request_payload)

    await handler.handle_google_batch_create(FakeRequest(raw), "gemini-2.0-flash")

    assert applied == []
    assert handler.metrics.record_calls[-1]["tokens_saved"] == 0


@pytest.mark.asyncio
async def test_google_batch_bypass_beats_the_empty_requests_return(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bypass must be checked above the empty-`requests` early return.

    A file-input batch carries no inline `requests`, so it takes that return —
    which hands `_google_batch_passthrough` the parsed dict and re-serializes
    canonically, dropping the client's whitespace. Below the return, bypass
    silently mutated exactly the bodies it promised not to touch.
    """
    handler = DummyBatchHandler()
    handler.config.optimize = True
    handler.http_client.post_response = FakeResponse(content=b"ok")

    # No inline requests (file input), and deliberately non-canonical spacing:
    # a canonical re-serialize collapses it, so the byte compare catches it.
    raw = '{"batch": {"input_config": {"file_name":  "files/abc"}},  "spaced":  true}'

    async def request_payload(request):  # noqa: ANN001
        return json.loads(raw)

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", request_payload)

    await handler.handle_google_batch_create(
        FakeRequest(raw, headers={"x-headroom-bypass": "true"}),
        "gemini-2.0-flash",
    )

    assert handler.http_client.posts[-1]["content"] == raw.encode("utf-8")


# ── passthrough_reason: the slice label apply_to_tags gives every other handler ──
# batch.py calls apply_to_tags zero times (the non-batch handlers call it at nine
# sites), so a bypassed or license-denied batch reached the funnel indistinguishable
# from an ordinary zero-savings one. Tags only reach RequestLog, not record_request,
# so these assert through a logger double.


def _tag_recorder(handler: DummyBatchHandler) -> list[dict]:
    """Attach a logger double and return the list its RequestLog tags land in."""
    seen: list[dict] = []
    handler.logger = SimpleNamespace(log=lambda entry: seen.append(dict(entry.tags)))
    return seen


@pytest.mark.asyncio
async def test_batch_bypass_tags_the_outcome_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bypassed OpenAI batch must be sliceable as bypass_header, not just zero-savings."""
    handler = DummyBatchHandler()
    handler.config.optimize = True
    tags_seen = _tag_recorder(handler)

    async def request_payload(request):  # noqa: ANN001
        return {"input_file_id": "file-1", "endpoint": "/v1/chat/completions"}

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", request_payload)

    await handler.handle_batch_create(FakeRequest("{}", headers={"x-headroom-bypass": "true"}))

    assert tags_seen[-1]["passthrough_reason"] == "bypass_header"


@pytest.mark.asyncio
async def test_google_batch_bypass_tags_the_outcome_with_its_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same slice label on the Google bypass path."""
    handler = DummyBatchHandler()
    handler.config.optimize = True
    handler.http_client.post_response = FakeResponse(content=b"ok")
    tags_seen = _tag_recorder(handler)

    raw = (
        '{"batch": {"input_config": {"requests": {"requests": '
        '[{"request": {"contents": [{"parts": [{"text": "hi"}]}]}, '
        '"metadata": {"key": "r1"}}]}}}}'
    )

    async def request_payload(request):  # noqa: ANN001
        return json.loads(raw)

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", request_payload)

    await handler.handle_google_batch_create(
        FakeRequest(raw, headers={"x-headroom-bypass": "true"}),
        "gemini-2.0-flash",
    )

    assert tags_seen[-1]["passthrough_reason"] == "bypass_header"


@pytest.mark.asyncio
async def test_google_batch_license_denied_tags_the_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """License denial is the other cause this PR introduced — tag it too."""
    handler = DummyBatchHandler()
    handler.config.optimize = True
    handler.usage_reporter = SimpleNamespace(should_compress=False)
    handler.http_client.post_response = FakeResponse(content=b'{"name":"batches/b1"}')
    install_batch_support_modules(monkeypatch)
    tags_seen = _tag_recorder(handler)

    raw = (
        '{"batch": {"input_config": {"requests": {"requests": '
        '[{"request": {"contents": [{"parts": [{"text": "hello"}]}]}, '
        '"metadata": {"key": "r1"}}]}}}}'
    )

    async def request_payload(request):  # noqa: ANN001
        return json.loads(raw)

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", request_payload)

    await handler.handle_google_batch_create(FakeRequest(raw), "gemini-2.0-flash")

    assert tags_seen[-1]["passthrough_reason"] == "license_denied"


@pytest.mark.asyncio
async def test_batch_license_denied_tags_the_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The OpenAI license gate lives inside _compress_batch_jsonl, which holds no
    tags — so the reason rides back on its stats dict. Runs the real method rather
    than stubbing it, or the assertion would only be testing the stub."""
    handler = DummyBatchHandler()
    handler.config.optimize = True
    handler.usage_reporter = SimpleNamespace(should_compress=False)
    handler.http_client.post_response = FakeResponse(content=b'{"id":"batch_123","object":"batch"}')
    install_batch_support_modules(monkeypatch, tokenizer_count=7)
    tags_seen = _tag_recorder(handler)

    async def request_payload(request):  # noqa: ANN001
        return {"input_file_id": "file-1", "endpoint": "/v1/chat/completions"}

    async def fake_download(file_id, headers):  # noqa: ANN001
        return json.dumps(
            {"body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}}
        )

    async def fake_upload(content, filename, headers):  # noqa: ANN001
        return "file-2"

    monkeypatch.setattr("headroom.proxy.helpers._read_request_json", request_payload)
    monkeypatch.setattr(handler, "_download_openai_file", fake_download)
    monkeypatch.setattr(handler, "_upload_openai_file", fake_upload)

    await handler.handle_batch_create(FakeRequest("{}", headers={"authorization": "Bearer t"}))

    assert tags_seen[-1]["passthrough_reason"] == "license_denied"
