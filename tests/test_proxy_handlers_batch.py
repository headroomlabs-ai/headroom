from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from headroom.proxy.handlers import batch as batch_module


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


class DummyBatchHandler(batch_module.BatchHandlerMixin):
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
        self._request_counter = 0
        self._retry_response = FakeResponse()

    async def _next_request_id(self) -> str:
        self._request_counter += 1
        return f"req-{self._request_counter}"

    async def handle_passthrough(self, request, base_url):  # noqa: ANN001, ANN201
        return {"request": request, "base_url": base_url}

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

    async def body(self) -> bytes:
        return self._body


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
    }


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
