from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


class _FakePrefixTracker:
    def get_frozen_message_count(self) -> int:
        return 0

    def get_last_original_messages(self) -> list[dict]:
        return []

    def get_last_forwarded_messages(self) -> list[dict]:
        return []

    def update_from_response(self, **kwargs):  # noqa: ANN003
        return None


class _BufferedPassthroughClient:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    async def request(self, method, url, headers=None, content=None, timeout=None):  # noqa: ANN001
        self.calls.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "content": content,
                "timeout": timeout,
            }
        )
        _assert_buffered_timeout(timeout)
        return self.response

    async def get(self, url, headers=None, timeout=None):  # noqa: ANN001
        return await self.request("GET", url, headers=headers, timeout=timeout)

    async def post(self, url, headers=None, content=None, timeout=None):  # noqa: ANN001
        return await self.request("POST", url, headers=headers, content=content, timeout=timeout)

    async def aclose(self) -> None:
        return None


def _assert_buffered_timeout(timeout: httpx.Timeout | None) -> None:
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 3.0
    assert timeout.read == 19.0
    assert timeout.write == 7.0
    assert timeout.pool == 3.0


def _make_config() -> ProxyConfig:
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
        connect_timeout_seconds=3,
        request_timeout_seconds=7,
        anthropic_buffered_request_timeout_seconds=19,
    )


def _install_prefix_tracker(proxy) -> None:
    tracker = _FakePrefixTracker()
    proxy.session_tracker_store.compute_session_id = lambda request, model, messages: "s1"
    proxy.session_tracker_store.get_or_create = lambda session_id, provider: tracker


def _anthropic_message_response() -> dict[str, object]:
    return {
        "id": "msg_test_1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "ok"}],
        "usage": {
            "input_tokens": 12,
            "output_tokens": 3,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    }


def _anthropic_batch_response() -> dict[str, object]:
    return {"id": "batch_test_1", "object": "batch", "status": "in_progress"}


def _anthropic_list_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={"object": "list", "data": [], "first_id": None, "last_id": None},
        headers={"content-type": "application/json"},
    )


def test_anthropic_messages_buffered_timeout_override_reaches_retry_request():
    config = _make_config()
    app = create_app(config)
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        _install_prefix_tracker(proxy)
        captured: dict[str, object] = {}

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            timeout = kwargs.get("timeout")
            captured["timeout"] = timeout
            _assert_buffered_timeout(timeout)
            return httpx.Response(200, json=_anthropic_message_response())

        proxy._retry_request = _fake_retry  # type: ignore[assignment]

        response = client.post(
            "/v1/messages",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert "timeout" in captured, response.text
    assert isinstance(captured["timeout"], httpx.Timeout)


def test_anthropic_batch_create_buffered_timeout_override_reaches_retry_request():
    config = _make_config()
    app = create_app(config)
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        captured: dict[str, object] = {}

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            timeout = kwargs.get("timeout")
            captured["timeout"] = timeout
            _assert_buffered_timeout(timeout)
            return httpx.Response(200, json=_anthropic_batch_response())

        proxy._retry_request = _fake_retry  # type: ignore[assignment]

        response = client.post(
            "/v1/messages/batches",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "requests": [
                    {
                        "custom_id": "req-1",
                        "params": {
                            "model": "claude-sonnet-4-6",
                            "max_tokens": 64,
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                    }
                ]
            },
        )

    assert response.status_code == 200, response.text
    assert isinstance(captured["timeout"], httpx.Timeout)


def test_anthropic_batch_passthrough_buffered_timeout_override_reaches_http_client():
    config = _make_config()
    app = create_app(config)
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http_client = _BufferedPassthroughClient(_anthropic_list_response())
        proxy.http_client = http_client

        response = client.get(
            "/v1/messages/batches",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
            },
        )

    assert response.status_code == 200, response.text
    assert len(http_client.calls) == 1
    assert isinstance(http_client.calls[0]["timeout"], httpx.Timeout)


def test_anthropic_batch_results_buffered_timeout_override_reaches_http_client_get():
    config = _make_config()
    app = create_app(config)
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        http_client = _BufferedPassthroughClient(
            httpx.Response(
                200,
                content=b'{"custom_id":"req-1","result":{"type":"succeeded"}}\n',
                headers={"content-type": "application/jsonl"},
            )
        )
        proxy.http_client = http_client

        response = client.get(
            "/v1/messages/batches/batch_test_1/results",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
            },
        )

    assert response.status_code == 200, response.text
    assert len(http_client.calls) == 1
    assert http_client.calls[0]["method"] == "GET"
    assert isinstance(http_client.calls[0]["timeout"], httpx.Timeout)


def test_anthropic_ccr_continuation_uses_buffered_timeout() -> None:
    config = _make_config()
    config.ccr_inject_tool = True
    config.ccr_handle_responses = True
    app = create_app(config)

    class _CCRHandler:
        def has_ccr_tool_calls(self, response, provider):  # noqa: ANN001
            return True

        async def handle_response(  # noqa: ANN001
            self,
            response,
            optimized_messages,
            tools,
            api_call_fn,
            provider,
        ):
            return await api_call_fn(
                optimized_messages
                + [{"role": "assistant", "content": response.get("content", [])}],
                tools,
            )

    with TestClient(app) as client:
        proxy = client.app.state.proxy
        _install_prefix_tracker(proxy)
        proxy.ccr_response_handler = _CCRHandler()
        http_client = _BufferedPassthroughClient(
            httpx.Response(200, json=_anthropic_message_response())
        )
        proxy.http_client = http_client

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            return httpx.Response(200, json=_anthropic_message_response())

        proxy._retry_request = _fake_retry  # type: ignore[assignment]

        client.post(
            "/v1/messages",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert len(http_client.calls) == 1
    assert http_client.calls[0]["method"] == "POST"
    assert isinstance(http_client.calls[0]["timeout"], httpx.Timeout)


def test_anthropic_memory_continuation_uses_buffered_timeout() -> None:
    config = _make_config()
    config.memory_enabled = True
    app = create_app(config)

    class _MemoryHandler:
        def __init__(self) -> None:
            self.config = type(
                "MemoryConfig",
                (),
                {
                    "inject_context": False,
                    "inject_tools": False,
                    "project_root_override": "",
                },
            )()
            self.initialized = False
            self.backend = None

        def get_beta_headers(self) -> dict[str, str]:
            return {}

        def has_memory_tool_calls(self, response, provider):  # noqa: ANN001
            return True

        async def handle_memory_tool_calls(  # noqa: ANN001
            self,
            response,
            user_id,
            provider,
            **kwargs,
        ):
            return [{"type": "tool_result", "tool_use_id": "mem_1", "content": "memory"}]

    with TestClient(app) as client:
        proxy = client.app.state.proxy
        _install_prefix_tracker(proxy)
        proxy.memory_handler = _MemoryHandler()
        captured_timeouts: list[httpx.Timeout | None] = []

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            timeout = kwargs.get("timeout")
            captured_timeouts.append(timeout)
            if len(body["messages"]) > 1:
                _assert_buffered_timeout(timeout)
            return httpx.Response(200, json=_anthropic_message_response())

        proxy._retry_request = _fake_retry  # type: ignore[assignment]

        client.post(
            "/v1/messages",
            headers={
                "x-api-key": "test-key",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "x-headroom-user-id": "user-1",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert len(captured_timeouts) == 2
    assert all(isinstance(timeout, httpx.Timeout) for timeout in captured_timeouts)


def test_retry_request_without_override_uses_client_default_timeout():
    config = _make_config()
    app = create_app(config)
    with TestClient(app) as client:
        proxy = client.app.state.proxy
        captured: list[dict[str, object]] = []

        async def _fake_retry(method, url, headers, body, stream=False, **kwargs):  # noqa: ANN001
            captured.append(kwargs)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "ok"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                },
            )

        proxy._retry_request = _fake_retry  # type: ignore[assignment]

        response = client.post(
            "/v1/chat/completions",
            headers={"authorization": "Bearer test-key", "content-type": "application/json"},
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert response.status_code == 200, response.text
    assert len(captured) == 1
    assert "timeout" not in captured[0], (
        "_retry_request without an override must not pass timeout=None to httpx"
    )


def test_generic_proxy_timeout_defaults_stay_unchanged():
    app = create_app(ProxyConfig())
    with TestClient(app) as client:
        timeout = client.app.state.proxy.http_client.timeout

    assert timeout.connect == 10.0
    assert timeout.read == 300.0
    assert timeout.write == 300.0
    assert timeout.pool == 10.0
