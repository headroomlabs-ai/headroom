"""Tests for ratelimit header forwarding in streaming responses.

Verifies that anthropic-ratelimit-* headers from the upstream API response
are forwarded to the client in StreamingResponse, even in SSE streaming mode.

This was a bug where non-streaming responses correctly forwarded all headers
via dict(response.headers), but streaming responses used StreamingResponse
without passing any upstream headers — silently dropping ratelimit info.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

import headroom.proxy.handlers.streaming as streaming_module
from headroom.proxy.server import HeadroomProxy


@pytest.fixture(autouse=True)
def _reset_codex_rate_limit_singleton():
    """Isolate the process-global CodexRateLimitState across tests.

    The tracker is a module singleton; save/restore ``_latest`` around every
    test so a captured snapshot never leaks into (or depends on) another test.
    """
    from headroom.subscription.codex_rate_limits import get_codex_rate_limit_state

    state = get_codex_rate_limit_state()
    saved = state._latest
    state._latest = None
    try:
        yield
    finally:
        state._latest = saved


class TestStreamingRatelimitHeaderForwarding:
    """Test that upstream ratelimit headers are forwarded in streaming responses."""

    def _create_mock_proxy(self):
        """Create a HeadroomProxy with mocked internals for unit testing."""
        proxy = object.__new__(HeadroomProxy)
        proxy.http_client = MagicMock(spec=httpx.AsyncClient)
        proxy.metrics = MagicMock()
        proxy.metrics.record_request = AsyncMock(return_value=None)
        proxy.metrics.record_failed = AsyncMock(return_value=None)
        proxy.cost_tracker = MagicMock()
        proxy.cost_tracker.estimate_cost.return_value = 0.001
        proxy.cost_tracker.record_request.return_value = None
        proxy.stats = {
            "requests_total": 0,
            "requests_optimized": 0,
            "tokens": {"original": 0, "optimized": 0, "saved": 0},
            "cost": {"total_usd": 0, "savings_usd": 0},
            "errors": 0,
            "active_requests": 0,
            "requests_per_model": {},
        }
        proxy.memory_manager = None
        proxy._config = MagicMock()
        proxy._config.memory_enabled = False
        proxy._config.ccr_inject_tool = False
        proxy._config.retry_max_attempts = 3
        proxy._config.retry_base_delay_ms = 0
        proxy._config.retry_max_delay_ms = 0
        proxy.config = proxy._config
        proxy._parse_sse_usage_from_buffer = MagicMock(return_value=None)
        proxy.memory_handler = None
        return proxy

    def _create_mock_upstream_response(self, extra_headers=None):
        """Create a mock httpx streaming response with ratelimit headers."""
        mock_response = AsyncMock()
        headers = {
            "content-type": "text/event-stream",
            "anthropic-ratelimit-tokens-limit": "80000",
            "anthropic-ratelimit-tokens-remaining": "75000",
            "anthropic-ratelimit-tokens-reset": "2026-03-25T12:00:00Z",
            "anthropic-ratelimit-requests-limit": "60",
            "anthropic-ratelimit-requests-remaining": "59",
            "anthropic-ratelimit-requests-reset": "2026-03-25T12:00:00Z",
            "anthropic-ratelimit-input-tokens-limit": "50000",
            "anthropic-ratelimit-input-tokens-remaining": "48000",
            "anthropic-ratelimit-input-tokens-reset": "2026-03-25T12:00:00Z",
            "anthropic-ratelimit-output-tokens-limit": "30000",
            "anthropic-ratelimit-output-tokens-remaining": "27000",
            "anthropic-ratelimit-output-tokens-reset": "2026-03-25T12:00:00Z",
            # request-id is forwarded (clients record it per transcript turn);
            # cf-ray is a non-allowlisted header that must NOT be forwarded.
            "x-request-id": "req-12345",
            "cf-ray": "abc123",
        }
        if extra_headers:
            headers.update(extra_headers)
        mock_response.headers = httpx.Headers(headers)
        mock_response.status_code = 200

        # Simulate a simple SSE stream
        sse_data = (
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01"}}\n\n'
            b'event: message_stop\ndata: {"type":"message_stop"}\n\n'
        )

        async def aiter_bytes():
            yield sse_data

        mock_response.aiter_bytes = aiter_bytes
        mock_response.aclose = AsyncMock()
        return mock_response

    @pytest.mark.asyncio
    async def test_ratelimit_headers_forwarded_in_streaming(self):
        """Ratelimit headers from upstream should appear in the StreamingResponse."""
        proxy = self._create_mock_proxy()
        mock_response = self._create_mock_upstream_response()

        # Mock build_request + send to return our mock response
        mock_request = MagicMock()
        proxy.http_client.build_request = MagicMock(return_value=mock_request)
        proxy.http_client.send = AsyncMock(return_value=mock_response)

        result = await proxy._stream_response(
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": "sk-test", "anthropic-version": "2023-06-01"},
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            request_id="test-123",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        # Verify ratelimit headers are present in the StreamingResponse
        assert result.headers.get("anthropic-ratelimit-tokens-limit") == "80000"
        assert result.headers.get("anthropic-ratelimit-tokens-remaining") == "75000"
        assert result.headers.get("anthropic-ratelimit-tokens-reset") == "2026-03-25T12:00:00Z"
        assert result.headers.get("anthropic-ratelimit-requests-limit") == "60"
        assert result.headers.get("anthropic-ratelimit-requests-remaining") == "59"
        assert result.headers.get("anthropic-ratelimit-input-tokens-limit") == "50000"
        assert result.headers.get("anthropic-ratelimit-output-tokens-limit") == "30000"

    @pytest.mark.asyncio
    async def test_non_ratelimit_headers_not_forwarded(self):
        """Arbitrary upstream headers stay dropped; the request-id family is allowed."""
        proxy = self._create_mock_proxy()
        mock_response = self._create_mock_upstream_response()

        mock_request = MagicMock()
        proxy.http_client.build_request = MagicMock(return_value=mock_request)
        proxy.http_client.send = AsyncMock(return_value=mock_response)

        result = await proxy._stream_response(
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": "sk-test"},
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            request_id="test-456",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        # request-id is forwarded; other non-ratelimit headers are not.
        assert result.headers.get("x-request-id") == "req-12345"
        assert result.headers.get("cf-ray") is None

    @pytest.mark.asyncio
    async def test_request_id_headers_forwarded_in_streaming(self):
        """The request-id family is forwarded on the streaming path (#1100)."""
        proxy = self._create_mock_proxy()
        mock_response = self._create_mock_upstream_response()
        mock_response.headers = httpx.Headers(
            {
                "content-type": "text/event-stream",
                "request-id": "req-aaa",
                "anthropic-request-id": "req-bbb",
                "x-request-id": "req-ccc",
                # Non-allowlisted header: must NOT be forwarded.
                "cf-ray": "ray-123",
            }
        )

        mock_request = MagicMock()
        proxy.http_client.build_request = MagicMock(return_value=mock_request)
        proxy.http_client.send = AsyncMock(return_value=mock_response)

        result = await proxy._stream_response(
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": "sk-test"},
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            request_id="test-1100",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        assert result.media_type == "text/event-stream"
        assert result.headers.get("request-id") == "req-aaa"
        assert result.headers.get("anthropic-request-id") == "req-bbb"
        assert result.headers.get("x-request-id") == "req-ccc"
        assert result.headers.get("cf-ray") is None

    @pytest.mark.asyncio
    async def test_no_ratelimit_headers_still_works(self):
        """When upstream has no ratelimit headers, streaming should still work."""
        proxy = self._create_mock_proxy()
        mock_response = self._create_mock_upstream_response()
        # Remove all ratelimit headers
        mock_response.headers = httpx.Headers(
            {
                "content-type": "text/event-stream",
                "x-request-id": "req-999",
            }
        )

        mock_request = MagicMock()
        proxy.http_client.build_request = MagicMock(return_value=mock_request)
        proxy.http_client.send = AsyncMock(return_value=mock_response)

        result = await proxy._stream_response(
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": "sk-test"},
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            request_id="test-789",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        # Should still return a valid StreamingResponse
        assert result.media_type == "text/event-stream"
        # No ratelimit headers to forward
        assert result.headers.get("anthropic-ratelimit-tokens-limit") is None

    @pytest.mark.asyncio
    async def test_upstream_http_error_preserves_status_body_and_metrics(self, monkeypatch):
        """Upstream non-200 streaming responses should preserve status/body and metrics."""
        proxy = self._create_mock_proxy()
        mock_response = self._create_mock_upstream_response()
        mock_response.status_code = 503
        mock_response.headers = httpx.Headers(
            {
                "content-type": "application/json",
                "content-encoding": "gzip",
                "content-length": "42",
            }
        )
        mock_response.aread = AsyncMock(return_value=b'{"error":{"message":"capacity exhausted"}}')
        mock_response.aclose = AsyncMock()

        mock_request = MagicMock()
        proxy.http_client.build_request = MagicMock(return_value=mock_request)
        proxy.http_client.send = AsyncMock(return_value=mock_response)
        fake_logger = MagicMock()
        monkeypatch.setattr(streaming_module, "logger", fake_logger)

        result = await proxy._stream_response(
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": "sk-test"},
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            request_id="test-http-error",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        assert result.status_code == 503
        assert result.body == b'{"error":{"message":"capacity exhausted"}}'
        assert result.headers.get("content-encoding") is None
        fake_logger.warning.assert_any_call(
            "[%s] Forwarding upstream streaming error status=%s url=%s",
            "test-http-error",
            503,
            "https://api.anthropic.com/v1/messages",
        )
        proxy.metrics.record_request.assert_awaited_once()
        proxy.cost_tracker.record_tokens.assert_called_once()
        mock_response.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_upstream_http_error_closes_response_when_body_read_fails(self, monkeypatch):
        """Reading a streaming error body should still close the upstream response."""
        proxy = self._create_mock_proxy()
        mock_response = self._create_mock_upstream_response()
        mock_response.status_code = 502
        mock_response.aread = AsyncMock(side_effect=RuntimeError("boom"))
        mock_response.aclose = AsyncMock()

        mock_request = MagicMock()
        proxy.http_client.build_request = MagicMock(return_value=mock_request)
        proxy.http_client.send = AsyncMock(return_value=mock_response)
        fake_logger = MagicMock()
        monkeypatch.setattr(streaming_module, "logger", fake_logger)

        result = await proxy._stream_response(
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": "sk-test"},
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            request_id="test-http-error-read-fail",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        assert result.status_code == 502
        assert result.headers.get("content-type") == "application/json"
        assert b"Failed to read upstream error response body" in result.body
        fake_logger.warning.assert_any_call(
            "[%s] Failed reading upstream error body status=%s url=%s error=%s",
            "test-http-error-read-fail",
            502,
            "https://api.anthropic.com/v1/messages",
            mock_response.aread.side_effect,
        )
        mock_response.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_error_returns_sse_error(self):
        """Connection errors should return an SSE error event (not crash)."""
        proxy = self._create_mock_proxy()

        mock_request = MagicMock()
        proxy.http_client.build_request = MagicMock(return_value=mock_request)
        proxy.http_client.send = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))

        result = await proxy._stream_response(
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": "sk-test"},
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            request_id="test-error",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        # Should return a StreamingResponse with error SSE event
        assert result.media_type == "text/event-stream"

        # Consume the generator to get the error event
        chunks = []
        async for chunk in result.body_iterator:
            chunks.append(chunk)

        assert len(chunks) == 1
        raw = chunks[0].decode("utf-8")
        assert "event: error" in raw
        error_data = json.loads(raw.split("data: ")[1].strip())
        assert error_data["error"]["type"] == "connection_error"
        assert "Connection refused" in error_data["error"]["message"]

    @pytest.mark.asyncio
    async def test_connect_timeout_retries_before_returning_stream(self):
        """Transient connect timeouts should retry before failing the stream."""
        proxy = self._create_mock_proxy()
        mock_response = self._create_mock_upstream_response()

        mock_request = MagicMock()
        proxy.http_client.build_request = MagicMock(return_value=mock_request)
        attempts = {"count": 0}

        async def flaky_send(*args, **kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise httpx.ConnectTimeout("timed out")
            return mock_response

        proxy.http_client.send = AsyncMock(side_effect=flaky_send)

        result = await proxy._stream_response(
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": "sk-test"},
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            request_id="test-retry",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        chunks = []
        async for chunk in result.body_iterator:
            chunks.append(chunk)

        assert attempts["count"] == 2
        assert chunks

    @pytest.mark.asyncio
    async def test_codex_rate_limit_headers_captured_and_forwarded_in_streaming(self):
        """Codex x-codex-* headers must refresh /stats state AND reach the client.

        Regression guard for the bug where Codex session/weekly usage never
        updated on the streaming SSE transport: the proxy neither captured the
        ``x-codex-*`` headers into ``CodexRateLimitState`` nor forwarded them to
        the client (the old ``"ratelimit" in k`` filter dropped them, so the
        Codex CLI's own usage display also went stale).
        """
        from headroom.subscription.codex_rate_limits import get_codex_rate_limit_state

        state = get_codex_rate_limit_state()

        proxy = self._create_mock_proxy()
        mock_response = self._create_mock_upstream_response(
            extra_headers={
                "x-codex-primary-used-percent": "42.0",
                "x-codex-primary-window-minutes": "300",
                "x-codex-secondary-used-percent": "8.0",
                "x-codex-secondary-window-minutes": "10080",
                "x-codex-limit-name": "gpt-5.4-codex",
            }
        )

        mock_request = MagicMock()
        proxy.http_client.build_request = MagicMock(return_value=mock_request)
        proxy.http_client.send = AsyncMock(return_value=mock_response)

        result = await proxy._stream_response(
            url="https://chatgpt.com/backend-api/codex/responses",
            headers={"authorization": "Bearer sk-test"},
            body={"model": "gpt-5.4", "stream": True, "input": "hi"},
            provider="openai",
            model="gpt-5.4",
            request_id="test-codex-sse",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        # 1. Rate-limit state refreshed from the *streaming* response.
        snap = state.latest
        assert snap is not None
        assert snap.primary is not None
        assert snap.primary.used_percent == 42.0
        assert snap.primary.window_minutes == 300
        assert snap.secondary is not None
        assert snap.secondary.used_percent == 8.0
        assert snap.secondary.window_minutes == 10080
        assert snap.limit_name == "gpt-5.4-codex"

        # 2. x-codex headers forwarded so the Codex CLI's native usage display
        #    keeps working through the proxy on the streaming path.
        assert result.headers.get("x-codex-primary-used-percent") == "42.0"
        assert result.headers.get("x-codex-limit-name") == "gpt-5.4-codex"
        # 3. Generic ratelimit headers and the request-id family forwarded;
        #    other unrelated headers dropped.
        assert result.headers.get("anthropic-ratelimit-tokens-limit") == "80000"
        assert result.headers.get("x-request-id") == "req-12345"

    @pytest.mark.asyncio
    async def test_codex_rate_limit_captured_on_streaming_429(self):
        """A streaming 429 carrying x-codex-* must still refresh /stats.

        The capture runs *before* the >=400 early-return, matching the
        non-streaming HTTP handlers (which capture on all statuses). A 429 is
        exactly when the session/weekly windows are most worth surfacing, so the
        previous success-only placement left the most important update missing.
        """
        from headroom.subscription.codex_rate_limits import get_codex_rate_limit_state

        state = get_codex_rate_limit_state()

        proxy = self._create_mock_proxy()
        mock_response = self._create_mock_upstream_response()
        mock_response.status_code = 429
        mock_response.headers = httpx.Headers(
            {
                "content-type": "application/json",
                "x-codex-primary-used-percent": "99.5",
                "x-codex-primary-window-minutes": "300",
            }
        )
        mock_response.aread = AsyncMock(return_value=b'{"error":{"message":"rate limited"}}')
        mock_response.aclose = AsyncMock()

        mock_request = MagicMock()
        proxy.http_client.build_request = MagicMock(return_value=mock_request)
        proxy.http_client.send = AsyncMock(return_value=mock_response)

        result = await proxy._stream_response(
            url="https://chatgpt.com/backend-api/codex/responses",
            headers={"authorization": "Bearer sk-test"},
            body={"model": "gpt-5.4", "stream": True, "input": "hi"},
            provider="openai",
            model="gpt-5.4",
            request_id="test-codex-429",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        assert result.status_code == 429
        snap = state.latest
        assert snap is not None
        assert snap.primary is not None
        assert snap.primary.used_percent == 99.5

    @pytest.mark.asyncio
    async def test_anthropic_stream_leaves_codex_state_untouched(self):
        """The now-unconditional capture must be a no-op for non-Codex streams."""
        from headroom.subscription.codex_rate_limits import get_codex_rate_limit_state

        state = get_codex_rate_limit_state()

        proxy = self._create_mock_proxy()
        mock_response = self._create_mock_upstream_response()  # anthropic-ratelimit-* only

        mock_request = MagicMock()
        proxy.http_client.build_request = MagicMock(return_value=mock_request)
        proxy.http_client.send = AsyncMock(return_value=mock_response)

        await proxy._stream_response(
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": "sk-test"},
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            request_id="test-anthropic-noop",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
        )

        assert state.latest is None
