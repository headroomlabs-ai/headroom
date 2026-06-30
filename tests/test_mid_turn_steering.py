import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from headroom.proxy.server import HeadroomProxy


class TestMidTurnSteering:
    def test_mid_turn_queue_exists_on_streaming_mixin(self):
        """StreamingMixin has _mid_turn_queues class attribute after the fix."""
        from headroom.proxy.handlers.streaming import StreamingMixin

        assert hasattr(StreamingMixin, "_mid_turn_queues")
        assert hasattr(StreamingMixin, "_active_streams")

    def test_mid_turn_message_queued_when_stream_active(self):
        """When a session has an active stream, mid-turn messages are queued."""
        from headroom.proxy.handlers.streaming import StreamingMixin

        mixin = StreamingMixin()
        session_key = "test-session-123"
        mixin._active_streams.add(session_key)
        body = {"messages": [{"role": "user", "content": "follow-up"}]}
        result = mixin._queue_mid_turn_message(session_key, body)
        assert result["status"] == 202
        assert result["event"] == "headroom_queued"
        assert not mixin._mid_turn_queues[session_key].empty()
        queued = mixin._mid_turn_queues[session_key].get_nowait()
        assert queued == body
        # Cleanup
        mixin._active_streams.discard(session_key)
        del mixin._mid_turn_queues[session_key]

    def test_no_queue_when_no_prior_stream(self):
        """When no stream is active, _mid_turn_queues stays empty for the session."""
        from headroom.proxy.handlers.streaming import StreamingMixin

        mixin = StreamingMixin()
        session_key = "inactive-session"
        assert session_key not in mixin._active_streams
        assert session_key not in mixin._mid_turn_queues

    def _create_mock_proxy(self):
        proxy = object.__new__(HeadroomProxy)
        proxy.http_client = MagicMock(spec=httpx.AsyncClient)
        proxy._config = MagicMock()
        proxy._config.memory_enabled = False
        proxy._config.ccr_inject_tool = False
        proxy._config.retry_max_attempts = 1
        proxy._config.retry_base_delay_ms = 0
        proxy._config.retry_max_delay_ms = 0
        proxy.config = proxy._config
        proxy.memory_handler = None
        proxy._parse_sse_usage_from_buffer = MagicMock(return_value=None)
        proxy._finalize_stream_response = AsyncMock(return_value=None)
        return proxy

    @staticmethod
    def _create_mock_upstream_response(
        chunks: list[bytes], *, terminal_exception: BaseException | None = None
    ):
        mock_response = AsyncMock()
        mock_response.headers = httpx.Headers({"content-type": "text/event-stream"})
        mock_response.status_code = 200

        async def aiter_bytes():
            for chunk in chunks:
                yield chunk
            if terminal_exception is not None:
                raise terminal_exception

        mock_response.aiter_bytes = aiter_bytes
        mock_response.aclose = AsyncMock()
        return mock_response

    @pytest.mark.asyncio
    async def test_mid_turn_stream_cancellation_clears_active_session_and_queue(self):
        proxy = self._create_mock_proxy()
        session_key = "cancelled-session"
        mock_response = self._create_mock_upstream_response(
            [
                b'event: message_start\ndata: {"type":"message_start"}\n\n',
            ],
            terminal_exception=asyncio.CancelledError(),
        )

        proxy.http_client.build_request = MagicMock(return_value=MagicMock())
        proxy.http_client.send = AsyncMock(return_value=mock_response)

        result = await proxy._stream_response(
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": "sk-test", "x-headroom-session-id": session_key},
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            request_id="test-cancelled",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
            session_key=session_key,
        )
        proxy._queue_mid_turn_message(
            session_key,
            {"messages": [{"role": "user", "content": "follow-up"}]},
        )

        try:
            with pytest.raises(asyncio.CancelledError):
                async for _chunk in result.body_iterator:
                    pass
            assert session_key not in proxy._active_streams
            assert session_key not in proxy._mid_turn_queues
            mock_response.aclose.assert_awaited_once()
        finally:
            proxy._active_streams.discard(session_key)
            proxy._mid_turn_queues.pop(session_key, None)

    @pytest.mark.asyncio
    async def test_mid_turn_stream_exception_clears_active_session_and_queue(self):
        proxy = self._create_mock_proxy()
        session_key = "errored-session"
        mock_response = self._create_mock_upstream_response(
            [
                b'event: message_start\ndata: {"type":"message_start"}\n\n',
            ],
            terminal_exception=RuntimeError("stream exploded"),
        )

        proxy.http_client.build_request = MagicMock(return_value=MagicMock())
        proxy.http_client.send = AsyncMock(return_value=mock_response)

        result = await proxy._stream_response(
            url="https://api.anthropic.com/v1/messages",
            headers={"x-api-key": "sk-test", "x-headroom-session-id": session_key},
            body={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 100,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            request_id="test-errored",
            original_tokens=10,
            optimized_tokens=10,
            tokens_saved=0,
            transforms_applied=[],
            tags={},
            optimization_latency=0.0,
            session_key=session_key,
        )
        proxy._queue_mid_turn_message(
            session_key,
            {"messages": [{"role": "user", "content": "follow-up"}]},
        )

        try:
            chunks = [chunk async for chunk in result.body_iterator]
            assert any(b"event: error" in chunk for chunk in chunks)
            assert session_key not in proxy._active_streams
            assert session_key not in proxy._mid_turn_queues
            mock_response.aclose.assert_awaited_once()
        finally:
            proxy._active_streams.discard(session_key)
            proxy._mid_turn_queues.pop(session_key, None)
