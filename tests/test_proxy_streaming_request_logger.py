"""Tests that the Anthropic streaming finalizer logs requests for the feed.

Without this, the streaming Anthropic path (which is what Claude Code uses)
silently bypassed the request logger, leaving `/stats.recent_requests` and
`/transformations/feed` permanently empty even when `--log-messages` was set.
The non-streaming Anthropic path and the Bedrock streaming path were the
only ones that called `self.logger.log(...)`.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from headroom.proxy.request_logger import RequestLogger
from headroom.proxy.server import HeadroomProxy


def _build_proxy_with_real_logger(*, log_full_messages: bool) -> HeadroomProxy:
    """Build a HeadroomProxy with mocks for everything except the request logger,
    so we can assert what actually gets recorded."""
    proxy = object.__new__(HeadroomProxy)
    proxy.http_client = MagicMock(spec=httpx.AsyncClient)
    proxy.metrics = MagicMock()
    proxy.metrics.record_request = AsyncMock(return_value=None)
    proxy.cost_tracker = MagicMock()
    proxy.cost_tracker.record_tokens.return_value = None
    proxy.memory_manager = None
    proxy.memory_handler = None
    proxy._config = MagicMock()
    proxy._config.log_full_messages = log_full_messages
    proxy._config.ccr_inject_tool = False
    proxy.config = proxy._config
    proxy.logger = RequestLogger(log_file=None, log_full_messages=log_full_messages)
    return proxy


def _stream_state(output_tokens: int = 42) -> dict:
    return {
        "output_tokens": output_tokens,
        "total_bytes": 200,
        "ttfb_ms": 35.0,
        "input_tokens": 1000,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_creation_ephemeral_5m_input_tokens": 0,
        "cache_creation_ephemeral_1h_input_tokens": 0,
        "sse_buffer": "",
    }


def test_parse_openai_responses_completed_usage_from_sse_buffer():
    proxy = _build_proxy_with_real_logger(log_full_messages=False)
    completed = {
        "type": "response.completed",
        "response": {
            "id": "resp_1",
            "usage": {
                "input_tokens": 844_000,
                "input_tokens_details": {"cached_tokens": 657_400},
                "output_tokens": 6_635,
            },
        },
    }
    state = {
        "sse_buffer": bytearray(
            f"event: response.completed\ndata: {json.dumps(completed)}\n\n".encode()
        )
    }

    usage = proxy._parse_sse_usage_from_buffer(state, "openai")

    assert usage == {
        "input_tokens": 844_000,
        "output_tokens": 6_635,
        "cache_read_input_tokens": 657_400,
    }
    assert state["sse_buffer"] == bytearray()


@pytest.mark.asyncio
async def test_finalize_stream_response_logs_request_for_feed():
    proxy = _build_proxy_with_real_logger(log_full_messages=False)

    await proxy._finalize_stream_response(
        body={"messages": [{"role": "user", "content": "hi"}]},
        provider="anthropic",
        model="claude-sonnet-4-6",
        request_id="req-stream-1",
        original_tokens=1000,
        optimized_tokens=600,
        tokens_saved=400,
        transforms_applied=["smart_crusher"],
        optimization_latency=12.0,
        stream_state=_stream_state(),
        start_time=0.0,
        tags={"stack": "wrap_claude"},
    )

    entries = proxy.logger.get_recent(10)
    assert len(entries) == 1, "streaming finalizer must log exactly one entry per request"
    entry = entries[0]
    assert entry["request_id"] == "req-stream-1"
    assert entry["provider"] == "anthropic"
    assert entry["model"] == "claude-sonnet-4-6"
    assert entry["input_tokens_original"] == 1000
    assert entry["input_tokens_optimized"] == 600
    assert entry["tokens_saved"] == 400
    assert entry["savings_percent"] == pytest.approx(40.0)
    assert entry["transforms_applied"] == ["smart_crusher"]
    assert entry["tags"] == {"stack": "wrap_claude"}
    assert entry["cache_hit"] is False


@pytest.mark.asyncio
async def test_finalize_stream_response_logs_original_and_compressed_messages():
    """With log_full_messages enabled, both sides of the compression are
    recorded: `request_messages` is the pre-compression snapshot the caller
    threads in via `original_messages`, `compressed_messages` is what was
    actually sent upstream (i.e. `body["messages"]` after in-place mutation)."""
    proxy = _build_proxy_with_real_logger(log_full_messages=True)
    # `body["messages"]` models the post-compression list - the proxy mutates
    # `body` in place before calling `_finalize_stream_response`, so this is
    # already what was shipped over the wire.
    body = {"messages": [{"role": "user", "content": "[compressed]"}]}
    original = [{"role": "user", "content": "[original, pre-compression]"}]

    await proxy._finalize_stream_response(
        body=body,
        provider="anthropic",
        model="claude-sonnet-4-6",
        request_id="req-stream-2",
        original_tokens=10,
        optimized_tokens=8,
        tokens_saved=2,
        transforms_applied=[],
        optimization_latency=1.0,
        stream_state=_stream_state(output_tokens=5),
        start_time=0.0,
        original_messages=original,
    )

    entries = proxy.logger.get_recent_with_messages(10)
    assert len(entries) == 1
    assert entries[0]["request_messages"] == original
    assert entries[0]["compressed_messages"] == body["messages"]


@pytest.mark.asyncio
async def test_finalize_stream_response_omits_messages_when_log_full_messages_disabled():
    proxy = _build_proxy_with_real_logger(log_full_messages=False)

    await proxy._finalize_stream_response(
        body={"messages": [{"role": "user", "content": "hello"}]},
        provider="anthropic",
        model="claude-sonnet-4-6",
        request_id="req-stream-3",
        original_tokens=10,
        optimized_tokens=8,
        tokens_saved=2,
        transforms_applied=[],
        optimization_latency=1.0,
        stream_state=_stream_state(output_tokens=5),
        start_time=0.0,
        original_messages=[{"role": "user", "content": "dropped"}],
    )

    entries = proxy.logger.get_recent_with_messages(10)
    assert len(entries) == 1
    # Both sides share the same gate - neither leaks when log_full_messages
    # is off.
    assert entries[0]["request_messages"] is None
    assert entries[0]["compressed_messages"] is None


@pytest.mark.asyncio
async def test_finalize_stream_response_handles_zero_original_tokens():
    proxy = _build_proxy_with_real_logger(log_full_messages=False)

    await proxy._finalize_stream_response(
        body={"messages": []},
        provider="anthropic",
        model="claude-sonnet-4-6",
        request_id="req-stream-4",
        original_tokens=0,
        optimized_tokens=0,
        tokens_saved=0,
        transforms_applied=[],
        optimization_latency=0.0,
        stream_state=_stream_state(output_tokens=0),
        start_time=0.0,
    )

    entries = proxy.logger.get_recent(10)
    assert len(entries) == 1
    assert entries[0]["savings_percent"] == 0


@pytest.mark.asyncio
async def test_finalize_openai_responses_stream_uses_provider_usage_for_dashboard():
    proxy = _build_proxy_with_real_logger(log_full_messages=False)
    state = _stream_state(output_tokens=6_635)
    state["input_tokens"] = 844_000
    state["cache_read_input_tokens"] = 657_400

    await proxy._finalize_stream_response(
        body={"model": "gpt-5.5", "input": [{"type": "message", "role": "user"}]},
        provider="openai",
        model="gpt-5.5",
        request_id="req-openai-responses-stream",
        original_tokens=0,
        optimized_tokens=0,
        tokens_saved=663_000,
        transforms_applied=["openai_responses_live_zone"],
        optimization_latency=26.0,
        stream_state=state,
        start_time=0.0,
    )

    entries = proxy.logger.get_recent(10)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["input_tokens_optimized"] == 844_000
    assert entry["input_tokens_original"] == 1_507_000
    assert entry["tokens_saved"] == 663_000
    assert entry["savings_percent"] == pytest.approx(663_000 / 1_507_000 * 100)
    assert entry["output_tokens"] == 6_635

    proxy.metrics.record_request.assert_awaited_once()
    metrics_kwargs = proxy.metrics.record_request.await_args.kwargs
    assert metrics_kwargs["input_tokens"] == 844_000
    assert metrics_kwargs["output_tokens"] == 6_635
    assert metrics_kwargs["tokens_saved"] == 663_000
    assert metrics_kwargs["cache_read_tokens"] == 657_400
    assert metrics_kwargs["uncached_input_tokens"] == 186_600

    proxy.cost_tracker.record_tokens.assert_called_once()
    cost_args, cost_kwargs = proxy.cost_tracker.record_tokens.call_args
    assert cost_args[:3] == ("gpt-5.5", 663_000, 844_000)
    assert cost_kwargs["cache_read_tokens"] == 657_400
    assert cost_kwargs["uncached_tokens"] == 186_600


@pytest.mark.asyncio
async def test_finalize_stream_response_recovers_usage_from_truncated_buffer() -> None:
    """When upstream truncates mid-event (no trailing \\n\\n), the per-chunk
    parser leaves the message_start usage event sitting in sse_buffer and
    PERF logs cache_read=cache_write=0 — which then poisons the freeze
    heuristic on the next request. The finalizer must flush the residual
    buffer so the real cache_read / cache_creation tokens still land in
    the log even on aborted streams.
    """
    proxy = _build_proxy_with_real_logger(log_full_messages=False)

    partial_message_start = (
        b"event: message_start\n"
        b'data: {"type":"message_start","message":{"id":"msg_x",'
        b'"type":"message","role":"assistant","model":"claude-sonnet-4-6",'
        b'"content":[],"stop_reason":null,"usage":{'
        b'"input_tokens":1234,"cache_read_input_tokens":50000,'
        b'"cache_creation_input_tokens":2500,"output_tokens":1}}}'
    )

    state = {
        "output_tokens": None,
        "total_bytes": len(partial_message_start),
        "ttfb_ms": 35.0,
        "input_tokens": None,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_creation_ephemeral_5m_input_tokens": 0,
        "cache_creation_ephemeral_1h_input_tokens": 0,
        "sse_buffer": bytearray(partial_message_start),
    }

    await proxy._finalize_stream_response(
        body={"messages": [{"role": "user", "content": "hi"}]},
        provider="anthropic",
        model="claude-sonnet-4-6",
        request_id="req-stream-truncated",
        original_tokens=2000,
        optimized_tokens=1800,
        tokens_saved=200,
        transforms_applied=[],
        optimization_latency=5.0,
        stream_state=state,
        start_time=0.0,
    )

    assert state["input_tokens"] == 1234
    assert state["cache_read_input_tokens"] == 50000
    assert state["cache_creation_input_tokens"] == 2500


@pytest.mark.asyncio
async def test_finalize_stream_response_no_op_when_logger_disabled():
    proxy = _build_proxy_with_real_logger(log_full_messages=False)
    proxy.logger = None  # `--no-log-requests` would put us here

    # Should not raise.
    await proxy._finalize_stream_response(
        body={"messages": []},
        provider="anthropic",
        model="claude-sonnet-4-6",
        request_id="req-stream-5",
        original_tokens=10,
        optimized_tokens=8,
        tokens_saved=2,
        transforms_applied=[],
        optimization_latency=1.0,
        stream_state=_stream_state(),
        start_time=0.0,
    )
