"""Independent byte and terminal oracles for the HTTP stream observer."""

import asyncio
import time

import pytest

from headroom.proxy.gateway.config import LimitsConfig
from headroom.proxy.gateway.errors import GatewayPublicError


async def chunks(*parts):
    for part in parts:
        yield part


@pytest.mark.asyncio
async def test_native_frame_observer_all_split_points():
    from headroom.proxy.gateway.streaming import StreamObserver

    literal = (
        ': keepalive\r\n\r\ndata: {"choices":[{"delta":{"content":"雪",'
        '"tool_calls":[{"function":{"arguments":"{}"}}]},\r\n'
        'data: "finish_reason":"tool_calls"}],"usage":{"prompt_tokens":3,'
        '"completion_tokens":2}}\r\n\r\ndata: [DONE]\r\n\r\n'
    ).encode()
    for split in range(len(literal) + 1):
        observer = StreamObserver("openai-chat", LimitsConfig(), time.monotonic() + 5)
        received = b"".join(
            [part async for part in observer.observe(chunks(literal[:split], literal[split:]))]
        )
        assert received == literal
        assert observer.terminal == "success"
        assert (observer.usage.input_tokens, observer.usage.output_tokens) == (3, 2)
    # Bound individual frames, not a network chunk containing many complete frames.
    observer = StreamObserver(
        "openai-chat", LimitsConfig(max_frame_bytes=1024), time.monotonic() + 5
    )
    large = b": heartbeat\n\n" * 1000 + literal
    assert b"".join([part async for part in observer.observe(chunks(large))]) == large


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", [b"", b"data: {", b"data: \xe9"])
async def test_eof_requires_protocol_terminal(tail):
    from headroom.proxy.gateway.streaming import StreamObserver

    observer = StreamObserver("openai-chat", LimitsConfig(), time.monotonic() + 5)
    with pytest.raises(GatewayPublicError, match="stream"):
        _ = [part async for part in observer.observe(chunks(b'data: {"choices":[]}\n\n', tail))]
    assert observer.terminal != "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("part", [b": heartbeat\n\n", b"data: {"])
async def test_pending_read_deadline_needs_no_new_bytes(part):
    from headroom.proxy.gateway.streaming import StreamObserver

    closed = asyncio.Event()

    async def stalled():
        try:
            yield part
            await asyncio.Event().wait()
        finally:
            closed.set()

    observer = StreamObserver(
        "openai-chat",
        LimitsConfig(stream_content_idle_seconds=0.03, partial_frame_seconds=0.03),
        time.monotonic() + 1,
    )
    with pytest.raises(GatewayPublicError, match="stream"):
        _ = [part async for part in observer.observe(stalled())]
    assert closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "literal", [b"data: {broken}\n\n", b"data: \xff\n\n", b"data: " + b"x" * 1030]
)
async def test_malformed_and_oversized_frames_fail(literal):
    from headroom.proxy.gateway.streaming import StreamObserver

    observer = StreamObserver(
        "openai-chat", LimitsConfig(max_frame_bytes=1024), time.monotonic() + 1
    )
    with pytest.raises(GatewayPublicError):
        _ = [part async for part in observer.observe(chunks(literal))]


@pytest.mark.asyncio
async def test_error_events_never_expose_provider_text():
    from headroom.proxy.gateway.streaming import StreamObserver

    observer = StreamObserver("openai-responses", LimitsConfig(), time.monotonic() + 1)
    with pytest.raises(GatewayPublicError) as exc:
        _ = [
            part
            async for part in observer.observe(
                chunks(
                    b'data: {"type":"response.failed","response":{"error":{"message":"SECRET"}}}\n\n'
                )
            )
        ]
    assert "SECRET" not in str(exc.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "protocol,terminal",
    [
        ("openai-responses", b'data: {"type":"response.completed"}\n\n'),
        ("anthropic-messages", b'data: {"type":"message_stop"}\n\n'),
        (
            "gemini-generate",
            b'data: {"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}\n\n',
        ),
        (
            "vertex-generate",
            b'data: {"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":1,"candidatesTokenCount":1}}\n\n',
        ),
    ],
)
async def test_terminal_closes_without_waiting_for_provider_eof(protocol, terminal):
    from headroom.proxy.gateway.streaming import StreamObserver

    closed = asyncio.Event()

    async def upstream():
        try:
            yield terminal
            await asyncio.Event().wait()
        finally:
            closed.set()

    observer = StreamObserver(
        protocol, LimitsConfig(stream_content_idle_seconds=0.02), time.monotonic() + 1
    )
    assert b"".join([part async for part in observer.observe(upstream())]) == terminal
    assert closed.is_set()


@pytest.mark.asyncio
async def test_content_idle_budget_starts_when_observation_starts() -> None:
    from headroom.proxy.gateway.streaming import StreamObserver

    terminal = b'data: {"type":"message_stop"}\n\n'
    observer = StreamObserver(
        "anthropic-messages",
        LimitsConfig(stream_content_idle_seconds=0.01),
        time.monotonic() + 1,
    )
    await asyncio.sleep(0.02)

    assert b"".join([part async for part in observer.observe(chunks(terminal))]) == terminal


@pytest.mark.asyncio
async def test_first_read_idle_budget_starts_when_provider_wait_begins(monkeypatch) -> None:
    import headroom.proxy.gateway.streaming as streaming

    class PreemptedClock:
        def __init__(self) -> None:
            self.calls = 0

        def monotonic(self) -> float:
            self.calls += 1
            return 100.0 if self.calls < 3 else 100.021

    monkeypatch.setattr(streaming, "time", PreemptedClock())
    terminal = b'data: {"type":"message_stop"}\n\n'
    observer = streaming.StreamObserver(
        "anthropic-messages",
        LimitsConfig(stream_content_idle_seconds=0.02),
        101.0,
    )

    assert b"".join([part async for part in observer.observe(chunks(terminal))]) == terminal


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "protocol,terminal",
    [
        (
            "openai-chat",
            b'data: {"choices":[{"finish_reason":"content_filter"}]}\n\ndata: [DONE]\n\n',
        ),
        (
            "anthropic-messages",
            b'data: {"type":"message_delta","delta":{"stop_reason":"refusal"}}\n\ndata: {"type":"message_stop"}\n\n',
        ),
        ("gemini-generate", b'data: {"candidates":[{"finishReason":"SAFETY"}]}\n\n'),
    ],
)
async def test_incomplete_and_refusal_are_not_success(protocol, terminal):
    from headroom.proxy.gateway.streaming import StreamObserver

    observer = StreamObserver(protocol, LimitsConfig(), time.monotonic() + 1)
    with pytest.raises(GatewayPublicError):
        _ = [part async for part in observer.observe(chunks(terminal))]
    assert observer.terminal == "failed"
