"""Image compression must stay off the event loop and behind the isolation runner.

The handlers no longer call `compressor.compress(...)` on the thread pool. They
delegate to `run_image_compression_isolated(...)`, which moves the native image
stack into a spawned subprocess so OpenCV crashes fail open without taking down
the proxy.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from headroom.proxy import image_isolation
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.handlers.openai import OpenAIHandlerMixin


@pytest.fixture(autouse=True)
def _reset_image_pool() -> None:
    image_isolation._reset_image_pool()
    try:
        yield
    finally:
        image_isolation._reset_image_pool()
        image_isolation._IMAGE_WORKER = image_isolation._compress_messages_worker


def test_image_blocks_use_isolation_runner_and_fail_open() -> None:
    for mixin, method in (
        (AnthropicHandlerMixin, "handle_anthropic_messages"),
        (OpenAIHandlerMixin, "handle_openai_chat"),
    ):
        fn = getattr(mixin, method)
        assert inspect.iscoroutinefunction(fn), f"{method} must be async to await the isolation"
        src = inspect.getsource(fn)
        assert "run_image_compression_isolated(" in src, f"{method}: isolation runner missing"
        assert "COMPRESSION_TIMEOUT_SECONDS" in src, f"{method}: isolation missing a timeout"
        assert "Image compression failed" in src, f"{method}: image compress not fail-open"


async def test_image_isolation_keeps_event_loop_responsive() -> None:
    ticks = 0
    image_isolation._IMAGE_WORKER = image_isolation._sleep_worker

    async def _ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    tick_task = asyncio.create_task(_ticker())
    try:
        returned, result = await image_isolation.run_image_compression_isolated(
            [{"role": "user", "content": "image payload"}],
            provider="openai",
            timeout=1.0,
        )
    finally:
        tick_task.cancel()

    assert returned == [{"role": "user", "content": "image payload"}]
    assert result is None
    assert ticks >= 5
