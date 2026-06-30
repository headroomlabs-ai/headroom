"""Image compression offload (perf): the Anthropic and OpenAI handlers must run the
CPU-bound `ImageCompressor.compress()` on the bounded compression executor, not inline
on the event loop, and fail open if it raises — matching the text-compression path.

Mirrors test_gemini_compression_offload.py. The wiring (each image block awaits
`_run_compression_in_executor(lambda: compressor.compress(...))`) reuses the proven
text path; these tests assert the observable properties that wiring delivers: the
blocks are async + offloaded + fail open, and the executor keeps the loop responsive
while a slow compression runs on a worker thread.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time

from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.handlers.openai import OpenAIHandlerMixin
from headroom.proxy.server import ProxyConfig, create_app


def _make_proxy():  # noqa: ANN202 — returns the internal HeadroomProxy
    app = create_app(
        ProxyConfig(
            optimize=True,
            cache_enabled=False,
            rate_limit_enabled=False,
            cost_tracking_enabled=False,
        )
    )
    return app.state.proxy


def test_image_blocks_offload_compress_and_fail_open() -> None:
    """Each image-compress block must be async, offload compress() onto the executor with a
    timeout, and fail open. Guards a future refactor from silently re-inlining compress()."""
    for mixin, method in (
        (AnthropicHandlerMixin, "handle_anthropic_messages"),
        (OpenAIHandlerMixin, "handle_openai_chat"),
    ):
        fn = getattr(mixin, method)
        assert inspect.iscoroutinefunction(fn), f"{method} must be async to await the offload"
        src = inspect.getsource(fn)
        assert "compressor.compress(" in src, f"{method}: image compress call missing"
        assert "_run_compression_in_executor(" in src, f"{method}: compress not offloaded"
        assert "COMPRESSION_TIMEOUT_SECONDS" in src, f"{method}: offload missing a timeout"
        assert "Image compression failed" in src, f"{method}: image compress not fail-open"


async def test_image_compress_offload_runs_on_worker_thread() -> None:
    """The exact call the image blocks make — _run_compression_in_executor(lambda: compress()) —
    runs compress() on a 'headroom-compress' executor thread, not the event-loop thread."""
    proxy = _make_proxy()
    loop_thread = threading.current_thread().name
    seen: dict[str, str] = {}

    def _slow_compress():  # noqa: ANN202 — stands in for ImageCompressor.compress
        seen["thread"] = threading.current_thread().name
        time.sleep(0.1)
        return [{"role": "user", "content": "compressed"}]

    result = await proxy._run_compression_in_executor(_slow_compress, timeout=10)

    assert result == [{"role": "user", "content": "compressed"}]
    assert seen["thread"].startswith("headroom-compress")
    assert seen["thread"] != loop_thread


async def test_image_compress_offload_keeps_event_loop_responsive() -> None:
    """While a slow image compression runs on the executor, the loop keeps scheduling
    coroutines. The bug this fixes — a bare inline compress() — would starve them to ~0 ticks."""
    proxy = _make_proxy()
    ticks = 0

    async def _ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    def _slow_compress():  # noqa: ANN202
        time.sleep(0.3)
        return "x"

    tick_task = asyncio.create_task(_ticker())
    try:
        result = await proxy._run_compression_in_executor(_slow_compress, timeout=10)
    finally:
        tick_task.cancel()

    assert result == "x"
    assert ticks >= 5  # ~30 expected at 10ms over 0.3s; a blocked loop yields near zero
