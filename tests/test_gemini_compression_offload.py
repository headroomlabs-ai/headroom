"""Gemini compression offload (perf): the 3 Gemini handlers must run the CPU-bound
`openai_pipeline.apply()` on the compression executor, not inline on the event loop.

The wiring (each handler awaits `_run_compression_in_executor(lambda: apply(...))`) mirrors
the proven openai/anthropic paths; these tests assert the two observable properties that
wiring delivers — apply runs on a worker thread, and the loop stays responsive during a
slow compression — plus a sanity check that the handlers are async and import the timeout.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time

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


def test_gemini_handlers_are_async_and_import_the_timeout() -> None:
    """Wiring sanity: the offload uses `await`, so the handlers must be coroutines, and the
    timeout constant must be importable in the module (a missing import would NameError)."""
    from headroom.proxy.handlers import gemini

    for name in (
        "handle_gemini_generate_content",
        "handle_google_cloudcode_stream",
        "handle_gemini_count_tokens",
    ):
        fn = getattr(gemini.GeminiHandlerMixin, name)
        assert inspect.iscoroutinefunction(fn), f"{name} must be async to await the offload"

    assert hasattr(gemini, "COMPRESSION_TIMEOUT_SECONDS")


async def test_compression_offload_runs_on_worker_thread() -> None:
    """apply() runs on a 'headroom-compress' executor thread, not the event-loop thread."""
    proxy = _make_proxy()
    loop_thread_name = threading.current_thread().name
    seen: dict[str, str] = {}

    def _slow_apply() -> str:
        seen["thread"] = threading.current_thread().name
        time.sleep(0.1)
        return "compressed"

    result = await proxy._run_compression_in_executor(_slow_apply, timeout=10)

    assert result == "compressed"
    assert seen["thread"].startswith("headroom-compress")
    assert seen["thread"] != loop_thread_name


async def test_compression_offload_keeps_event_loop_responsive() -> None:
    """While a slow compression runs on the executor, the loop keeps scheduling coroutines.
    A bare sync apply() on the loop (the bug this fixes) would starve them to ~0 ticks."""
    proxy = _make_proxy()
    ticks = 0

    async def _ticker() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    def _slow_apply() -> str:
        time.sleep(0.3)
        return "x"

    tick_task = asyncio.create_task(_ticker())
    try:
        result = await proxy._run_compression_in_executor(_slow_apply, timeout=10)
    finally:
        tick_task.cancel()

    assert result == "x"
    # ~30 ticks expected at 10ms over 0.3s; a blocked loop would yield near zero.
    assert ticks >= 5
