from __future__ import annotations

import asyncio
import inspect
import threading
import time

import pytest

from headroom.proxy import image_isolation
from headroom.proxy.handlers.anthropic import AnthropicHandlerMixin
from headroom.proxy.handlers.openai import OpenAIHandlerMixin


@pytest.fixture(autouse=True)
def _reset_image_pool() -> None:
    image_isolation._reset_image_pool()
    image_isolation._IMAGE_POOL_INFLIGHT = 0
    try:
        yield
    finally:
        image_isolation._reset_image_pool()
        image_isolation._IMAGE_POOL_INFLIGHT = 0
        image_isolation._IMAGE_WORKER = image_isolation._compress_messages_worker


def _messages() -> list[dict[str, object]]:
    return [{"role": "user", "content": "image payload"}]


async def test_worker_sigsegv_fails_open_parent_survives(monkeypatch) -> None:
    messages = _messages()
    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._abort_worker)

    returned, result = await image_isolation.run_image_compression_isolated(
        messages,
        provider="openai",
        timeout=5.0,
    )

    assert returned is messages
    assert result is None

    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._success_worker)
    recovered, recovered_result = await image_isolation.run_image_compression_isolated(
        messages,
        provider="openai",
        timeout=5.0,
    )

    assert recovered[-1]["content"] == "compressed:openai"
    assert recovered_result is not None


async def test_worker_exception_fails_open(monkeypatch) -> None:
    messages = _messages()
    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._raise_worker)

    returned, result = await image_isolation.run_image_compression_isolated(
        messages,
        provider="anthropic",
        timeout=5.0,
    )

    assert returned is messages
    assert result is None


async def test_worker_timeout_fails_open(monkeypatch) -> None:
    messages = _messages()
    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._sleep_worker)

    returned, result = await image_isolation.run_image_compression_isolated(
        messages,
        provider="openai",
        timeout=0.05,
    )

    assert returned is messages
    assert result is None

    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._success_worker)
    recovered, recovered_result = await image_isolation.run_image_compression_isolated(
        messages,
        provider="openai",
        timeout=5.0,
    )

    assert recovered[-1]["content"] == "compressed:openai"
    assert recovered_result is not None


async def test_worker_timeout_kills_the_abandoned_worker(monkeypatch) -> None:
    # shutdown(wait=False) alone left the worker running the abandoned call to
    # the end while the next request cold-started a fresh worker beside it.
    messages = _messages()
    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._success_worker)
    await image_isolation.run_image_compression_isolated(messages, provider="openai", timeout=30.0)
    (worker,) = image_isolation._IMAGE_POOL._processes.values()

    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._hang_worker)
    returned, result = await image_isolation.run_image_compression_isolated(
        messages,
        provider="openai",
        timeout=0.5,
    )

    assert returned is messages
    assert result is None
    worker.join(timeout=5.0)
    assert not worker.is_alive()


def _interference_worker(
    messages: list[dict[str, object]],
    provider: str,
    deadline: float | None = None,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    if messages[-1]["content"] == "slow":
        time.sleep(0.3)
        raise RuntimeError("boom")
    return [{"role": "user", "content": "compressed:" + str(messages[-1]["content"])}], {
        "technique": "preserve"
    }


async def test_one_timeout_does_not_destroy_a_concurrent_request(monkeypatch) -> None:
    # #4418: request A times out while request B's image job is still queued on
    # the single shared worker. A's reset used to shutdown(cancel_futures=True)
    # and kill the worker, so B lost its compression and failed open. A timeout
    # in one call must leave the other request's job alone.
    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", _interference_worker)
    slow = [{"role": "user", "content": "slow"}]
    fast = [{"role": "user", "content": "fast"}]

    async def _call(messages: list[dict[str, object]], timeout: float):
        return await image_isolation.run_image_compression_isolated(
            messages, provider="openai", timeout=timeout
        )

    a_task = asyncio.create_task(_call(slow, 0.1))
    await asyncio.sleep(0.03)
    b_task = asyncio.create_task(_call(fast, 5.0))

    a_returned, a_result = await a_task
    b_returned, b_result = await b_task

    # A failed open: its own original payload, no result.
    assert a_returned is slow
    assert a_result is None

    # B was not harmed: it still got its compressed payload.
    assert b_returned[-1]["content"] == "compressed:fast"
    assert b_result == {"technique": "preserve"}


async def test_executor_future_cancellation_fails_open(monkeypatch) -> None:
    # A future cancelled by a pool teardown (shutdown(cancel_futures=True))
    # must fail open on Python 3.10 as well as 3.11+. Task.cancelling() does
    # not exist on 3.10; discriminating via that API leaked AttributeError
    # instead of forwarding the original payload.
    messages = _messages()
    loop = asyncio.get_running_loop()
    cancelled: asyncio.Future[object] = loop.create_future()
    cancelled.cancel()
    monkeypatch.setattr(loop, "run_in_executor", lambda *args, **kwargs: cancelled)

    returned, result = await image_isolation.run_image_compression_isolated(
        messages, provider="openai", timeout=5.0
    )

    assert returned is messages
    assert result is None


async def test_caller_cancellation_propagates(monkeypatch) -> None:
    # Genuine task cancellation must still propagate, including on Python 3.10
    # where Task.cancelling() is unavailable.
    entered = asyncio.Event()
    orig_wait_for = image_isolation.asyncio.wait_for

    async def _tapping_wait_for(aw, *args, **kwargs):
        entered.set()
        return await orig_wait_for(aw, *args, **kwargs)

    monkeypatch.setattr(image_isolation.asyncio, "wait_for", _tapping_wait_for)
    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._hang_worker)

    messages = _messages()
    task = asyncio.create_task(
        image_isolation.run_image_compression_isolated(messages, provider="openai", timeout=30.0)
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_acquire_counts_before_returning_the_pool() -> None:
    # Acquire must publish the pool and increment inflight under one lock so a
    # concurrent reset cannot unpublished it as idle before this call is counted.
    pool = image_isolation._acquire_image_pool()
    try:
        assert pool is image_isolation._IMAGE_POOL
        assert image_isolation._IMAGE_POOL_INFLIGHT == 1
    finally:
        image_isolation._release_image_pool(pool)
    assert image_isolation._IMAGE_POOL_INFLIGHT == 0


def test_release_of_unpublished_pool_does_not_underflow_replacement() -> None:
    old = image_isolation._acquire_image_pool()
    image_isolation._reset_image_pool(old)
    replacement = image_isolation._acquire_image_pool()
    try:
        assert replacement is not old
        assert image_isolation._IMAGE_POOL_INFLIGHT == 1
        image_isolation._release_image_pool(old)
        assert image_isolation._IMAGE_POOL_INFLIGHT == 1
    finally:
        image_isolation._release_image_pool(replacement)


def test_concurrent_event_loop_threads_keep_compression(monkeypatch) -> None:
    # Separate event-loop threads share the pool lock. A timeout on one thread
    # used to reset the pool after the other had already observed it but before
    # its inflight count, so that call submitted to a shut-down pool and
    # forwarded the image uncompressed.
    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", _interference_worker)
    slow = [{"role": "user", "content": "slow"}]
    fast = [{"role": "user", "content": "fast"}]
    outcomes: dict[str, tuple[list[dict[str, object]], dict[str, object] | None]] = {}
    errors: list[BaseException] = []

    def _run(name: str, messages: list[dict[str, object]], timeout: float) -> None:
        async def _call():
            return await image_isolation.run_image_compression_isolated(
                messages, provider="openai", timeout=timeout
            )

        try:
            outcomes[name] = asyncio.run(_call())
        except Exception as exc:
            errors.append(exc)

    slow_thread = threading.Thread(target=_run, args=("slow", slow, 0.1))
    fast_thread = threading.Thread(target=_run, args=("fast", fast, 5.0))
    slow_thread.start()
    time.sleep(0.03)
    fast_thread.start()
    slow_thread.join(timeout=10)
    fast_thread.join(timeout=10)

    assert errors == []
    a_returned, a_result = outcomes["slow"]
    b_returned, b_result = outcomes["fast"]
    assert a_returned is slow
    assert a_result is None
    assert b_returned[-1]["content"] == "compressed:fast"
    assert b_result == {"technique": "preserve"}


def test_late_failure_leaves_a_rebuilt_pool_alone() -> None:
    # A call that fails after another request already replaced its pool must
    # not drop (and kill the worker of) the replacement.
    old = image_isolation._image_pool()
    image_isolation._reset_image_pool(old)
    rebuilt = image_isolation._image_pool()

    image_isolation._reset_image_pool(old)

    assert image_isolation._IMAGE_POOL is rebuilt


async def test_worker_success_returns_compressed(monkeypatch) -> None:
    messages = _messages()
    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", image_isolation._success_worker)

    returned, result = await image_isolation.run_image_compression_isolated(
        messages,
        provider="anthropic",
        timeout=5.0,
    )

    assert returned[-1]["content"] == "compressed:anthropic"
    assert result == {
        "technique": "preserve",
        "original_tokens": 100,
        "compressed_tokens": 60,
        "confidence": 1.0,
        "savings_percent": 40.0,
    }


def test_success_marks_mutation_and_logs() -> None:
    anthropic_src = inspect.getsource(AnthropicHandlerMixin.handle_anthropic_messages)
    openai_src = inspect.getsource(OpenAIHandlerMixin.handle_openai_chat)

    assert "run_image_compression_isolated(" in anthropic_src
    assert 'body_mutation_tracker.mark_mutated("image_compression")' in anthropic_src
    assert "Image compression:" in anthropic_src

    assert "run_image_compression_isolated(" in openai_src
    assert "Image:" in openai_src
