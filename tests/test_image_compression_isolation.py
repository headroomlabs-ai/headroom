from __future__ import annotations

import asyncio
import inspect
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


async def test_worker_cancelled_while_awaiting_fails_open(monkeypatch) -> None:
    # A future cancelled by a pool teardown surfaces as asyncio.CancelledError
    # (a BaseException since 3.8), which the call sites' ``except Exception``
    # never caught. The runner must forward the payload instead of raising.
    async def _cancelled(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(image_isolation.asyncio, "wait_for", _cancelled)
    messages = _messages()

    returned, result = await image_isolation.run_image_compression_isolated(
        messages, provider="openai", timeout=5.0
    )

    assert returned is messages
    assert result is None


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
