from __future__ import annotations

import asyncio
import inspect
import threading
import time
from pathlib import Path

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


def _queued_abandon_worker(
    messages: list[dict[str, object]],
    provider: str,
    deadline: float | None = None,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    payload = messages[-1]["content"]
    if not isinstance(payload, dict):
        raise TypeError(f"expected coordination payload, got {type(payload)!r}")
    kind = payload["kind"]
    if kind == "peer":
        Path(str(payload["started"])).write_text("1", encoding="utf-8", newline="\n")
        release = Path(str(payload["release"]))
        give_up = time.time() + 15
        while not release.exists() and time.time() < give_up:
            time.sleep(0.01)
        Path(str(payload["done"])).write_text("1", encoding="utf-8", newline="\n")
        return [{"role": "user", "content": "compressed:peer"}], {"technique": "preserve"}
    if kind == "filler":
        return [{"role": "user", "content": "compressed:filler"}], {"technique": "preserve"}
    if kind == "target":
        Path(str(payload["ran"])).write_text("1", encoding="utf-8", newline="\n")
        return [{"role": "user", "content": "compressed:target"}], {"technique": "preserve"}
    raise RuntimeError(f"unknown kind:{kind}")


@pytest.mark.parametrize("abandon", ["timeout", "cancel"])
async def test_abandoned_queued_job_never_runs_while_peer_completes(
    monkeypatch, tmp_path, abandon: str
) -> None:
    # A timeout or caller cancel used to complete only the private waiter, so
    # the still-queued executor job could run after the request already failed
    # open. Isolation correctly refuses to tear down the pool while a peer is
    # inflight; this call's pending future must still be cancelled so the
    # abandoned job never runs and the unrelated peer still finishes.
    monkeypatch.setattr(image_isolation, "_IMAGE_WORKER", _queued_abandon_worker)

    started = tmp_path / "peer-started"
    release = tmp_path / "peer-release"
    done = tmp_path / "peer-done"
    ran = tmp_path / "target-ran"
    peer_msgs = [
        {
            "role": "user",
            "content": {
                "kind": "peer",
                "started": str(started),
                "release": str(release),
                "done": str(done),
            },
        }
    ]
    target_msgs = [{"role": "user", "content": {"kind": "target", "ran": str(ran)}}]
    # ProcessPoolExecutor prefetches max_workers + 1 items onto its call
    # queue and marks those futures running, which makes cancel() a no-op.
    # Queue fillers so the abandoned target stays pending behind that prefetch.
    filler_count = 3

    loop = asyncio.get_running_loop()
    real_run_in_executor = loop.run_in_executor
    executor_futures: list[asyncio.Future[object]] = []

    def _track_run_in_executor(*args: object, **kwargs: object):
        future = real_run_in_executor(*args, **kwargs)
        executor_futures.append(future)
        return future

    monkeypatch.setattr(loop, "run_in_executor", _track_run_in_executor)

    peer_task = asyncio.create_task(
        image_isolation.run_image_compression_isolated(peer_msgs, provider="openai", timeout=15.0)
    )
    deadline = time.time() + 5
    while not started.exists() and time.time() < deadline:
        await asyncio.sleep(0.01)
    assert started.exists(), "peer never occupied the shared worker"

    filler_tasks = [
        asyncio.create_task(
            image_isolation.run_image_compression_isolated(
                [{"role": "user", "content": {"kind": "filler", "n": n}}],
                provider="openai",
                timeout=15.0,
            )
        )
        for n in range(filler_count)
    ]
    target_task = asyncio.create_task(
        image_isolation.run_image_compression_isolated(
            target_msgs,
            provider="openai",
            timeout=0.05 if abandon == "timeout" else 15.0,
        )
    )
    expected_jobs = 1 + filler_count + 1
    submitted = time.time() + 5
    while len(executor_futures) < expected_jobs and time.time() < submitted:
        await asyncio.sleep(0.01)
    assert len(executor_futures) >= expected_jobs
    target_future = executor_futures[-1]
    assert not target_future.done()

    if abandon == "timeout":
        target_returned, target_result = await target_task
        assert target_returned is target_msgs
        assert target_result is None
    else:
        target_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await target_task

    assert target_future.cancelled()
    assert not ran.exists()

    release.write_text("1", encoding="utf-8", newline="\n")
    peer_returned, peer_result = await peer_task
    assert peer_returned[-1]["content"] == "compressed:peer"
    assert peer_result == {"technique": "preserve"}
    assert not executor_futures[0].cancelled()

    for filler_returned, filler_result in await asyncio.gather(*filler_tasks):
        assert filler_returned[-1]["content"] == "compressed:filler"
        assert filler_result == {"technique": "preserve"}

    await asyncio.sleep(0.2)
    assert not ran.exists()
    assert done.read_text(encoding="utf-8") == "1"


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
