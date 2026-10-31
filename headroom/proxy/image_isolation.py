from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from typing import Any

# Share of the call's timeout spent starting new OCR. One OCR already running
# when it runs out still finishes, so the rest of the budget is its margin.
_OCR_BUDGET_FRACTION = 2 / 3

logger = logging.getLogger("headroom.proxy")

_IMAGE_POOL_LOCK = threading.Lock()
_IMAGE_POOL: ProcessPoolExecutor | None = None

# Number of calls currently awaiting a result from the *published*
# ``_IMAGE_POOL``. A timeout-driven reset tears down the whole single-worker
# pool (#4418), so it must only run when the failing call is the *only* one
# using that pool. The counter is paired with the published pool under
# ``_IMAGE_POOL_LOCK``: acquire creates/returns the pool and increments in one
# critical section, and unpublishing zeros the counter so a release of an
# already-reset pool cannot underflow a replacement.
_IMAGE_POOL_INFLIGHT = 0

# Per-worker-process cached compressor. The image pool is a persistent
# single-worker ProcessPoolExecutor, so building a fresh ImageCompressor (and,
# inside it, a fresh ONNX router loading native ort.InferenceSession models) on
# every call accumulated native memory in the worker and grew RSS to 1+ GB over
# a day (#2513). Load the models once per worker and reuse them.
_WORKER_COMPRESSOR: Any = None


def _get_worker_compressor() -> Any:
    global _WORKER_COMPRESSOR
    if _WORKER_COMPRESSOR is None:
        from headroom.image import ImageCompressor

        instance = ImageCompressor()
        # Shared across calls in this worker: don't let a per-call close() unload
        # the models the next call reuses.
        instance._is_singleton = True
        _WORKER_COMPRESSOR = instance
    return _WORKER_COMPRESSOR


def _compress_messages_worker(
    messages: list[dict[str, Any]],
    provider: str,
    deadline: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    compressor = _get_worker_compressor()
    compressed = compressor.compress(messages, provider=provider, deadline=deadline)
    if compressor.last_result is None:
        return compressed, None
    return compressed, {
        "technique": compressor.last_result.technique.value,
        "original_tokens": compressor.last_result.original_tokens,
        "compressed_tokens": compressor.last_result.compressed_tokens,
        "confidence": compressor.last_result.confidence,
        "savings_percent": compressor.last_result.savings_percent,
    }


def _success_worker(
    messages: list[dict[str, Any]],
    provider: str,
    deadline: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    compressed = [dict(message) for message in messages]
    compressed[-1] = {**compressed[-1], "content": f"compressed:{provider}"}
    return compressed, {
        "technique": "preserve",
        "original_tokens": 100,
        "compressed_tokens": 60,
        "confidence": 1.0,
        "savings_percent": 40.0,
    }


def _raise_worker(
    messages: list[dict[str, Any]],
    provider: str,
    deadline: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    raise RuntimeError(f"boom:{provider}")


def _sleep_worker(
    messages: list[dict[str, Any]],
    provider: str,
    deadline: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    time.sleep(0.2)
    return messages, None


def _hang_worker(
    messages: list[dict[str, Any]],
    provider: str,
    deadline: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    time.sleep(60)
    return messages, None


def _abort_worker(
    messages: list[dict[str, Any]],
    provider: str,
    deadline: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    os.abort()


_IMAGE_WORKER: Callable[
    [list[dict[str, Any]], str, float | None],
    tuple[list[dict[str, Any]], dict[str, Any] | None],
] = _compress_messages_worker


class _PoolJobCancelled(Exception):
    """The process-pool job was cancelled (typically by a pool reset).

    ``ProcessPoolExecutor.shutdown(cancel_futures=True)`` cancels the wrapped
    concurrent.futures job, which asyncio surfaces as ``CancelledError``. That
    is not a request to cancel *this* task, and ``Task.cancelling()`` (the
    discriminator) only exists on Python 3.11+. Mapping pool-job cancellation
    onto a regular exception lets genuine caller cancellation propagate from
    ``wait_for`` on 3.10 as well as 3.11+.
    """


def _ensure_image_pool_locked() -> ProcessPoolExecutor:
    """Return the published pool, creating it if needed. Caller holds the lock."""
    global _IMAGE_POOL
    if _IMAGE_POOL is None:
        _IMAGE_POOL = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
        )
    return _IMAGE_POOL


def _image_pool() -> ProcessPoolExecutor:
    with _IMAGE_POOL_LOCK:
        return _ensure_image_pool_locked()


def _acquire_image_pool() -> ProcessPoolExecutor:
    """Return the shared pool and record one more call waiting on it.

    Pool lookup/creation and the inflight increment share one lock so a
    concurrent reset cannot unpublished the pool after we observed it and
    before we counted this call.
    """
    global _IMAGE_POOL_INFLIGHT
    with _IMAGE_POOL_LOCK:
        pool = _ensure_image_pool_locked()
        _IMAGE_POOL_INFLIGHT += 1
        return pool


def _release_image_pool(pool: ProcessPoolExecutor) -> None:
    """Record that a call waiting on ``pool`` has finished.

    Inflight counts the currently published pool. A release for a pool that
    was already unpublished is a no-op so it cannot underflow a replacement.
    """
    global _IMAGE_POOL_INFLIGHT
    with _IMAGE_POOL_LOCK:
        if _IMAGE_POOL is pool and _IMAGE_POOL_INFLIGHT > 0:
            _IMAGE_POOL_INFLIGHT -= 1


def _reset_image_pool(pool: ProcessPoolExecutor | None = None) -> None:
    """Drop ``pool`` (default: the current one) and kill its worker.

    ``shutdown(wait=False)`` alone never stops a worker that is still running
    an abandoned call: after a timeout it ran the image to the end, its result
    discarded, while the next request cold-loaded the models in a fresh worker
    beside it. Only the current pool is unpublished, so a call that fails
    after another request already rebuilt the pool leaves the rebuilt one alone.

    The reset kills the single shared worker, so it also cancels or kills every
    other call's queued or running job (#4418). Refuse the destructive teardown
    while any other call is still using the pool: those requests fail open on
    their own and keep their compression instead of losing it to this one's
    timeout. Usage is checked under the same lock as acquire.
    """
    global _IMAGE_POOL, _IMAGE_POOL_INFLIGHT
    with _IMAGE_POOL_LOCK:
        if pool is None:
            pool = _IMAGE_POOL
        if _IMAGE_POOL_INFLIGHT > 1 and _IMAGE_POOL is pool:
            return
        if _IMAGE_POOL is pool:
            _IMAGE_POOL = None
            _IMAGE_POOL_INFLIGHT = 0
    if pool is None:
        return
    # shutdown() sets _processes to None, so take the workers first.
    workers = list((pool._processes or {}).values())
    pool.shutdown(wait=False, cancel_futures=True)
    for worker in workers:
        worker.kill()


async def _await_pool_job(
    future: asyncio.Future[Any],
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Await ``future`` without treating its cancellation as task cancellation.

    A cancelled executor future and a cancelled caller both used to surface as
    ``asyncio.CancelledError``. On Python 3.10 there is no ``Task.cancelling()``
    to tell them apart, so the previous handler either leaked ``AttributeError``
    or could not propagate genuine cancellation. Complete a private waiter
    instead: executor cancellation becomes ``_PoolJobCancelled``, while
    ``wait_for`` only raises ``CancelledError`` when *this* task is cancelled.

    Timeout and caller cancellation complete only that private waiter, so they
    would otherwise leave the ``run_in_executor`` future queued. Cancel it
    best-effort: a still-pending job must not run after this call already
    failed open. A job that is already running cannot be cancelled; isolation
    still refuses to kill the worker while another call is using the pool.
    """
    loop = asyncio.get_running_loop()
    waiter: asyncio.Future[Any] = loop.create_future()

    def _propagate(done: asyncio.Future[Any]) -> None:
        if waiter.done():
            return
        try:
            if done.cancelled():
                waiter.set_exception(_PoolJobCancelled())
                return
            exc = done.exception()
            if exc is not None:
                waiter.set_exception(exc)
                return
            waiter.set_result(done.result())
        except asyncio.InvalidStateError:
            return

    future.add_done_callback(_propagate)
    try:
        return await asyncio.wait_for(waiter, timeout=timeout)
    finally:
        if not future.done():
            future.cancel()


async def run_image_compression_isolated(
    messages: list[dict[str, Any]],
    provider: str,
    *,
    timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    loop = asyncio.get_running_loop()
    pool: ProcessPoolExecutor | None = None
    acquired = False
    try:
        deadline = time.time() + timeout * _OCR_BUDGET_FRACTION
        pool = _acquire_image_pool()
        acquired = True
        future = loop.run_in_executor(pool, _IMAGE_WORKER, messages, provider, deadline)
        return await _await_pool_job(future, timeout)
    except _PoolJobCancelled:
        logger.warning("Image compression worker cancelled; forwarding original image payload")
        _reset_image_pool(pool)
        return messages, None
    except BrokenProcessPool:
        logger.warning("Image compression worker crashed; forwarding original image payload")
        _reset_image_pool(pool)
        return messages, None
    except TimeoutError:
        logger.warning("Image compression worker timed out; forwarding original image payload")
        _reset_image_pool(pool)
        return messages, None
    except Exception as exc:
        logger.warning(
            "Image compression worker failed (%s); forwarding original image payload",
            type(exc).__name__,
        )
        _reset_image_pool(pool)
        return messages, None
    finally:
        if acquired and pool is not None:
            _release_image_pool(pool)
