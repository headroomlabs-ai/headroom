"""Off-path background compression (Phase 3, #1171).

The request path must never block on ML compression. When a cold-start-large
request would otherwise run kompress synchronously under the 30s budget (and
leak a non-preemptible worker on timeout -> executor saturation -> cascade),
it instead forwards the already-cached/uncompressed messages immediately and
enqueues the compression here. A single per-process drain runs it with NO
request-coupled deadline and stores the result in the session
``CompressionCache``, so the next turn is a cache hit and the forwarded bytes
become (and stay) the compressed form.

This is per-process by design: ``CompressionCache`` is already per-process
(``HeadroomProxy._compression_caches``), and multi-worker deployments are
already warned to use ``--workers 1`` or sticky sessions, so a per-process
drain matches the existing cache semantics without any new cross-process lock.

Limitations, all fail-open (lost savings, never lost correctness): only the
token-mode cold-start path defers here -- other modes compress synchronously;
the queue is in-memory, so a restart mid-drain drops queued jobs (they re-defer
on a later turn); and a full queue or a duplicate in-flight key drops the job,
surfaced to telemetry as ``deferred:dropped``. Background work is bounded by the
Phase 1 kompress deadline (a non-terminating compressor would pin the single
drain thread, but the compressors terminate).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _Job:
    key: str
    compress: Callable[[], Any]  # sync callable, runs in the executor (no timeout)
    store: Callable[[Any], None]  # sync callable, stores the result into the cache


class BackgroundCompressor:
    """Single per-process async drain that compresses enqueued work off the
    request path, with no request-coupled deadline.

    ``run_in_executor`` is injected so the drain reuses the proxy's compression
    ThreadPoolExecutor (without the request-path ``asyncio.wait_for`` timeout),
    and so tests can supply a trivial runner.
    """

    def __init__(
        self,
        run_in_executor: Callable[[Callable[[], Any]], Awaitable[Any]],
        *,
        max_queue: int = 256,
    ) -> None:
        self._run_in_executor = run_in_executor
        self._queue: asyncio.Queue[_Job] = asyncio.Queue(maxsize=max_queue)
        self._pending: set[str] = set()
        self._task: asyncio.Task[None] | None = None
        self._processed = 0
        self._dropped = 0
        self._errors = 0

    def enqueue(
        self,
        key: str,
        compress: Callable[[], Any],
        store: Callable[[Any], None],
    ) -> bool:
        """Queue a compression job. Returns False (and drops) if the key is
        already in flight or the queue is full -- both are safe: the request
        has already been forwarded uncompressed, so a drop just defers the
        savings to a later turn."""
        if key in self._pending:
            return False  # already queued / in flight -- dedup
        # Claim the slot BEFORE the job is observable in the queue so dedup is
        # atomic against another enqueue of the same key.
        self._pending.add(key)
        try:
            self._queue.put_nowait(_Job(key, compress, store))
        except asyncio.QueueFull:
            self._pending.discard(key)
            self._dropped += 1
            logger.warning(
                "background compression queue full (%d); dropping %s "
                "(request already forwarded uncompressed)",
                self._queue.maxsize,
                key,
            )
            return False
        return True

    async def _process_one(self, job: _Job) -> None:
        try:
            result = await self._run_in_executor(job.compress)
            job.store(result)
            self._processed += 1
        except Exception as e:  # noqa: BLE001 -- fail-open: request already went out uncompressed
            self._errors += 1
            logger.warning("background compression failed for %s: %s", job.key, e)
        finally:
            self._pending.discard(job.key)

    async def _drain(self) -> None:
        while True:
            job = await self._queue.get()
            try:
                await self._process_one(job)
            finally:
                self._queue.task_done()

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._drain(), name="headroom-bg-compress")

    async def stop(self, *, drain: bool = True, timeout: float = 5.0) -> None:
        if self._task is None:
            return
        if drain:
            try:
                await asyncio.wait_for(self._queue.join(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    "background compression drain timed out with %d queued",
                    self._queue.qsize(),
                )
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def stats(self) -> dict[str, int]:
        return {
            "queued": self._queue.qsize(),
            "pending": len(self._pending),
            "processed": self._processed,
            "dropped": self._dropped,
            "errors": self._errors,
        }
