"""Phase 3 (#1171): off-path BackgroundCompressor.

A single per-process drain compresses enqueued work with NO request-coupled
deadline and stores the result -- so the request path never blocks on ML.
Tests use asyncio.run so they do not depend on a pytest-asyncio config.
"""

from __future__ import annotations

import asyncio

from headroom.proxy.background_compression import BackgroundCompressor


async def _passthrough_executor(fn):
    return fn()


def test_enqueue_compresses_and_stores():
    async def main():
        stored: list[str] = []
        bc = BackgroundCompressor(_passthrough_executor)
        await bc.start()
        ok = bc.enqueue("k1", lambda: "COMPRESSED", lambda r: stored.append(r))
        await asyncio.wait_for(bc._queue.join(), timeout=2)
        await bc.stop()
        return ok, stored, bc.stats()

    ok, stored, stats = asyncio.run(main())
    assert ok is True
    assert stored == ["COMPRESSED"]
    assert stats["processed"] == 1
    assert stats["pending"] == 0


def test_dedup_skips_inflight_key():
    async def main():
        calls: list[int] = []

        async def slow_executor(fn):
            await asyncio.sleep(0.05)
            return fn()

        bc = BackgroundCompressor(slow_executor)
        await bc.start()
        first = bc.enqueue("same", lambda: calls.append(1), lambda r: None)
        second = bc.enqueue("same", lambda: calls.append(1), lambda r: None)
        await asyncio.wait_for(bc._queue.join(), timeout=2)
        await bc.stop()
        return first, second, len(calls)

    first, second, n = asyncio.run(main())
    assert first is True
    assert second is False  # duplicate key skipped
    assert n == 1


def test_fail_open_on_compress_error():
    async def main():
        stored: list[str] = []
        bc = BackgroundCompressor(_passthrough_executor)
        await bc.start()

        def boom():
            raise RuntimeError("kompress exploded")

        bc.enqueue("bad", boom, lambda r: stored.append(r))
        bc.enqueue("good", lambda: "ok", lambda r: stored.append(r))
        await asyncio.wait_for(bc._queue.join(), timeout=2)
        await bc.stop()
        return stored, bc.stats()

    stored, stats = asyncio.run(main())
    assert stored == ["ok"]  # failing job did not store; later job still processed
    assert stats["errors"] == 1
    assert stats["processed"] == 1
    assert stats["pending"] == 0  # key released even on error


def test_queue_full_drops_without_raising():
    async def main():
        # Never start the drain, so the queue fills and stays full.
        bc = BackgroundCompressor(_passthrough_executor, max_queue=2)
        results = [bc.enqueue(f"k{i}", lambda: None, lambda r: None) for i in range(5)]
        return results, bc.stats()

    results, stats = asyncio.run(main())
    assert results[:2] == [True, True]
    assert results[2:] == [False, False, False]  # overflow dropped, no exception
    assert stats["dropped"] == 3


def test_stop_drains_remaining_jobs():
    async def main():
        stored: list[str] = []

        async def slow_executor(fn):
            await asyncio.sleep(0.02)
            return fn()

        bc = BackgroundCompressor(slow_executor)
        await bc.start()
        for i in range(4):
            bc.enqueue(f"k{i}", (lambda i=i: f"c{i}"), lambda r: stored.append(r))
        await bc.stop()  # drain=True by default -> waits for queue.join()
        return sorted(stored)

    stored = asyncio.run(main())
    assert stored == ["c0", "c1", "c2", "c3"]
