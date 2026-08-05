"""Phase 3 (#1171) byte-identity + off-path data flow.

The off-path design's correctness argument is: forwarding the uncompressed
messages on turn N and the compressed form on turn N+1 does NOT corrupt the
upstream prefix cache, because ``apply_cached`` swaps in the stored compressed
bytes verbatim (one-time miss, then stable). These tests pin that claim and the
end-to-end enqueue -> drain -> store -> cache-hit flow the handler gate relies
on, without standing up the full Anthropic handler.
"""

from __future__ import annotations

import asyncio

from headroom.cache.compression_cache import CompressionCache
from headroom.proxy.background_compression import BackgroundCompressor


def _tool(content: str) -> dict:
    return {"role": "tool", "content": content}


def test_apply_cached_is_byte_identical_and_stable():
    cache = CompressionCache()
    originals = [_tool("x " * 10000)]  # a large tool result
    compressed = [_tool("COMPRESSED")]

    cache.update_from_result(originals, compressed)

    # One-time miss already paid; from here the swap is verbatim AND stable
    # across repeated turns (no every-turn thrash).
    out1 = cache.apply_cached(originals)
    out2 = cache.apply_cached(originals)
    assert out1[0]["content"] == "COMPRESSED"
    assert out1 == out2
    # apply_cached never mutates its input.
    assert originals[0]["content"] == "x " * 10000


def test_unchanged_content_is_not_swapped():
    cache = CompressionCache()
    originals = [_tool("same bytes")]
    # Pipeline returned identical content (nothing to compress) -> no mapping.
    cache.update_from_result(originals, [_tool("same bytes")])
    assert cache.apply_cached(originals)[0]["content"] == "same bytes"


def test_offpath_enqueue_drain_store_then_cache_hit():
    async def main():
        cache = CompressionCache()
        originals = [_tool("x " * 10000)]

        async def run(fn):  # trivial in-loop executor stand-in
            return fn()

        bc = BackgroundCompressor(run)
        await bc.start()
        # Exactly the two lambdas the deferral gate passes: compress -> produce
        # the compressed messages; store -> fold them into the session cache.
        bc.enqueue(
            "sess:42",
            lambda: [_tool("COMPRESSED")],
            lambda result: cache.update_from_result(originals, result),
        )
        await bc._queue.join()
        await bc.stop()
        return cache.apply_cached(originals), bc.stats()

    out, stats = asyncio.run(main())
    # After the background job ran, the next turn is a byte-identical cache hit.
    assert out[0]["content"] == "COMPRESSED"
    assert stats["processed"] == 1
    assert stats["errors"] == 0
