"""A cached result whose CCR original is gone must not be served (#4796).

The result cache outlives the CCR entries its markers point at: the store
evicts by capacity and expires on its own TTL, while a re-compression of the
same content hashes to the SAME cache key. So a cached marker could point at a
deleted original for the rest of the process, and every turn handed the model a
pointer that 404s on retrieve. Serving it is worse than a cache miss.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from headroom.cache.backends import InMemoryBackend
from headroom.cache.compression_store import (
    CompressionStore,
    clear_request_compression_store,
    set_request_compression_store,
)
from headroom.transforms.content_router import (
    CompressionCache,
    CompressionStrategy,
    ContentRouter,
    ContentRouterConfig,
    _ccr_marker_hashes,
)

MARKER_RE = re.compile(r"<<ccr:([0-9a-f]{24})>>")


@pytest.fixture
def ccr_store():
    """A small request-scoped store, so capacity eviction is easy to trigger."""
    store = CompressionStore(max_entries=2, backend=InMemoryBackend())
    set_request_compression_store(store)
    try:
        yield store
    finally:
        clear_request_compression_store()


@pytest.mark.parametrize(
    "marker",
    [
        "<<ccr:{hash}>>",
        "<<ccr:{hash},base64,2.0KB>>",
        "<<ccr:{hash} 12_rows_offloaded>>",
        "[80 lines compressed to 8. Retrieve more: hash={hash}]",
        "[128 tokens compressed. Retrieve original: hash={hash}]",
    ],
)
def test_a_cached_marker_is_a_miss_once_its_original_is_gone(ccr_store, marker):
    hash_key = ccr_store.store("the original tool output", "the compressed view")
    cache = CompressionCache()
    cache.put("k", marker.format(hash=hash_key), 0.2, "x")
    assert cache.get("k") is not None, "a live marker is still served"

    ccr_store.store("an unrelated tool output", "its compressed view")
    ccr_store.store("a third tool output", "its compressed view")  # over capacity
    assert ccr_store.exists(hash_key) is False
    assert cache.get("k") is None
    assert cache.stats["cache_evictions"] == 1


def test_a_marker_free_result_is_never_checked_against_the_store(ccr_store, monkeypatch):
    cache = CompressionCache()
    cache.put("k", "no pointer here", 0.5, "kompress")

    def explode():
        raise AssertionError("marker-free results must not touch the CCR store")

    monkeypatch.setattr("headroom.cache.compression_store.get_compression_store", explode)
    assert cache.get("k") == ("no pointer here", 0.5, "kompress")


def test_a_check_that_cannot_run_keeps_serving_the_cached_result(monkeypatch):
    # Fail open: a store that is absent or failing is not evidence the original
    # is gone, and a miss would only cost another compression.
    cache = CompressionCache()
    cache.put("k", "head <<ccr:0123456789abcdef>>", 0.2, "x")

    def explode():
        raise RuntimeError("ccr store unavailable")

    monkeypatch.setattr("headroom.cache.compression_store.get_compression_store", explode)
    assert cache.get("k") == ("head <<ccr:0123456789abcdef>>", 0.2, "x")


def test_every_marker_shape_is_read_for_its_hash():
    assert _ccr_marker_hashes(
        "a <<ccr:abcdef012345>> b <<ccr:1234567890ab,base64,2.0KB>> "
        "c [80 lines compressed to 8. Retrieve more: hash=fedcba987654]"
    ) == {"abcdef012345", "1234567890ab", "fedcba987654"}
    assert _ccr_marker_hashes("Retrieve original: hash=ABCDEF0123456789") == {"abcdef0123456789"}
    # Shorter than 12 hex chars is not a hash this store ever issues.
    assert _ccr_marker_hashes("hash=abcdef01234") == set()


class Tokenizer:
    def count_text(self, text: str) -> int:
        return max(1, len(str(text)) // 4)


def tool_log(case: str) -> str:
    return "\n".join(
        f"2026-01-01 12:00:{i % 60:02d} INFO case{case} processed record {i}" for i in range(251)
    )


def payload(case: str) -> list[dict]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{case}",
                    "type": "function",
                    "function": {"name": "run", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"call_{case}", "content": tool_log(case)},
    ]


def tool_output(result) -> str:
    return result.messages[-1]["content"]


def test_an_evicted_original_makes_the_router_recompress(ccr_store, monkeypatch):
    router = ContentRouter(ContentRouterConfig(ccr_inject_marker=False))
    monkeypatch.setattr(router, "_kompress_model_ready", lambda: True)
    calls: list[str] = []

    def compress(content, context="", bias=1.0, **kwargs):
        calls.append(content)
        compressed = f"compressed copy {len(calls)}"
        hash_key = ccr_store.store(content, compressed)
        return SimpleNamespace(
            original=content,
            compressed=f"{compressed} <<ccr:{hash_key}>>",
            compression_ratio=0.1,
            strategy_used=CompressionStrategy.KOMPRESS,
            strategy_chain=[],
        )

    monkeypatch.setattr(router, "compress", compress)

    first = tool_output(router.apply(payload("a"), Tokenizer()))
    first_hash = MARKER_RE.search(first).group(1)
    assert ccr_store.exists(first_hash)

    # Two distinct logs fill the store; "a" and then "b" are evicted, exactly
    # the capacity pressure that leaves a stale marker in the cache.
    router.apply(payload("b"), Tokenizer())
    router.apply(payload("c"), Tokenizer())
    assert ccr_store.exists(first_hash) is False
    assert len(calls) == 3

    again = tool_output(router.apply(payload("a"), Tokenizer()))
    assert len(calls) == 4, "a stale marker must be a cache miss, not a hit"
    # Recompressing the same content hashes to the same key, which is exactly
    # why the stale marker could never heal on its own: the cache has to miss
    # so the store gets the original back under that same hash.
    assert MARKER_RE.search(again).group(1) == first_hash
    assert ccr_store.exists(first_hash), "the recompression must restore the original"
    assert router._cache.stats["cache_hits"] == 0

    # The fresh result is cached in turn, so the next apply reuses it.
    assert tool_output(router.apply(payload("a"), Tokenizer())) == again
    assert len(calls) == 4
