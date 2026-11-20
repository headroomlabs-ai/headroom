"""Never replay a cached reference after its original expires or disappears."""

from __future__ import annotations

import pytest

from headroom.cache import compression_store
from headroom.cache.compression_cache import CompressionCache as SessionCache
from headroom.cache.compression_store import CompressionStore
from headroom.transforms.content_router import CompressionCache as RouterCache


@pytest.mark.parametrize("cache_kind", ["router", "session"])
@pytest.mark.parametrize("failure", ["expired", "evicted", "backend_error"])
def test_cached_reference_requires_a_live_original(monkeypatch, cache_kind, failure):
    store = CompressionStore(max_entries=1, default_ttl=2)
    monkeypatch.setattr(compression_store, "get_compression_store", lambda: store)
    key = store.store("Complete original dialogue", "Short dialogue")
    text = f"[100 lines compressed to 10. Retrieve more: hash={key}]"
    cache = RouterCache() if cache_kind == "router" else SessionCache()
    if cache_kind == "router":
        cache.put(1, text, 0.1, "text")

        def read():
            return cache.get(1)
    else:
        cache.store_compressed("content", text, 100)

        def read():
            return cache.get_compressed("content")

    assert read() is not None
    if failure == "expired":
        timestamp = compression_store.time.time()
        monkeypatch.setattr(compression_store.time, "time", lambda: timestamp + 3)
    elif failure == "evicted":
        store.store("Other original", "Other summary")
    else:

        def fail(*args, **kwargs):
            raise OSError("Storage unavailable")

        monkeypatch.setattr(store, "get_entry_status", fail)
    assert read() is None


def test_expired_session_reference_cannot_freeze_raw_evidence(monkeypatch):
    store = CompressionStore(default_ttl=2)
    monkeypatch.setattr(compression_store, "get_compression_store", lambda: store)
    key = store.store("Original evidence", "Summary")
    cache = SessionCache()
    message = {"role": "tool", "tool_call_id": "call1", "content": "Original evidence"}
    cache.store_compressed(cache.content_hash(message["content"]), f"<<ccr:{key},text,1KB>>", 100)
    timestamp = compression_store.time.time()
    monkeypatch.setattr(compression_store.time, "time", lambda: timestamp + 3)
    assert cache.compute_frozen_count([message, {"role": "user", "content": "Continue"}]) == 0
    assert cache.apply_cached([message]) == [message]


@pytest.mark.parametrize(
    "marker",
    [
        "<<ccr:{key},text,1KB>>",
        "[10 items compressed. hash={key}]",
        "Retrieve original: hash={key}",
    ],
)
def test_all_marker_formats_require_live_originals(monkeypatch, marker):
    store = CompressionStore(default_ttl=2)
    monkeypatch.setattr(compression_store, "get_compression_store", lambda: store)
    key = store.store("Original", "Summary")
    cache = RouterCache()
    cache.put(1, marker.format(key=key), 0.1, "text")
    assert cache.get(1) is not None
    timestamp = compression_store.time.time()
    monkeypatch.setattr(compression_store.time, "time", lambda: timestamp + 3)
    assert cache.get(1) is None
