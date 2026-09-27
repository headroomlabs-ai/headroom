from __future__ import annotations

import pytest
from headroom.cache.backends.memory import InMemoryBackend
from headroom.cache.compression_store import CompressionStore

from headroom_edge_layer.cache_safety import (
    already_compressed,
    cap_input,
    config_scoped_hash,
    format_elided_marker,
    format_unchanged_marker,
    store_original,
)


@pytest.fixture
def store() -> CompressionStore:
    return CompressionStore(backend=InMemoryBackend())


def test_config_scoped_hash_is_deterministic() -> None:
    a = config_scoped_hash("hello", config_version="v1")
    b = config_scoped_hash("hello", config_version="v1")
    assert a == b
    assert len(a) == 24
    assert all(c in "0123456789abcdef" for c in a)


def test_config_scoped_hash_changes_with_config_version() -> None:
    a = config_scoped_hash("hello", config_version="v1")
    b = config_scoped_hash("hello", config_version="v2")
    assert a != b  # a config change must never silently reuse a stale entry


def test_config_scoped_hash_changes_with_salt() -> None:
    a = config_scoped_hash("hello", config_version="v1", salt="grep:a.py")
    b = config_scoped_hash("hello", config_version="v1", salt="grep:b.py")
    assert a != b


def test_store_original_round_trips(store: CompressionStore) -> None:
    hash_key = store_original(store, "the original text", "elided", config_version="v1")
    entry = store.retrieve(hash_key)
    assert entry is not None
    assert entry.original_content == "the original text"


def test_store_original_refresh_overwrites_in_place(store: CompressionStore) -> None:
    """Re-storing the same content under the same key (config-scoped hash)
    must succeed every time it's replayed, refreshing the entry's TTL —
    never error, never duplicate (cache-safety rule #5).
    """
    hash_1 = store_original(store, "unchanged content", "unchanged content", config_version="v1")
    hash_2 = store_original(store, "unchanged content", "unchanged content", config_version="v1")
    assert hash_1 == hash_2
    assert store.retrieve(hash_1) is not None


def test_format_unchanged_marker_matches_read_maturation_convention() -> None:
    hash_key = config_scoped_hash("x", config_version="v1")
    marker = format_unchanged_marker(origin_index=2, path="/a.py", line_range="1-50", hash_key=hash_key)
    assert f"Retrieve original: hash={hash_key}" in marker
    assert "/a.py" in marker


def test_format_elided_marker_matches_search_compressor_convention() -> None:
    hash_key = config_scoped_hash("x", config_version="v1")
    marker = format_elided_marker(count=60, unit="lines", kept=20, hash_key=hash_key)
    assert f"Retrieve more: hash={hash_key}" in marker
    assert "60" in marker and "20" in marker


def test_already_compressed_recognizes_all_three_engine_marker_shapes() -> None:
    hash_key = config_scoped_hash("x", config_version="v1")
    assert already_compressed(f"... Retrieve original: hash={hash_key} ...")
    assert already_compressed(f"... Retrieve more: hash={hash_key} ...")
    assert already_compressed(f"...<<ccr:{hash_key}>>...")
    assert not already_compressed("plain text with no marker")


def test_cap_input_passes_through_under_limit() -> None:
    assert cap_input("short", max_chars=100) == "short"


def test_cap_input_returns_none_over_limit() -> None:
    assert cap_input("x" * 101, max_chars=100) is None
