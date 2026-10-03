"""Store-level integration for token-efficient CCR labels (HEADROOM_CCR_SHORT_LABELS).

With the flag on, store() keys entries by the allocator's short label (a prefix
of the content hash) and returns it, so emission sites embed a ~1-2 token id
instead of the 24-hex hash. retrieve() is unchanged — the label IS the key.
With the flag off, behavior is byte-identical to today (full 24-hex key).
"""

from __future__ import annotations

import hashlib

from headroom.cache.backends import InMemoryBackend
from headroom.cache.compression_store import CompressionStore


def _store(short_labels: bool) -> CompressionStore:
    return CompressionStore(backend=InMemoryBackend(), short_labels=short_labels)


def _full_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:24]


def test_flag_off_returns_full_24hex_key():
    s = _store(short_labels=False)
    content = "some long tool output " * 20
    key = s.store(content, "compressed")
    assert key == _full_hash(content)  # unchanged legacy behavior
    assert len(key) == 24


def test_flag_on_returns_short_label_that_roundtrips():
    s = _store(short_labels=True)
    content = "some long tool output " * 20
    key = s.store(content, "compressed")
    assert len(key) < 24  # token-efficient
    assert _full_hash(content).startswith(key)  # label is a prefix of the hash
    entry = s.retrieve(key)  # retrieve unchanged — label is the storage key
    assert entry is not None
    assert entry.original_content == content


def test_flag_on_same_content_dedups_to_same_label():
    s = _store(short_labels=True)
    content = "repeated block " * 30
    k1 = s.store(content, "c1")
    k2 = s.store(content, "c2")
    assert k1 == k2  # idempotent label -> dedup


def test_flag_on_distinct_content_distinct_labels():
    s = _store(short_labels=True)
    keys = {s.store(f"distinct output number {i} " * 10, "c") for i in range(200)}
    assert len(keys) == 200  # every distinct block gets its own label


def test_flag_on_label_is_prefix_and_retrievable_across_many():
    s = _store(short_labels=True)
    contents = [f"block {i} " * 25 for i in range(200)]
    keys = [s.store(c, "z") for c in contents]
    for c, k in zip(contents, keys):
        assert _full_hash(c).startswith(k)  # prefix invariant
        assert s.retrieve(k).original_content == c  # every label resolves


def test_explicit_hash_path_is_not_relabeled():
    # SmartCrusher's Rust row-drop path supplies its own hash and emits the
    # marker itself; the store must honor it verbatim, not relabel it.
    s = _store(short_labels=True)
    explicit = "a" * 12  # hex SHA-256[:12]-style key
    key = s.store("content", "compressed", explicit_hash=explicit)
    assert key == explicit
    assert s.retrieve(explicit).original_content == "content"


def test_env_flag_enables_short_labels(monkeypatch):
    monkeypatch.setenv("HEADROOM_CCR_SHORT_LABELS", "1")
    s = CompressionStore(backend=InMemoryBackend())  # reads env when unset
    key = s.store("env-driven content " * 20, "compressed")
    assert len(key) < 24
