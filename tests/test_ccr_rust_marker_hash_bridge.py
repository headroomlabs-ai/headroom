"""Issue #816: Rust search/diff/log CCR markers must be retrievable.

The Rust side embeds ``MD5(original)[:24]`` in the emitted
``Retrieve more: hash=...`` marker, but since PR #395
``CompressionStore.store()`` defaults to ``SHA-256(original)[:24]``.
PR #395 fixed the SmartCrusher path by passing ``explicit_hash``
(see ``test_ccr_row_drop_store_bridge.py``); the CCR write-through
used by the Rust-accelerated transforms was never migrated, so every
marker they emitted dangled — retrieval returned "Entry not found or
expired" inside any TTL.

These tests pin the cross-language contract at the shared
``persist_rust_ccr_entry`` helper: the store entry must be keyed by the
exact hash the marker embeds.
"""

from __future__ import annotations

import hashlib

import pytest

from headroom.cache.compression_store import (
    get_compression_store,
    reset_compression_store,
)
from headroom.transforms.base import persist_rust_ccr_entry

pytest.importorskip("headroom._core", reason="Rust extension required")


@pytest.fixture(autouse=True)
def _fresh_store():
    reset_compression_store()
    yield
    reset_compression_store()


def _rust_marker_key(original: str) -> str:
    """The hash the Rust side embeds in emitted markers."""
    return hashlib.md5(original.encode()).hexdigest()[:24]


def _assert_round_trip(original: str) -> None:
    """Entry must be retrievable under the marker's key, not SHA-256."""
    marker_key = _rust_marker_key(original)
    store = get_compression_store()

    entry = store.retrieve(marker_key)
    assert entry is not None, (
        f"store has no entry under the Rust marker key {marker_key!r}; "
        f"the marker dangles (issue #816)"
    )
    assert entry.original_content == original

    sha_key = hashlib.sha256(original.encode()).hexdigest()[:24]
    assert store.retrieve(sha_key) is None, (
        "entry stored under the SHA-256 default key instead of the "
        "marker's MD5 key — explicit_hash was not passed through"
    )


@pytest.mark.parametrize(
    ("source", "original"),
    [
        ("search", "src/app.py:12: def handle_request(payload):\n" * 40),
        ("diff", "+added line of code\n-removed line of code\n" * 40),
        ("log", "2026-06-11T09:00:00Z INFO worker heartbeat ok seq=1\n" * 40),
    ],
)
def test_persist_stores_under_marker_key(source: str, original: str) -> None:
    persist_rust_ccr_entry(
        original, f"compressed {source} output", _rust_marker_key(original), source=source
    )
    _assert_round_trip(original)
