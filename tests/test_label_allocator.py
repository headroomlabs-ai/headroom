"""Unit tests for the adaptive CCR label allocator.

Pins the two load-bearing invariants — earliest-wins (prefix-monotonic, so an
issued label never changes → prompt-cache-safe) and content-derived prefixes
(dedup + convergence to the full hash as a ceiling) — plus round-trip resolve
and the token-efficiency payoff.
"""

from __future__ import annotations

import hashlib

from headroom.cache.label_allocator import DEFAULT_MIN_WIDTH, CcrLabelAllocator


def _h(i: int) -> str:
    """Deterministic 24-hex content hash, like the store's SHA-256[:24]."""
    return hashlib.sha256(str(i).encode()).hexdigest()[:24]


def test_first_label_is_min_width():
    a = CcrLabelAllocator(min_width=2)
    assert a.label_for("abcdef0123") == "ab"
    assert DEFAULT_MIN_WIDTH == 2


def test_idempotent_same_hash_same_label():
    a = CcrLabelAllocator(min_width=2)
    assert a.label_for("abcdef") == "ab"
    assert a.label_for("abcdef") == "ab"  # repeat -> identical, no new entry
    assert len(a) == 1


def test_collision_widens_newcomer_earliest_wins():
    a = CcrLabelAllocator(min_width=2)
    first = a.label_for("ab1111")  # "ab"
    second = a.label_for("ab2222")  # "ab" taken -> widen the newcomer to "ab2"
    assert first == "ab"  # earliest keeps the short label
    assert second == "ab2"
    assert a.label_for("ab1111") == "ab"  # unchanged after the collision


def test_labels_are_unique_and_are_prefixes():
    a = CcrLabelAllocator(min_width=2)
    hashes = [_h(i) for i in range(300)]
    labels = [a.label_for(h) for h in hashes]
    assert len(set(labels)) == len(labels)  # every label distinct
    for h, label in zip(hashes, labels):
        assert h.startswith(label)  # label is always a prefix of its own hash


def test_prefix_monotonic_replay_is_cache_safe():
    # The cache-safety invariant: for every k, the labels from the first k
    # hashes are byte-identical whether or not later hashes exist. So appending
    # a stashed block never rewrites an earlier block's marker.
    hashes = [_h(i) for i in range(60)]
    full = CcrLabelAllocator(min_width=2)
    full_labels = [full.label_for(h) for h in hashes]
    for k in range(1, len(hashes) + 1):
        partial = CcrLabelAllocator(min_width=2)
        partial_labels = [partial.label_for(h) for h in hashes[:k]]
        assert partial_labels == full_labels[:k]


def test_deterministic_across_instances():
    hashes = [_h(i) for i in range(120)]
    a = CcrLabelAllocator(min_width=2)
    b = CcrLabelAllocator(min_width=2)
    assert [a.label_for(h) for h in hashes] == [b.label_for(h) for h in hashes]


def test_converges_to_full_hash_as_ceiling():
    # When every shorter prefix is already taken, the label falls back to the
    # full hash and never grows beyond it.
    a = CcrLabelAllocator(min_width=1)
    assert a.label_for("a") == "a"
    assert a.label_for("ab") == "ab"  # "a" taken -> "ab"
    assert a.label_for("abc") == "abc"  # "a","ab" taken -> full hash "abc"
    for label, h in [("a", "a"), ("ab", "ab"), ("abc", "abc")]:
        assert len(label) <= len(h)


def test_resolve_roundtrips():
    a = CcrLabelAllocator(min_width=2)
    label = a.label_for("deadbeefcafe")
    assert a.resolve(label) == "deadbeefcafe"
    assert a.resolve("zz") is None  # unknown -> None (caller treats key as a hash)


def test_token_efficiency_payoff():
    # 50 stashed items keep labels tiny vs the 24-hex default.
    a = CcrLabelAllocator(min_width=2)
    labels = [a.label_for(_h(i)) for i in range(50)]
    avg_len = sum(len(x) for x in labels) / len(labels)
    assert avg_len < 4  # vs 24 today
    assert max(len(x) for x in labels) <= 6


def test_rejects_bad_input():
    import pytest

    with pytest.raises(ValueError):
        CcrLabelAllocator(min_width=0)
    with pytest.raises(ValueError):
        CcrLabelAllocator().label_for("")
