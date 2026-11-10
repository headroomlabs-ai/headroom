"""Unit tests for :mod:`perfkit.attribution` (T013).

Covers stable-id determinism, sanitization, parent-chain tracking, and the
display form used by Markdown drill-down links.
"""

from __future__ import annotations

import pytest

from benchmarks.perfkit.attribution import (
    attribution_from,
    build_id,
    diff_ids,
    display_form,
)


def test_build_id_is_deterministic_for_same_inputs() -> None:
    a = build_id("G2", "/v1/messages", "compression", "smart_crusher", "apply")
    b = build_id("G2", "/v1/messages", "compression", "smart_crusher", "apply")
    assert a == b


def test_build_id_pads_empty_leaf_slots() -> None:
    """Coarse-grained tiers still emit a stable 5-slot shape."""

    assert build_id("G4", "/v1/messages") == "G4:/v1/messages:-:-:-"


def test_build_id_preserves_path_slashes() -> None:
    """``/`` is a legal character inside request paths and passes through.

    The drill-down links in US2 click through exactly this string, so
    non-preserving the slash would break Markdown links.
    """

    sid = build_id("G3", "/v1/messages", stage="compression")
    head = sid.split("@")[0]
    request_slot = head.split(":")[1]
    assert request_slot == "/v1/messages"


def test_build_id_with_parents_uses_at_separator() -> None:
    full = build_id(
        "G2",
        "/v1/messages",
        "compression",
        "smart_crusher",
        "apply",
        parents=("G4:/v1/messages:-:-:-",),
    )
    # Parents are already-canonical IDs; their inner ':' is preserved,
    # only the '@' boundary is rewritten so the parser can find the chain.
    assert full.endswith("@G4:/v1/messages:-:-:-")


def test_build_id_truncates_oversized_segments() -> None:
    long = "x" * 500
    out = build_id("G1", long)
    assert len(out.split(":")[1]) <= 96


def test_attribution_from_materializes_parents() -> None:
    parents = ("G4:/v1/messages:-:-:-", "G3:/v1/messages:streamed-response:-:-")
    attr = attribution_from(
        "G2",
        "/v1/messages",
        "compression",
        "smart_crusher.apply",
        "apply",
        parents=parents,
    )
    assert attr.parents == parents
    # The "." inside the transform slot is allowed by the sanitizer (it's the
    # legal ``function.member`` form every Headroom transform uses).
    assert attr.transform == "smart_crusher.apply"
    assert attr.tier == "G2"


def test_attribution_from_chain_ends_with_self() -> None:
    parents = ("G4:/v1/messages:-:-:-",)
    attr = attribution_from(
        "G1",
        "/v1/messages",
        "smart_crusher",
        parents=parents,
    )
    assert attr.to_chain()[-1] == attr.stable_id
    assert attr.to_chain()[0] == parents[0]


def test_diff_ids_equal_returns_no_diff() -> None:
    sid = build_id("G1", "/v1/messages", "smart_crusher", "apply")
    assert diff_ids(sid, sid).differs is False


def test_diff_ids_different_head_signals_head_mismatch() -> None:
    a = build_id("G1", "/v1/messages", "smart_crusher", "apply")
    b = build_id("G2", "/v1/messages", "smart_crusher", "apply")
    diff = diff_ids(a, b)
    assert diff.differs is True
    assert diff.diff_reason == "head_mismatch"


def test_diff_ids_same_head_different_parents() -> None:
    """A change only in the parent chain suffix is detectable."""

    head = build_id("G2", "/v1/messages", "compression", "smart_crusher", "apply")
    a = head + "@G4:/v1/messages:-:-:-"
    b = head + "@G4:/v1/chat/completions:-:-:-"
    diff = diff_ids(a, b)
    assert diff.differs is True
    assert diff.diff_reason == "parents_changed"


def test_display_form_is_compact() -> None:
    sid = build_id("G4", "/v1/messages")
    assert display_form(sid) == "[G4] /v1/messages"


def test_build_id_raises_on_invalid_inputs_via_implicit_pads() -> None:
    """Empty slots are tolerated (they become ``-``); nothing else required."""

    assert build_id("G4", "/v1/chat/completions") == "G4:/v1/chat/completions:-:-:-"


def test_build_id_rejects_garbage_chars_gracefully() -> None:
    """Disallowed characters become ``-`` rather than crashing.

    ``#``, ``?`` and spaces survive as ``-``; ``/`` is still preserved.
    """

    sid = build_id("G4", "?#/v1/foo bar?")
    request_slot = sid.split("@")[0].split(":")[1]
    assert "/" in request_slot
    assert " " not in sid
    assert "?" not in sid
    assert "#" not in sid
