"""Tests for :class:`headroom.proxy.memory_injection.MemoryInjectionBudget`.

Pre-PR Headroom had **no token cap on injected memory**: top_k=10
candidates × ~400 tokens each = up to ~4000 tokens injected per
request. None of Letta/Mem0/Cognee/Supermemory ship a token-uncapped
injection path on the hot wire.

``MemoryInjectionBudget`` is the single configurable cap applied to
every injection site so all 5 sites are uniformly bounded — set the
budget once, apply at every handler, dashboards see the same shape.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

from headroom.proxy.memory_injection import MemoryInjectionBudget

# ── Value-type contract ───────────────────────────────────────────────


def test_budget_is_frozen() -> None:
    b = MemoryInjectionBudget()
    try:
        b.max_tokens = 99  # type: ignore[misc]
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("MemoryInjectionBudget must be frozen")


def test_budget_defaults() -> None:
    """Default budget is conservative — 1024 tokens, 10 entries, 0.3
    similarity floor. Operators can override via config; the default
    is hard-set so a misconfiguration can't accidentally unbound
    injection."""
    b = MemoryInjectionBudget()
    assert b.max_tokens == 1024
    assert b.max_entries == 10
    assert b.min_similarity == 0.3


def test_budget_value_equal() -> None:
    a = MemoryInjectionBudget(max_tokens=512, max_entries=5, min_similarity=0.5)
    b = MemoryInjectionBudget(max_tokens=512, max_entries=5, min_similarity=0.5)
    assert a == b


# ── apply_to_text — bounding the formatted context block ─────────────


def test_apply_to_text_returns_input_when_under_budget() -> None:
    """Short context passes through unchanged — no spurious mutation."""
    b = MemoryInjectionBudget(max_tokens=1024)
    text = "## Relevant Memories\n1. small fact\n"
    out = b.apply_to_text(text)
    assert out == text


def test_apply_to_text_truncates_when_over_budget() -> None:
    """Large context is bounded — truncated at the budget. The
    truncation here is on the OUTPUT (the formatted injection block),
    NOT on the INPUT (which keeps full fidelity per MemoryQuery
    contract)."""
    # 4 tokens/char heuristic in our cap — make the input clearly
    # over even the most generous budget.
    b = MemoryInjectionBudget(max_tokens=128)  # ~512 chars at 4 char/token
    huge = "x" * 100000
    out = b.apply_to_text(huge)
    # Output should be substantially smaller than input.
    assert len(out) < len(huge)


def test_apply_to_text_preserves_full_lines() -> None:
    """When truncating, prefer cutting at line boundaries so the
    dashboard renders intact memory entries (no half-truncated bullet
    point)."""
    b = MemoryInjectionBudget(max_tokens=64)  # very tight
    text = "## Relevant Memories\n" + "".join(f"{i}. fact {i}\n" for i in range(100))
    out = b.apply_to_text(text)
    # No partial last line — every retained line ends in newline or is
    # the final line.
    if out and not out.endswith("\n"):
        # The last char is the closing of the final line; it must not
        # be in the middle of "fact " — easy heuristic: must not end
        # mid-word with a hanging digit-then-period.
        assert ". fact" not in out[-15:] or out.rstrip().endswith(("fact 0", "fact 1", "fact 2"))


def test_apply_to_text_handles_empty_input() -> None:
    """Empty input → empty output."""
    assert MemoryInjectionBudget().apply_to_text("") == ""


# ── apply_to_entries — bounding the list before formatting ───────────


def test_apply_to_entries_caps_entry_count() -> None:
    """Even if the backend returns 100 candidates, the budget caps
    entry count to ``max_entries``."""
    b = MemoryInjectionBudget(max_entries=3)
    entries = [{"content": f"entry {i}", "score": 0.9 - i * 0.01} for i in range(20)]
    out = b.apply_to_entries(entries)
    assert len(out) == 3


def test_apply_to_entries_preserves_order_of_input() -> None:
    """Budget doesn't re-rank — the backend's order is preserved. (The
    backend should already have ranked by score; budget just caps.)"""
    b = MemoryInjectionBudget(max_entries=2)
    entries = [
        {"content": "alpha", "score": 0.9},
        {"content": "beta", "score": 0.8},
        {"content": "gamma", "score": 0.7},
    ]
    out = b.apply_to_entries(entries)
    assert [e["content"] for e in out] == ["alpha", "beta"]


def test_apply_to_entries_filters_below_min_similarity() -> None:
    """Entries below ``min_similarity`` are dropped, regardless of
    entry-count budget remaining."""
    b = MemoryInjectionBudget(max_entries=10, min_similarity=0.5)
    entries = [
        {"content": "kept", "score": 0.9},
        {"content": "dropped", "score": 0.3},
        {"content": "kept2", "score": 0.55},
    ]
    out = b.apply_to_entries(entries)
    assert {e["content"] for e in out} == {"kept", "kept2"}
