"""Phase G PR-G3 remediation (C4) — wrap-CLI RTK metrics primitive.

The Rust proxy previously held a dead `wrap_rtk_invocations_total`
counter. C4 remediation moved it Python-side because the wrap CLI
(headroom.cli.wrap) is where RTK invocations are actually counted.
These tests cover the counter primitives in isolation.
"""

from __future__ import annotations

import threading

import pytest

from headroom.cli.wrap_rtk_metrics import (
    record_rtk_invocation,
    reset_rtk_invocations,
    rtk_invocation_counts,
)


@pytest.fixture(autouse=True)
def _reset_between_tests():
    """Reset the module-level counter map between tests so each
    test owns a clean slate."""
    reset_rtk_invocations()
    yield
    reset_rtk_invocations()


def test_record_increments_default_delta_one():
    record_rtk_invocation("git")
    counts = rtk_invocation_counts()
    assert counts == {"git": 1}


def test_record_accumulates_per_tool():
    record_rtk_invocation("git")
    record_rtk_invocation("git")
    record_rtk_invocation("ls")
    record_rtk_invocation("cargo")
    record_rtk_invocation("cargo")
    record_rtk_invocation("cargo")
    counts = rtk_invocation_counts()
    assert counts == {"git": 2, "ls": 1, "cargo": 3}


def test_record_with_explicit_delta():
    record_rtk_invocation("git", delta=5)
    record_rtk_invocation("git", delta=2)
    counts = rtk_invocation_counts()
    assert counts == {"git": 7}


def test_record_zero_delta_is_noop_record():
    # delta=0 is legal — caller may want to "touch" the counter to
    # ensure the key exists before later increments.
    record_rtk_invocation("git", delta=0)
    counts = rtk_invocation_counts()
    assert counts == {"git": 0}


def test_record_rejects_negative_delta():
    with pytest.raises(ValueError, match="must be non-negative"):
        record_rtk_invocation("git", delta=-1)


def test_record_rejects_non_string_tool():
    with pytest.raises(TypeError, match="tool must be a str"):
        record_rtk_invocation(123, delta=1)  # type: ignore[arg-type]


def test_record_rejects_non_int_delta():
    with pytest.raises(TypeError, match="delta must be an int"):
        record_rtk_invocation("git", delta="1")  # type: ignore[arg-type]


def test_counts_returns_snapshot_not_view():
    # The returned mapping must be a plain dict copy, not the
    # internal defaultdict — otherwise callers could pollute the
    # counter map by reading absent keys.
    record_rtk_invocation("git")
    counts = rtk_invocation_counts()
    # Reading a key that's not present must not add it to the
    # internal map.
    _ = counts.get("nonexistent_tool", 0)
    counts2 = rtk_invocation_counts()
    assert "nonexistent_tool" not in counts2


def test_thread_safe_concurrent_increments():
    # 10 threads each bumping `git` 100 times: final count must be
    # exactly 1000. The threading.Lock guards the dict update so
    # races are impossible.
    def worker():
        for _ in range(100):
            record_rtk_invocation("git")

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    counts = rtk_invocation_counts()
    assert counts == {"git": 1000}


def test_reset_clears_counts():
    record_rtk_invocation("git", delta=42)
    record_rtk_invocation("ls", delta=7)
    assert rtk_invocation_counts() != {}
    reset_rtk_invocations()
    assert rtk_invocation_counts() == {}
