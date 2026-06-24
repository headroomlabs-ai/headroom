"""Smoke tests for the PR4 lossless-first default behavior.

Verifies that `SmartCrusher()` (default constructor, no args) produces
the lossless compaction output for cleanly tabular input — the
user-visible win from PR4. Complements the legacy parity fixtures in
`test_smart_crusher_rust_parity.py` which exercise `without_compaction()`.
"""

from __future__ import annotations

import json

import pytest


def _build_extension() -> None:
    try:
        from headroom._core import SmartCrusher  # noqa: F401
    except ImportError:
        pytest.skip(
            "headroom._core not built — run `bash scripts/build_rust_extension.sh`",
            allow_module_level=True,
        )


_build_extension()


def test_default_lossless_wins_on_uniform_tabular() -> None:
    """50 uniform tabular dicts → CSV+schema beats threshold (>=30%)
    by a wide margin. Default `SmartCrusher()` ships the compacted
    string in place of the array."""
    from headroom._core import SmartCrusher

    items = [{"id": i, "level": "info", "msg": "ok"} for i in range(50)]
    content = json.dumps(items)
    crusher = SmartCrusher()
    result = crusher.crush(content, "", 1.0)

    # Output should be SHORTER than input (lossless win).
    assert len(result.compressed) < len(content), (
        f"compressed not smaller: {len(result.compressed)} >= {len(content)}"
    )
    # Strategy string surfaces the lossless tag.
    assert result.strategy.startswith("lossless:table"), (
        f"expected lossless:table, got {result.strategy!r}"
    )
    # was_modified flips because the JSON has a string where the
    # array used to be.
    assert result.was_modified


def test_default_falls_through_to_lossy_when_below_threshold() -> None:
    """Heterogeneous / non-tabular input → compactor declines (or
    savings below threshold) → lossy path runs."""
    from headroom._core import SmartCrusher

    # 30 unique-id objects with no schema redundancy → not enough
    # tabular signal for lossless to clear 30% by itself.
    items = [{"id": i, "user_unique_field_name": f"u_{i}"} for i in range(30)]
    content = json.dumps(items)
    crusher = SmartCrusher()
    result = crusher.crush(content, "", 1.0)

    # Either lossless wins (above threshold), OR lossy ran. Both are
    # acceptable; what we're guarding against is "nothing happened."
    # If the strategy is `passthrough` something is wrong.
    assert result.strategy != "passthrough", (
        f"expected some compression, got passthrough: {result.compressed[:80]!r}"
    )


def test_without_compaction_preserves_pre_pr4_behavior() -> None:
    """Opt-out constructor matches the legacy parity path."""
    from headroom._core import SmartCrusher

    items = [{"id": i, "level": "info", "msg": "ok"} for i in range(50)]
    content = json.dumps(items)
    crusher = SmartCrusher.without_compaction()
    result = crusher.crush(content, "", 1.0)

    # No lossless attempt → output stays JSON-array-shaped
    # (pre-PR4 contract).
    parsed = json.loads(result.compressed)
    assert isinstance(parsed, list), (
        f"expected JSON array, got {type(parsed).__name__}: {result.compressed[:80]!r}"
    )
