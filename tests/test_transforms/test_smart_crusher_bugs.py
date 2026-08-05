"""Regression tests for SmartCrusher bugs.

Bug 1: _crush_number_array mixes types (string summary + numbers),
       violating the schema-preserving guarantee.
Bug 2: _current_field_semantics is shared instance state, creating
       a race condition when crushing concurrently.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from headroom import SmartCrusherConfig
from headroom.transforms.smart_crusher import SmartCrusher

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_crusher(max_items: int = 10, min_items: int = 3) -> SmartCrusher:
    """Build a SmartCrusher with deterministic small-K config for tests."""
    config = SmartCrusherConfig(
        enabled=True,
        min_items_to_analyze=min_items,
        min_tokens_to_crush=0,
        max_items_after_crush=max_items,
        variance_threshold=2.0,
    )
    return SmartCrusher(config=config)


# Bug #1 (number array schema preservation) — invariant pinned by the
# Rust port (`crates/headroom-core/src/transforms/smart_crusher/crushers.rs::
# crush_number_array` + its unit tests) and the parity fixtures
# (`tests/parity/fixtures/smart_crusher/number_array_40_changepoint*`).
# The Python `_crush_number_array` helper that the previous tests
# probed was removed when the Python implementation was retired in
# Stage 3c.1b.

# ---------------------------------------------------------------------------
# Bug 2: Race condition on _current_field_semantics
# ---------------------------------------------------------------------------


class TestFieldSemanticsThreadSafety:
    """_current_field_semantics must not leak between concurrent crushes.

    Previously it was stored as instance state (self._current_field_semantics)
    which created a race condition when the same SmartCrusher instance
    was used from multiple threads.
    """

    def test_concurrent_crushes_no_cross_contamination(self) -> None:
        """Two concurrent crushes must not share field_semantics state."""
        crusher = _make_crusher(max_items=5)

        # Two different array payloads
        payload_a = json.dumps([{"name": f"item_{i}", "value": i} for i in range(20)])
        payload_b = json.dumps([{"key": f"k_{i}", "score": i * 0.1} for i in range(20)])

        results: dict[str, str] = {}
        errors: list[Exception] = []

        def crush_task(label: str, content: str) -> None:
            try:
                result, modified, info = crusher._smart_crush_content(content)
                results[label] = result
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            # Run many concurrent crushes to increase race probability
            for i in range(20):
                futures.append(executor.submit(crush_task, f"a_{i}", payload_a))
                futures.append(executor.submit(crush_task, f"b_{i}", payload_b))
            for f in as_completed(futures):
                f.result()  # Re-raise exceptions

        assert not errors, f"Concurrent crushes raised errors: {errors}"

        # After all crushes, thread-local state must be clean
        tl = getattr(crusher, "_thread_local", None)
        if tl is not None:
            semantics = getattr(tl, "field_semantics", None)
            assert semantics is None, f"field_semantics leaked in thread-local: {semantics}"


# ---------------------------------------------------------------------------
# Issue 7: Recursion depth limit
# ---------------------------------------------------------------------------


class TestRecursionDepthLimit:
    """_process_value must not crash on deeply nested JSON."""

    def test_deeply_nested_json_does_not_crash(self) -> None:
        """Nesting deeper than _MAX_PROCESS_DEPTH should return value unchanged."""
        crusher = _make_crusher()
        # Build a 100-level nested structure
        nested: dict = {"leaf": "value"}
        for _i in range(100):
            nested = {"level": nested}

        content = json.dumps(nested)
        result, was_modified, info = crusher._smart_crush_content(content)
        # Should not raise RecursionError
        parsed = json.loads(result)
        # The deep structure should be preserved (returned as-is past depth limit)
        assert isinstance(parsed, dict)

    def test_deeply_nested_list_does_not_crash(self) -> None:
        """Deeply nested lists should also be handled safely."""
        crusher = _make_crusher()
        nested: list = ["leaf"]
        for _i in range(100):
            nested = [nested]

        content = json.dumps(nested)
        result, was_modified, info = crusher._smart_crush_content(content)
        parsed = json.loads(result)
        assert isinstance(parsed, list)


class TestLosslessOnlyMode:
    """`lossless_only` produces marker-free, byte-recoverable output.

    Strict mode: lossless tabular compaction still applies, but any path
    that would need a CCR marker (lossy row-drop OR opaque-blob offload)
    leaves the content uncompacted instead — so the result is always
    marker-free and decodes back to the original input without loss.
    """

    def _droppable_rows(self) -> list[dict]:
        return [{"path": "a.py", "line": i, "content": "x" * 300} for i in range(50)]

    def test_lossless_only_is_marker_free_and_byte_recoverable(self) -> None:
        rows = self._droppable_rows()
        config = SmartCrusherConfig(
            min_items_to_analyze=3,
            min_tokens_to_crush=0,
            lossless_min_savings_ratio=0.99,  # force the would-be-lossy path
            lossless_only=True,
        )
        out = SmartCrusher(config=config).crush(json.dumps(rows))
        assert "<<ccr:" not in out.compressed
        assert json.loads(out.compressed) == rows

    def test_crush_kwarg_overrides_configured_mode(self) -> None:
        # Configured non-strict, but the per-call kwarg forces strict mode
        # for this call: marker-free and fully recoverable.
        rows = self._droppable_rows()
        config = SmartCrusherConfig(
            min_items_to_analyze=3,
            min_tokens_to_crush=0,
            lossless_min_savings_ratio=0.99,
        )
        crusher = SmartCrusher(config=config)
        out = crusher.crush(json.dumps(rows), lossless_only=True)
        assert "<<ccr:" not in out.compressed
        assert json.loads(out.compressed) == rows


# Stage 3c.1 lockstep bug-fix tests previously lived here; they probed
# Python helpers (`_percentile_linear`, `_detect_sequential_pattern`,
# `_detect_rare_status_values`, `_compute_k_split`) that were removed
# along with the Python implementation in Stage 3c.1b. The Rust port
# pins the same invariants — see the `bug1_*` / `bug2_*` / `bug3_*` /
# `bug4_*` tests in `crates/headroom-core/src/transforms/smart_crusher/`
# (notably `crushers.rs` and `analyzer.rs`). Parity fixtures
# (`tests/parity/fixtures/smart_crusher/`) byte-compare the post-fix
# behavior across the language boundary.
