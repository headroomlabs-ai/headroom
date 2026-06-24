"""Parity test: Rust-backed `DiffCompressor` vs recorded fixtures.

Stage 3b verification, post-deletion. The Python implementation has
been retired; the only remaining `DiffCompressor` is Rust-backed via
PyO3 (`headroom._core`). These tests guard against regressions by
replaying every recorded fixture in
`tests/parity/fixtures/diff_compressor/` through the Rust backend and
asserting the output matches the recording byte-for-byte.

Skipped automatically when the `headroom._core` wheel isn't installed
(e.g. CI lane without the maturin step). The Rust crate's own
`cargo run -p headroom-parity` covers the same fixtures from the Rust
side; this Python-side test catches PyO3 bridge regressions
(input/output mistranslation) that the Rust-only test cannot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _has_core() -> bool:
    try:
        import headroom._core  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_core(),
    reason="headroom._core wheel not installed (run `scripts/build_rust_extension.sh`)",
)


_FIXTURES_DIR = Path(__file__).parent.parent / "parity" / "fixtures" / "diff_compressor"


def _all_fixtures() -> list[Path]:
    return sorted(_FIXTURES_DIR.glob("*.json"))


def test_at_least_27_fixtures_present():
    """Sanity check: the bug-fix and routing-gap fixtures landed."""
    fixtures = _all_fixtures()
    assert len(fixtures) >= 27, (
        f"expected >= 27 fixtures, found {len(fixtures)}. "
        "If you re-recorded and got fewer, something deleted them."
    )


@pytest.mark.parametrize("fixture_path", _all_fixtures(), ids=lambda p: p.name)
def test_rust_backend_matches_recorded_output(fixture_path: Path):
    """Replay each recorded input through the Rust backend; every output
    field must match the recording. Any mismatch is a PyO3 bridge bug or a
    real Rust regression (cross-check with `cargo run -p headroom-parity`
    to localize).
    """
    from headroom.transforms.diff_compressor import DiffCompressor, DiffCompressorConfig

    rec = json.loads(fixture_path.read_text())
    cfg = DiffCompressorConfig(**rec["config"])
    result = DiffCompressor(cfg).compress(rec["input"])

    expected = rec["output"]
    # Field-by-field surfaces a single divergence rather than dumping the
    # whole result on failure.
    assert result.compressed == expected["compressed"], f"{fixture_path.name}: compressed differs"
    assert result.original_line_count == expected["original_line_count"]
    assert result.compressed_line_count == expected["compressed_line_count"]
    assert result.files_affected == expected["files_affected"]
    assert result.additions == expected["additions"]
    assert result.deletions == expected["deletions"]
    assert result.hunks_kept == expected["hunks_kept"]
    assert result.hunks_removed == expected["hunks_removed"]
    assert result.cache_key == expected["cache_key"]


def test_compress_with_stats_returns_python_dataclass_and_pyo3_stats():
    """The sidecar stats API returns a `(DiffCompressionResult,
    _core.DiffCompressorStats)` tuple. Verify both halves are usable from
    Python — PyO3 doesn't auto-promote the stats class to a dataclass.
    """
    from headroom.transforms.diff_compressor import (
        DiffCompressionResult,
        DiffCompressor,
        DiffCompressorConfig,
    )

    diff = (
        "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
    ) + "# pad\n" * 60
    result, stats = DiffCompressor(DiffCompressorConfig()).compress_with_stats(diff)

    assert isinstance(result, DiffCompressionResult)
    assert stats.input_lines >= 60
    assert stats.processing_duration_us >= 0
    assert isinstance(stats.parse_warnings, list)


def test_content_router_uses_rust_backend():
    """Stage 3b deletion check: ContentRouter._get_diff_compressor must
    return the (now Rust-only) DiffCompressor — there's no other backend
    left, so this guards against an accidental re-introduction of a
    Python-side stub or fallback.
    """
    from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
    from headroom.transforms.diff_compressor import DiffCompressor

    router = ContentRouter(ContentRouterConfig())
    compressor = router._get_diff_compressor()
    assert isinstance(compressor, DiffCompressor)
    # The Rust delegation handle must be live — guards against a
    # half-deleted shim that defines the class but not `_rust`.
    assert hasattr(compressor, "_rust")
