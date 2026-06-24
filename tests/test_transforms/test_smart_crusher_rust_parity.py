"""Parity test: PyO3-backed `SmartCrusher` vs recorded fixtures.

Stage 3c.1b verification — guards the PyO3 bridge against regressions
by replaying every recorded fixture in
`tests/parity/fixtures/smart_crusher/` through `headroom._core.SmartCrusher`
and asserting the output matches the recording byte-for-byte.

Twin of `test_diff_compressor_rust_parity.py`. The Rust side runs the
same fixtures via `cargo run -p headroom-parity --bin parity-run --
run --only smart_crusher`; this Python test specifically catches PyO3
bridge regressions (input/output mistranslation) that the Rust-only
binary cannot.

Skipped automatically when the `headroom._core` wheel isn't installed
(e.g. CI lane without the maturin step).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _has_core() -> bool:
    try:
        from headroom._core import SmartCrusher  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _has_core(),
    reason="headroom._core wheel not installed (run `scripts/build_rust_extension.sh`)",
)


_FIXTURES_DIR = Path(__file__).parent.parent / "parity" / "fixtures" / "smart_crusher"


def _all_fixtures() -> list[Path]:
    return sorted(_FIXTURES_DIR.glob("*.json"))


def test_at_least_17_fixtures_present():
    """Sanity check: the recorded fixture suite landed."""
    fixtures = _all_fixtures()
    assert len(fixtures) >= 17, (
        f"expected >= 17 fixtures, found {len(fixtures)}. "
        "If you re-recorded and got fewer, something deleted them."
    )


@pytest.mark.parametrize("fixture_path", _all_fixtures(), ids=lambda p: p.name)
def test_rust_backend_matches_recorded_output(fixture_path: Path):
    """Replay each recorded input through the PyO3 bridge; every output
    field must match the recording. Any mismatch is a bridge bug or a
    Rust regression — cross-check with `cargo run -p headroom-parity`.
    """
    from headroom._core import SmartCrusher, SmartCrusherConfig

    fixture = json.loads(fixture_path.read_text())
    inp = fixture["input"]
    cfg_dict = fixture["config"]
    expected = fixture["output"]

    cfg = SmartCrusherConfig(**cfg_dict)
    # Legacy fixtures were recorded against the pre-PR4 lossy-only
    # path. Use `without_compaction()` to preserve byte-equal coverage
    # of that path; the new lossless default has its own coverage in
    # `test_smart_crusher_lossless_default.py`.
    crusher = SmartCrusher.without_compaction(cfg)
    actual = crusher.crush(inp["content"], inp["query"], inp["bias"])

    assert actual.compressed == expected["compressed"], (
        f"compressed bytes differ for {fixture_path.name}\n"
        f"  expected: {expected['compressed'][:120]!r}\n"
        f"  actual  : {actual.compressed[:120]!r}"
    )
    assert actual.original == expected["original"], f"original bytes differ for {fixture_path.name}"
    assert actual.was_modified == expected["was_modified"], (
        f"was_modified differs for {fixture_path.name}: "
        f"expected={expected['was_modified']} actual={actual.was_modified}"
    )
    assert actual.strategy == expected["strategy"], (
        f"strategy differs for {fixture_path.name}: "
        f"expected={expected['strategy']!r} actual={actual.strategy!r}"
    )
