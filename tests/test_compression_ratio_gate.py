"""Compression-ratio drift gate (deterministic, offline, zero-model).

Pipes a committed corpus through every text compressor and asserts the
measured compression percentages stay inside the committed baseline bands.
Scoring is pure byte/token accounting (``headroom.evals.runners.compression_ratio``)
— no LLM calls, no network, no randomness — so this runs in the standard
``[dev]`` CI shard as a blocking PR check, and again under ``[all]`` extras
in the eval workflow where the tree-sitter / trafilatura / Kompress
compressors are also present.

Why this exists: compression percentage used to drift release-to-release
with no indicator. A failure here means a changeset moved a compressor's
measured ratio beyond its tolerance band — intended improvements require
regenerating the baseline so the drift is visible in the diff:

    python tests/fixtures/compression_ratio_golden/_generate.py

The corpus and baseline live under ``tests/fixtures/compression_ratio_golden/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.evals.runners.compression_ratio import (
    COMPRESSOR_NAMES,
    CompressorUnavailable,
    compressor_available,
    measure_case,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "compression_ratio_golden"
CASES: list[dict] = json.loads((FIXTURE_DIR / "cases.json").read_text())
BASELINE: dict = json.loads((FIXTURE_DIR / "baseline.json").read_text())

REGEN_HINT = (
    "If this compression change is intended, regenerate the baseline with "
    "`python tests/fixtures/compression_ratio_golden/_generate.py` and review "
    "the baseline.json diff in the PR."
)


def _case_key(case: dict) -> str:
    return f"{case['compressor']}/{case['id']}"


@pytest.mark.parametrize("case", CASES, ids=[_case_key(c) for c in CASES])
def test_compression_ratio_within_baseline(case: dict) -> None:
    """Each case's measured byte/token ratios must stay inside the band."""
    key = _case_key(case)
    entry = BASELINE["cases"].get(key)
    assert entry is not None, f"missing baseline entry for {key!r} — regenerate the baseline"

    ok, reason = compressor_available(case["compressor"])
    if not ok:
        pytest.skip(f"{case['compressor']} unavailable here: {reason}")
    if "unavailable" in entry:
        pytest.skip(
            f"{key} baseline was generated without the compressor backend "
            f"({entry['unavailable']}); regenerate on a host with the backend installed"
        )

    try:
        m = measure_case(case)
    except CompressorUnavailable as e:
        pytest.skip(str(e))

    tol = entry["tolerance"]
    for metric in ("byte_ratio", "token_ratio"):
        expected = entry[metric]
        actual = getattr(m, metric)
        assert expected - tol <= actual <= expected + tol, (
            f"RATIO DRIFT in {key}: {metric} {actual:.4f} outside baseline band "
            f"{expected:.4f} ± {tol} (bytes {m.bytes_before}->{m.bytes_after}, "
            f"tokens {m.tokens_before}->{m.tokens_after}). {REGEN_HINT}"
        )


def test_corpus_covers_every_compressor() -> None:
    """Every registered text compressor must have corpus cases.

    Guards the inventory: a new built-in compressor added to
    ``_BUILTIN_COMPRESSOR_DESCRIPTORS`` without corpus coverage fails here
    instead of drifting unmeasured. ``image`` is excluded — it transforms
    image message blocks, not piped text.
    """
    covered = {case["compressor"] for case in CASES}
    missing = [name for name in COMPRESSOR_NAMES if name not in covered]
    assert not missing, (
        f"corpus has no cases for compressor(s): {missing}. Add cases in "
        "tests/fixtures/compression_ratio_golden/_generate.py and regenerate."
    )

    try:
        from headroom.transforms.content_router import _BUILTIN_COMPRESSOR_DESCRIPTORS
    except ImportError:
        return
    registry_names = {d.name for d in _BUILTIN_COMPRESSOR_DESCRIPTORS} - {"image"}
    uncovered = registry_names - covered
    assert not uncovered, (
        f"built-in compressor(s) {sorted(uncovered)} have no corpus cases. "
        "Add cases in tests/fixtures/compression_ratio_golden/_generate.py and regenerate."
    )


def test_aggregate_ratio_within_band() -> None:
    """Per-compressor mean byte ratio must not drift as a group.

    A systematic shift across a compressor's cases (all moving the same
    direction, each inside its own band) still trips the gate.
    """
    measured: dict[str, list[float]] = {}
    for case in CASES:
        name = case["compressor"]
        key = _case_key(case)
        entry = BASELINE["cases"].get(key, {})
        if "unavailable" in entry:
            continue
        ok, _reason = compressor_available(name)
        if not ok:
            continue
        try:
            measured.setdefault(name, []).append(measure_case(case).byte_ratio)
        except CompressorUnavailable:
            continue

    for name, ratios in measured.items():
        expected = BASELINE["aggregates"].get(name)
        if expected is None:
            continue
        tol = BASELINE["cases"][
            f"{name}/{next(c['id'] for c in CASES if c['compressor'] == name)}"
        ]["tolerance"]
        actual = sum(ratios) / len(ratios)
        assert expected - tol <= actual <= expected + tol, (
            f"AGGREGATE RATIO DRIFT in {name}: mean byte_ratio {actual:.4f} outside "
            f"baseline band {expected:.4f} ± {tol}. {REGEN_HINT}"
        )
