"""perfkit tier packages.

Phase 2 wires the real G1..G6 implementations here. Until then, this
``__init__`` provides the dispatch surface that :mod:`perfkit.cli` imports
so the CLI can be invoked without raising at import time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from ..cli import EXIT_OK


def run_tiers(
    *,
    tiers: Sequence[str],
    skip: Iterable[str] = (),
    quick: bool = False,
    artifacts_dir: Path | str,
    baseline_dir: Path | str,
    markdown: bool = True,
) -> int:
    """Run the listed tiers and emit artifacts. Stub until Phase 2 lands."""

    return EXIT_OK


def pr_guard(
    *,
    tiers: Sequence[str],
    skip: Iterable[str] = (),
    quick: bool = True,
    artifacts_dir: Path | str,
    baseline_dir: Path | str,
    markdown: bool = True,
) -> int:
    """Per-PR regression guard; stub until Phase 3 lands."""

    return EXIT_OK


def release_perf(
    *,
    tiers: Sequence[str],
    skip: Iterable[str] = (),
    artifacts_dir: Path | str,
    baseline_dir: Path | str,
    markdown: bool = True,
) -> int:
    """Release-perf suite; stub until Phase 5 lands."""

    return EXIT_OK


def g5_soak(
    *,
    artifacts_dir: Path | str,
    baseline_dir: Path | str,
    tiers: Sequence[str],
) -> int:
    """G5 long-running soak; stub until Phase 6 lands."""

    return EXIT_OK


def compare(
    *,
    tiers: Sequence[str],
    artifacts_dir: Path | str,
    baseline_dir: Path | str,
) -> int:
    """Two-ref comparison; stub until Phase 7 lands."""

    return EXIT_OK


def update_baseline(
    *,
    reason: str,
    artifacts_dir: Path | str,
    baseline_dir: Path | str,
) -> int:
    """Explicit baseline update; stub until Phase 3 lands."""

    return EXIT_OK
