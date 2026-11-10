"""Source-only maintainer CLI for the perfkit suite.

Subcommands (per FR-015 + spec US1/US2/US3/US4/US5):

* ``run``        — execute one or more tiers and emit artifacts.
* ``pr-guard``   — per-PR regression guard (G1+G2+G3 vs. pinned baselines).
* ``release``    — full release-perf (G1+G2+G3+G4+G6) with release-notes report.
* ``soak``       — long-running G5 trend capture.
* ``compare``    — two-ref, per-tier delta table with significance verdict.
* ``update-baseline`` — explicit first-record / intentional-shift policy.

Run ``python -m benchmarks.perfkit.cli ...`` from a repository checkout.
This maintainer tool is not installed as a console script.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

# Default repo-relative directories.
DEFAULT_BASELINES_DIR = Path("benchmarks/baselines")
DEFAULT_ARTIFACTS_DIR = Path("benchmarks/artifacts")
DEFAULT_BASELINE_DIR_DOC = "Directory with pinned baseline JSON files."
DEFAULT_ARTIFACTS_DIR_DOC = "Directory in which run artifacts (JSON + .md) are written."


# Exit codes per contracts/regression-verdict.md §7.
EXIT_OK = 0
EXIT_FAIL = 2  # regression verdict
EXIT_BASELINE_MISSING = 3
EXIT_BASELINE_INVALIDATED = 4
EXIT_TOOL_ERROR = 10


def build_parser() -> argparse.ArgumentParser:
    """Compose the argument parser used by ``main`` and unit tests."""

    p = argparse.ArgumentParser(
        prog="python -m benchmarks.perfkit.cli",
        description="Multi-granularity performance harness for Headroom.",
    )
    p.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINES_DIR,
        help=DEFAULT_BASELINE_DIR_DOC,
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help=DEFAULT_ARTIFACTS_DIR_DOC,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="Run one or more benchmark tiers and emit artifacts.")
    sub.add_parser("pr-guard", help="Per-PR regression guard (G1+G2+G3).")
    sub.add_parser("release", help="Full release-perf suite.")
    sub.add_parser("soak", help="Long-running G5 trend capture.")
    sub.add_parser("compare", help="Two-ref delta table with significance verdict.")

    upd = sub.add_parser(
        "update-baseline",
        help="Explicitly record or amend a baseline (CODEOWNER-gated).",
    )
    upd.add_argument(
        "--reason",
        required=True,
        choices=[
            "first_record",
            "approved_intended_shift",
            "schema_bump",
            "fixture_change",
            "floor_change",
        ],
        help="Governance reason for the baseline update.",
    )

    for sp in sub.choices.values():
        sp.add_argument("--tiers", help="Comma-separated tier list (G1..G6).")
        sp.add_argument("--skip-tier", help="Comma-separated tiers to skip.")
        sp.add_argument("--quick", action="store_true", help="Run a fast subset for PR CI.")
        sp.add_argument(
            "--markdown", action="store_true", help="Emit Markdown summary next to each JSON."
        )

    return p


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse ``argv`` (defaults to ``sys.argv[1:]``)."""

    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the maintainer CLI from a source checkout."""

    args = parse_args(sys.argv[1:] if argv is None else argv)

    # Dispatch is lazy to keep ``import time`` low (the parser already pulls in
    # the heavy ``benchmark/*`` modules implicitly via the dispatched commands).
    cmd = args.cmd
    if cmd == "run":
        return _cmd_run(args)
    if cmd == "pr-guard":
        return _cmd_pr_guard(args)
    if cmd == "release":
        return _cmd_release(args)
    if cmd == "soak":
        return _cmd_soak(args)
    if cmd == "compare":
        return _cmd_compare(args)
    if cmd == "update-baseline":
        return _cmd_update_baseline(args)

    print(f"unknown subcommand: {cmd!r}", file=sys.stderr)
    return EXIT_TOOL_ERROR


# ---------------------------------------------------------------------------
# Subcommand dispatch.
#
# These are intentionally tiny so unit tests can invoke them with mocked
# tiers. Phase 2 wires the real implementations.
# ---------------------------------------------------------------------------


def _split_tiers(arg: str | None) -> list[str] | None:
    if not arg:
        return None
    return [t.strip() for t in arg.split(",") if t.strip()]


def _cmd_run(args: argparse.Namespace) -> int:
    """Execute one or more tiers and emit artifacts.

    Phase 2 (T014-T022) supplies the tier implementations; the dispatcher
    records ``run`` as the no-op happy path until then.
    """

    from .tiers import run_tiers  # local import to avoid heavy CLI-time cost

    tiers = _split_tiers(args.tiers) or ["G1", "G2", "G3"]
    skip = set(_split_tiers(args.skip_tier) or ())
    code = run_tiers(
        tiers=tiers,
        skip=skip,
        quick=bool(args.quick),
        artifacts_dir=args.artifacts_dir,
        baseline_dir=args.baseline_dir,
        markdown=bool(args.markdown),
    )
    return code


def _cmd_pr_guard(args: argparse.Namespace) -> int:
    """Per-PR guard: G1+G2+G3 vs. pinned baseline; exit 2 on regression."""

    from .tiers import pr_guard

    tiers = _split_tiers(args.tiers) or ["G1", "G2", "G3"]
    skip = set(_split_tiers(args.skip_tier) or ())
    return pr_guard(
        tiers=tiers,
        skip=skip,
        quick=bool(args.quick),
        artifacts_dir=args.artifacts_dir,
        baseline_dir=args.baseline_dir,
        markdown=bool(args.markdown),
    )


def _cmd_release(args: argparse.Namespace) -> int:
    """Release-perf suite (G1+G2+G3+G4+G6, no G5)."""

    from .tiers import release_perf

    tiers = _split_tiers(args.tiers) or ["G1", "G2", "G3", "G4", "G6"]
    skip = set(_split_tiers(args.skip_tier) or ())
    return release_perf(
        tiers=tiers,
        skip=skip,
        artifacts_dir=args.artifacts_dir,
        baseline_dir=args.baseline_dir,
        markdown=bool(args.markdown),
    )


def _cmd_soak(args: argparse.Namespace) -> int:
    """Long-running G5 trend capture."""

    from .tiers import g5_soak

    tiers = _split_tiers(args.tiers) or ["G5"]
    return g5_soak(
        artifacts_dir=args.artifacts_dir,
        baseline_dir=args.baseline_dir,
        tiers=tiers,
    )


def _cmd_compare(args: argparse.Namespace) -> int:
    """Two-ref per-tier delta with significance verdict."""

    from .tiers import compare

    tiers = _split_tiers(args.tiers) or ["G4", "G3", "G2", "G1"]
    return compare(
        tiers=tiers,
        artifacts_dir=args.artifacts_dir,
        baseline_dir=args.baseline_dir,
    )


def _cmd_update_baseline(args: argparse.Namespace) -> int:
    """Explicit baseline update (CODEOWNER-gated; check governance fields)."""

    from .tiers import update_baseline

    return update_baseline(
        reason=args.reason,
        artifacts_dir=args.artifacts_dir,
        baseline_dir=args.baseline_dir,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
