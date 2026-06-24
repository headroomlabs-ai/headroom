#!/usr/bin/env python3
"""Entry point for recording Rust-vs-Python parity fixtures.

Usage:
    python scripts/record_fixtures.py

This driver:
  1. Monkey-patches the Phase-1 transform classes via
     `tests.parity.recorder.record_all()`.
  2. Runs a small deterministic synthetic workload (no network, no real LLM
     calls) that hits each transform at least 20 times with varied inputs.
  3. Prints a summary and exits non-zero if any transform was blocked.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Make `tests/parity/recorder.py` importable without installing it.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from tests.parity.recorder import record_all, run_default_workload  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("record_fixtures")

    statuses = record_all()
    log.info("patch status:")
    blocked = []
    for name, status in statuses.items():
        log.info("  %-16s %s", name, status)
        if status.startswith("blocked:"):
            blocked.append((name, status))

    counts = run_default_workload()
    log.info("fixture counts:")
    shortfall = []
    for name, n in counts.items():
        log.info("  %-16s %d", name, n)
        if n < 20:
            shortfall.append((name, n))

    if blocked:
        log.error("blocked transforms: %s", blocked)
    if shortfall:
        log.error("transforms with <20 fixtures: %s", shortfall)

    # Exit non-zero if anything was wholly blocked; short recordings are
    # a soft warning.
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
