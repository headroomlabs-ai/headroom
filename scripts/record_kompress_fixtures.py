#!/usr/bin/env python3
"""Record standard parity fixtures for the Kompress transform only.

Drives the Python `KompressCompressor` (enable_ccr=False) over the shared
`_varied_kompress_inputs()` workload while `record_all()` has the compress
method patched, so only `tests/parity/fixtures/kompress/` is (re)written —
no churn to other transforms' fixtures.

Run after the model is cached:
    python scripts/record_kompress_fixtures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    from tests.parity.recorder import _varied_kompress_inputs, record_all

    statuses = record_all()
    if not statuses.get("kompress", "").startswith("patched"):
        print(f"kompress not patched: {statuses.get('kompress')}", file=sys.stderr)
        return 1

    from headroom.transforms.kompress_compressor import (
        KompressCompressor,
        KompressConfig,
    )

    kc = KompressCompressor(KompressConfig(enable_ccr=False))
    inputs = _varied_kompress_inputs()
    for s in inputs:
        kc.compress(s)

    out_dir = REPO / "tests" / "parity" / "fixtures" / "kompress"
    n = len(list(out_dir.glob("*.json")))
    print(f"recorded {n} kompress fixtures from {len(inputs)} inputs -> {out_dir}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
