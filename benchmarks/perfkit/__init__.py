"""perfkit: canonical, multi-granularity performance testing harness for Headroom.

This package is the measurement infrastructure (not the optimization PRs that
land separately). It implements the contracts in
``specs/002-perf-testing-granularities/contracts/``. It is source-only; run
``python -m benchmarks.perfkit.cli`` from a repository checkout.

Layering (kept strictly bottom-up so each tier depends only on what's below it):

* :mod:`perfkit.schema`      - canonical artifact dataclasses (schema 1.0.0)
* :mod:`perfkit.fixtures`    - immutable, content-hashed fixture loader
* :mod:`perfkit.attribution` - stable-id chain across tiers
* :mod:`perfkit.noise_floor` - per-tier/per-metric noise floor table
* :mod:`perfkit.baseline`    - pinned baseline + first-record policy
* :mod:`perfkit.artifact_io` - read/write JSON + Markdown rendering
* :mod:`perfkit.comparator`  - regression verdict engine (G6)
* :mod:`perfkit.cli`         - source-only maintainer CLI
* :mod:`perfkit.tiers.*`     - benchmark tiers G1..G6
* :mod:`perfkit.mock_upstream` - deterministic, loop-back upstream server
"""

from __future__ import annotations

__all__ = ["__version__"]
__version__ = "1.0.0"
