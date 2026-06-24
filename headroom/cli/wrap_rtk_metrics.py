"""RTK invocation metrics for the wrap CLI.

Phase G PR-G3 remediation (C4): RTK lives wrap-side, not proxy-side
(see ``docs/rtk-architecture.md``). The wrap CLI tails
``rtk gain --format json`` and bumps a process-local counter keyed
by rewritten command name (`git`, `ls`, `cargo`, ...). The Python
proxy's ``/metrics`` endpoint then surfaces the counter as
``wrap_rtk_invocations_total{tool=...}`` for fleet-wide scrape.

The counter primitives live here (not in ``wrap.py``) so the
proxy's prometheus exporter can import them without dragging in the
full ``wrap.py`` module — that module owns subprocess-level CLI
spawning and is heavyweight to import at proxy startup.

Per realignment build-constraint "no silent fallbacks":
``record_rtk_invocation`` raises on a non-string tool name rather
than coercing; a caller passing the wrong type is a bug, not a
runtime fallback condition.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Mapping

# Module-level counter — process-local. Multiple worker processes
# (uvicorn workers) each maintain their own; the Python proxy already
# documents this in ``docs/observability.md``. Reset is exposed for
# tests; production code never reaches for it.
_rtk_invocation_counts: dict[str, int] = defaultdict(int)
_lock = threading.Lock()


def record_rtk_invocation(tool: str, delta: int = 1) -> None:
    """Record one (or `delta`) RTK invocation(s) for the given tool.

    `tool` is the rewritten command name as observed in the
    ``rtk gain --format json`` output (e.g. ``"git"``, ``"ls"``,
    ``"cargo"``). The counter is keyed verbatim.

    `delta` defaults to 1 for the common "one invocation seen" path
    but accepts arbitrary positive deltas so the wrap tail can bump
    by a JSON-reported batch count.

    Raises:
        TypeError: if `tool` is not a `str` or `delta` is not an `int`.
        ValueError: if `delta` is negative.
    """
    if not isinstance(tool, str):
        raise TypeError(f"tool must be a str, got {type(tool).__name__}")
    if not isinstance(delta, int):
        raise TypeError(f"delta must be an int, got {type(delta).__name__}")
    if delta < 0:
        raise ValueError(f"delta must be non-negative, got {delta}")
    with _lock:
        _rtk_invocation_counts[tool] += delta


def rtk_invocation_counts() -> Mapping[str, int]:
    """Return a snapshot of the current invocation counts.

    Returns a plain dict (not the defaultdict) so callers cannot
    accidentally pollute the counter map by reading absent keys.
    """
    with _lock:
        return dict(_rtk_invocation_counts)


def reset_rtk_invocations() -> None:
    """Reset the counter map. Test-only — never called from production."""
    with _lock:
        _rtk_invocation_counts.clear()
