"""Extension seam: model-routing statistics for ``/stats`` and the dashboard.

A proxy extension that rewrites which model serves a request can register a
provider here; the ``/stats`` payload then carries a ``routing`` section and
the dashboard renders a "Model Routing" table. Mirrors the lossless-provider
seam (:mod:`headroom.transforms.lossless_provider`): the core defines the
contract and renders the data — it does not know who computes it, and it
stays inert when nothing registers.

Provider contract — return ``None`` (nothing to show) or::

    {
      "decisions": int,          # total routing decisions observed
      "downgrades": int,         # requests actually served on a cheaper model
      "upgrades": int,           # requests actually served on a pricier model
      "unchanged": int,          # decisions that left the request untouched
      "pairs": [                 # requested -> served aggregation
        {"requested": str, "served": str,
         "direction": "downgrade" | "upgrade" | "same" | "lateral" | "unknown",
         "count": int,           # decisions for this pair
         "enforced": int,        # of those, how many rewrote the wire
         "holdout": int,         # kept unrouted as a control arm
         "measured": int,        # rows with a measured cost baseline
         "savings_usd": float},  # signed; negative for upgrades
        ...
      ],

      # Optional, and strongly preferred. Routing savings accrue ACROSS
      # sessions, so a durable provider should report all time above -- scoping
      # the table to one proxy process understates the layer badly. But a
      # lifetime total moves ~0.1% per turn, so on its own it looks frozen while
      # the router is in fact evaluating every request. Send ``session`` too and
      # the dashboard renders a live counter beside the table.
      "window": "lifetime" | "session",   # scope of the numbers above
      "session": {"decisions": int, "downgrades": int, "upgrades": int,
                  "unchanged": int, "since": float},   # this process only
    }

Unknown keys are passed through untouched, so a provider may add fields ahead
of the dashboard learning to render them. Absent ``window``, the dashboard
labels the table "all decisions on record" rather than claiming a scope it
cannot verify.

The provider is called on ``/stats`` builds (the dashboard polls a cached
snapshot), so it should be cheap — an aggregate query, not a table walk.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

log = logging.getLogger(__name__)

_provider: Callable[[], dict | None] | None = None


def set_routing_stats_provider(fn: Callable[[], dict | None]) -> None:
    """Register (or replace) the routing-stats provider."""
    global _provider
    _provider = fn


def clear_routing_stats_provider() -> None:
    """Test/reset helper."""
    global _provider
    _provider = None


def get_routing_stats() -> dict | None:
    """The registered provider's snapshot, or ``None``.

    Never raises: a broken provider must not take down ``/stats``. A payload
    without a non-empty ``pairs`` list is treated as "nothing to show" so the
    dashboard has one rule for hiding the section.
    """
    if _provider is None:
        return None
    try:
        out = _provider()
    except Exception:
        log.debug("routing stats provider failed", exc_info=True)
        return None
    if not isinstance(out, dict) or not out.get("pairs"):
        return None
    return out
