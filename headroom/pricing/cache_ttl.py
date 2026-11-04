"""Prompt-cache TTL pricing structure.

Anthropic prices prompt-cache traffic as fixed multiples of a model's base input
rate rather than as independent per-model numbers:

===============  ==========  ==================================================
Traffic          Multiplier  Notes
===============  ==========  ==================================================
cache read       ``0.10x``   Any TTL — reads are the same price either way.
cache write 5m   ``1.25x``   The default TTL.
cache write 1h   ``2.00x``   Survives 12x longer for 1.6x the write price.
===============  ==========  ==================================================

Because the ratios are structural, they are derived from the base input price
here instead of being transcribed into every row of
:mod:`headroom.pricing.anthropic_prices` — a per-model table would be 3x the
columns, would drift, and would let one tier be mis-typed without anything
noticing. :class:`~headroom.pricing.registry.ModelPricing` already carries
``cached_input_per_1m`` for reads; this module adds the *write* side, which the
registry has no field for.

Scope: these ratios are Anthropic's. Other providers differ structurally, not
just numerically — OpenAI, for instance, discounts cached input but does not
bill for cache writes at all, so there is no 5m/1h trade to make there.

The TTL trade has two terms and both must be counted. Switching a workload to
the 1h TTL turns idle-gap rewrites into cheap reads, but it also raises the
price of *every* write that still happens. Modelling only the recovery
overstates the saving — measured at 1.9x on a real 531-transcript corpus — so
:func:`ttl_breakeven_share` exists to make the threshold explicit.
"""

from __future__ import annotations

from typing import Literal

CacheTTL = Literal["5m", "1h"]

#: Cache reads cost this multiple of the base input rate, at either TTL.
CACHE_READ_MULTIPLIER = 0.10

#: Cache writes cost this multiple of the base input rate, by TTL.
CACHE_WRITE_MULTIPLIERS: dict[str, float] = {"5m": 1.25, "1h": 2.00}

#: Anthropic's default when a request does not ask for the extended TTL.
DEFAULT_CACHE_TTL: CacheTTL = "5m"


def cache_write_multiplier(ttl: str) -> float:
    """Return the base-input multiplier for a cache write at ``ttl``.

    Raises ``ValueError`` on an unknown TTL rather than silently falling back to
    the cheaper 5m rate, which would understate cost.
    """
    try:
        return CACHE_WRITE_MULTIPLIERS[ttl]
    except KeyError:
        known = ", ".join(sorted(CACHE_WRITE_MULTIPLIERS))
        raise ValueError(f"unknown cache TTL {ttl!r} (known: {known})") from None


def cache_rates_per_1m(input_per_1m: float) -> dict[str, float]:
    """Derive per-1M-token cache rates from a model's base input rate.

    Returns ``{"read": …, "write_5m": …, "write_1h": …}`` in the same units as
    ``input_per_1m`` (USD per 1M tokens).
    """
    return {
        "read": input_per_1m * CACHE_READ_MULTIPLIER,
        "write_5m": input_per_1m * CACHE_WRITE_MULTIPLIERS["5m"],
        "write_1h": input_per_1m * CACHE_WRITE_MULTIPLIERS["1h"],
    }


def ttl_breakeven_share() -> float:
    """Return the share of cache writes above which the 1h TTL is cheaper.

    Let ``W`` be total cache-write tokens and ``G`` the subset written again
    after an idle gap longer than the 5m TTL but shorter than 1h. Under the 5m
    default the workload pays ``W * write_5m``. Under the 1h TTL the ``G``
    tokens become reads while the remaining ``W - G`` writes pay the higher
    rate::

        (W - G) * write_1h + G * read  <  W * write_5m
        =>  G / W  >  (write_1h - write_5m) / (write_1h - read)

    The threshold is scale- and model-independent because every term is a
    multiple of the same base input rate: ``(2.00 - 1.25) / (2.00 - 0.10)``,
    i.e. **39.5%**.
    """
    w5 = CACHE_WRITE_MULTIPLIERS["5m"]
    w1h = CACHE_WRITE_MULTIPLIERS["1h"]
    return (w1h - w5) / (w1h - CACHE_READ_MULTIPLIER)
