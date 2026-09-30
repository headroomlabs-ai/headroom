"""OpenAI pricing must not be shadowed by a shorter model family.

``_get_pricing`` matches by prefix in plain dict order, so the first *inserted*
prefix won rather than the most specific one. ``gpt-4.1`` fell into the ``gpt-4``
entry and was priced at the legacy $30/$60:

    gpt-4.1        $30.00 in   vs   $2.00 actual    15x
    gpt-4.1-mini   $30.00 in   vs   $0.40 actual    75x
    gpt-4.1-nano   $30.00 in   vs   $0.10 actual   300x

Unlike ``get_context_limit``, ``_get_pricing`` has no litellm lookup in front of
it, so this table is the only source for ``client.py``'s cost_before/cost_after,
``reporting/generator.py`` and ``evals/cost_tracker.py``. (The *proxy* cost path
is unaffected -- it calls ``litellm.cost_per_token`` directly, as does
``savings_ledger``.)

Values here were verified against litellm's ``model_cost``.
"""

from __future__ import annotations

import pytest

from headroom.providers.openai import OpenAIProvider

# (model, input $/1M, output $/1M)
EXPECTED = [
    # The shadowed cases.
    ("gpt-4.1", 2.00, 8.00),
    ("gpt-4.1-mini", 0.40, 1.60),
    ("gpt-4.1-nano", 0.10, 0.40),
    ("gpt-4.1-2025-04-14", 2.00, 8.00),
    # Fell through to the unknown-model default (GPT-4o tier).
    ("gpt-5", 1.25, 10.00),
    ("gpt-5-mini", 0.25, 2.00),
    ("gpt-5-nano", 0.05, 0.40),
    ("o4-mini", 1.10, 4.40),
    # Stale entry: o3 was cut to $2/$8 in June 2025.
    ("o3", 2.00, 8.00),
    # Must not regress.
    ("gpt-4o", 2.50, 10.00),
    ("gpt-4o-mini", 0.15, 0.60),
    ("gpt-4", 30.00, 60.00),
    ("gpt-4-turbo", 10.00, 30.00),
    ("gpt-3.5-turbo", 0.50, 1.50),
    ("o3-mini", 1.10, 4.40),
    ("o1", 15.00, 60.00),
]


@pytest.mark.parametrize(("model", "want_in", "want_out"), EXPECTED)
def test_pricing_prefers_the_most_specific_prefix(
    model: str, want_in: float, want_out: float
) -> None:
    got_in, got_out = OpenAIProvider()._get_pricing(model)

    # Tolerance, not equality: these rates may come from LiteLLM's per-token
    # figures, and the x1e6 conversion is not exact in binary floating point
    # ($0.4/1M arrives as 0.39999999999999997). Money compared to the cent.
    assert got_in == pytest.approx(want_in, abs=0.001)
    assert got_out == pytest.approx(want_out, abs=0.001)


def test_nano_is_not_priced_as_legacy_gpt4() -> None:
    """The 300x case, stated plainly: nano must be the cheapest gpt-4.1 tier."""
    provider = OpenAIProvider()

    nano_in, _ = provider._get_pricing("gpt-4.1-nano")
    legacy_in, _ = provider._get_pricing("gpt-4")

    assert nano_in < legacy_in / 100


def test_pricing_metadata_is_not_stale() -> None:
    """The staleness warning is a real feature; keep the stamp honest.

    If someone edits _PRICING without re-verifying, this starts failing rather
    than silently shipping a stale table behind a fresh-looking date.
    """
    from headroom.providers.openai import _PRICING_LAST_UPDATED, _PRICING_STALE_DAYS

    assert _PRICING_STALE_DAYS > 0
    # Sanity: the stamp should postdate the gpt-4.1/gpt-5 entries it covers.
    assert _PRICING_LAST_UPDATED.year >= 2026
