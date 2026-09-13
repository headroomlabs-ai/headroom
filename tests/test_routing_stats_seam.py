"""The routing-stats seam: what makes the dashboard's Model Routing panel appear.

The panel is gated on a non-empty ``pairs`` list, so anything that empties it
removes the whole section from the dashboard with no error anywhere. That is
not hypothetical: scoping a durable provider to the current proxy process made
``pairs`` empty on every fresh start, and the panel silently vanished.
"""

from __future__ import annotations

import pytest

from headroom.proxy.routing_stats import (
    clear_routing_stats_provider,
    get_routing_stats,
    set_routing_stats_provider,
)


@pytest.fixture(autouse=True)
def _clean():
    clear_routing_stats_provider()
    yield
    clear_routing_stats_provider()


def _matrix(**over):
    base = {
        "decisions": 3,
        "downgrades": 1,
        "upgrades": 0,
        "unchanged": 2,
        "pairs": [
            {
                "requested": "claude-opus-5",
                "served": "claude-sonnet-5",
                "direction": "downgrade",
                "count": 1,
                "enforced": 1,
                "holdout": 0,
                "measured": 1,
                "savings_usd": 0.42,
            }
        ],
    }
    base.update(over)
    return base


def test_no_provider_is_inert():
    assert get_routing_stats() is None


def test_provider_payload_passes_through():
    set_routing_stats_provider(lambda: _matrix())
    out = get_routing_stats()
    assert out["decisions"] == 3
    assert out["pairs"][0]["served"] == "claude-sonnet-5"


def test_empty_pairs_hides_the_section():
    """The rule the dashboard depends on -- and the trap it sets."""
    set_routing_stats_provider(lambda: _matrix(pairs=[], decisions=0))
    assert get_routing_stats() is None


def test_a_broken_provider_never_breaks_stats():
    def boom():
        raise RuntimeError("decision log is locked")

    set_routing_stats_provider(boom)
    assert get_routing_stats() is None


def test_unknown_keys_survive_so_providers_can_lead_the_dashboard():
    """`window` and `session` reach the template even on an older core."""
    set_routing_stats_provider(
        lambda: _matrix(
            window="lifetime",
            session={
                "decisions": 4,
                "downgrades": 3,
                "upgrades": 0,
                "unchanged": 1,
                "since": 1788140194.8,
            },
        )
    )
    out = get_routing_stats()
    assert out["window"] == "lifetime"
    assert out["session"]["downgrades"] == 3


def test_live_session_counter_can_be_zero_while_the_panel_still_renders():
    """The regression this file exists for.

    A durable provider reports lifetime pairs (so the panel renders and shows
    accrued savings) while the session block legitimately reads zero on a proxy
    that has only just started. Reporting the SESSION in ``pairs`` instead
    emptied it and took the whole section down.
    """
    set_routing_stats_provider(
        lambda: _matrix(
            window="lifetime",
            session={
                "decisions": 0,
                "downgrades": 0,
                "upgrades": 0,
                "unchanged": 0,
                "since": 1788140194.8,
            },
        )
    )
    out = get_routing_stats()
    assert out is not None, "panel must survive a session with no decisions yet"
    assert out["pairs"], "lifetime pairs keep the section on screen"
    assert out["session"]["decisions"] == 0


def test_non_dict_payload_is_ignored():
    set_routing_stats_provider(lambda: ["not", "a", "matrix"])
    assert get_routing_stats() is None
