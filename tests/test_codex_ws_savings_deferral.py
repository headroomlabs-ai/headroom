"""Codex WS Responses: never record compression savings without input accounting.

``tokens_saved`` accumulates at compression time (our own count); input tokens
only arrive with a usage frame on ``response.completed``. A cancelled or failed
turn therefore produces a savings delta with no input delta — recording that
pair writes a savings-with-zero-spend checkpoint into the savings tracker and
desyncs every downstream funnel (dashboards flag "compression savings but zero
tokens spent").
"""

from __future__ import annotations

import re
from pathlib import Path

from headroom.proxy.handlers.openai import _deferrable_savings_delta

OPENAI_HANDLER = Path(__file__).parent.parent / "headroom" / "proxy" / "handlers" / "openai.py"


def test_deferrable_savings_delta_gates_on_input() -> None:
    # Normal turn: usage arrived, savings recorded as computed.
    assert _deferrable_savings_delta(500, 120) == 120
    # Usage-less turn (cancelled/failed): savings deferred.
    assert _deferrable_savings_delta(0, 120) == 0
    assert _deferrable_savings_delta(-1, 120) == 0
    # Non-positive savings pass through unchanged regardless of input, so the
    # recorded-total bookkeeping keeps its normal resync behaviour.
    assert _deferrable_savings_delta(0, 0) == 0
    assert _deferrable_savings_delta(500, 0) == 0
    assert _deferrable_savings_delta(0, -5) == -5
    assert _deferrable_savings_delta(500, -5) == -5


def test_deferred_savings_land_with_next_usage_turn() -> None:
    """Walk the recorded-total bookkeeping the handler performs per turn.

    The handler computes ``saved_delta = _deferrable_savings_delta(input_delta,
    tokens_saved - recorded)`` and then advances ``recorded += saved_delta``.
    A deferred turn must leave the savings pending so they ride along with the
    next usage-carrying turn instead of being dropped.
    """

    recorded = 0

    # Turn 1: compressed (100 saved) but cancelled before any usage frame.
    tokens_saved = 100
    delta = _deferrable_savings_delta(0, tokens_saved - recorded)
    assert delta == 0  # nothing recorded...
    recorded += delta
    assert recorded == 0  # ...and the 100 stays pending.

    # Turn 2: compressed (50 more saved) and completed with usage.
    tokens_saved = 150
    delta = _deferrable_savings_delta(4_000, tokens_saved - recorded)
    assert delta == 150  # turn 2's 50 plus the deferred 100.
    recorded += delta
    assert recorded == tokens_saved


def test_per_turn_and_residual_sites_use_the_gate() -> None:
    """Source-level guard for the closure-internal wiring.

    The per-turn metrics closure and the session-end residual flush both live
    inside ``handle_openai_responses_ws`` and cannot be reached by unit tests
    (see the pending-harness note in test_codex_ws_compression_scheduler.py),
    so guard the wiring in source: both sites must gate their savings delta
    through ``_deferrable_savings_delta``, and the recorded-savings total must
    advance by the recorded delta (``+= saved_delta``) — a naked
    ``= tokens_saved`` assignment would silently drop deferred savings.
    """

    source = OPENAI_HANDLER.read_text()
    assert source.count("_deferrable_savings_delta(") >= 3, (
        "Expected the per-turn WS metrics closure and the session-end "
        "residual flush to both gate savings through "
        "_deferrable_savings_delta (plus its def). A savings delta recorded "
        "without input accounting writes a savings-with-zero-spend "
        "checkpoint into the savings tracker."
    )
    assert re.search(r"ws_recorded_tokens_saved_total\s*\+=\s*saved_delta", source), (
        "The recorded-savings total must advance by the recorded delta so "
        "savings deferred from usage-less turns stay pending for the next "
        "usage-carrying turn."
    )
    assert not re.search(r"ws_recorded_tokens_saved_total\s*=\s*tokens_saved\b", source), (
        "Naked `ws_recorded_tokens_saved_total = tokens_saved` reintroduced: "
        "this silently drops savings deferred from usage-less turns."
    )
