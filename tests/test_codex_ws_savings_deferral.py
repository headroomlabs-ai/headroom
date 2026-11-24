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
    """A frame whose response ends without usage hands its savings to the
    next completion with usage on the connection; its measure is released."""
    from headroom.proxy.handlers.openai import _WsFrame, _WsTurnMeasures

    released: list[object] = []
    turns = _WsTurnMeasures(release=released.append)
    # Frame 1: compressed (100 saved) but its response fails before any usage.
    turns.sent(_WsFrame("m1", saved=100))  # type: ignore[arg-type]
    turns.created("r1")
    turns.ended("r1", completed=False)
    assert turns.take_completed() is None  # nothing booked...
    assert released == ["m1"]  # ...and frame 1 never becomes a baseline.

    # Frame 2: compressed (50 saved) and completed with usage.
    turns.sent(_WsFrame(None, saved=50))
    turns.created("r2")
    turns.ended("r2", completed=True)
    booking = turns.take_completed()
    assert booking is not None and booking.saved == 150  # its 50 plus the deferred 100.


def test_per_turn_and_residual_sites_use_the_gate() -> None:
    """Source-level guard for the closure-internal wiring: the per-turn record
    books only the frame its response completed with usage (plus deferred
    savings), and the session-end residual gates its savings on input through
    ``_deferrable_savings_delta``. Booking a savings delta without input
    accounting writes a savings-with-zero-spend checkpoint."""

    source = OPENAI_HANDLER.read_text()
    assert source.count("_deferrable_savings_delta(") >= 2, (
        "Expected the session-end residual flush to gate savings through "
        "_deferrable_savings_delta (plus its def)."
    )
    assert "booking = ws_measures.take_completed()" in source, (
        "The per-turn WS record must book the frame whose response completed "
        "with usage, not a connection-wide savings delta."
    )
    assert not re.search(r"tokens_saved\s*-\s*ws_recorded_tokens_saved_total", source), (
        "A connection-wide savings delta in the per-turn record books the "
        "savings of frames still awaiting their response."
    )
