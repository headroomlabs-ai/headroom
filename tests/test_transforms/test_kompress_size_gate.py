"""Phase 0 (#1171): kompress input-size gate.

Kompress (ModernBERT ONNX) inference scales O(tokens) and runs synchronously on
the request thread under the 30s compression budget. Above a size ceiling the
router must route around the ML path (to the fast LogCompressor, else
passthrough) so a large/cold context can't blow the timeout and leak a
non-preemptible worker. The gate lives inside ``_try_ml_compressor`` so it
covers EVERY kompress entry point (TEXT, KOMPRESS-direct, CODE_AWARE, and the
strategy-fallback path), all of which funnel through that single boundary.
"""

from __future__ import annotations

import pytest

from headroom.transforms import ContentRouter
from headroom.transforms.content_router import CompressionStrategy


def test_oversized_input_is_gated_away_from_kompress(monkeypatch):
    router = ContentRouter()
    router._kompress_max_tokens = 50  # tiny ceiling (×4 = 200 chars)

    def _boom():  # kompress must never be fetched for a gated input
        raise AssertionError("kompress must not be invoked for oversized (gated) input")

    monkeypatch.setattr(router, "_get_kompress", _boom)

    big = "the quick brown fox jumps over the lazy dog. " * 100  # >200 chars
    out, ntok = router._try_ml_compressor(big, "")

    assert router._kompress_gate_fires == 1
    assert isinstance(out, str) and ntok > 0


def test_small_input_still_routes_to_kompress(monkeypatch):
    router = ContentRouter()
    router._kompress_max_tokens = 50000  # default ceiling; small input is under it

    # Return no kompressor so the ML block is a fast no-op (no model load in test).
    monkeypatch.setattr(router, "_get_kompress", lambda: None)

    out, ntok = router._try_ml_compressor("short text", "")

    assert router._kompress_gate_fires == 0  # gate did NOT fire for small input
    assert isinstance(out, str)


def test_gate_disabled_when_threshold_zero(monkeypatch):
    router = ContentRouter()
    router._kompress_max_tokens = 0  # disabled

    monkeypatch.setattr(router, "_get_kompress", lambda: None)

    big = "x " * 100000
    out, ntok = router._try_ml_compressor(big, "")

    assert router._kompress_gate_fires == 0  # disabled → never gates


@pytest.mark.parametrize("strategy", [CompressionStrategy.KOMPRESS, CompressionStrategy.TEXT])
def test_gate_fires_through_strategy_dispatch(monkeypatch, strategy):
    # Funnel check: drive the strategy dispatch (not _try_ml_compressor directly)
    # and confirm both ML strategies reach the single gate boundary.
    router = ContentRouter()
    router._kompress_max_tokens = 50

    def _boom():
        raise AssertionError("kompress must not run for gated input")

    monkeypatch.setattr(router, "_get_kompress", _boom)

    big = "the quick brown fox jumps over the lazy dog. " * 100
    router._apply_strategy_to_content(big, strategy, "")

    assert router._kompress_gate_fires == 1
