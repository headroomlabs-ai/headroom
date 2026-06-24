"""Tests for CO₂ saved estimation in savings_tracker.

Validates estimate_co2_saved_mg() and its presence in stats_preview().
"""

from __future__ import annotations

import pytest

from headroom.proxy.savings_tracker import estimate_co2_saved_mg


def test_zero_tokens_returns_zero() -> None:
    assert estimate_co2_saved_mg(0) == 0.0
    assert estimate_co2_saved_mg(-100) == 0.0


def test_positive_tokens_returns_positive() -> None:
    result = estimate_co2_saved_mg(1_000_000)
    assert result > 0


def test_claude_sonnet_uses_correct_factor() -> None:
    # claude-sonnet: 0.0003 Wh/token × 0.4 kg/kWh × 1000 mg/g = 0.12 mg/token
    result = estimate_co2_saved_mg(1000, model="claude-sonnet-4-5")
    assert abs(result - 120.0) < 1.0  # 0.12 mg × 1000 = 120 mg


def test_unknown_model_uses_default_factor() -> None:
    # default: 0.0002 Wh/token × 0.4 = 0.08 mg/token
    result = estimate_co2_saved_mg(1000, model="unknown-model-xyz")
    assert abs(result - 80.0) < 1.0


def test_claude_haiku_lower_than_sonnet() -> None:
    haiku = estimate_co2_saved_mg(1000, model="claude-haiku")
    sonnet = estimate_co2_saved_mg(1000, model="claude-sonnet")
    assert haiku < sonnet


def test_large_savings_in_grams() -> None:
    # 1M tokens at claude-sonnet = 120,000 mg = 120 g
    mg = estimate_co2_saved_mg(1_000_000, model="claude-sonnet")
    g = mg / 1000
    assert abs(g - 120.0) < 1.0


def test_stats_preview_includes_co2() -> None:
    """stats_preview() must include a co2 block with co2_saved_mg."""
    from headroom.proxy.savings_tracker import SavingsTracker
    import tempfile, pathlib

    with tempfile.TemporaryDirectory() as tmp:
        tracker = SavingsTracker(path=str(pathlib.Path(tmp) / "savings.json"))
        tracker.record_compression_savings(
            model="claude-sonnet-4-5",
            tokens_saved=10_000,
        )
        preview = tracker.stats_preview(model="claude-sonnet-4-5")

    assert "co2" in preview
    co2 = preview["co2"]
    assert "co2_saved_mg" in co2
    assert co2["co2_saved_mg"] > 0
    assert "co2_saved_g" in co2
    assert co2["co2_saved_g"] == round(co2["co2_saved_mg"] / 1000, 6)
    assert "methodology" in co2
