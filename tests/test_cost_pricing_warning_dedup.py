"""Pricing-lookup warnings for an unresolvable model must fire once, not per request.

#2504: a custom / OpenAI-compatible model LiteLLM can't price (e.g. glm-5.2)
logged an identical WARNING on every single request, flooding proxy.log.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture
def cost_tracker(monkeypatch: pytest.MonkeyPatch):
    import headroom.proxy.cost as cost_mod

    # Reset the per-process dedup set so tests are order-independent.
    cost_mod._warned_pricing_models.clear()

    class _FakeLiteLLM:
        @staticmethod
        def cost_per_token(**_kwargs):
            raise RuntimeError("LLM Provider NOT provided.")

    monkeypatch.setattr(cost_mod, "_get_litellm_module", lambda: _FakeLiteLLM())
    return cost_mod.CostTracker()


def test_pricing_failure_warns_once_per_model(cost_tracker, caplog):
    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        for _ in range(5):
            assert cost_tracker.estimate_cost("glm-5.2", 100, 50) is None

    warnings = [
        r for r in caplog.records if "Failed to get pricing for model glm-5.2" in r.getMessage()
    ]
    assert len(warnings) == 1


def test_distinct_models_each_warn_once(cost_tracker, caplog):
    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        cost_tracker.estimate_cost("glm-5.2", 10, 5)
        cost_tracker.estimate_cost("glm-5.2", 10, 5)
        cost_tracker.estimate_cost("mystery-model", 10, 5)
        cost_tracker.estimate_cost("mystery-model", 10, 5)

    msgs = [r.getMessage() for r in caplog.records if "Failed to get pricing" in r.getMessage()]
    assert sum("for model glm-5.2:" in m for m in msgs) == 1
    assert sum("for model mystery-model:" in m for m in msgs) == 1


def test_litellm_unavailable_warns_once_per_model(monkeypatch, caplog):
    import headroom.proxy.cost as cost_mod

    cost_mod._warned_pricing_models.clear()
    monkeypatch.setattr(cost_mod, "_get_litellm_module", lambda: None)
    tracker = cost_mod.CostTracker()

    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        for _ in range(3):
            assert tracker.estimate_cost("glm-5.2", 10, 5) is None

    unavailable = [r for r in caplog.records if "LiteLLM not available" in r.getMessage()]
    assert len(unavailable) == 1
