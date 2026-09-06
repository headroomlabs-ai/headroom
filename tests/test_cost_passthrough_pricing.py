"""Internal ``passthrough:*`` model labels must never reach the pricing lookup.

#2578: passthrough handlers (e.g. Anthropic's ``/v1/messages/count_tokens``)
tag their requests with a synthetic ``passthrough:<endpoint>`` model name for
dashboard visibility. Those labels aren't billable model calls, so pricing
them through LiteLLM raised ``LLM Provider NOT provided`` and logged a
spurious WARNING on every such request.
"""

from __future__ import annotations

import logging

import pytest


@pytest.fixture
def cost_tracker(monkeypatch: pytest.MonkeyPatch):
    import headroom.proxy.cost as cost_mod

    cost_mod._warned_pricing_models.clear()

    class _BoomLiteLLM:
        """LiteLLM stand-in that fails the test if its pricing path is hit."""

        @staticmethod
        def cost_per_token(**_kwargs):
            raise AssertionError("passthrough model must not reach litellm.cost_per_token")

    monkeypatch.setattr(cost_mod, "_get_litellm_module", lambda: _BoomLiteLLM())
    return cost_mod.CostTracker()


def test_passthrough_model_skips_pricing(cost_tracker, caplog):
    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        assert cost_tracker.estimate_cost("passthrough:count_tokens", 100, 50) is None

    # No pricing warning, and no "LiteLLM not available" warning either.
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_passthrough_model_skips_even_without_litellm(monkeypatch, caplog):
    import headroom.proxy.cost as cost_mod

    cost_mod._warned_pricing_models.clear()
    monkeypatch.setattr(cost_mod, "_get_litellm_module", lambda: None)
    tracker = cost_mod.CostTracker()

    with caplog.at_level(logging.WARNING, logger="headroom.proxy"):
        assert tracker.estimate_cost("passthrough:batches", 10, 5) is None

    # The LiteLLM-availability warning must NOT fire for passthrough labels.
    unavailable = [r for r in caplog.records if "LiteLLM not available" in r.getMessage()]
    assert unavailable == []


def test_real_model_still_priced(cost_tracker, monkeypatch):
    # A non-passthrough model still flows through to LiteLLM pricing. Record the
    # cost_per_token call rather than relying on an exception (which estimate_cost
    # catches and converts to a warning), then assert a real cost comes back.
    import headroom.proxy.cost as cost_mod

    calls = {}

    class _RecordingLiteLLM:
        @staticmethod
        def cost_per_token(**kwargs):
            calls["hit"] = True
            return (0.001, 0.002)

    monkeypatch.setattr(cost_mod, "_get_litellm_module", lambda: _RecordingLiteLLM())
    result = cost_tracker.estimate_cost("gpt-4o", 100, 50)

    assert calls.get("hit") is True
    assert result is not None
