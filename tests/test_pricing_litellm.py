from __future__ import annotations

from types import SimpleNamespace

from headroom.pricing import litellm_pricing


def test_litellm_helpers_when_dependency_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", False)
    monkeypatch.setattr(litellm_pricing, "litellm", None)

    assert litellm_pricing.get_litellm_model_cost() == {}
    assert litellm_pricing.get_model_pricing("gpt-4o") is None
    assert litellm_pricing.estimate_cost("gpt-4o", input_tokens=1, output_tokens=1) is None
    assert litellm_pricing.list_available_models() == []


def test_litellm_model_pricing_exact_match_and_defaults(monkeypatch) -> None:
    fake_litellm = SimpleNamespace(
        model_cost={
            "gpt-4o": {
                "input_cost_per_token": 0.0000025,
                "output_cost_per_token": 0.00001,
                "max_tokens": 128000,
            }
        }
    )
    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(litellm_pricing, "litellm", fake_litellm)

    assert litellm_pricing.get_litellm_model_cost() == fake_litellm.model_cost
    pricing = litellm_pricing.get_model_pricing("gpt-4o")
    assert pricing is not None
    assert pricing.model == "gpt-4o"
    assert pricing.input_cost_per_1m == 2.5
    assert pricing.output_cost_per_1m == 10.0
    assert pricing.max_tokens == 128000
    assert pricing.max_input_tokens is None
    assert pricing.max_output_tokens is None
    assert pricing.supports_vision is False
    assert pricing.supports_function_calling is False
    assert (
        litellm_pricing.estimate_cost("gpt-4o", input_tokens=200_000, output_tokens=300_000) == 3.5
    )
    assert litellm_pricing.list_available_models() == ["gpt-4o"]


def test_litellm_model_pricing_uses_provider_prefixes(monkeypatch) -> None:
    fake_litellm = SimpleNamespace(
        model_cost={
            "openai/gpt-4o-mini": {
                "input_cost_per_token": 0.00000015,
                "output_cost_per_token": 0.0000006,
                "supports_vision": True,
                "supports_function_calling": True,
                "max_input_tokens": 64000,
                "max_output_tokens": 16000,
            }
        }
    )
    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(litellm_pricing, "litellm", fake_litellm)

    pricing = litellm_pricing.get_model_pricing("gpt-4o-mini")
    assert pricing is not None
    assert pricing.input_cost_per_1m == 0.15
    assert pricing.output_cost_per_1m == 0.6
    assert pricing.max_input_tokens == 64000
    assert pricing.max_output_tokens == 16000
    assert pricing.supports_vision is True
    assert pricing.supports_function_calling is True


def test_litellm_model_pricing_uses_aliases_and_zero_cost_defaults(monkeypatch) -> None:
    fake_litellm = SimpleNamespace(
        model_cost={
            "claude-sonnet-4-20250514": {
                "input_cost_per_token": None,
                "output_cost_per_token": None,
            }
        }
    )
    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(litellm_pricing, "litellm", fake_litellm)

    pricing = litellm_pricing.get_model_pricing("claude-3-5-sonnet-20241022")
    assert pricing is not None
    assert pricing.model == "claude-3-5-sonnet-20241022"
    assert pricing.input_cost_per_1m == 0
    assert pricing.output_cost_per_1m == 0
    assert litellm_pricing.estimate_cost("claude-3-5-sonnet-20241022", input_tokens=1) == 0


def test_litellm_model_pricing_returns_none_for_unknown_models(monkeypatch) -> None:
    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(litellm_pricing, "litellm", SimpleNamespace(model_cost={}))
    assert litellm_pricing.get_model_pricing("missing") is None
