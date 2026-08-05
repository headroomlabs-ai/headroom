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


def test_litellm_minimax_mixed_case_with_provider_prefix(monkeypatch) -> None:
    """MiniMax-M3 must resolve via the `minimax/` prefix even though its
    model name uses mixed case.

    `resolve_litellm_model()` is what callers in `proxy/cost.py`,
    `proxy/savings_tracker.py`, and `perf/analyzer.py` use to get a
    key LiteLLM's own cost DB recognises. The upstream DB only stores
    the entry under `minimax/MiniMax-M3`, so bare `MiniMax-M3` would
    otherwise miss and the resolver would return the input unchanged.
    """

    def fake_cost_per_token(
        model: str, prompt_tokens: int = 0, completion_tokens: int = 0
    ) -> tuple[float, float]:
        if model in fake_litellm.model_cost:
            entry = fake_litellm.model_cost[model]
            return (
                entry["input_cost_per_token"] * prompt_tokens,
                entry["output_cost_per_token"] * completion_tokens,
            )
        raise KeyError(f"unknown model: {model}")

    fake_litellm = SimpleNamespace(
        model_cost={
            "minimax/MiniMax-M3": {
                "input_cost_per_token": 0.0000006,
                "output_cost_per_token": 0.0000024,
            }
        },
        cost_per_token=fake_cost_per_token,
    )
    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(litellm_pricing, "litellm", fake_litellm)

    # Bare mixed-case name resolves via the case-insensitive `minimax-` prefix.
    assert litellm_pricing.resolve_litellm_model("MiniMax-M3") == "minimax/MiniMax-M3"


def test_litellm_minimax_preregistration_safety_net(monkeypatch) -> None:
    """When LiteLLM only ships the prefixed `minimax/MiniMax-M3` entry, the
    module-load pre-registration should also expose the bare `MiniMax-M3`
    key so `estimate_cost()` works on a cold resolver cache (since
    `get_model_pricing` does not know about the `minimax/` prefix).
    """
    fake_litellm = SimpleNamespace(
        model_cost={
            "minimax/MiniMax-M3": {
                "input_cost_per_token": 0.0000006,
                "output_cost_per_token": 0.0000024,
            }
        }
    )
    monkeypatch.setattr(litellm_pricing, "LITELLM_AVAILABLE", True)
    monkeypatch.setattr(litellm_pricing, "litellm", fake_litellm)

    litellm_pricing._register_minimax_pricing()

    assert "MiniMax-M3" in fake_litellm.model_cost
    assert fake_litellm.model_cost["MiniMax-M3"]["input_cost_per_token"] == 0.0000006
    # After pre-registration, bare-name estimate_cost works end-to-end.
    assert (
        litellm_pricing.estimate_cost("MiniMax-M3", input_tokens=1_000_000, output_tokens=100_000)
        == 0.84
    )
    # Pre-registration must not clobber a user-customised bare entry.
    fake_litellm.model_cost["MiniMax-M3"] = {"customised": True}
    litellm_pricing._register_minimax_pricing()
    assert fake_litellm.model_cost["MiniMax-M3"] == {"customised": True}
