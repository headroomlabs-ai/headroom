from __future__ import annotations

from headroom.pricing.litellm_model_resolution import (
    MODEL_ALIASES,
    LiteLLMModelPrefixRule,
    _strip_vertex_version_suffix,
    pricing_lookup_candidates,
    resolution_candidates,
    resolve_litellm_model_name,
)


def test_prefix_rule_matches_case_insensitively() -> None:
    rule = LiteLLMModelPrefixRule("minimax-", "minimax/")

    assert rule.candidate_for("MiniMax-M3") == "minimax/MiniMax-M3"
    assert rule.candidate_for("gpt-4o") is None


def test_resolution_candidates_try_bare_then_matching_prefix_then_alias() -> None:
    assert resolution_candidates("gpt-4o") == ("gpt-4o", "openai/gpt-4o")
    assert resolution_candidates("MiniMax-M3") == ("MiniMax-M3", "minimax/MiniMax-M3")

    retired = "claude-3-5-sonnet-20241022"
    candidates = resolution_candidates(retired)
    assert candidates[0] == retired
    assert f"anthropic/{retired}" in candidates
    assert f"vertex_ai/{retired}" not in candidates  # no @YYYYMMDD suffix = not Vertex
    assert MODEL_ALIASES[retired] in candidates


def test_pricing_lookup_candidates_include_provider_prefixes_and_aliases() -> None:
    candidates = pricing_lookup_candidates("claude-3-5-sonnet-20241022")

    assert candidates[0] == "claude-3-5-sonnet-20241022"
    assert "anthropic/claude-3-5-sonnet-20241022" in candidates
    assert "minimax/claude-3-5-sonnet-20241022" in candidates
    assert candidates[-1] == MODEL_ALIASES["claude-3-5-sonnet-20241022"]


def test_retired_claude_3_sonnet_aliases_to_sonnet_tier_not_haiku() -> None:
    """Retired claude-3-sonnet must map to a Sonnet-tier price, not Haiku.

    claude-3-sonnet-20240229 was a $3/$15-per-1M model; aliasing it to
    claude-3-haiku-20240307 ($0.25/$1.25) underpriced its cost/savings ~12x.
    """
    alias = MODEL_ALIASES["claude-3-sonnet-20240229"]

    assert "haiku" not in alias
    # Same-tier target as the other retired-Sonnet aliases.
    assert alias == MODEL_ALIASES["claude-3-5-sonnet-20241022"]
    assert alias == "claude-sonnet-4-20250514"


def test_resolve_litellm_model_name_returns_first_known_candidate() -> None:
    known = {"openai/gpt-4o"}

    assert resolve_litellm_model_name("gpt-4o", known.__contains__) == "openai/gpt-4o"


def test_resolve_litellm_model_name_returns_original_when_unknown() -> None:
    assert resolve_litellm_model_name("mystery-model", lambda _: False) == "mystery-model"


def test_strip_vertex_version_suffix() -> None:
    assert _strip_vertex_version_suffix("claude-haiku-4-5@20251001") == "claude-haiku-4-5"
    assert _strip_vertex_version_suffix("claude-opus-4@20250514") == "claude-opus-4"
    assert _strip_vertex_version_suffix("claude-sonnet-4-6") == "claude-sonnet-4-6"
    assert _strip_vertex_version_suffix("claude-sonnet-4-20250514") == "claude-sonnet-4-20250514"


def test_resolution_candidates_vertex_versioned_models() -> None:
    # Vertex appends @YYYYMMDD — bare name and vertex_ai/ must be candidates
    candidates = resolution_candidates("claude-haiku-4-5@20251001")
    assert "claude-haiku-4-5" in candidates
    assert "anthropic/claude-haiku-4-5" in candidates
    assert "vertex_ai/claude-haiku-4-5" in candidates  # vertex_ai/ only for versioned

    candidates = resolution_candidates("claude-opus-4@20250514")
    assert "claude-opus-4" in candidates
    assert "anthropic/claude-opus-4" in candidates
    assert "vertex_ai/claude-opus-4" in candidates

    # Non-versioned names should NOT get vertex_ai/ candidates
    candidates = resolution_candidates("claude-sonnet-4-6")
    assert candidates[0] == "claude-sonnet-4-6"
    assert "vertex_ai/claude-sonnet-4-6" not in candidates
    assert "anthropic/claude-sonnet-4-6" in candidates


def test_pricing_lookup_candidates_vertex_versioned_models() -> None:
    candidates = pricing_lookup_candidates("claude-haiku-4-5@20251001")
    # Bare name and vertex_ai/ prefix must both be candidates
    assert "claude-haiku-4-5" in candidates
    assert "vertex_ai/claude-haiku-4-5" in candidates
    assert "anthropic/claude-haiku-4-5" in candidates

    candidates = pricing_lookup_candidates("claude-opus-4@20250514")
    assert "claude-opus-4" in candidates
    assert "vertex_ai/claude-opus-4" in candidates

    # Non-versioned names should NOT get vertex_ai/ pricing candidates
    candidates = pricing_lookup_candidates("claude-sonnet-4-6")
    assert "vertex_ai/claude-sonnet-4-6" not in candidates
    assert "anthropic/claude-sonnet-4-6" in candidates


def test_vertex_versioned_model_resolves_to_known_key() -> None:
    # Simulate LiteLLM knowing the bare model name (not the versioned one)
    known = {"claude-haiku-4-5", "anthropic/claude-sonnet-4-6"}
    assert (
        resolve_litellm_model_name("claude-haiku-4-5@20251001", known.__contains__)
        == "claude-haiku-4-5"
    )
    assert (
        resolve_litellm_model_name("claude-sonnet-4-6", known.__contains__)
        == "anthropic/claude-sonnet-4-6"
    )
    # Unknown versioned model falls back to original
    assert (
        resolve_litellm_model_name("claude-unknown@20251001", lambda _: False)
        == "claude-unknown@20251001"
    )
