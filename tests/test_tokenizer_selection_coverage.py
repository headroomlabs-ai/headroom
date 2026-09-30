"""Model names must resolve to the tokenizer their model actually uses.

Two selection gaps, both measured against real counters on identical text:

1. ``MODEL_PATTERNS`` stopped at ``^gpt-4``/``^o1``/``^o3``, so the current
   flagships — ``gpt-5``, ``gpt-5.1``, ``o4-mini`` — fell through to the char
   estimator. Deviation vs the correct o200k encoding: +20% English, -33% JSON,
   -44% logs.

2. Every pattern is ``^``-anchored, which is right for a bare model id and wrong
   for the wrapped ids gateways send. ``bedrock/anthropic.claude-3-5-sonnet``,
   ``vertex_ai/claude-…``, ``openrouter/anthropic/claude-…``, ``azure/gpt-4o``
   and Bedrock's ``us.anthropic.claude-…`` all matched nothing. LiteLLM's
   ``headroom`` guardrail passes exactly these forms.

The estimator is a legitimate FALLBACK; the bug is reaching it when a real
tokenizer for that family exists.
"""

from __future__ import annotations

import pytest

from headroom.tokenizers import get_tokenizer
from headroom.tokenizers.registry import _name_candidates

_TIKTOKEN = "TiktokenCounter"


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5",
        "gpt-5.1",
        "gpt-5-mini",
        "gpt-5.1-codex",
        "o4-mini",
    ],
)
def test_current_openai_flagships_get_a_real_tokenizer(model: str) -> None:
    """These fell to EstimatingTokenCounter before ^gpt-5 / ^o4 were added."""
    assert type(get_tokenizer(model)).__name__ == _TIKTOKEN


@pytest.mark.parametrize(
    "model",
    [
        # gateway path prefixes
        "bedrock/anthropic.claude-3-5-sonnet",
        "vertex_ai/claude-sonnet-4-6",
        "openrouter/anthropic/claude-sonnet-4-6",
        "anthropic/claude-opus-4",
        "litellm/claude-sonnet-4-6",
        # Bedrock dotted ids, with and without a region segment
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "us.anthropic.claude-sonnet-4-6-v1:0",
        "eu.anthropic.claude-sonnet-4-6-v1:0",
        # OpenAI behind a gateway
        "azure/gpt-4o",
        "openrouter/openai/gpt-4o",
    ],
)
def test_gateway_wrapped_names_resolve_like_their_bare_form(model: str) -> None:
    assert type(get_tokenizer(model)).__name__ == _TIKTOKEN


def test_wrapped_gemini_matches_the_bare_form_exactly() -> None:
    """Prefix stripping must reach the google backend, not the generic fallback."""
    text = "hello world " * 200
    assert get_tokenizer("vertex_ai/gemini-2.5-pro").count_text(text) == get_tokenizer(
        "gemini-2.5-pro"
    ).count_text(text)


def test_bare_names_are_unaffected() -> None:
    """The exact-match candidate is tried first, so nothing already-correct moves."""
    for model, expected in (
        ("gpt-4o", _TIKTOKEN),
        ("gpt-3.5-turbo", _TIKTOKEN),
        ("o1-preview", _TIKTOKEN),
        ("o3-mini", _TIKTOKEN),
        ("claude-sonnet-4-6", _TIKTOKEN),
    ):
        assert type(get_tokenizer(model)).__name__ == expected, model


def test_unknown_alias_still_falls_back_to_estimation() -> None:
    """Prefix stripping must not invent a match for a genuinely unknown model."""
    assert type(get_tokenizer("my-gateway/big-model")).__name__ == "EstimatingTokenCounter"
    assert type(get_tokenizer("totally-unknown-xyz")).__name__ == "EstimatingTokenCounter"


def test_name_candidates_orders_most_specific_first() -> None:
    """The full name must be candidate 0 so exact registrations always win."""
    got = _name_candidates("openrouter/anthropic/claude-sonnet-4-6")
    assert got[0] == "openrouter/anthropic/claude-sonnet-4-6"
    assert "anthropic/claude-sonnet-4-6" in got
    assert "claude-sonnet-4-6" in got

    dotted = _name_candidates("us.anthropic.claude-sonnet-4-6-v1:0")
    assert dotted[0] == "us.anthropic.claude-sonnet-4-6-v1:0"
    assert "claude-sonnet-4-6-v1:0" in dotted


def test_name_candidates_is_deduplicated_and_finite() -> None:
    got = _name_candidates("a/b/c.d.e")
    assert len(got) == len(set(got))
    assert got[0] == "a/b/c.d.e"
