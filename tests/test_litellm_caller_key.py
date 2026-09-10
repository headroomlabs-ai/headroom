"""A caller's key must not be dropped unless we are certain it cannot work.

The proxy forwards the inbound credential to the upstream provider. When a
routing extension rewrites the model across families mid-request, that key stops
matching the target and the 401 that follows is indistinguishable, downstream,
from "the cheap model failed the task".

Refusing to forward is the fix, but it is also the more dangerous direction: a
false positive silently strips a credential from a deployment that was working,
and litellm then falls back to an env key that may not exist. So the rule is
positive evidence only -- an unrecognised credential always travels.
"""

from __future__ import annotations

import pytest

from headroom.backends.litellm import _caller_key_travels_to

ANTHROPIC_KEY = "sk-ant-api03-abc123"


@pytest.mark.parametrize(
    "model",
    ["gpt-5-mini", "gpt-4o", "azure/gpt-4", "gemini/gemini-2.0-flash"],
)
def test_anthropic_key_is_refused_for_a_provider_that_cannot_accept_it(model: str) -> None:
    """The bug this exists for: claude-* rewritten to a non-Anthropic target."""
    pytest.importorskip("litellm")
    assert _caller_key_travels_to(model, ANTHROPIC_KEY) is False


@pytest.mark.parametrize(
    "model",
    ["claude-opus-4-5-20251101", "anthropic/claude-sonnet-4-5-20250929"],
)
def test_anthropic_key_travels_to_anthropic(model: str) -> None:
    pytest.importorskip("litellm")
    assert _caller_key_travels_to(model, ANTHROPIC_KEY) is True


@pytest.mark.parametrize(
    "key",
    [
        "sk-proj-openai-style",  # a dozen vendors mint this shape
        "Bearer-ish-opaque-token",  # a plain gateway token
        "hf_abc123",
        "sk-ant",  # near miss, not the prefix
        "",
    ],
)
def test_only_the_anthropic_prefix_is_ever_classified(key: str) -> None:
    """Everything else is unclassifiable from the string, so it passes through.

    This is the regression the review caught: the first version returned
    `not provider.startswith("anthropic")`, which dropped every one of these
    against an Anthropic-class target.
    """
    assert _caller_key_travels_to("gpt-5-mini", key) is True
    assert _caller_key_travels_to("claude-opus-4-5-20251101", key) is True


def test_unknown_provider_keeps_the_pass_through() -> None:
    """A compatible or self-hosted gateway we cannot classify must not lose its
    key -- including when `get_llm_provider` raises on the model string."""
    assert _caller_key_travels_to("some-self-hosted-thing", ANTHROPIC_KEY) is True
    assert _caller_key_travels_to("", ANTHROPIC_KEY) is True
