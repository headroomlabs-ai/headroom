"""Pick a model litellm actually prices, instead of hardcoding one it may retire.

``litellm.model_cost`` is downloaded from GitHub at import time, so it is live
third-party data. BerriAI prunes retired models from it: on 2026-09-23
``claude-sonnet-4-20250514`` disappeared and every test that priced it began
failing with ``KeyError: 'input_cost_per_token'`` — on every open pull request
at once, with no change on our side.

The model id in those tests is incidental. They assert that Headroom's cost
arithmetic agrees with litellm's numbers, not that any particular model is
priced correctly, so the fix is to stop naming a specific release and instead
ask for *a* model carrying the fields the test needs.

Pinning to the copy of the table vendored in the litellm wheel is not the
alternative it looks like: the two maps are complementary, not ordered. The
vendored map keeps retired ids but predates current models
(``claude-sonnet-5``), and its older entries lack newer fields such as
``input_cost_per_token_above_200k_tokens`` entirely.
"""

from __future__ import annotations

import pytest

#: Preference order, newest first. A test takes the first entry that carries
#: every field it needs, so retiring one is a no-op until the list runs dry.
_CANDIDATES: tuple[str, ...] = (
    "claude-sonnet-4-5-20250929",
    "claude-sonnet-4-5",
    "claude-sonnet-4-20250514",
    "claude-opus-4-5-20251101",
    "claude-opus-4-5",
)

_BASE_FIELDS = ("input_cost_per_token", "output_cost_per_token")


def anthropic_pricing_model(*required_fields: str) -> str:
    """Return a currently-priced Anthropic model carrying ``required_fields``.

    Always includes the base input/output costs. Raises with an actionable
    message rather than skipping: if litellm prices none of these, the pricing
    tests are not measuring anything and that should be loud.
    """
    import litellm

    needed = set(_BASE_FIELDS) | set(required_fields)
    for model in _CANDIDATES:
        info = litellm.model_cost.get(model)
        if isinstance(info, dict) and needed <= set(info):
            return model

    raise AssertionError(
        "litellm prices none of the candidate models with the fields "
        f"{sorted(needed)}. It most likely retired them from "
        "model_prices_and_context_window.json; add a current model id to "
        "_CANDIDATES in tests/_pricing_models.py (newest first)."
    )


@pytest.fixture(scope="session")
def anthropic_model() -> str:
    """Session fixture wrapper for tests that prefer injection over a constant."""
    return anthropic_pricing_model()
