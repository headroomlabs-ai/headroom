"""An alias target that litellm no longer prices is worse than no alias.

`MODEL_ALIASES` exists so a retired model id still prices at its own tier.
Every target pointed at `claude-sonnet-4-20250514` until litellm pruned it on
2026-09-23; all three aliases then resolved to nothing and those models fell
through to the unknown-model default -- the GPT-4o tier -- which silently
misprices them. Nothing failed, because no test asserted the targets were live.
"""

from __future__ import annotations

import pytest

from tests._dotenv import importorskip_no_env_leak

importorskip_no_env_leak("litellm")

from headroom.pricing.litellm_model_resolution import MODEL_ALIASES  # noqa: E402
from headroom.pricing.litellm_pricing import get_model_pricing  # noqa: E402


@pytest.mark.parametrize(("retired", "target"), sorted(MODEL_ALIASES.items()))
def test_model_aliases_point_at_something_litellm_still_prices(retired: str, target: str) -> None:
    """The target must be priced, or the alias is a silent no-op."""
    assert get_model_pricing(target) is not None, (
        f"MODEL_ALIASES[{retired!r}] points at {target!r}, which litellm no longer "
        "prices. Repoint it at a current model of the same tier "
        "(headroom/pricing/litellm_model_resolution.py)."
    )


@pytest.mark.parametrize(("retired", "target"), sorted(MODEL_ALIASES.items()))
def test_a_retired_id_prices_at_its_alias_rate(retired: str, target: str) -> None:
    """The point of the alias: the retired id resolves to the target's numbers."""
    priced = get_model_pricing(retired)
    assert priced is not None, f"{retired!r} did not resolve through its alias"

    target_priced = get_model_pricing(target)
    assert target_priced is not None
    assert (priced.input_cost_per_1m, priced.output_cost_per_1m) == (
        target_priced.input_cost_per_1m,
        target_priced.output_cost_per_1m,
    )


def test_retired_sonnets_are_not_priced_at_the_unknown_default() -> None:
    """Regression for the failure mode this whole file is about.

    The unknown-model fallback is the GPT-4o tier. A Sonnet-tier model landing
    there is wrong in both directions and is exactly what happened when the
    alias targets went stale.
    """
    gpt4o = get_model_pricing("gpt-4o")
    assert gpt4o is not None

    for retired in MODEL_ALIASES:
        priced = get_model_pricing(retired)
        assert priced is not None
        assert (priced.input_cost_per_1m, priced.output_cost_per_1m) != (
            gpt4o.input_cost_per_1m,
            gpt4o.output_cost_per_1m,
        ), f"{retired!r} is priced at the GPT-4o tier — its alias is not resolving"
