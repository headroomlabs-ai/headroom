"""``original_tokens`` and ``optimized_tokens`` must share a tokenizer scale.

Every derived quantity is a delta between the two — ``tokens_saved``,
``tokens_inflated``, ``attempted_input_tokens``, and the beacon's
``eligible_pct`` / ``yield_pct``. Handlers used to pass the provider's
``usage.prompt_tokens`` as ``optimized_tokens`` because it also fed billing,
which put a provider count against a locally-estimated ``original_tokens``.

A real beacon payload from a gpt-4o-mini session, where our estimator
undercounted by 2 tokens on a 10-token request:

    "tokens": {"original": 10, "attempted": 12, "input": 12, "saved": 0}
    "rates":  {"eligible_pct": 120}

120% is structurally impossible — you cannot attempt to compress more than
arrived. The same mismatch also produces a phantom ``tok_inflated``.
"""

from __future__ import annotations

from headroom.proxy.outcome import RequestOutcome

# The exact numbers from the reported beacon entry.
_LOCAL_ORIGINAL = 10
_LOCAL_OPTIMIZED = 10  # nothing compressed: a 10-token request is below every floor
_PROVIDER_COUNT = 12  # gpt-4o-mini's own prompt_tokens — a different ruler


def _outcome(**kw) -> RequestOutcome:
    base: dict = {
        "request_id": "r1",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "original_tokens": _LOCAL_ORIGINAL,
        "optimized_tokens": _LOCAL_OPTIMIZED,
        "output_tokens": 5,
        "tokens_saved": 0,
        "attempted_input_tokens": _LOCAL_OPTIMIZED,
    }
    base.update(kw)
    return RequestOutcome(**base)


def test_provider_count_does_not_contaminate_the_local_pair() -> None:
    """The provider's number rides alongside instead of replacing tok_after."""
    o = _outcome(provider_input_tokens=_PROVIDER_COUNT)

    assert o.original_tokens == _LOCAL_ORIGINAL
    assert o.optimized_tokens == _LOCAL_OPTIMIZED
    assert o.provider_input_tokens == _PROVIDER_COUNT


def test_no_phantom_inflation_when_the_provider_counts_higher() -> None:
    """The regression: provider 12 vs local 10 reported 2 tokens of growth."""
    fixed = _outcome(provider_input_tokens=_PROVIDER_COUNT)
    assert fixed.tokens_inflated == 0

    # What the old shape produced — optimized carrying the provider count.
    contaminated = _outcome(optimized_tokens=_PROVIDER_COUNT)
    assert contaminated.tokens_inflated == 2, (
        "guard is inverted; this asserts the OLD behaviour to document the bug"
    )


def test_attempted_cannot_exceed_original_on_a_no_op_turn() -> None:
    """`eligible_pct = attempted / original` must stay <= 100 here.

    attempted is built as optimized + saved, so once optimized is local the
    ratio is bounded by original for any turn that did not grow.
    """
    o = _outcome(
        provider_input_tokens=_PROVIDER_COUNT,
        attempted_input_tokens=_LOCAL_OPTIMIZED + 0,
    )
    assert o.attempted_input_tokens <= o.original_tokens
    assert 100 * o.attempted_input_tokens / o.original_tokens == 100.0


def test_provider_count_is_optional_and_defaults_to_zero() -> None:
    """18 pre-existing emit sites pass nothing; billing must fall back."""
    o = _outcome()
    assert o.provider_input_tokens == 0


def test_real_inflation_is_still_reported() -> None:
    """Fixing the scale must not mute genuine post-compression growth.

    Same tokenizer both sides, request forwarded larger (memory injection /
    proactive expansion) — that is real and must survive.
    """
    o = _outcome(
        original_tokens=55_161,
        optimized_tokens=57_845,
        provider_input_tokens=58_000,
        tokens_saved=0,
    )
    assert o.tokens_inflated == 2_684
