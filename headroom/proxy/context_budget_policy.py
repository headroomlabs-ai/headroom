"""Pre-forward context budget policy for the Anthropic messages handler.

Pure-logic module: no token counting, no message inspection, no I/O beyond
os.environ in the two resolver helpers.  Does not import FastAPI, the handler,
or the provider.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class BudgetDecision:
    mode: str
    counted_tokens: int
    declared_limit: int | None
    reserve: int
    threshold: int
    overage: int
    should_reject: bool
    reason: str
    # Retained for diagnostic access; not used in evaluation logic.
    _extra: dict = field(default_factory=dict, compare=False)


def resolve_mode() -> str:
    """Read HEADROOM_CONTEXT_LIMIT_MODE; default 'observe'.

    Accepts 'observe' and 'reject' (case-insensitive).
    Raises ValueError naming the bad value and the accepted set.
    """
    raw = os.environ.get("HEADROOM_CONTEXT_LIMIT_MODE", "observe").strip().lower()
    if raw not in ("observe", "reject"):
        raise ValueError(
            f"HEADROOM_CONTEXT_LIMIT_MODE={raw!r} is not accepted; "
            "accepted values: 'observe', 'reject'"
        )
    return raw


def resolve_safety_margin() -> int:
    """Read HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN; default 0.

    Raises ValueError on non-integer or negative value.
    """
    raw = os.environ.get("HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN", "0").strip()
    try:
        val = int(raw)
    except ValueError as exc:
        raise ValueError(f"HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN={raw!r} is not an integer") from exc
    if val < 0:
        raise ValueError(f"HEADROOM_CONTEXT_LIMIT_SAFETY_MARGIN={val} must be >= 0")
    return val


def evaluate(
    *,
    counted_tokens: int,
    declared_limit: int | None,
    max_output_tokens: int,
    mode: str,
    safety_margin: int,
) -> BudgetDecision:
    """Evaluate the context budget for a finalized request.

    Branches (in order):
    1. declared_limit is None  -> reason='no_declared_limit', should_reject=False
    2. threshold <= 0          -> reason='degenerate_threshold', overage reported,
                                  should_reject=(mode=='reject')
    3. counted_tokens <= threshold -> reason='under_threshold', should_reject=False
    4. else                    -> reason='over_threshold', should_reject=(mode=='reject')

    reserve = max(safety_margin, max_output_tokens)
    threshold = declared_limit - reserve
    overage = max(0, counted_tokens - threshold)
    """
    if declared_limit is None:
        return BudgetDecision(
            mode=mode,
            counted_tokens=counted_tokens,
            declared_limit=None,
            reserve=0,
            threshold=0,
            overage=0,
            should_reject=False,
            reason="no_declared_limit",
        )

    reserve = max(safety_margin, max_output_tokens)
    threshold = declared_limit - reserve

    if threshold <= 0:
        # The reserved output alone consumes the whole declared window, so no
        # request can fit by construction. In reject mode that is over budget:
        # refusing keeps the proxy's promise that an impossible request never
        # reaches upstream. Observe mode still logs and forwards. The overage
        # uses the same counted - threshold formula as the over-threshold branch,
        # which is positive whenever the threshold is non-positive.
        return BudgetDecision(
            mode=mode,
            counted_tokens=counted_tokens,
            declared_limit=declared_limit,
            reserve=reserve,
            threshold=threshold,
            overage=counted_tokens - threshold,
            should_reject=(mode == "reject"),
            reason="degenerate_threshold",
        )

    if counted_tokens <= threshold:
        return BudgetDecision(
            mode=mode,
            counted_tokens=counted_tokens,
            declared_limit=declared_limit,
            reserve=reserve,
            threshold=threshold,
            overage=0,
            should_reject=False,
            reason="under_threshold",
        )

    overage = counted_tokens - threshold
    return BudgetDecision(
        mode=mode,
        counted_tokens=counted_tokens,
        declared_limit=declared_limit,
        reserve=reserve,
        threshold=threshold,
        overage=overage,
        should_reject=(mode == "reject"),
        reason="over_threshold",
    )
