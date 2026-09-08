"""Policy for how estimate-based cost records feed budget enforcement.

Every cost record the proxy books carries a *basis*: where the input-token
count came from. ``measured`` means the provider reported a usage breakdown;
``estimated`` means the response carried none and Headroom substituted its own
``tokens_sent`` count so input cost wasn't silently dropped from the budget
(#2713).

The substitution is the right default — dropping input cost entirely would
under-enforce far worse — but a budget is a hard control, and an estimate can
drift in either direction. This module names the two bases and the three
policies an operator can pick for what estimated spend does to the budget:

* ``count``  — estimated spend counts toward the limit (the historical
  behavior, and still the default; nothing changes except that the record is
  now marked, logged, and separable in ``/stats``).
* ``ignore`` — the record is still booked to the ledger and reported, but only
  provider-measured spend consumes the budget. Fails open on the estimate.
* ``block``  — fail closed: once the period holds any estimated spend, refuse
  rather than enforce a hard limit against a guess.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

logger = logging.getLogger("headroom.proxy")

# Provenance of the input-token count behind a booked cost record.
COST_BASIS_MEASURED = "measured"
COST_BASIS_ESTIMATED = "estimated"

# What estimated spend does to budget enforcement.
BUDGET_BASIS_COUNT = "count"
BUDGET_BASIS_IGNORE = "ignore"
BUDGET_BASIS_BLOCK = "block"

VALID_POLICIES = (BUDGET_BASIS_COUNT, BUDGET_BASIS_IGNORE, BUDGET_BASIS_BLOCK)
DEFAULT_POLICY = BUDGET_BASIS_COUNT

ENV_VAR = "HEADROOM_BUDGET_ESTIMATED_BASIS"

# A typo'd knob must not fail proxy startup, but it also must not warn on every
# resolve. Bounded by the number of distinct bad values seen (realistically 1).
_warned_invalid_policies: set[str] = set()


def resolve_estimated_basis_policy(
    configured: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the estimated-basis budget policy.

    Precedence: explicit ``configured`` value, then ``env[ENV_VAR]``, then
    :data:`DEFAULT_POLICY`. An unrecognized value falls back to the default and
    warns once — a misconfigured knob should never stop the proxy from booting,
    but it must not silently change enforcement either.
    """
    raw = configured
    if raw is None and env is not None:
        raw = env.get(ENV_VAR)
    if raw is None:
        return DEFAULT_POLICY

    candidate = raw.strip().lower()
    if candidate in VALID_POLICIES:
        return candidate
    if not candidate:
        return DEFAULT_POLICY

    if candidate not in _warned_invalid_policies:
        _warned_invalid_policies.add(candidate)
        logger.warning(
            "Unknown %s=%r — falling back to %r (valid: %s)",
            ENV_VAR,
            raw,
            DEFAULT_POLICY,
            ", ".join(VALID_POLICIES),
        )
    return DEFAULT_POLICY
