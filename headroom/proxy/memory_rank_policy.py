"""Pure memory ranking policy helpers.

This module owns timestamp parsing and recency score math for proxy memory
ranking. It deliberately avoids backend objects and ranker classes so the
formula can be tested, ported, and reused independently of retrieval adapters.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

UTC = timezone.utc


def parse_memory_created_at(value: object) -> datetime | None:
    """Best-effort parse of a memory timestamp into a UTC-aware datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def memory_recency_factor(
    *,
    now: datetime,
    created_at: datetime | None,
    decay_days: float,
) -> float:
    """Compute the recency multiplier for one memory candidate.

    Missing timestamps and future timestamps are neutral. For normal historical
    timestamps the multiplier is ``exp(-age_days / decay_days)``.
    """
    if created_at is None:
        return 1.0
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    age_days = (now - created_at).total_seconds() / 86400.0
    if age_days <= 0:
        return 1.0
    return math.exp(-age_days / decay_days)


def boost_memory_score(
    *,
    score: float,
    now: datetime,
    created_at: datetime | None,
    decay_days: float,
) -> float:
    """Apply the recency multiplier to a backend similarity score."""
    return score * memory_recency_factor(
        now=now,
        created_at=created_at,
        decay_days=decay_days,
    )
