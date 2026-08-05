"""Tests for pure memory rank policy formulas."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from headroom.proxy.memory_rank_policy import (
    boost_memory_score,
    memory_recency_factor,
    parse_memory_created_at,
)

_UTC = timezone.utc


def test_parse_memory_created_at_accepts_zulu_iso_string() -> None:
    parsed = parse_memory_created_at("2026-05-19T12:00:00Z")
    assert parsed == datetime(2026, 5, 19, 12, 0, tzinfo=_UTC)


def test_parse_memory_created_at_normalizes_naive_datetime_to_utc() -> None:
    parsed = parse_memory_created_at(datetime(2026, 5, 19, 12, 0))
    assert parsed == datetime(2026, 5, 19, 12, 0, tzinfo=_UTC)


def test_parse_memory_created_at_invalid_values_are_neutral() -> None:
    assert parse_memory_created_at("not-a-date") is None
    assert parse_memory_created_at(123) is None
    assert parse_memory_created_at(None) is None


def test_memory_recency_factor_uses_exponential_decay() -> None:
    now = datetime(2026, 5, 31, tzinfo=_UTC)
    created_at = now - timedelta(days=30)
    factor = memory_recency_factor(now=now, created_at=created_at, decay_days=30.0)
    assert math.isclose(factor, math.exp(-1), rel_tol=1e-12)


def test_memory_recency_factor_treats_missing_and_future_dates_as_neutral() -> None:
    now = datetime(2026, 5, 31, tzinfo=_UTC)
    future = now + timedelta(days=3)
    assert memory_recency_factor(now=now, created_at=None, decay_days=30.0) == 1.0
    assert memory_recency_factor(now=now, created_at=future, decay_days=30.0) == 1.0


def test_boost_memory_score_applies_recency_factor() -> None:
    now = datetime(2026, 5, 31, tzinfo=_UTC)
    created_at = now - timedelta(days=60)
    boosted = boost_memory_score(
        score=0.9,
        now=now,
        created_at=created_at,
        decay_days=30.0,
    )
    assert math.isclose(boosted, 0.9 * math.exp(-2), rel_tol=1e-12)
