"""Version-gate tests for thinking_compactor.bills_prior_thinking.

The gate decides whether a model re-bills prior-turn thinking (so compaction
pays) or strips it server-side (so compacting would turn free thinking into
billed text). It parses the version out of the model id, and must not mistake
the ``YYYYMMDD`` release-date suffix for the minor version.
"""

from __future__ import annotations

import pytest

from headroom.transforms.thinking_compactor import bills_prior_thinking


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        # Major-only ids with a date suffix: the date must NOT be read as the
        # minor version. Claude 4.0 strips prior thinking server-side -> False.
        ("claude-sonnet-4-20250514", False),
        ("claude-opus-4-20250514", False),
        ("claude-opus-4", False),
        # Explicit minor below the 4.6 threshold -> strips -> False.
        ("claude-sonnet-4-5-20250929", False),
        ("claude-haiku-4-5-20251001", False),
        ("claude-opus-4-1-20250805", False),
        ("claude-opus-4-5-20250101", False),
        # 4.6+ and the 5 family re-bill prior thinking -> True.
        ("claude-opus-4-6", True),
        ("claude-sonnet-4-6", True),
        ("claude-opus-4-8", True),
        ("claude-sonnet-5", True),
        ("claude-sonnet-5-20260101", True),
        # Legacy 3.x naming (with date) stays below threshold.
        ("claude-3-5-sonnet-20241022", False),
        ("claude-3-opus-20240229", False),
        # No version at all -> conservative False.
        ("some-unversioned-model", False),
    ],
)
def test_bills_prior_thinking_version_gate(model: str, expected: bool) -> None:
    assert bills_prior_thinking(model) is expected


def test_date_suffix_is_not_read_as_minor_version() -> None:
    """Regression for the specific inversion: a major-only id whose 8-digit date
    suffix was parsed as the minor version, flipping a strip-model to 'bills'."""
    # Same major (4), only the date suffix differs from an explicit-minor id.
    assert bills_prior_thinking("claude-sonnet-4-20250514") is False
    # Sanity: the real 4.6 id still bills.
    assert bills_prior_thinking("claude-sonnet-4-6-20260101") is True
