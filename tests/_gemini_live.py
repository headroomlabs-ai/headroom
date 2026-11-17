"""Helpers for optional live Gemini tests."""

from __future__ import annotations

from typing import Any

import pytest

GEMINI_QUOTA_STATUS = 429
_GEMINI_QUOTA_MARKERS = (
    "quota",
    "rate limit",
    "rate-limit",
    "too many requests",
)


def skip_if_gemini_quota_exhausted(response: Any) -> None:
    """Skip live Gemini tests when the configured key has no available quota."""
    if getattr(response, "status_code", None) != GEMINI_QUOTA_STATUS:
        return
    body = getattr(response, "text", "") or ""
    if any(marker in body.lower() for marker in _GEMINI_QUOTA_MARKERS):
        pytest.skip("Gemini live API quota/rate limit exhausted")
