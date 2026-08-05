"""Pure token-bucket rate-limit policy helpers."""

from __future__ import annotations


def refilled_tokens(
    *,
    current_tokens: float,
    last_update: float,
    now: float,
    rate_per_minute: float,
) -> float:
    """Return token count after time-based refill, capped at bucket capacity."""
    elapsed = max(0.0, now - last_update)
    refill = elapsed * (rate_per_minute / 60.0)
    return min(rate_per_minute, current_tokens + refill)


def consume_from_bucket(
    *,
    available_tokens: float,
    requested_tokens: float,
    rate_per_minute: float,
) -> tuple[bool, float, float]:
    """Return ``(allowed, remaining_tokens, wait_seconds)`` for a token request."""
    if available_tokens >= requested_tokens:
        return True, available_tokens - requested_tokens, 0.0

    wait_seconds = (requested_tokens - available_tokens) * (60.0 / rate_per_minute)
    return False, available_tokens, wait_seconds


def stale_bucket_keys(
    last_updates: dict[str, float],
    *,
    now: float,
    stale_after_seconds: float,
) -> list[str]:
    """Return bucket keys whose last update is older than the stale threshold."""
    stale_before = now - stale_after_seconds
    return [key for key, last_update in last_updates.items() if last_update < stale_before]
