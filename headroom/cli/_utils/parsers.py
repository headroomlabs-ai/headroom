"""Parsing utilities for CLI input."""

import re
from datetime import timedelta

import click


def parse_duration(duration_str: str) -> timedelta:
    """Parse a duration string into a timedelta.

    Supported formats:
        - "7d" - 7 days
        - "2w" - 2 weeks
        - "1m" - 1 month (30 days)
        - "3h" - 3 hours

    Args:
        duration_str: Duration string to parse.

    Returns:
        A timedelta representing the duration.

    Raises:
        click.BadParameter: If the format is invalid.
    """
    pattern = r"^(\d+)([dwmh])$"
    match = re.match(pattern, duration_str.strip().lower())

    if not match:
        raise click.BadParameter(
            f"Invalid duration format: '{duration_str}'. "
            "Use format like '7d' (days), '2w' (weeks), '1m' (months), '3h' (hours)."
        )

    value = int(match.group(1))
    unit = match.group(2)

    if value <= 0:
        raise click.BadParameter("Duration value must be positive.")

    unit_to_timedelta = {
        "h": timedelta(hours=value),
        "d": timedelta(days=value),
        "w": timedelta(weeks=value),
        "m": timedelta(days=value * 30),  # Approximate month as 30 days
    }

    return unit_to_timedelta[unit]
