"""Telemetry opt-in state for Headroom.

Headroom collects only **local**, aggregate telemetry (tokens saved, compression
ratios, timings) for the in-process collector and the ``/stats`` /
``/v1/telemetry`` endpoints. **Nothing is sent to Headroom Labs:** the anonymous
telemetry beacon that previously shipped aggregate stats has been removed.
Operational metrics can still be exported to *your own* OpenTelemetry collector
via ``HEADROOM_OTEL_METRICS_*`` (a destination you control).

This module holds the ``HEADROOM_TELEMETRY`` opt-in predicate (off by default)
that gates local collection, plus the CLI notice helpers.
"""

from __future__ import annotations

import os

_OFF_VALUES = frozenset(("off", "false", "0", "no", "disable", "disabled"))
_ON_VALUES = frozenset(("on", "true", "1", "yes", "enable", "enabled"))


def is_telemetry_enabled() -> bool:
    """Check if local telemetry collection is enabled (off by default, opt-in).

    Fail-closed: only enabled when HEADROOM_TELEMETRY is set to an explicit
    on-value (on/true/1/yes/enable/enabled). Anything else — including unset,
    empty, or an unrecognized value — leaves it disabled. Local collection only
    feeds the in-process collector and the ``/stats`` endpoint; nothing is
    transmitted to Headroom Labs.
    """
    val = os.environ.get("HEADROOM_TELEMETRY", "").lower().strip()
    return val in _ON_VALUES


def is_telemetry_warn_enabled() -> bool:
    """Check if telemetry warnings are enabled (feature flag, on by default).

    Set HEADROOM_TELEMETRY_WARN=off to suppress startup/wrap notices.
    This is a build/pack-time feature flag intended for operators who want
    to disable the notice without disabling telemetry itself.
    """
    val = os.environ.get("HEADROOM_TELEMETRY_WARN", "on").lower().strip()
    return val not in _OFF_VALUES


def format_telemetry_notice(*, prefix: str = "") -> str:
    """Return a single-line telemetry notice suitable for CLI output.

    Args:
        prefix: Optional leading whitespace / box-drawing prefix.

    Returns an empty string when telemetry or warnings are disabled so callers
    can unconditionally include the result in their output.
    """
    if not is_telemetry_enabled() or not is_telemetry_warn_enabled():
        return ""
    return (
        f"{prefix}Telemetry:    ENABLED (local aggregate stats only — nothing sent externally) | "
        "Disable: HEADROOM_TELEMETRY=off or --no-telemetry"
    )
