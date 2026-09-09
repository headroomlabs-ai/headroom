"""Telemetry state for Headroom (local-only build).

``HEADROOM_TELEMETRY`` — **off by default, opt-in.** Aggregates stats locally
for the in-process collector and the ``/stats`` / ``/v1/telemetry`` endpoints.
Never leaves the machine.

The upstream project's anonymous upload beacon (``HEADROOM_BEACON``) and the
license usage reporter have been removed from this build entirely: there is no
code path that transmits telemetry off this machine.

Operational metrics, if enabled, go to *your own* OpenTelemetry collector via
``HEADROOM_OTEL_METRICS_*`` — a destination you control.
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
    transmitted anywhere.
    """
    from headroom.offline import is_offline

    if is_offline():
        return False
    val = os.environ.get("HEADROOM_TELEMETRY", "").lower().strip()
    return val in _ON_VALUES


def is_beacon_enabled() -> bool:
    """The upload beacon does not exist in this build; always ``False``."""
    return False


def is_telemetry_warn_enabled() -> bool:
    """Check if telemetry warnings are enabled (feature flag, on by default).

    Set HEADROOM_TELEMETRY_WARN=off to suppress startup/wrap notices.
    """
    val = os.environ.get("HEADROOM_TELEMETRY_WARN", "on").lower().strip()
    return val not in _OFF_VALUES


def format_telemetry_notice(*, prefix: str = "") -> str:
    """Return a single-line telemetry notice suitable for CLI output.

    Returns an empty string when local telemetry or warnings are disabled so
    callers can unconditionally include the result in their output.
    """
    if not is_telemetry_warn_enabled():
        return ""
    if not is_telemetry_enabled():
        return ""
    return (
        f"{prefix}Telemetry:    ENABLED (local aggregate stats only — nothing sent externally) | "
        "Disable: HEADROOM_TELEMETRY=off or --no-telemetry"
    )
