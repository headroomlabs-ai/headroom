"""Telemetry state for Headroom.

Two independent switches, because they answer two different questions:

``HEADROOM_TELEMETRY`` — **off by default, opt-in.** Aggregates stats locally
for the in-process collector and the ``/stats`` / ``/v1/telemetry`` endpoints.
Never leaves the machine.

``HEADROOM_BEACON`` — **on by default, opt-out.** Uploads an anonymous,
content-free session summary to Headroom Labs. Disable with
``HEADROOM_BEACON=off``, ``DO_NOT_TRACK=1``, or offline mode. See
:mod:`headroom.telemetry.session` for the exact payload and
``deploy/beacon/worker.js`` for what the receiver does with it.

Operational metrics are separate again, and go to *your own* OpenTelemetry
collector via ``HEADROOM_OTEL_METRICS_*`` — a destination you control, that
Headroom Labs never sees.

Keeping these separate matters: an operator who turned on local stats has not
thereby agreed to upload anything, and must not start doing so on upgrade.
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
    from headroom.offline import is_offline

    if is_offline():
        return False
    val = os.environ.get("HEADROOM_TELEMETRY", "").lower().strip()
    return val in _ON_VALUES


# Whether the upload beacon runs when the operator has expressed no preference.
#
# True makes Headroom opt-OUT: every install uploads anonymous session summaries
# unless it is told not to. Flip to False for opt-in and nothing uploads without
# an explicit HEADROOM_BEACON=on.
#
# This single constant is the whole policy — deliberately, so the decision is
# one line to audit and one line to reverse.
BEACON_DEFAULT_ON = True


def is_beacon_enabled() -> bool:
    """Check if the anonymous upload beacon is enabled.

    Default is :data:`BEACON_DEFAULT_ON`. Three things switch it off regardless,
    in this order:

    * offline mode — no network is no network;
    * ``DO_NOT_TRACK`` — the cross-tool convention, honoured because a user who
      has already stated this preference should not have to state it again;
    * ``HEADROOM_BEACON=off`` (or false/0/no/disable/disabled).

    Deliberately a *separate* switch from ``HEADROOM_TELEMETRY``. That flag
    means "aggregate stats locally so ``/stats`` has something to show" — data
    that never leaves the machine. Overloading it to also authorise upload
    would mean every operator who had turned on local stats begins transmitting
    the moment they upgrade, having answered a different question.

    Note the asymmetry with :func:`is_telemetry_enabled`, which is fail-closed:
    this one is fail-open by construction while ``BEACON_DEFAULT_ON`` is True,
    so an unrecognised value uploads rather than staying silent.
    """
    from headroom.offline import is_offline

    if is_offline():
        return False
    dnt = os.environ.get("DO_NOT_TRACK", "").lower().strip()
    if dnt and dnt not in _OFF_VALUES:
        return False
    raw = os.environ.get("HEADROOM_BEACON", "").lower().strip()
    if raw in _OFF_VALUES:
        return False
    if raw in _ON_VALUES:
        return True
    return BEACON_DEFAULT_ON


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
    if not is_telemetry_warn_enabled():
        return ""

    beacon = is_beacon_enabled()
    local = is_telemetry_enabled()
    if not beacon and not local:
        return ""

    # The beacon line comes first and is unconditional when on. It is opt-out,
    # so this notice is the only place a user learns it is running at all —
    # surprise is what turns anonymous telemetry into a trust incident.
    #
    # Wording: lead with what the data *is* (counters) and why it is useful
    # (understanding when compression works), not with the fact of
    # transmission. "Stats are sent to Headroom Labs" is accurate but reads like
    # extraction, which misrepresents a payload that is entirely counters. The
    # reassurance has to stay concrete, though — "never prompts, code, or file
    # paths" names the things people actually worry about, and every one of
    # those is enforced by the allowlist, not just promised here.
    #
    # "compression stats" until schema v2, which widened the payload to cache
    # behaviour, session shape, harness and configuration. All still counters,
    # all still content-free — but this notice is the only place an opt-out
    # beacon is disclosed at all, so it has to describe what is actually sent
    # rather than the narrowest true thing. `headroom telemetry --show` prints
    # the exact payload for anyone who wants the specifics.
    if beacon:
        return (
            f"{prefix}Telemetry:    anonymous usage counters — never prompts, code, or file "
            "paths. See: headroom telemetry --show | Off: HEADROOM_BEACON=off"
        )
    return (
        f"{prefix}Telemetry:    ENABLED (local aggregate stats only — nothing sent externally) | "
        "Disable: HEADROOM_TELEMETRY=off or --no-telemetry"
    )
