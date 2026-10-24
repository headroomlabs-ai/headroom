"""One licence variable, and an explicit switch for usage reporting.

Headroom reads its licence token from ``HEADROOM_LICENSE``. Every licensed
extension reads the same variable, so core does too.

``HEADROOM_LICENSE_KEY`` was the older name read only by core's cloud usage
reporter. It is still honoured for one release as a deprecated alias, with a
warning, so existing managed deployments keep their licence while they rename.

Having a licence set never turns on outbound reporting by itself. The usage
reporter validates the licence against the Headroom cloud and posts aggregate
usage counts; it runs only when the operator also sets
``HEADROOM_USAGE_REPORTING=1``. ``HEADROOM_OFFLINE`` still suppresses it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

logger = logging.getLogger("headroom.license_env")

LICENSE_ENV = "HEADROOM_LICENSE"
DEPRECATED_LICENSE_ENV = "HEADROOM_LICENSE_KEY"
USAGE_REPORTING_ENV = "HEADROOM_USAGE_REPORTING"

_TRUE = frozenset({"1", "true", "yes", "on"})


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def resolve_license_token(environ: Mapping[str, str] | None = None) -> str | None:
    """Return the licence token, preferring ``HEADROOM_LICENSE``.

    Falls back to the deprecated ``HEADROOM_LICENSE_KEY`` with a warning. When
    both are set to different values, ``HEADROOM_LICENSE`` wins and a warning
    says so, because silently picking one would hide a misconfiguration.
    """
    env = os.environ if environ is None else environ
    current = _clean(env.get(LICENSE_ENV))
    legacy = _clean(env.get(DEPRECATED_LICENSE_ENV))

    if current and legacy and current != legacy:
        logger.warning(
            "event=license_env_conflict using=%s ignored=%s hint=unset_%s_it_is_deprecated",
            LICENSE_ENV,
            DEPRECATED_LICENSE_ENV,
            DEPRECATED_LICENSE_ENV,
        )
        return current
    if current:
        return current
    if legacy:
        logger.warning(
            "event=license_env_deprecated variable=%s replacement=%s "
            "hint=rename_it_the_alias_is_removed_in_the_next_release",
            DEPRECATED_LICENSE_ENV,
            LICENSE_ENV,
        )
        return legacy
    return None


def usage_reporting_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """True only when the operator explicitly opted in to usage reporting."""
    env = os.environ if environ is None else environ
    return (env.get(USAGE_REPORTING_ENV) or "").strip().lower() in _TRUE
