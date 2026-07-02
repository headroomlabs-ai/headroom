"""Air-gap / no-egress master switch (``HEADROOM_OFFLINE``).

A single predicate the individual egress paths consult so a regulated or
air-gapped deployment can disable **all** outbound network access with one
flag: the telemetry beacon, the update check, the license/usage reporter, and
HuggingFace model downloads. Each of those already had its own opt-out; this
is the one switch that turns them all off together and fails closed.

Kept at the top level (depends only on the stdlib) so any layer — telemetry,
proxy, model code — can import it without creating a package cycle.
"""

from __future__ import annotations

import os

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})

OFFLINE_ENV = "HEADROOM_OFFLINE"


def is_offline() -> bool:
    """Return True when ``HEADROOM_OFFLINE`` selects fully-offline operation."""
    return os.environ.get(OFFLINE_ENV, "").strip().lower() in _TRUE_VALUES


def apply_offline_env() -> None:
    """Force HuggingFace/Transformers offline so model code uses only locally
    cached artifacts and never reaches the Hub.

    Idempotent and uses ``setdefault`` so an explicit operator override (e.g.
    ``HF_HUB_OFFLINE=0``) still wins. Call once early in startup.
    """
    if is_offline():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
