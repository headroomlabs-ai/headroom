"""Deployment context detection for telemetry.

Derives two orthogonal identity fields the beacon reports:

* ``install_mode`` — how the proxy process is deployed
  (``persistent`` / ``on_demand`` / ``wrapped`` / ``unknown``).
* ``headroom_stack`` — how Headroom is being invoked
  (``proxy``, ``wrap_claude``, ``adapter_ts_openai``, ...).

Both helpers are best-effort and never raise: telemetry is fire-and-forget and
must not break the proxy.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


_KNOWN_WRAP_AGENTS = frozenset(
    {"claude", "copilot", "codex", "aider", "cursor", "openclaw", "opencode"}
)

# Stack slugs must start with a letter and contain only [a-z0-9_], max 64 chars.
# Applied at every ingress (env var, HTTP header, stats aggregation) so downstream
# sinks (Prometheus labels, Supabase column, JSONB payload) see a bounded vocabulary.
_STACK_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# Cardinality cap on the per-process requests_by_stack dict. Protects the
# Prometheus scrape, the in-memory counter, and the JSONB telemetry payload
# from unbounded label explosion when clients send arbitrary X-Headroom-Stack
# header values.
MAX_DISTINCT_STACKS = 32


def normalize_stack(raw: str | None) -> str | None:
    """Validate and normalize a stack slug.

    Returns the lowercased/stripped slug if it matches ``^[a-z][a-z0-9_]{0,63}$``,
    else ``None``. All external stack identifiers (env var, HTTP header, stats
    keys) must pass through this function — it is the single chokepoint that
    bounds cardinality and rejects garbage before it reaches Prometheus or the
    Supabase telemetry row.
    """

    if not raw:
        return None
    slug = raw.strip().lower()
    if not _STACK_SLUG_RE.match(slug):
        return None
    return slug


def _slug_from_agent_type(agent_type: str) -> str:
    """Return ``wrap_<agent>`` for known agents, otherwise ``unknown``."""

    agent_type = agent_type.strip().lower()
    if agent_type and agent_type in _KNOWN_WRAP_AGENTS:
        return f"wrap_{agent_type}"
    return "unknown"


def detect_install_mode(port: int) -> str:
    """Classify how the proxy is deployed.

    Resolution order:

    1. ``HEADROOM_AGENT_TYPE`` env var set → ``wrapped`` (spawned by ``headroom wrap``).
    2. A ``DeploymentManifest`` on disk whose port matches ``port`` → ``persistent``.
    3. Otherwise → ``on_demand``.

    Any failure falls back to ``unknown`` so a broken install subsystem
    doesn't silence telemetry.
    """

    try:
        if os.environ.get("HEADROOM_AGENT_TYPE"):
            return "wrapped"

        try:
            from headroom.install.state import list_manifests

            for manifest in list_manifests():
                if getattr(manifest, "port", None) == port:
                    return "persistent"
        except Exception:
            logger.debug(
                "Beacon: manifest lookup failed during install_mode detection",
                exc_info=True,
            )

        return "on_demand"
    except Exception:
        logger.debug("Beacon: detect_install_mode crashed", exc_info=True)
        return "unknown"


def detect_stack(stats: dict[str, Any] | None = None) -> str:
    """Classify how Headroom is being invoked.

    Resolution order:

    1. ``HEADROOM_STACK`` env var set → use that slug verbatim.
    2. ``HEADROOM_AGENT_TYPE`` env var set → ``wrap_<agent>``.
    3. ``stats['requests']['by_stack']`` dict populated →
       pick the stack with >80% of requests, else ``mixed``.
    4. Otherwise → ``proxy``.

    Any failure falls back to ``unknown``.
    """

    try:
        explicit = normalize_stack(os.environ.get("HEADROOM_STACK"))
        if explicit:
            return explicit

        agent_type = os.environ.get("HEADROOM_AGENT_TYPE")
        if agent_type:
            return _slug_from_agent_type(agent_type)

        if stats:
            by_stack = (stats.get("requests") or {}).get("by_stack") or {}
            if by_stack:
                total = sum(by_stack.values())
                if total > 0:
                    dominant, count = max(by_stack.items(), key=lambda kv: kv[1])
                    if count / total >= 0.8:
                        return normalize_stack(str(dominant)) or "unknown"
                    return "mixed"

        return "proxy"
    except Exception:
        logger.debug("Beacon: detect_stack crashed", exc_info=True)
        return "unknown"
