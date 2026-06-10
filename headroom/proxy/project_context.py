"""Per-request project attribution for the proxy.

``headroom wrap`` launches agents with an ``X-Headroom-Project`` header
(via ``ANTHROPIC_CUSTOM_HEADERS`` for Claude Code and ``env_http_headers``
for Codex) naming the project directory the agent is working in. The proxy
captures that header once per request — in the HTTP middleware for regular
requests and at the WebSocket accept for Codex responses-WS sessions —
into a :mod:`contextvars` variable, so the outcome funnel can attribute
savings to a project without threading a parameter through every handler.

The value is sanitized (printable characters only, length-capped) before it
is stored; an absent or unusable header simply leaves attribution off for
that request, matching pre-feature behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any

from headroom.proxy.savings_tracker import sanitize_project_name

PROJECT_HEADER = "x-headroom-project"

_current_project: ContextVar[str | None] = ContextVar("headroom_current_project", default=None)


def classify_project(headers: Mapping[str, Any] | Any) -> str | None:
    """Extract a sanitized project name from request headers, if present."""
    get = getattr(headers, "get", None)
    if get is None:
        return None
    value = get(PROJECT_HEADER) or get("X-Headroom-Project")
    return sanitize_project_name(value)


def set_current_project(project: str | None) -> None:
    """Bind the active request's project for downstream outcome recording."""
    _current_project.set(sanitize_project_name(project))


def get_current_project() -> str | None:
    """Project bound to the current request context, or ``None``."""
    return _current_project.get()


__all__ = [
    "PROJECT_HEADER",
    "classify_project",
    "get_current_project",
    "set_current_project",
]
