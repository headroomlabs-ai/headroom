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

from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any

from headroom.proxy.project_policy import (
    PROJECT_HEADER,
    PROJECT_PATH_PREFIX,
    classify_project,
    split_project_path,
    with_project_prefix,
)
from headroom.proxy.request_scope import normalize_scope_path
from headroom.proxy.savings_tracker import sanitize_project_name

_current_project: ContextVar[str | None] = ContextVar("headroom_current_project", default=None)


def set_current_project(project: str | None) -> None:
    """Bind the active request's project for downstream outcome recording."""
    _current_project.set(sanitize_project_name(project))


def get_current_project() -> str | None:
    """Project bound to the current request context, or ``None``."""
    return _current_project.get()


def strip_project_path_prefix(scope: MutableMapping[str, Any]) -> str | None:
    """Strip a ``/p/<name>`` prefix from an ASGI scope, returning the name.

    Mutates ``scope["path"]`` (and ``raw_path``) so routing sees the
    canonical path. Must run before anything caches the request URL.
    """
    project, stripped = split_project_path(scope.get("path", ""))
    if project is not None:
        normalize_scope_path(scope, stripped)
    return project


__all__ = [
    "PROJECT_HEADER",
    "PROJECT_PATH_PREFIX",
    "classify_project",
    "get_current_project",
    "set_current_project",
    "split_project_path",
    "strip_project_path_prefix",
    "with_project_prefix",
]
