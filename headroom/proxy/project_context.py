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

A second contextvar, ``_current_registered_cwd``, is what disk verification
trusts as a workspace root. It is never a raw header value — it's the
result of ``workspace_registry.resolve_registered_cwd()``, which matches a
session token against a workspace root ``headroom wrap`` itself registered
over a trusted local channel (a 0700 marker file). A request can only
*reference* a pre-established root this way, never assert an arbitrary one.
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

# Server-verified, not header-derived -- see module docstring.
_current_registered_cwd: ContextVar[str | None] = ContextVar(
    "headroom_current_registered_cwd", default=None
)


def set_current_project(project: str | None) -> None:
    """Bind the active request's project for downstream outcome recording."""
    _current_project.set(sanitize_project_name(project))


def get_current_project() -> str | None:
    """Project bound to the current request context, or ``None``."""
    return _current_project.get()


def set_registered_cwd(cwd: str | None) -> None:
    """Bind the server-verified workspace root resolved for this request."""
    _current_registered_cwd.set(cwd)


def get_registered_cwd() -> str | None:
    """Server-verified workspace root for the current request, or ``None``."""
    return _current_registered_cwd.get()


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
    "get_registered_cwd",
    "set_current_project",
    "set_registered_cwd",
    "split_project_path",
    "strip_project_path_prefix",
    "with_project_prefix",
]
