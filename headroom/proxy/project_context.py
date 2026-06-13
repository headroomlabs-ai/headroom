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

from collections.abc import Mapping, MutableMapping
from contextvars import ContextVar
from typing import Any
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from headroom.proxy.savings_tracker import sanitize_project_name

PROJECT_HEADER = "x-headroom-project"
PROJECT_PATH_PREFIX = "/p/"

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


def split_project_path(path: str) -> tuple[str | None, str]:
    """Split ``/p/<name>/rest`` into ``(name, /rest)``.

    Clients that cannot send custom headers (aider, Copilot BYOK, Cursor)
    are pointed at a project-prefixed base URL instead; the first path
    segment after ``/p/`` is the URL-encoded project name. Returns
    ``(None, path)`` unchanged when the prefix is absent or unusable.
    """
    if not path.startswith(PROJECT_PATH_PREFIX):
        return None, path
    remainder = path[len(PROJECT_PATH_PREFIX) :]
    segment, sep, rest = remainder.partition("/")
    project = sanitize_project_name(unquote(segment)) if segment else None
    if project is None:
        return None, path
    return project, ("/" + rest) if sep else "/"


def strip_project_path_prefix(scope: MutableMapping[str, Any]) -> str | None:
    """Strip a ``/p/<name>`` prefix from an ASGI scope, returning the name.

    Mutates ``scope["path"]`` (and ``raw_path``) so routing sees the
    canonical path. Must run before anything caches the request URL.
    """
    project, stripped = split_project_path(scope.get("path", ""))
    if project is not None:
        scope["path"] = stripped
        if "raw_path" in scope:
            scope["raw_path"] = quote(stripped).encode("ascii")
    return project


def with_project_prefix(base_url: str, project: str | None) -> str:
    """Insert ``/p/<name>`` ahead of the path of a local proxy base URL.

    Producer-side counterpart of :func:`split_project_path`, used by
    ``headroom wrap`` for clients that cannot send custom headers.
    Returns ``base_url`` unchanged when the project name is unusable.
    """
    name = sanitize_project_name(project)
    if name is None:
        return base_url
    parts = urlsplit(base_url)
    prefixed = f"{PROJECT_PATH_PREFIX}{quote(name, safe='')}{parts.path}"
    return urlunsplit(parts._replace(path=prefixed.rstrip("/")))


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
