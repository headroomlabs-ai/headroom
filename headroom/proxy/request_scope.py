"""ASGI request-scope mutation helpers."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

if TYPE_CHECKING:
    from fastapi import Request


def normalize_scope_path(scope: MutableMapping[str, Any], path: str) -> None:
    """Set an ASGI scope path and keep ``raw_path`` aligned when present."""
    scope["path"] = path
    if "raw_path" in scope:
        scope["raw_path"] = quote(path).encode("ascii")


def normalize_request_path(request: Request, path: str) -> None:
    """Set a FastAPI request path and clear its cached URL, if any."""
    normalize_scope_path(request.scope, path)
    if hasattr(request, "_url"):
        delattr(request, "_url")
