"""Storage modules for Headroom SDK."""

import logging
import os

from headroom.log_safety import describe_exception

from .base import Storage
from .jsonl import JSONLStorage
from .sqlite import SQLiteStorage

__all__ = [
    "Storage",
    "SQLiteStorage",
    "JSONLStorage",
]

logger = logging.getLogger(__name__)


def _builtin_storage_path(store_url: str, scheme: str) -> str:
    """Return the filesystem portion of a built-in storage URL."""
    path = store_url.removeprefix(f"{scheme}://")
    if (
        os.name == "nt"
        and len(path) >= 4
        and path[0] == "/"
        and path[1].isalpha()
        and path[2] == ":"
        and path[3] in ("/", "\\")
    ):
        return path[1:]
    return path


def create_storage(store_url: str) -> Storage:
    """
    Create a storage instance from URL.

    Supported URLs (built-in):
    - sqlite:///path/to/file.db
    - jsonl:///path/to/file.jsonl

    Other schemes (e.g. postgres://) can be provided by packages that register
    the setuptools entry point headroom.storage_backend with name=<scheme>.

    Args:
        store_url: Storage URL.

    Returns:
        Storage instance.
    """
    if store_url.startswith("sqlite://"):
        path = _builtin_storage_path(store_url, "sqlite")
        return SQLiteStorage(path)
    elif store_url.startswith("jsonl://"):
        path = _builtin_storage_path(store_url, "jsonl")
        return JSONLStorage(path)
    else:
        # Unknown scheme: try entry point headroom.storage_backend[name=scheme]
        scheme = store_url.split("://", 1)[0].lower() if "://" in store_url else ""
        if scheme:
            try:
                from importlib.metadata import entry_points

                all_eps = entry_points(group="headroom.storage_backend")
                ep = next((e for e in all_eps if e.name == scheme), None)
                if ep is not None:
                    create_fn = ep.load()
                    result: Storage = create_fn(store_url)
                    return result
                logger.warning(
                    "No headroom.storage_backend entry point for scheme %r; "
                    "falling back to SQLite storage",
                    scheme,
                )
            except Exception as e:
                # The plugin's exception can echo the store URL, credentials included,
                # so no level logs its message; DEBUG gets describe_exception (types and frames).
                logger.warning(
                    "Failed to load storage backend for scheme %r (%s); "
                    "falling back to SQLite storage",
                    scheme,
                    type(e).__name__,
                )
                logger.debug("Storage backend %r load failure: %s", scheme, describe_exception(e))
        # Default to SQLite (legacy behavior)
        return SQLiteStorage(store_url)
