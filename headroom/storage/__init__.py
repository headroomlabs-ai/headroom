"""Storage modules for Headroom SDK."""

from .base import Storage
from .jsonl import JSONLStorage
from .sqlite import SQLiteStorage

__all__ = [
    "Storage",
    "SQLiteStorage",
    "JSONLStorage",
]


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
        path = store_url.replace("sqlite://", "")
        # Handle sqlite:/// (3 slashes for absolute path)
        if path.startswith("/"):
            path = path  # Already absolute
        return SQLiteStorage(path)
    elif store_url.startswith("jsonl://"):
        path = store_url.replace("jsonl://", "")
        if path.startswith("/"):
            path = path
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
            except Exception:
                pass
        # Default to SQLite (legacy behavior)
        return SQLiteStorage(store_url)
