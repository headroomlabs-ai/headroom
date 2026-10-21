"""Resolve cognee backend settings from environment variables.

Provides a single source of truth for ``HEADROOM_COGNEE_*`` env vars so that
``CogneeConfig``, the ``Memory`` facade, and the proxy all pick up the same
defaults when the caller does not pass an explicit value.

Supported environment variables:

- ``HEADROOM_COGNEE_DATASET``       Dataset name used for all headroom memories.
                                     Default: ``headroom_memories``.
- ``HEADROOM_COGNEE_SYSTEM_ROOT``   Directory for cognee system state (databases,
                                     caches). Unset means cognee's own default.
- ``HEADROOM_COGNEE_DATA_ROOT``     Directory for cognee data storage. Unset means
                                     cognee's own default.
- ``HEADROOM_COGNEE_SEARCH_TYPE``   cognee ``SearchType`` name used for searches
                                     (e.g. ``CHUNKS``, ``GRAPH_COMPLETION``).
                                     Default: ``CHUNKS``.
- ``HEADROOM_COGNEE_AUTO_COGNIFY``  ``true``/``false``. Whether to run
                                     ``cognee.cognify()`` after each save.
                                     Default: ``true``.
- ``HEADROOM_COGNEE_METADATA_DB``   Path to the SQLite file holding the cognee
                                     backend's durable memory registry and
                                     delete/update tombstones. Unset means
                                     ``headroom_cognee_meta.db`` under the data
                                     root (or system root, or ``~/.headroom``).

Explicit constructor arguments always win over environment values; the env
vars only fill in defaults when the caller omits the argument (dataclasses
use ``field(default_factory=...)`` to defer resolution to instantiation time).
"""

from __future__ import annotations

import os

DEFAULT_COGNEE_DATASET = "headroom_memories"
DEFAULT_COGNEE_SEARCH_TYPE = "CHUNKS"

_TRUTHY = frozenset({"1", "true", "yes", "y", "on"})
_FALSY = frozenset({"0", "false", "no", "n", "off"})


def _strip_env(name: str) -> str | None:
    """Return the trimmed env var value, or ``None`` if unset/empty."""
    raw = os.environ.get(name)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def _parse_bool(raw: str | None) -> bool | None:
    """Parse a bool env value. Returns ``None`` if unset, else True/False.

    Unknown strings raise ``ValueError`` so misconfiguration is visible.
    """
    if raw is None:
        return None
    lowered = raw.lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    raise ValueError(f"Invalid boolean value {raw!r}; expected one of {sorted(_TRUTHY | _FALSY)}")


def cognee_env_dataset() -> str:
    """Return ``HEADROOM_COGNEE_DATASET`` or the ``headroom_memories`` default."""
    return _strip_env("HEADROOM_COGNEE_DATASET") or DEFAULT_COGNEE_DATASET


def cognee_env_system_root() -> str | None:
    """Return ``HEADROOM_COGNEE_SYSTEM_ROOT`` or ``None`` if unset."""
    return _strip_env("HEADROOM_COGNEE_SYSTEM_ROOT")


def cognee_env_data_root() -> str | None:
    """Return ``HEADROOM_COGNEE_DATA_ROOT`` or ``None`` if unset."""
    return _strip_env("HEADROOM_COGNEE_DATA_ROOT")


def cognee_env_search_type() -> str:
    """Return ``HEADROOM_COGNEE_SEARCH_TYPE`` or the ``CHUNKS`` default."""
    return _strip_env("HEADROOM_COGNEE_SEARCH_TYPE") or DEFAULT_COGNEE_SEARCH_TYPE


def cognee_env_auto_cognify() -> bool:
    """Return ``HEADROOM_COGNEE_AUTO_COGNIFY`` parsed as bool. Default: ``True``."""
    parsed = _parse_bool(_strip_env("HEADROOM_COGNEE_AUTO_COGNIFY"))
    return True if parsed is None else parsed


def cognee_env_metadata_db() -> str | None:
    """Return ``HEADROOM_COGNEE_METADATA_DB`` or ``None`` if unset."""
    return _strip_env("HEADROOM_COGNEE_METADATA_DB")
