"""Resolve Qdrant connection settings from environment variables.

Provides a single source of truth for ``HEADROOM_QDRANT_*`` env vars so that
``Memory``, ``Mem0Config``, ``DirectMem0Config``, and the proxy all pick up
the same defaults when the caller does not pass an explicit value.

Supported environment variables:

- ``HEADROOM_QDRANT_URL``        Full URL (e.g. ``https://xyz.cloud.qdrant.io:6333``).
                                  When set, takes precedence over host/port.
- ``HEADROOM_QDRANT_HOST``       Hostname. Default: ``localhost``.
- ``HEADROOM_QDRANT_PORT``       HTTP port. Default: ``6333``.
- ``HEADROOM_QDRANT_API_KEY``    API key for hosted Qdrant (e.g. Qdrant Cloud).
- ``HEADROOM_QDRANT_HTTPS``      ``true``/``false``. Forces HTTPS on/off.
- ``HEADROOM_QDRANT_PREFER_GRPC````true``/``false``. Use gRPC instead of HTTP.
- ``HEADROOM_QDRANT_GRPC_PORT``  gRPC port. Default: ``6334``.

Explicit constructor arguments always win over environment values; the env
vars only fill in defaults when the caller passes ``None`` (or omits the
argument on a dataclass that uses ``field(default_factory=...)``).
"""

from __future__ import annotations

import os

DEFAULT_QDRANT_HOST = "localhost"
DEFAULT_QDRANT_PORT = 6333
DEFAULT_QDRANT_GRPC_PORT = 6334

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


def _parse_port(raw: str | None, var_name: str) -> int | None:
    """Parse a port env value. Returns ``None`` if unset."""
    if raw is None:
        return None
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(f"{var_name}={raw!r} is not a valid integer port") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{var_name}={port} is outside the valid port range 1-65535")
    return port


def qdrant_env_url() -> str | None:
    """Return ``HEADROOM_QDRANT_URL`` or ``None`` if unset."""
    return _strip_env("HEADROOM_QDRANT_URL")


def qdrant_env_host() -> str:
    """Return ``HEADROOM_QDRANT_HOST`` or the ``localhost`` default."""
    return _strip_env("HEADROOM_QDRANT_HOST") or DEFAULT_QDRANT_HOST


def qdrant_env_port() -> int:
    """Return ``HEADROOM_QDRANT_PORT`` or the ``6333`` default."""
    return (
        _parse_port(_strip_env("HEADROOM_QDRANT_PORT"), "HEADROOM_QDRANT_PORT")
        or DEFAULT_QDRANT_PORT
    )


def qdrant_env_api_key() -> str | None:
    """Return ``HEADROOM_QDRANT_API_KEY`` or ``None`` if unset."""
    return _strip_env("HEADROOM_QDRANT_API_KEY")


def qdrant_env_https() -> bool | None:
    """Return ``HEADROOM_QDRANT_HTTPS`` parsed as bool, or ``None`` if unset."""
    return _parse_bool(_strip_env("HEADROOM_QDRANT_HTTPS"))


def qdrant_env_prefer_grpc() -> bool:
    """Return ``HEADROOM_QDRANT_PREFER_GRPC`` parsed as bool. Default: ``False``."""
    return _parse_bool(_strip_env("HEADROOM_QDRANT_PREFER_GRPC")) or False


def qdrant_env_grpc_port() -> int:
    """Return ``HEADROOM_QDRANT_GRPC_PORT`` or the ``6334`` default."""
    return (
        _parse_port(_strip_env("HEADROOM_QDRANT_GRPC_PORT"), "HEADROOM_QDRANT_GRPC_PORT")
        or DEFAULT_QDRANT_GRPC_PORT
    )


def build_qdrant_client_kwargs(
    *,
    url: str | None = None,
    host: str | None = None,
    port: int | None = None,
    api_key: str | None = None,
    https: bool | None = None,
    prefer_grpc: bool | None = None,
    grpc_port: int | None = None,
) -> dict[str, object]:
    """Build a kwargs dict suitable for ``qdrant_client.QdrantClient(**kwargs)``.

    URL takes precedence over host/port: if ``url`` is a non-empty string the
    returned dict contains ``url`` and omits ``host``/``port``. Otherwise
    ``host``/``port`` are populated (falling back to ``localhost:6333``).

    Optional fields (``api_key``, ``https``, ``prefer_grpc``, ``grpc_port``)
    are only included when they have a value, so callers that don't need them
    don't accidentally pass ``None`` into Qdrant client options that would
    override sensible library defaults.
    """
    kwargs: dict[str, object] = {}

    effective_url = url if url else None
    if effective_url:
        kwargs["url"] = effective_url
    else:
        kwargs["host"] = host or DEFAULT_QDRANT_HOST
        kwargs["port"] = port or DEFAULT_QDRANT_PORT

    if api_key:
        kwargs["api_key"] = api_key
    if https is not None:
        kwargs["https"] = https
    if prefer_grpc:
        kwargs["prefer_grpc"] = True
        kwargs["grpc_port"] = grpc_port or DEFAULT_QDRANT_GRPC_PORT

    return kwargs
