"""Python forwarder mode resolution policy."""

from __future__ import annotations

from typing import Literal, cast

PYTHON_FORWARDER_MODE_ENV = "HEADROOM_PROXY_PYTHON_FORWARDER_MODE"
PythonForwarderMode = Literal["byte_faithful", "legacy_json_kwarg"]
PYTHON_FORWARDER_MODE_DEFAULT: PythonForwarderMode = "byte_faithful"


def resolve_python_forwarder_mode(raw: str | None) -> PythonForwarderMode:
    """Resolve the active Python-forwarder mode from an optional value."""
    normalized = (raw or "").strip().lower()
    if not normalized:
        return PYTHON_FORWARDER_MODE_DEFAULT
    if normalized in ("byte_faithful", "legacy_json_kwarg"):
        return cast(PythonForwarderMode, normalized)
    raise ValueError(
        f"Invalid {PYTHON_FORWARDER_MODE_ENV}={normalized!r}; "
        "expected 'byte_faithful' or 'legacy_json_kwarg'"
    )
