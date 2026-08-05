from __future__ import annotations

import pytest

from headroom.proxy.python_forwarder_mode_policy import (
    PYTHON_FORWARDER_MODE_DEFAULT,
    resolve_python_forwarder_mode,
)


def test_resolve_python_forwarder_mode_defaults_for_missing_value() -> None:
    assert resolve_python_forwarder_mode(None) == PYTHON_FORWARDER_MODE_DEFAULT
    assert resolve_python_forwarder_mode("") == PYTHON_FORWARDER_MODE_DEFAULT
    assert resolve_python_forwarder_mode("   ") == PYTHON_FORWARDER_MODE_DEFAULT


def test_resolve_python_forwarder_mode_accepts_known_values_case_insensitively() -> None:
    assert resolve_python_forwarder_mode("byte_faithful") == "byte_faithful"
    assert resolve_python_forwarder_mode(" LEGACY_JSON_KWARG ") == "legacy_json_kwarg"


def test_resolve_python_forwarder_mode_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="HEADROOM_PROXY_PYTHON_FORWARDER_MODE"):
        resolve_python_forwarder_mode("json")
