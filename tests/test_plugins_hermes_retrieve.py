"""Tests for the Hermes headroom_retrieve plugin (plugins/hermes/).

The plugin targets the Hermes Agent runtime, which provides a
``tools.registry`` module. That module does not exist inside the headroom
codebase, so these tests stub it before loading the plugin file directly
via importlib.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import httpx
import pytest

PLUGIN_PATH = (
    Path(__file__).resolve().parent.parent
    / "plugins"
    / "hermes"
    / "headroom_retrieve"
    / "__init__.py"
)


def _load_plugin() -> types.ModuleType:
    """Load the plugin file with a stubbed Hermes ``tools.registry``."""
    registry = types.ModuleType("tools.registry")
    registry.tool_error = lambda msg: json.dumps({"error": msg})  # type: ignore[attr-defined]
    registry.tool_result = lambda data: json.dumps({"result": data})  # type: ignore[attr-defined]
    tools_pkg = types.ModuleType("tools")
    tools_pkg.registry = registry  # type: ignore[attr-defined]
    sys.modules.setdefault("tools", tools_pkg)
    sys.modules["tools.registry"] = registry

    spec = importlib.util.spec_from_file_location("hermes_headroom_retrieve", PLUGIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self) -> dict[str, Any]:
        return self._payload


@pytest.fixture()
def plugin() -> types.ModuleType:
    return _load_plugin()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("abc123", "abc123"),
        ("ccr:abc123", "abc123"),
        ("<<ccr:abc123>>", "abc123"),
        ("<<ccr:abc123,base64,4.5KB>>", "abc123"),
        ("hash=abc123", "abc123"),
        ("  <<ccr:abc123>>  ", "abc123"),
    ],
)
def test_handler_normalizes_marker_shapes_to_bare_hash(
    plugin: types.ModuleType, monkeypatch: pytest.MonkeyPatch, raw: str, expected: str
) -> None:
    # Arrange
    seen: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], timeout: int) -> _FakeResponse:  # noqa: A002
        seen["payload"] = json
        return _FakeResponse(200, {"original_content": "data", "original_tokens": 1})

    monkeypatch.setattr(plugin.httpx, "post", fake_post)

    # Act
    plugin._handle_headroom_retrieve({"hash": raw})

    # Assert
    assert seen["payload"]["hash"] == expected


def test_handler_passes_optional_query_through(
    plugin: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    seen: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], timeout: int) -> _FakeResponse:  # noqa: A002
        seen["payload"] = json
        return _FakeResponse(200, {"original_content": "data"})

    monkeypatch.setattr(plugin.httpx, "post", fake_post)

    # Act
    plugin._handle_headroom_retrieve({"hash": "abc123", "query": "error lines"})

    # Assert
    assert seen["payload"] == {"hash": "abc123", "query": "error lines"}


def test_handler_returns_original_content_on_200(
    plugin: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    payload = {"original_content": "full text", "original_tokens": 42, "tool_name": "read_file"}
    monkeypatch.setattr(plugin.httpx, "post", lambda *a, **kw: _FakeResponse(200, payload))

    # Act
    out = json.loads(plugin._handle_headroom_retrieve({"hash": "abc123"}))

    # Assert
    assert out["result"]["original_content"] == "full text"
    assert out["result"]["original_tokens"] == 42
    assert out["result"]["tool_name"] == "read_file"


def test_handler_reports_expired_entry_on_404(
    plugin: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    monkeypatch.setattr(plugin.httpx, "post", lambda *a, **kw: _FakeResponse(404))

    # Act
    out = json.loads(plugin._handle_headroom_retrieve({"hash": "deadbeef"}))

    # Assert: actionable message, not a bare error
    assert "expired" in out["error"]
    assert "re-run" in out["error"].lower()


def test_handler_reports_unreachable_proxy_on_connection_error(
    plugin: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange
    def raise_connect_error(*a: Any, **kw: Any) -> None:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(plugin.httpx, "post", raise_connect_error)

    # Act
    out = json.loads(plugin._handle_headroom_retrieve({"hash": "abc123"}))

    # Assert
    assert "unreachable" in out["error"]


def test_handler_rejects_empty_hash(plugin: types.ModuleType) -> None:
    # Act
    out = json.loads(plugin._handle_headroom_retrieve({"hash": "   "}))

    # Assert
    assert "required" in out["error"]


def test_register_exposes_tool_with_marker_aware_schema(plugin: types.ModuleType) -> None:
    # Arrange
    registered: dict[str, Any] = {}

    class FakeCtx:
        def register_tool(self, **kwargs: Any) -> None:
            registered.update(kwargs)

    # Act
    plugin.register(FakeCtx())

    # Assert
    assert registered["name"] == "headroom_retrieve"
    assert registered["toolset"] == "headroom"
    assert registered["schema"]["parameters"]["required"] == ["hash"]
    # The description must teach both marker formats so the model recognizes them.
    description = registered["schema"]["description"]
    assert "<<ccr:" in description
    assert "hash=" in description
