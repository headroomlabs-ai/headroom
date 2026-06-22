"""Tests for the OpenCode MCP registrar.

OpenCode stores MCP servers in ``opencode.json`` under the ``mcp`` key as
``{"type": "local", "command": [...], "environment": {...}, "enabled": true}``
entries. These exercise the registrar against a temp config file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.mcp_registry import (
    OpenCodeRegistrar,
    RegisterStatus,
    ServerSpec,
    build_headroom_spec,
    get_all_registrars,
)


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    return tmp_path / "opencode.json"


@pytest.fixture
def registrar(config_path: Path) -> OpenCodeRegistrar:
    return OpenCodeRegistrar(config_path=config_path)


def test_opencode_registrar_in_fleet() -> None:
    names = {r.name for r in get_all_registrars()}
    assert "opencode" in names


def test_register_writes_local_mcp_shape(registrar: OpenCodeRegistrar, config_path: Path) -> None:
    result = registrar.register_server(build_headroom_spec("http://127.0.0.1:9999"))
    assert result.status == RegisterStatus.REGISTERED

    data = json.loads(config_path.read_text())
    entry = data["mcp"]["headroom"]
    assert entry["type"] == "local"
    assert entry["command"] == ["headroom", "mcp", "serve"]
    assert entry["enabled"] is True
    # Non-default proxy URL flows into the environment block.
    assert entry["environment"] == {"HEADROOM_PROXY_URL": "http://127.0.0.1:9999"}


def test_register_default_url_omits_environment(
    registrar: OpenCodeRegistrar, config_path: Path
) -> None:
    registrar.register_server(build_headroom_spec())  # default 8787
    entry = json.loads(config_path.read_text())["mcp"]["headroom"]
    assert "environment" not in entry


def test_get_server_round_trips(registrar: OpenCodeRegistrar) -> None:
    spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    registrar.register_server(spec)
    got = registrar.get_server("headroom")
    assert got == spec
    assert registrar.get_server("absent") is None


def test_register_is_idempotent(registrar: OpenCodeRegistrar) -> None:
    spec = build_headroom_spec()
    assert registrar.register_server(spec).status == RegisterStatus.REGISTERED
    assert registrar.register_server(spec).status == RegisterStatus.ALREADY


def test_register_mismatch_without_force(registrar: OpenCodeRegistrar) -> None:
    registrar.register_server(ServerSpec(name="headroom", command="headroom", args=("mcp",)))
    result = registrar.register_server(
        ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    )
    assert result.status == RegisterStatus.MISMATCH

    forced = registrar.register_server(
        ServerSpec(name="headroom", command="headroom", args=("mcp", "serve")), force=True
    )
    assert forced.status == RegisterStatus.REGISTERED
    assert registrar.get_server("headroom").args == ("mcp", "serve")


def test_unregister_prunes_empty_mcp(registrar: OpenCodeRegistrar, config_path: Path) -> None:
    registrar.register_server(build_headroom_spec())
    assert registrar.unregister_server("headroom") is True
    assert registrar.unregister_server("headroom") is False  # already gone
    # The now-empty mcp object is removed entirely.
    assert "mcp" not in json.loads(config_path.read_text())


def test_register_preserves_user_config(registrar: OpenCodeRegistrar, config_path: Path) -> None:
    config_path.write_text(
        json.dumps(
            {
                "$schema": "https://opencode.ai/config.json",
                "model": "anthropic/claude-sonnet-4-5",
                "provider": {"openai": {"options": {"baseURL": "http://127.0.0.1:8787/v1"}}},
                "mcp": {"other": {"type": "local", "command": ["x"], "enabled": True}},
            }
        )
    )
    registrar.register_server(build_headroom_spec())
    data = json.loads(config_path.read_text())
    # Our entry is added; everything else is untouched.
    assert data["mcp"]["headroom"]["command"] == ["headroom", "mcp", "serve"]
    assert data["mcp"]["other"]["command"] == ["x"]
    assert data["model"] == "anthropic/claude-sonnet-4-5"
    assert data["provider"]["openai"]["options"]["baseURL"] == "http://127.0.0.1:8787/v1"


def test_detect_true_when_config_present(registrar: OpenCodeRegistrar, config_path: Path) -> None:
    config_path.write_text("{}")
    assert registrar.detect() is True
