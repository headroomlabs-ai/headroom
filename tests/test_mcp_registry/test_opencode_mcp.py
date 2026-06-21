"""Tests for OpenCode MCP registrar."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from headroom.mcp_registry.base import RegisterResult, RegisterStatus, ServerSpec
from headroom.mcp_registry.opencode import OpencodeRegistrar


@pytest.fixture
def temp_config_path(tmp_path: Path) -> Path:
    return tmp_path / "opencode.json"


@pytest.fixture
def registrar(temp_config_path: Path) -> OpencodeRegistrar:
    return OpencodeRegistrar(config_path=temp_config_path)


class TestOpencodeRegistrarDetect:
    def test_detects_when_config_dir_exists(self, registrar: OpencodeRegistrar):
        registrar._config_path.parent.mkdir(parents=True, exist_ok=True)
        with patch("shutil.which", return_value=None):
            assert registrar.detect() is True

    def test_returns_false_when_no_dir(self, tmp_path: Path):
        r = OpencodeRegistrar(config_path=tmp_path / "nope" / "opencode.json")
        with patch("shutil.which", return_value=None):
            assert r.detect() is False


class TestOpencodeRegistrarRegister:
    def test_registers_new_server(self, registrar: OpencodeRegistrar):
        spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
        result = registrar.register_server(spec, force=True)
        assert result.status == RegisterStatus.REGISTERED

        data = json.loads(registrar._config_path.read_text())
        assert "mcp" in data
        assert "headroom" in data["mcp"]

    def test_already_registered_returns_already(
        self, registrar: OpencodeRegistrar
    ):
        spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
        registrar.register_server(spec, force=True)
        result = registrar.register_server(spec, force=False)
        assert result.status == RegisterStatus.ALREADY

    def test_mismatch_without_force_returns_mismatch(
        self, registrar: OpencodeRegistrar
    ):
        spec1 = ServerSpec(name="headroom", command="old", args=())
        registrar.register_server(spec1, force=True)

        spec2 = ServerSpec(name="headroom", command="new", args=())
        result = registrar.register_server(spec2, force=False)
        assert result.status == RegisterStatus.MISMATCH

    def test_force_overwrites_mismatch(self, registrar: OpencodeRegistrar):
        spec1 = ServerSpec(name="headroom", command="old", args=())
        registrar.register_server(spec1, force=True)

        spec2 = ServerSpec(name="headroom", command="new", args=())
        result = registrar.register_server(spec2, force=True)
        assert result.status == RegisterStatus.REGISTERED

    def test_unregister_removes_entry(self, registrar: OpencodeRegistrar):
        spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
        registrar.register_server(spec, force=True)
        assert "headroom" in json.loads(registrar._config_path.read_text())["mcp"]

        removed = registrar.unregister_server("headroom")
        assert removed is True
        data = json.loads(registrar._config_path.read_text())
        assert "headroom" not in data.get("mcp", {})

    def test_unregister_nonexistent(self, registrar: OpencodeRegistrar):
        assert registrar.unregister_server("nope") is False

    def test_unregister_removes_mcp_key_when_empty(
        self, registrar: OpencodeRegistrar
    ):
        spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
        registrar.register_server(spec, force=True)
        registrar.unregister_server("headroom")

        data = json.loads(registrar._config_path.read_text())
        assert "mcp" not in data

    def test_get_server_returns_spec(self, registrar: OpencodeRegistrar):
        spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
        registrar.register_server(spec, force=True)

        result = registrar.get_server("headroom")
        assert result is not None
        assert result.name == "headroom"
        assert result.command == "headroom"

    def test_get_server_nonexistent(self, registrar: OpencodeRegistrar):
        assert registrar.get_server("nope") is None

    def test_register_with_proxy_url_sets_entry(
        self, registrar: OpencodeRegistrar
    ):
        spec = ServerSpec(
            name="headroom",
            command="headroom",
            args=("mcp", "serve"),
            env={"HEADROOM_PROXY_URL": "http://127.0.0.1:9999/mcp"},
        )
        registrar.register_server(spec, force=True)
        data = json.loads(registrar._config_path.read_text())
        assert data["mcp"]["headroom"]["url"] == "http://127.0.0.1:9999/mcp"
        assert data["mcp"]["headroom"]["type"] == "remote"
        assert data["mcp"]["headroom"]["enabled"] is True

    def test_preserves_existing_mcp_entries(
        self, registrar: OpencodeRegistrar
    ):
        pre_spec = ServerSpec(
            name="context7",
            command="",
            args=("https://mcp.context7.com/mcp",),
        )
        registrar.register_server(pre_spec, force=True)

        headroom_spec = ServerSpec(
            name="headroom", command="headroom", args=("mcp", "serve")
        )
        registrar.register_server(headroom_spec, force=True)

        data = json.loads(registrar._config_path.read_text())
        assert "context7" in data["mcp"]
        assert "headroom" in data["mcp"]
