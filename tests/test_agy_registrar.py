"""Tests for headroom.mcp_registry.agy.AgyRegistrar.

All tests use a tmp_path home_dir seam so the real ~/.gemini is never touched.
"""

from __future__ import annotations

import json
from pathlib import Path

from headroom.mcp_registry.agy import AgyRegistrar
from headroom.mcp_registry.base import RegisterStatus, ServerSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPEC = ServerSpec(
    name="headroom",
    command="headroom",
    args=("mcp", "serve"),
    env={"HEADROOM_PROXY_URL": "http://127.0.0.1:9999"},
)

_OTHER_SPEC = ServerSpec(
    name="other-server",
    command="/usr/bin/other",
    args=("--flag",),
    env={},
)


def _make_reg(tmp_path: Path) -> AgyRegistrar:
    return AgyRegistrar(home_dir=tmp_path)


def _config_path(tmp_path: Path) -> Path:
    return tmp_path / ".gemini" / "antigravity-cli" / "mcp_config.json"


def _write_config(tmp_path: Path, data: dict) -> None:
    p = _config_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2) + "\n")


def _read_config(tmp_path: Path) -> dict:
    p = _config_path(tmp_path)
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# detect
# ---------------------------------------------------------------------------


class TestDetect:
    def test_returns_false_when_dir_absent(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        assert reg.detect() is False

    def test_returns_true_when_config_dir_exists(self, tmp_path: Path) -> None:
        _config_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        reg = _make_reg(tmp_path)
        assert reg.detect() is True

    def test_returns_true_when_config_file_exists(self, tmp_path: Path) -> None:
        _write_config(tmp_path, {"mcpServers": {}})
        reg = _make_reg(tmp_path)
        assert reg.detect() is True


# ---------------------------------------------------------------------------
# get_server
# ---------------------------------------------------------------------------


class TestGetServer:
    def test_returns_none_when_file_absent(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        assert reg.get_server("headroom") is None

    def test_returns_none_when_server_absent(self, tmp_path: Path) -> None:
        _write_config(tmp_path, {"mcpServers": {}})
        reg = _make_reg(tmp_path)
        assert reg.get_server("headroom") is None

    def test_returns_spec_when_present(self, tmp_path: Path) -> None:
        _write_config(
            tmp_path,
            {
                "mcpServers": {
                    "headroom": {
                        "command": "headroom",
                        "args": ["mcp", "serve"],
                        "env": {"HEADROOM_PROXY_URL": "http://127.0.0.1:9999"},
                    }
                }
            },
        )
        reg = _make_reg(tmp_path)
        spec = reg.get_server("headroom")
        assert spec is not None
        assert spec.name == "headroom"
        assert spec.command == "headroom"
        assert tuple(spec.args) == ("mcp", "serve")
        assert spec.env == {"HEADROOM_PROXY_URL": "http://127.0.0.1:9999"}

    def test_malformed_json_returns_none(self, tmp_path: Path) -> None:
        p = _config_path(tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{not valid json")
        reg = _make_reg(tmp_path)
        assert reg.get_server("headroom") is None


# ---------------------------------------------------------------------------
# register_server — REGISTERED
# ---------------------------------------------------------------------------


class TestRegisterServer:
    def test_registers_new_server(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        result = reg.register_server(_SPEC)
        assert result.status == RegisterStatus.REGISTERED
        config = _read_config(tmp_path)
        assert "headroom" in config["mcpServers"]
        entry = config["mcpServers"]["headroom"]
        assert entry["command"] == "headroom"
        assert entry["args"] == ["mcp", "serve"]

    def test_registers_creates_parent_dirs(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        assert not _config_path(tmp_path).parent.exists()
        reg.register_server(_SPEC)
        assert _config_path(tmp_path).exists()

    # ALREADY
    def test_already_when_spec_matches(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        reg.register_server(_SPEC)
        result = reg.register_server(_SPEC)
        assert result.status == RegisterStatus.ALREADY

    # MISMATCH without force
    def test_mismatch_when_command_differs_no_force(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        reg.register_server(_SPEC)
        different = ServerSpec(
            name="headroom",
            command="/different/path/headroom",
            args=("mcp", "serve"),
            env={"HEADROOM_PROXY_URL": "http://127.0.0.1:9999"},
        )
        result = reg.register_server(different)
        assert result.status == RegisterStatus.MISMATCH
        # Config must be unchanged
        config = _read_config(tmp_path)
        assert config["mcpServers"]["headroom"]["command"] == "headroom"

    def test_mismatch_when_env_differs_no_force(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        reg.register_server(_SPEC)
        different = ServerSpec(
            name="headroom",
            command="headroom",
            args=("mcp", "serve"),
            env={"HEADROOM_PROXY_URL": "http://127.0.0.1:1111"},
        )
        result = reg.register_server(different)
        assert result.status == RegisterStatus.MISMATCH

    # force overwrite
    def test_force_overwrites_existing(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        reg.register_server(_SPEC)
        updated = ServerSpec(
            name="headroom",
            command="/new/headroom",
            args=("mcp", "serve"),
            env={"HEADROOM_PROXY_URL": "http://127.0.0.1:9999"},
        )
        result = reg.register_server(updated, force=True)
        assert result.status == RegisterStatus.REGISTERED
        config = _read_config(tmp_path)
        assert config["mcpServers"]["headroom"]["command"] == "/new/headroom"

    # MERGE: other entries preserved
    def test_merge_preserves_other_user_servers(self, tmp_path: Path) -> None:
        # Pre-populate with a user-managed server.
        _write_config(
            tmp_path,
            {
                "mcpServers": {
                    "user-server": {
                        "command": "/usr/bin/user-mcp",
                        "args": ["--some-flag"],
                    }
                }
            },
        )
        reg = _make_reg(tmp_path)
        result = reg.register_server(_SPEC)
        assert result.status == RegisterStatus.REGISTERED
        config = _read_config(tmp_path)
        # Both entries must exist.
        assert "user-server" in config["mcpServers"]
        assert "headroom" in config["mcpServers"]
        # User entry untouched.
        assert config["mcpServers"]["user-server"]["command"] == "/usr/bin/user-mcp"

    def test_missing_file_treated_as_empty(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        result = reg.register_server(_SPEC)
        assert result.status == RegisterStatus.REGISTERED

    def test_registers_spec_without_env(self, tmp_path: Path) -> None:
        spec = ServerSpec(name="minimal", command="headroom", args=())
        reg = _make_reg(tmp_path)
        result = reg.register_server(spec)
        assert result.status == RegisterStatus.REGISTERED
        config = _read_config(tmp_path)
        entry = config["mcpServers"]["minimal"]
        # env key should not be present when empty.
        assert "env" not in entry


# ---------------------------------------------------------------------------
# unregister_server
# ---------------------------------------------------------------------------


class TestUnregisterServer:
    def test_returns_false_when_file_absent(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        assert reg.unregister_server("headroom") is False

    def test_returns_false_when_server_absent(self, tmp_path: Path) -> None:
        _write_config(tmp_path, {"mcpServers": {}})
        reg = _make_reg(tmp_path)
        assert reg.unregister_server("headroom") is False

    def test_removes_named_server(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        reg.register_server(_SPEC)
        removed = reg.unregister_server("headroom")
        assert removed is True
        config = _read_config(tmp_path)
        assert "headroom" not in config["mcpServers"]

    def test_unregister_only_removes_named_server(self, tmp_path: Path) -> None:
        """Unregistering 'headroom' MUST NOT remove other user entries."""
        _write_config(
            tmp_path,
            {
                "mcpServers": {
                    "user-server": {"command": "/bin/user-mcp"},
                    "headroom": {
                        "command": "headroom",
                        "args": ["mcp", "serve"],
                    },
                }
            },
        )
        reg = _make_reg(tmp_path)
        reg.unregister_server("headroom")
        config = _read_config(tmp_path)
        assert "user-server" in config["mcpServers"]
        assert "headroom" not in config["mcpServers"]

    def test_idempotent_double_unregister(self, tmp_path: Path) -> None:
        reg = _make_reg(tmp_path)
        reg.register_server(_SPEC)
        assert reg.unregister_server("headroom") is True
        assert reg.unregister_server("headroom") is False
