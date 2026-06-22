"""Tests for :class:`headroom.mcp_registry.opencode.OpencodeRegistrar`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from headroom.mcp_registry.base import RegisterStatus
from headroom.mcp_registry.opencode import (
    OpencodeRegistrar,
    _diff_specs,
    _entry_to_spec,
    _spec_to_entry,
    _specs_equivalent,
)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _registrar(tmp_path: Path) -> OpencodeRegistrar:
    return OpencodeRegistrar(config_path=tmp_path / "opencode.json")


def test_detect_when_binary_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Detection succeeds when the opencode binary is in PATH."""
    monkeypatch.setenv("PATH", str(tmp_path))
    (tmp_path / "opencode").write_text("#!/bin/sh\necho ok")
    (tmp_path / "opencode").chmod(0o755)
    registrar = _registrar(tmp_path)
    assert registrar.detect() is True


def test_detect_when_config_dir_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Detection succeeds when the config directory exists."""
    monkeypatch.setenv("PATH", "/nonexistent")
    config_dir = tmp_path / "opencode"
    config_dir.mkdir()
    registrar = OpencodeRegistrar(config_path=config_dir / "opencode.json")
    assert registrar.detect() is True


def test_detect_when_nothing_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Detection fails when neither binary nor config dir exists."""
    monkeypatch.setenv("PATH", "/nonexistent")
    registrar = OpencodeRegistrar(config_path=tmp_path / "nonexistent" / "opencode.json")
    assert registrar.detect() is False


def test_get_server_returns_none_when_absent(tmp_path: Path) -> None:
    """get_server returns None when the server is not configured."""
    registrar = _registrar(tmp_path)
    assert registrar.get_server("headroom") is None


def test_get_server_returns_spec_when_present(tmp_path: Path) -> None:
    """get_server parses the existing MCP entry correctly."""
    config = {
        "mcp": {
            "headroom": {
                "type": "remote",
                "url": "http://127.0.0.1:8787/mcp",
                "enabled": True,
            }
        }
    }
    _write_json(tmp_path / "opencode.json", config)
    registrar = _registrar(tmp_path)
    spec = registrar.get_server("headroom")
    assert spec is not None
    assert spec.name == "headroom"


def test_register_server_creates_config_when_missing(tmp_path: Path) -> None:
    """register_server creates the config file when it doesn't exist."""
    registrar = _registrar(tmp_path)
    from headroom.mcp_registry.base import ServerSpec

    spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    result = registrar.register_server(spec)
    assert result.status == RegisterStatus.REGISTERED
    config_path = tmp_path / "opencode.json"
    assert config_path.exists()


def test_register_server_idempotent(tmp_path: Path) -> None:
    """register_server is a no-op when the same spec is already present."""
    registrar = _registrar(tmp_path)
    from headroom.mcp_registry.base import ServerSpec

    spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    registrar.register_server(spec)
    result = registrar.register_server(spec)
    assert result.status == RegisterStatus.ALREADY


def test_unregister_server_removes_entry(tmp_path: Path) -> None:
    """unregister_server removes the server entry."""
    registrar = _registrar(tmp_path)
    from headroom.mcp_registry.base import ServerSpec

    spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    registrar.register_server(spec)
    assert registrar.unregister_server("headroom") is True
    assert registrar.get_server("headroom") is None


def test_unregister_server_returns_false_when_absent(tmp_path: Path) -> None:
    """unregister_server returns False when the server was not registered."""
    registrar = _registrar(tmp_path)
    assert registrar.unregister_server("headroom") is False


def test_specs_equivalent_true() -> None:
    from headroom.mcp_registry.base import ServerSpec

    a = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    b = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    assert _specs_equivalent(a, b) is True


def test_specs_equivalent_false() -> None:
    from headroom.mcp_registry.base import ServerSpec

    a = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    b = ServerSpec(name="headroom", command="other", args=("mcp", "serve"))
    assert _specs_equivalent(a, b) is False


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_register_server_force_overwrites_mismatch(tmp_path: Path) -> None:
    """register_server with force=True overwrites a mismatched existing server."""
    registrar = _registrar(tmp_path)
    from headroom.mcp_registry.base import ServerSpec

    spec_a = ServerSpec(name="headroom", command="old", args=("serve",))
    spec_b = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))

    registrar.register_server(spec_a)
    assert registrar.register_server(spec_b).status == RegisterStatus.MISMATCH
    assert registrar.register_server(spec_b, force=True).status == RegisterStatus.REGISTERED
    updated = registrar.get_server("headroom")
    assert updated is not None
    assert updated.command == "headroom"


def test_unregister_removes_mcp_key_when_empty(tmp_path: Path) -> None:
    """unregister_server removes the top-level 'mcp' key when it becomes empty."""
    registrar = _registrar(tmp_path)
    from headroom.mcp_registry.base import ServerSpec

    spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    registrar.register_server(spec)
    assert registrar.unregister_server("headroom") is True
    assert registrar.get_server("headroom") is None
    # mcp key should be removed entirely
    import json

    data = json.loads((tmp_path / "opencode.json").read_text())
    assert "mcp" not in data


def test_register_server_leaves_other_mcp_servers(tmp_path: Path) -> None:
    """register_server preserves other MCP servers in the config."""
    registrar = _registrar(tmp_path)
    from headroom.mcp_registry.base import ServerSpec

    # Pre-populate with a user-managed MCP server
    _write_json(
        tmp_path / "opencode.json",
        {
            "mcp": {
                "existing-server": {
                    "type": "remote",
                    "url": "https://example.com",
                    "enabled": True,
                },
            }
        },
    )

    spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    registrar.register_server(spec)

    data = json.loads((tmp_path / "opencode.json").read_text())
    assert "headroom" in data["mcp"]
    assert "existing-server" in data["mcp"]


def test_unregister_preserves_other_mcp_servers(tmp_path: Path) -> None:
    """unregister_server leaves other MCP servers intact."""
    registrar = _registrar(tmp_path)
    from headroom.mcp_registry.base import ServerSpec

    _write_json(
        tmp_path / "opencode.json",
        {
            "mcp": {
                "existing-server": {"type": "remote", "url": "https://example.com"},
            }
        },
    )
    spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    registrar.register_server(spec)
    registrar.unregister_server("headroom")

    data = json.loads((tmp_path / "opencode.json").read_text())
    assert "headroom" not in data["mcp"]
    assert "existing-server" in data["mcp"]


def test_get_server_returns_none_for_non_dict_mcp(tmp_path: Path) -> None:
    """get_server returns None when 'mcp' is not a dict."""
    registrar = _registrar(tmp_path)
    _write_json(tmp_path / "opencode.json", {"mcp": "not-a-dict"})
    assert registrar.get_server("headroom") is None


def test_get_server_handles_missing_config_file(tmp_path: Path) -> None:
    """get_server returns None when the config file doesn't exist."""
    registrar = _registrar(tmp_path)
    assert registrar.get_server("headroom") is None


def test_register_server_handles_config_with_no_mcp_key(tmp_path: Path) -> None:
    """register_server adds 'mcp' key when it doesn't exist."""
    registrar = _registrar(tmp_path)
    _write_json(tmp_path / "opencode.json", {"model": "gpt-4o"})
    from headroom.mcp_registry.base import ServerSpec

    spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    registrar.register_server(spec)

    data = json.loads((tmp_path / "opencode.json").read_text())
    assert data["model"] == "gpt-4o"  # preserved
    assert "headroom" in data["mcp"]


def test_register_server_with_env_vars(tmp_path: Path) -> None:
    """register_server handles ServerSpec with environment variables."""
    registrar = _registrar(tmp_path)
    from headroom.mcp_registry.base import ServerSpec

    spec = ServerSpec(
        name="headroom",
        command="headroom",
        args=("mcp", "serve"),
        env={"HEADROOM_PROXY_URL": "http://127.0.0.1:9090"},
    )
    registrar.register_server(spec)

    data = json.loads((tmp_path / "opencode.json").read_text())
    assert data["mcp"]["headroom"]["env"] == {"HEADROOM_PROXY_URL": "http://127.0.0.1:9090"}


def test_register_then_re_register_with_different_env_returns_mismatch(tmp_path: Path) -> None:
    """Re-registering with different env returns MISMATCH without force."""
    registrar = _registrar(tmp_path)
    from headroom.mcp_registry.base import ServerSpec

    spec_a = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    spec_b = ServerSpec(
        name="headroom",
        command="headroom",
        args=("mcp", "serve"),
        env={"NEW_VAR": "value"},
    )
    registrar.register_server(spec_a)
    result = registrar.register_server(spec_b)
    assert result.status == RegisterStatus.MISMATCH


def test_register_server_on_malformed_config_file(tmp_path: Path) -> None:
    """register_server overwrites a malformed config file, preserving nothing."""
    registrar = _registrar(tmp_path)
    (tmp_path / "opencode.json").write_text("not valid json at all")
    from headroom.mcp_registry.base import ServerSpec

    spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    result = registrar.register_server(spec)
    assert result.status == RegisterStatus.REGISTERED

    data = json.loads((tmp_path / "opencode.json").read_text())
    assert "headroom" in data["mcp"]


def test_entry_to_spec_command_as_string() -> None:
    """_entry_to_spec handles a string command (not a list)."""
    entry = {
        "type": "remote",
        "command": "some-command",
        "enabled": True,
    }
    spec = _entry_to_spec("test", entry)
    assert spec.name == "test"
    assert spec.command == "some-command"
    assert spec.args == ()


def test_entry_to_spec_no_command() -> None:
    """_entry_to_spec handles an entry without a 'command' key."""
    entry: dict[str, Any] = {
        "type": "remote",
        "url": "http://example.com",
        "enabled": True,
    }
    spec = _entry_to_spec("test", entry)
    assert spec.name == "test"
    assert spec.command == ""


def test_spec_to_entry_roundtrip() -> None:
    """_spec_to_entry and _entry_to_spec are inverses for local commands."""
    from headroom.mcp_registry.base import ServerSpec

    original = ServerSpec(
        name="test",
        command="python",
        args=("-m", "server"),
        env={"KEY": "VAL"},
    )
    entry = _spec_to_entry(original)
    assert entry["type"] == "remote"
    assert entry["command"] == ["python", "-m", "server"]
    assert entry["env"] == {"KEY": "VAL"}

    restored = _entry_to_spec("test", entry)
    assert restored.name == original.name
    assert restored.command == original.command
    assert restored.args == original.args
    assert restored.env == original.env


def test_diff_specs_all_fields() -> None:
    """_diff_specs reports differences in all fields."""
    from headroom.mcp_registry.base import ServerSpec

    a = ServerSpec(name="s", command="cmd_a", args=("a1",), env={"K": "A"})
    b = ServerSpec(name="s", command="cmd_b", args=("b1",), env={"K": "B"})
    diff = _diff_specs(a, b)
    assert "cmd_a" in diff
    assert "cmd_b" in diff
    assert "a1" in diff
    assert "b1" in diff


def test_diff_specs_no_difference_returns_generic_message() -> None:
    """_diff_specs returns a generic message when no identifiable field differs."""
    from headroom.mcp_registry.base import ServerSpec

    a = ServerSpec(name="s", command="x")
    b = ServerSpec(name="s", command="x")
    diff = _diff_specs(a, b)
    assert "unidentified field" in diff


def test_register_server_returns_already_status(
    tmp_path: Path,
) -> None:
    r = _registrar(tmp_path)
    from headroom.mcp_registry.base import ServerSpec

    spec = ServerSpec(name="already", command="cmd")
    r.register_server(spec)
    result = r.register_server(spec)
    assert result.status == RegisterStatus.ALREADY


def test_unregister_server_handles_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = _registrar(tmp_path)
    from headroom.mcp_registry.base import ServerSpec

    spec = ServerSpec(name="bad-unregister", command="cmd")
    r.register_server(spec)

    def _fail_write(*args: Any, **kwargs: Any) -> None:
        msg = "permission denied"
        raise OSError(msg)

    monkeypatch.setattr("headroom.mcp_registry.opencode._write_json", _fail_write)
    ok = r.unregister_server("bad-unregister")
    assert ok is False


def test_get_all_registrars_includes_opencode() -> None:
    from headroom.mcp_registry.install import get_all_registrars

    registrars = get_all_registrars()
    names = [r.name for r in registrars]
    assert "opencode" in names
