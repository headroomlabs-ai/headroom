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
                "type": "local",
                "command": ["headroom", "mcp", "serve"],
                "enabled": True,
            }
        }
    }
    _write_json(tmp_path / "opencode.json", config)
    registrar = _registrar(tmp_path)
    spec = registrar.get_server("headroom")
    assert spec is not None
    assert spec.name == "headroom"
    assert spec.command == "headroom"
    assert spec.args == ("mcp", "serve")


def test_register_server_creates_config_when_missing(tmp_path: Path) -> None:
    """register_server creates the config file when it doesn't exist."""
    registrar = _registrar(tmp_path)
    from headroom.mcp_registry.base import ServerSpec

    spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    result = registrar.register_server(spec)
    assert result.status == RegisterStatus.REGISTERED
    config_path = tmp_path / "opencode.json"
    assert config_path.exists()
    data = json.loads(config_path.read_text())
    assert data["mcp"]["headroom"] == {
        "type": "local",
        "command": ["headroom", "mcp", "serve"],
        "enabled": True,
    }


@pytest.mark.parametrize("contents", ["not json", "{", '{"theme": }', "[]"])
def test_register_server_preserves_malformed_config(tmp_path: Path, contents: str) -> None:
    """Registering must NOT clobber an existing but unparseable opencode.json.

    The file holds theme/model/provider and other MCP servers; before the fix a
    malformed file was read as {} and rewritten with only {"mcp": ...}."""
    from headroom.mcp_registry.base import ServerSpec

    config_path = tmp_path / "opencode.json"
    config_path.write_text(contents, encoding="utf-8")
    registrar = _registrar(tmp_path)

    spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    result = registrar.register_server(spec)

    assert result.status == RegisterStatus.FAILED
    assert "not valid JSON" in result.detail
    assert config_path.read_text(encoding="utf-8") == contents


def test_register_server_preserves_other_keys(tmp_path: Path) -> None:
    """The happy path merges: theme/model and an existing MCP server survive."""
    from headroom.mcp_registry.base import ServerSpec

    config_path = tmp_path / "opencode.json"
    _write_json(
        config_path,
        {"theme": "dark", "model": "anthropic/claude", "mcp": {"other": {"type": "local"}}},
    )
    registrar = _registrar(tmp_path)

    spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    result = registrar.register_server(spec)

    assert result.status == RegisterStatus.REGISTERED
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["theme"] == "dark"
    assert data["model"] == "anthropic/claude"
    assert data["mcp"]["other"] == {"type": "local"}
    assert "headroom" in data["mcp"]


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
    assert data["mcp"]["headroom"]["environment"] == {"HEADROOM_PROXY_URL": "http://127.0.0.1:9090"}


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
    """register_server refuses to overwrite a malformed config, preserving it.

    Updated for the clobber guard: opencode.json holds theme/model/provider and
    other MCP servers, so a present-but-unparseable file is left untouched and
    registration fails rather than silently wiping the user's config.
    """
    registrar = _registrar(tmp_path)
    (tmp_path / "opencode.json").write_text("not valid json at all")
    from headroom.mcp_registry.base import ServerSpec

    spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    result = registrar.register_server(spec)
    assert result.status == RegisterStatus.FAILED

    # The malformed file is left byte-for-byte untouched rather than clobbered.
    assert (tmp_path / "opencode.json").read_text() == "not valid json at all"


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


def test_entry_to_spec_reads_opencode_environment() -> None:
    """_entry_to_spec reads OpenCode's environment map."""
    entry = {
        "type": "local",
        "command": ["headroom", "mcp", "serve"],
        "enabled": True,
        "environment": {"HEADROOM_PROXY_URL": "http://127.0.0.1:9090"},
    }
    spec = _entry_to_spec("headroom", entry)
    assert spec.env == {"HEADROOM_PROXY_URL": "http://127.0.0.1:9090"}


def test_entry_to_spec_reads_legacy_env() -> None:
    """_entry_to_spec keeps compatibility with previously written env maps."""
    entry = {
        "type": "local",
        "command": ["headroom", "mcp", "serve"],
        "enabled": True,
        "env": {"HEADROOM_PROXY_URL": "http://127.0.0.1:9090"},
    }
    spec = _entry_to_spec("headroom", entry)
    assert spec.env == {"HEADROOM_PROXY_URL": "http://127.0.0.1:9090"}


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
    assert entry["type"] == "local"
    assert entry["command"] == ["python", "-m", "server"]
    assert entry["environment"] == {"KEY": "VAL"}

    restored = _entry_to_spec("test", entry)
    assert restored.name == original.name
    assert restored.command == original.command
    assert restored.args == original.args
    assert restored.env == original.env


def test_spec_to_entry_no_env_has_no_environment_key() -> None:
    """_spec_to_entry omits 'environment' key when spec has no env vars."""
    from headroom.mcp_registry.base import ServerSpec

    spec = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    entry = _spec_to_entry(spec)
    assert entry["type"] == "local"
    assert "url" not in entry
    assert "environment" not in entry
    assert "env" not in entry


def test_entry_to_spec_reads_environment_field() -> None:
    """_entry_to_spec reads the 'environment' field (not legacy 'env')."""
    entry = {
        "type": "local",
        "command": ["headroom", "mcp", "serve"],
        "environment": {"HEADROOM_PROXY_URL": "http://127.0.0.1:8787"},
        "enabled": True,
    }
    spec = _entry_to_spec("headroom", entry)
    assert spec.env == {"HEADROOM_PROXY_URL": "http://127.0.0.1:8787"}


def test_entry_to_spec_falls_back_to_legacy_env_field() -> None:
    """_entry_to_spec falls back to 'env' when 'environment' is absent."""
    entry = {
        "type": "local",
        "command": ["headroom", "mcp", "serve"],
        "env": {"LEGACY_KEY": "value"},
        "enabled": True,
    }
    spec = _entry_to_spec("headroom", entry)
    assert spec.env == {"LEGACY_KEY": "value"}


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
