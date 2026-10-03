"""Tests for the Antigravity IDE MCP registrar."""

from __future__ import annotations

import json
from pathlib import Path

from headroom.mcp_registry.antigravity import AntigravityRegistrar
from headroom.mcp_registry.base import RegisterStatus, ServerSpec
from headroom.mcp_registry.install import get_all_registrars, install_everywhere


def _make_registrar(tmp_path: Path) -> AntigravityRegistrar:
    return AntigravityRegistrar(home_dir=tmp_path)


def _spec() -> ServerSpec:
    return ServerSpec(
        name="headroom",
        command="/usr/bin/headroom",
        args=("mcp", "serve"),
        env={"HEADROOM_PROXY_URL": "http://127.0.0.1:8787"},
    )


def _config_path(tmp_path: Path) -> Path:
    return tmp_path / ".gemini" / "config" / "mcp_config.json"


def _legacy_config_path(tmp_path: Path) -> Path:
    return tmp_path / ".gemini" / "antigravity" / "mcp_config.json"


def test_detect_true_when_antigravity_dir_exists(tmp_path: Path) -> None:
    (tmp_path / ".gemini" / "antigravity").mkdir(parents=True)
    assert _make_registrar(tmp_path).detect() is True


def test_detect_true_when_config_dir_exists(tmp_path: Path) -> None:
    # The current Desktop config lives directly under ~/.gemini/config; an
    # install that has not created its MCP file yet still counts.
    (tmp_path / ".gemini" / "config").mkdir(parents=True)
    assert _make_registrar(tmp_path).detect() is True


def test_detect_true_when_only_config_file_exists(tmp_path: Path) -> None:
    config = _config_path(tmp_path)
    config.parent.mkdir(parents=True)
    config.write_text(json.dumps({"mcpServers": {}}))
    assert _make_registrar(tmp_path).detect() is True


def test_detect_false_when_nothing_exists(tmp_path: Path) -> None:
    assert _make_registrar(tmp_path).detect() is False


def test_register_creates_config_with_mcp_servers(tmp_path: Path) -> None:
    result = _make_registrar(tmp_path).register_server(_spec())

    assert result.status == RegisterStatus.REGISTERED
    config = _config_path(tmp_path)
    assert config.exists()
    payload = json.loads(config.read_text())
    entry = payload["mcpServers"]["headroom"]
    assert entry["command"] == "/usr/bin/headroom"
    assert entry["args"] == ["mcp", "serve"]
    assert entry["env"] == {"HEADROOM_PROXY_URL": "http://127.0.0.1:8787"}


def test_register_uses_legacy_location_when_current_missing(tmp_path: Path) -> None:
    legacy = _legacy_config_path(tmp_path)
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"mcpServers": {}}))

    result = _make_registrar(tmp_path).register_server(_spec())

    assert result.status == RegisterStatus.REGISTERED
    assert json.loads(legacy.read_text())["mcpServers"]["headroom"]["command"] == (
        "/usr/bin/headroom"
    )
    # The current location is left alone when the legacy one already exists.
    assert not _config_path(tmp_path).exists()


def test_register_prefers_current_location_when_both_exist(tmp_path: Path) -> None:
    current = _config_path(tmp_path)
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps({"mcpServers": {}}))
    legacy = _legacy_config_path(tmp_path)
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"mcpServers": {}}))

    result = _make_registrar(tmp_path).register_server(_spec())

    assert result.status == RegisterStatus.REGISTERED
    assert "headroom" in json.loads(current.read_text())["mcpServers"]
    # The stale legacy file is not touched: the IDE reads the current one.
    assert "headroom" not in json.loads(legacy.read_text())["mcpServers"]


def test_get_server_reads_current_location_when_both_exist(tmp_path: Path) -> None:
    current = _config_path(tmp_path)
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps({"mcpServers": {"headroom": {"command": "/current/headroom"}}}))
    legacy = _legacy_config_path(tmp_path)
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"mcpServers": {"headroom": {"command": "/stale/headroom"}}}))

    server = _make_registrar(tmp_path).get_server("headroom")

    assert server is not None
    assert server.command == "/current/headroom"


def test_register_is_idempotent(tmp_path: Path) -> None:
    registrar = _make_registrar(tmp_path)
    assert registrar.register_server(_spec()).status == RegisterStatus.REGISTERED
    second = registrar.register_server(_spec())
    assert second.status == RegisterStatus.ALREADY


def test_register_mismatch_without_force_leaves_file_alone(tmp_path: Path) -> None:
    registrar = _make_registrar(tmp_path)
    registrar.register_server(_spec())

    other = ServerSpec(name="headroom", command="/other/headroom", args=("mcp", "serve"))
    result = registrar.register_server(other)

    assert result.status == RegisterStatus.MISMATCH
    assert registrar.get_server("headroom") is not None
    assert registrar.get_server("headroom").command == "/usr/bin/headroom"  # type: ignore[union-attr]


def test_register_force_overwrites_mismatch(tmp_path: Path) -> None:
    registrar = _make_registrar(tmp_path)
    registrar.register_server(_spec())

    other = ServerSpec(name="headroom", command="/other/headroom", args=("mcp", "serve"))
    result = registrar.register_server(other, force=True)

    assert result.status == RegisterStatus.REGISTERED
    assert registrar.get_server("headroom") is not None
    assert registrar.get_server("headroom").command == "/other/headroom"  # type: ignore[union-attr]


def test_register_preserves_other_servers(tmp_path: Path) -> None:
    config = _config_path(tmp_path)
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps({"mcpServers": {"other": {"command": "other-server", "args": []}}})
    )

    result = _make_registrar(tmp_path).register_server(_spec())

    assert result.status == RegisterStatus.REGISTERED
    payload = json.loads(config.read_text())
    assert payload["mcpServers"]["other"] == {"command": "other-server", "args": []}
    assert payload["mcpServers"]["headroom"]["command"] == "/usr/bin/headroom"


def test_register_refuses_to_overwrite_unparseable_config(tmp_path: Path) -> None:
    config = _config_path(tmp_path)
    config.parent.mkdir(parents=True)
    config.write_text("{ not valid json")

    result = _make_registrar(tmp_path).register_server(_spec())

    assert result.status == RegisterStatus.FAILED
    assert "not valid JSON" in (result.detail or "")
    assert config.read_text() == "{ not valid json"


def test_get_server_returns_none_when_absent(tmp_path: Path) -> None:
    assert _make_registrar(tmp_path).get_server("headroom") is None


def test_get_server_reads_existing_entry(tmp_path: Path) -> None:
    registrar = _make_registrar(tmp_path)
    registrar.register_server(_spec())

    server = registrar.get_server("headroom")

    assert server is not None
    assert server.name == "headroom"
    assert server.command == "/usr/bin/headroom"
    assert tuple(server.args) == ("mcp", "serve")
    assert dict(server.env) == {"HEADROOM_PROXY_URL": "http://127.0.0.1:8787"}


def test_unregister_removes_entry_and_cache_dirs(tmp_path: Path) -> None:
    registrar = _make_registrar(tmp_path)
    registrar.register_server(_spec())
    cache_dir = tmp_path / ".gemini" / "antigravity-ide" / "mcp" / "headroom"
    cache_dir.mkdir(parents=True)

    assert registrar.unregister_server("headroom") is True

    assert registrar.get_server("headroom") is None
    assert not cache_dir.exists()


def test_unregister_returns_false_when_absent(tmp_path: Path) -> None:
    assert _make_registrar(tmp_path).unregister_server("headroom") is False


def test_unregister_removes_entry_from_all_locations(tmp_path: Path) -> None:
    registrar = _make_registrar(tmp_path)
    for path in (_config_path(tmp_path), _legacy_config_path(tmp_path)):
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "headroom": {"command": "/usr/bin/headroom"},
                        "other": {"command": "other-server"},
                    }
                }
            )
        )

    assert registrar.unregister_server("headroom") is True

    for path in (_config_path(tmp_path), _legacy_config_path(tmp_path)):
        payload = json.loads(path.read_text())
        assert "headroom" not in payload["mcpServers"]
        # Unrelated entries are preserved in both files.
        assert payload["mcpServers"]["other"] == {"command": "other-server"}


def test_registrar_name_and_display_name() -> None:
    registrar = AntigravityRegistrar(home_dir=Path("/nonexistent"))
    assert registrar.name == "antigravity"
    assert registrar.display_name == "Antigravity IDE"


def test_get_all_registrars_includes_antigravity() -> None:
    names = [registrar.name for registrar in get_all_registrars()]
    assert "antigravity" in names


def test_install_everywhere_installs_into_detected_antigravity(tmp_path: Path) -> None:
    (tmp_path / ".gemini" / "antigravity").mkdir(parents=True)
    registrar = AntigravityRegistrar(home_dir=tmp_path)

    results = install_everywhere(registrars=[registrar], agents=["antigravity"])

    assert set(results) == {"antigravity"}
    assert results["antigravity"].status == RegisterStatus.REGISTERED
    assert registrar.get_server("headroom") is not None


def test_install_everywhere_registers_with_only_config_dir(tmp_path: Path) -> None:
    # First-time setup: only the current Desktop config directory exists, no
    # MCP file yet. An explicit install must attempt registration instead of
    # bailing out as NOT_DETECTED.
    (tmp_path / ".gemini" / "config").mkdir(parents=True)
    registrar = AntigravityRegistrar(home_dir=tmp_path)

    results = install_everywhere(registrars=[registrar], agents=["antigravity"])

    assert set(results) == {"antigravity"}
    assert results["antigravity"].status == RegisterStatus.REGISTERED
    assert registrar.get_server("headroom") is not None
    assert (tmp_path / ".gemini" / "config" / "mcp_config.json").exists()


def test_register_prefers_current_location_when_config_dir_exists(
    tmp_path: Path,
) -> None:
    """Registration must land in the active Desktop config, not a stale legacy file.

    With ``~/.gemini/config/`` present (the current Desktop install marker)
    and a legacy ``~/.gemini/antigravity/mcp_config.json`` also on disk, the
    registration goes to ``~/.gemini/config/mcp_config.json`` — the location
    the running IDE actually reads.
    """
    legacy = _legacy_config_path(tmp_path)
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"mcpServers": {"other": {"command": "other-cmd"}}}))
    (tmp_path / ".gemini" / "config").mkdir(parents=True)

    result = _make_registrar(tmp_path).register_server(_spec())

    assert result.status == RegisterStatus.REGISTERED
    current = _config_path(tmp_path)
    servers = json.loads(current.read_text())["mcpServers"]
    assert servers["headroom"]["command"] == "/usr/bin/headroom"
    # The legacy file is left alone.
    assert "headroom" not in json.loads(legacy.read_text())["mcpServers"]


def test_register_uses_legacy_location_when_no_config_dir(tmp_path: Path) -> None:
    """A legacy-only install still registers into its existing config file."""
    legacy = _legacy_config_path(tmp_path)
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"mcpServers": {}}))

    result = _make_registrar(tmp_path).register_server(_spec())

    assert result.status == RegisterStatus.REGISTERED
    assert "headroom" in json.loads(legacy.read_text())["mcpServers"]
    assert not _config_path(tmp_path).exists()
