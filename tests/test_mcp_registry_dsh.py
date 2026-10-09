"""Tests for the DeepSeek Harness (dsh) MCP registrar."""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.mcp_registry.base import RegisterStatus, ServerSpec
from headroom.mcp_registry.dsh import (
    DshRegistrar,
    _block_to_entries,
    _render_block,
    _spec_to_entry,
)
from headroom.mcp_registry.install import get_all_registrars


def _spec(**overrides) -> ServerSpec:
    base = ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_detects_dsh_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DSH_HOME", str(tmp_path))
    assert DshRegistrar().detect() is True


def test_not_detected_without_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("DSH_HOME", str(tmp_path / "missing"))
    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))
    assert DshRegistrar().detect() is False


def test_ensure_home_creates_missing_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "dsh"
    monkeypatch.setenv("DSH_HOME", str(home))
    reg = DshRegistrar()
    assert reg.detect() is False
    reg.ensure_home()
    assert reg.detect() is True
    assert home.is_dir()


def test_register_after_ensure_home_without_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fresh install: no DSH home yet — the explicit wrap path must be able to
    register before dsh creates its home on first boot."""
    home = tmp_path / "dsh"
    monkeypatch.setenv("DSH_HOME", str(home))
    reg = DshRegistrar()
    reg.ensure_home()
    result = reg.register_server(_spec())
    assert result.status is RegisterStatus.REGISTERED
    text = (home / "cordis.patch.yml").read_text(encoding="utf-8")
    assert "@deepseek-ai/dsh-mcp-client" in text


def test_register_writes_patch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "dsh"
    home.mkdir()
    monkeypatch.setenv("DSH_HOME", str(home))
    result = DshRegistrar().register_server(_spec())
    assert result.status is RegisterStatus.REGISTERED
    text = (home / "cordis.patch.yml").read_text(encoding="utf-8")
    assert "@deepseek-ai/dsh-mcp-client" in text
    assert "serverName: headroom" in text
    assert "transport: stdio" in text


def test_register_is_idempotent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "dsh"
    home.mkdir()
    monkeypatch.setenv("DSH_HOME", str(home))
    reg = DshRegistrar()
    assert reg.register_server(_spec()).status is RegisterStatus.REGISTERED
    assert reg.register_server(_spec()).status is RegisterStatus.ALREADY


def test_mismatch_leaves_existing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "dsh"
    home.mkdir()
    monkeypatch.setenv("DSH_HOME", str(home))
    reg = DshRegistrar()
    reg.register_server(_spec())
    before = (home / "cordis.patch.yml").read_text(encoding="utf-8")
    result = reg.register_server(_spec(command="other-headroom"))
    assert result.status is RegisterStatus.MISMATCH
    assert (home / "cordis.patch.yml").read_text(encoding="utf-8") == before


def test_unregister_removes_only_headroom(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "dsh"
    home.mkdir()
    monkeypatch.setenv("DSH_HOME", str(home))
    patch = home / "cordis.patch.yml"
    patch.write_text(
        "- insert:\n    - id: user-plugin\n      name: '@deepseek-ai/other'\n", encoding="utf-8"
    )
    reg = DshRegistrar()
    reg.register_server(_spec())
    assert reg.unregister_server("headroom") is True
    text = patch.read_text(encoding="utf-8")
    assert "headroom" not in text
    assert "user-plugin" in text


def test_registry_includes_dsh() -> None:
    assert any(reg.name == "dsh" for reg in get_all_registrars())


def test_register_fails_on_truncated_block(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "dsh"
    home.mkdir()
    monkeypatch.setenv("DSH_HOME", str(home))
    patch = home / "cordis.patch.yml"
    truncated = "# --- Headroom MCP server ---\n"
    patch.write_text(truncated, encoding="utf-8")
    result = DshRegistrar().register_server(_spec())
    assert result.status is RegisterStatus.FAILED
    assert "unterminated" in (result.detail or "")
    assert patch.read_text(encoding="utf-8") == truncated


def test_register_fails_on_malformed_yaml_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "dsh"
    home.mkdir()
    monkeypatch.setenv("DSH_HOME", str(home))
    patch = home / "cordis.patch.yml"
    malformed = (
        "# --- Headroom MCP server ---\n- insert: [unclosed\n# --- end Headroom MCP server ---\n"
    )
    patch.write_text(malformed, encoding="utf-8")
    result = DshRegistrar().register_server(_spec())
    assert result.status is RegisterStatus.FAILED
    assert patch.read_text(encoding="utf-8") == malformed


def test_force_overwrites_truncated_block(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = tmp_path / "dsh"
    home.mkdir()
    monkeypatch.setenv("DSH_HOME", str(home))
    patch = home / "cordis.patch.yml"
    patch.write_text("# --- Headroom MCP server ---\n", encoding="utf-8")
    result = DshRegistrar().register_server(_spec(), force=True)
    assert result.status is RegisterStatus.REGISTERED
    text = patch.read_text(encoding="utf-8")
    assert text.count("# --- Headroom MCP server ---") == 1
    assert "@deepseek-ai/dsh-mcp-client" in text


def test_unregister_preserves_unrelated_bytes_exactly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "dsh"
    home.mkdir()
    monkeypatch.setenv("DSH_HOME", str(home))
    patch = home / "cordis.patch.yml"
    unrelated = "- insert:\n    - id: user-plugin\n      name: '@deepseek-ai/other'\n"
    patch.write_text(unrelated, encoding="utf-8")
    reg = DshRegistrar()
    assert reg.register_server(_spec()).status is RegisterStatus.REGISTERED
    assert reg.unregister_server("headroom") is True
    assert patch.read_text(encoding="utf-8") == unrelated


def test_unregister_truncated_block_preserves_unrelated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "dsh"
    home.mkdir()
    monkeypatch.setenv("DSH_HOME", str(home))
    patch = home / "cordis.patch.yml"
    unrelated = "- insert:\n    - id: user-plugin\n      name: '@deepseek-ai/other'\n"
    patch.write_text(unrelated + "# --- Headroom MCP server ---\n- insert:\n", encoding="utf-8")
    assert DshRegistrar().unregister_server("headroom") is True
    assert patch.read_text(encoding="utf-8") == unrelated


def test_block_round_trips_multiple_servers() -> None:
    headroom = _spec()
    serena = _spec(name="serena", command="uvx", args=("--from", "serena-agent"))
    block = _render_block([_spec_to_entry(headroom), _spec_to_entry(serena)])
    entries = _block_to_entries(block)
    assert entries is not None
    assert set(entries) == {"headroom", "serena"}
    assert entries["headroom"].command == "headroom"
    assert entries["serena"].command == "uvx"


def test_block_with_malformed_config_row_is_corrupt() -> None:
    block = (
        "# --- Headroom MCP server ---\n"
        "- insert:\n"
        "  - id: mcp-headroom\n"
        "    name: '@deepseek-ai/dsh-mcp-client'\n"
        "    config: null\n"
        "# --- end Headroom MCP server ---\n"
    )
    assert _block_to_entries(block) is None


def test_register_two_servers_and_unregister_one_preserves_the_other(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "dsh"
    home.mkdir()
    monkeypatch.setenv("DSH_HOME", str(home))
    reg = DshRegistrar()
    assert reg.register_server(_spec()).status is RegisterStatus.REGISTERED
    serena = _spec(name="serena", command="uvx", args=("--from", "serena-agent"))
    assert reg.register_server(serena).status is RegisterStatus.REGISTERED

    assert reg.get_server("headroom") is not None
    assert reg.get_server("serena") is not None

    assert reg.unregister_server("serena") is True
    assert reg.get_server("serena") is None
    assert reg.get_server("headroom") is not None
    text = (home / "cordis.patch.yml").read_text(encoding="utf-8")
    assert "serena" not in text
    assert "mcp-headroom" in text


def test_register_two_servers_is_idempotent_per_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "dsh"
    home.mkdir()
    monkeypatch.setenv("DSH_HOME", str(home))
    reg = DshRegistrar()
    serena = _spec(name="serena", command="uvx", args=("--from", "serena-agent"))
    assert reg.register_server(_spec()).status is RegisterStatus.REGISTERED
    assert reg.register_server(serena).status is RegisterStatus.REGISTERED
    assert reg.register_server(_spec()).status is RegisterStatus.ALREADY
    assert reg.register_server(serena).status is RegisterStatus.ALREADY


def test_register_starts_block_on_new_line_when_file_lacks_trailing_newline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "dsh"
    home.mkdir()
    monkeypatch.setenv("DSH_HOME", str(home))
    patch = home / "cordis.patch.yml"
    patch.write_text("- id: webserver\n  config:\n    port: 3080", encoding="utf-8")
    assert DshRegistrar().register_server(_spec()).status is RegisterStatus.REGISTERED
    text = patch.read_text(encoding="utf-8")
    assert "3080\n# --- Headroom MCP server ---" in text
    assert "3080# --- Headroom MCP server ---" not in text
