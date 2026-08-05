"""Tests for the Grok Build MCP registrar."""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.mcp_registry.base import RegisterStatus, ServerSpec
from headroom.mcp_registry.grok import GrokRegistrar


def _make_registrar(tmp_path: Path) -> GrokRegistrar:
    return GrokRegistrar(home_dir=tmp_path)


def _spec() -> ServerSpec:
    return ServerSpec(
        name="headroom",
        command="/usr/bin/python",
        args=("-m", "headroom.cli", "mcp", "serve"),
    )


def test_detect_true_when_grok_dir_exists(tmp_path: Path) -> None:
    (tmp_path / ".grok").mkdir()
    assert _make_registrar(tmp_path).detect() is True


def test_detect_false_when_grok_dir_missing(tmp_path: Path) -> None:
    assert _make_registrar(tmp_path).detect() is False


def test_register_uses_grok_home_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    grok_home = tmp_path / "custom-grok-home"
    monkeypatch.setenv("GROK_HOME", str(grok_home))

    result = GrokRegistrar().register_server(_spec())

    assert result.status == RegisterStatus.REGISTERED
    config = grok_home / "config.toml"
    assert config.exists()
    assert "[mcp_servers.headroom]" in config.read_text()
