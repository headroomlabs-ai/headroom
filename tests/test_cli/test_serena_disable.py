"""`--no-serena` must actively disable Serena, not merely skip adding it.

Serena is installed by default, so a prior `headroom wrap` persists a
`serena` MCP entry and the agent keeps launching it (dashboard popup and
all). These tests pin that a later `--no-serena` removes the entry Headroom
installed, leaves a user-managed entry alone, and that Codex unwrap also
removes Serena.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_cli
from headroom.cli.main import main
from headroom.mcp_registry import build_serena_spec
from headroom.mcp_registry.base import ServerSpec
from headroom.mcp_registry.ledger import record_install


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


class _FakeRegistrar:
    """Minimal registrar capturing unregister calls."""

    def __init__(self, name: str, *, detected: bool = True, server: ServerSpec | None = None):
        self.name = name
        self.display_name = name.capitalize()
        self._detected = detected
        self._server = server
        self.unregistered: list[str] = []

    def detect(self) -> bool:
        return self._detected

    def get_server(self, server_name: str) -> ServerSpec | None:
        return self._server if server_name == "serena" else None

    def unregister_server(self, server_name: str) -> bool:
        self.unregistered.append(server_name)
        self._server = None
        return True


def test_disable_removes_headroom_installed_serena(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    spec = build_serena_spec("claude-code")
    record_install("claude", spec)  # ledger now proves Headroom owns it
    registrar = _FakeRegistrar("claude", server=spec)

    wrap_cli._disable_serena_mcp(registrar, verbose=True)

    assert registrar.unregistered == ["serena"]
    assert "Removed previously-installed Serena MCP" in capsys.readouterr().out


def test_disable_preserves_user_managed_serena(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    # Present in the agent config but NOT in Headroom's ledger → user-owned.
    user_spec = ServerSpec(name="serena", command="/usr/local/bin/custom-serena")
    registrar = _FakeRegistrar("claude", server=user_spec)

    wrap_cli._disable_serena_mcp(registrar, verbose=True)

    assert registrar.unregistered == []  # never touch a user-managed entry
    assert "user-managed" in capsys.readouterr().out


def test_disable_noop_when_serena_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    registrar = _FakeRegistrar("claude", server=None)

    wrap_cli._disable_serena_mcp(registrar, verbose=True)

    assert registrar.unregistered == []
    assert "Skipping Serena MCP (--no-serena)" in capsys.readouterr().out


def test_disable_noop_when_agent_not_detected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    spec = build_serena_spec("claude-code")
    record_install("claude", spec)
    registrar = _FakeRegistrar("claude", detected=False, server=spec)

    wrap_cli._disable_serena_mcp(registrar, verbose=True)

    assert registrar.unregistered == []  # not detected → leave everything alone


def test_unwrap_codex_removes_headroom_installed_serena(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    spec = build_serena_spec("codex")
    record_install("codex", spec)
    registrar = _FakeRegistrar("codex", server=spec)

    with (
        patch("headroom.mcp_registry.CodexRegistrar", return_value=registrar),
        patch(
            "headroom.cli.wrap._restore_codex_provider_config",
            return_value=("noop", tmp_path / "config.toml"),
        ),
        patch("headroom.cli.wrap._stop_local_proxy_for_unwrap"),
    ):
        result = runner.invoke(main, ["unwrap", "codex"])

    assert result.exit_code == 0, result.output
    assert registrar.unregistered == ["serena"]
    assert "Removed Headroom-installed Serena MCP server from Codex" in result.output
