"""Integration tests for MCP CLI commands.

These are real tests that:
- Actually write/read config files
- Test actual CLI behavior
- Test MCP server initialization (when MCP SDK is available)
"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.cli.mcp import (
    get_headroom_command,
    load_mcp_config,
    save_mcp_config,
)
from headroom.mcp_registry.base import ServerSpec

# Check if MCP SDK is available
try:
    import mcp  # noqa: F401

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False


@pytest.fixture
def temp_claude_dir(tmp_path):
    """Create a temporary .claude directory for testing."""
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    return claude_dir


@pytest.fixture
def mock_claude_config_path(temp_claude_dir):
    """Patch the MCP config path to use temp directory."""
    config_path = temp_claude_dir / "mcp.json"
    with patch("headroom.cli.mcp.MCP_CONFIG_PATH", config_path):
        with patch("headroom.cli.mcp.CLAUDE_CONFIG_DIR", temp_claude_dir):
            yield config_path


@pytest.fixture
def mock_mcp_available():
    """Mock MCP SDK as available for testing install/uninstall commands."""
    mock_mcp = MagicMock()
    with patch.dict(sys.modules, {"mcp": mock_mcp}):
        yield mock_mcp


class FakeRegistrar:
    name = "fake"
    display_name = "Fake Agent"

    def __init__(self, *, configured: bool = True) -> None:
        self.configured = configured
        self.removed: list[str] = []

    def detect(self) -> bool:
        return True

    def get_server(self, server_name: str) -> ServerSpec | None:
        if self.configured and server_name == "headroom":
            return ServerSpec(name="headroom", command="headroom", args=("mcp", "serve"))
        return None

    def unregister_server(self, server_name: str) -> bool:
        if self.configured and server_name == "headroom":
            self.configured = False
            self.removed.append(server_name)
            return True
        return False


class TestMCPConfigFunctions:
    """Test config file handling functions."""

    def test_get_headroom_command_returns_list(self):
        """Command should be a list suitable for subprocess."""
        cmd = get_headroom_command()
        assert isinstance(cmd, list)
        assert len(cmd) >= 1
        # Should end with mcp serve args
        assert "mcp" in cmd or "-m" in cmd

    def test_load_mcp_config_empty_when_no_file(self, mock_claude_config_path):
        """Loading non-existent config returns empty structure."""
        config = load_mcp_config()
        assert config == {"mcpServers": {}}

    def test_save_and_load_config(self, mock_claude_config_path):
        """Config can be saved and loaded back."""
        test_config = {
            "mcpServers": {
                "headroom": {
                    "command": "headroom",
                    "args": ["mcp", "serve"],
                }
            }
        }
        save_mcp_config(test_config)

        # File should exist
        assert mock_claude_config_path.exists()

        # Load it back
        loaded = load_mcp_config()
        assert loaded == test_config

    def test_save_config_creates_directory(self, tmp_path):
        """save_mcp_config creates parent directory if needed."""
        claude_dir = tmp_path / "new_dir" / ".claude"
        config_path = claude_dir / "mcp.json"

        with patch("headroom.cli.mcp.MCP_CONFIG_PATH", config_path):
            with patch("headroom.cli.mcp.CLAUDE_CONFIG_DIR", claude_dir):
                save_mcp_config({"mcpServers": {}})

        assert config_path.exists()

    def test_load_config_preserves_other_servers(self, mock_claude_config_path):
        """Loading preserves other MCP servers in config."""
        # Write config with another server
        existing_config = {
            "mcpServers": {
                "other-server": {"command": "other", "args": []},
            }
        }
        mock_claude_config_path.write_text(json.dumps(existing_config))

        loaded = load_mcp_config()
        assert "other-server" in loaded["mcpServers"]


#
# Note: Tests for the 'mcp install' command's writes/idempotency/CLI-vs-file
# fallback used to live here, but they were tightly coupled to private
# globals (MCP_CONFIG_PATH, shutil.which) and exercised the same surface
# already covered by:
#   - tests/test_mcp_registry/test_claude_registrar.py (file/CLI behavior
#     with proper constructor injection — no patches)
#   - tests/test_mcp_registry/test_install.py (orchestrator semantics with
#     fake registrars)
# Removing the duplicates leaves the CLI as glue: argument parsing +
# output formatting, which is straightforward and not worth its own test
# layer.


class TestMCPUninstallCommand:
    """Test 'headroom mcp uninstall' command."""

    def test_uninstall_removes_headroom(self, mock_mcp_available):
        """Uninstall removes headroom through detected registrars."""
        registrar = FakeRegistrar()

        runner = CliRunner()
        with patch("headroom.mcp_registry.get_all_registrars", return_value=[registrar]):
            result = runner.invoke(main, ["mcp", "uninstall"])

        assert result.exit_code == 0
        assert "removed" in result.output.lower()
        assert registrar.removed == ["headroom"]

    def test_uninstall_checks_only_headroom_servers(self):
        """Uninstall only asks registrars to remove Headroom-owned servers."""
        registrar = FakeRegistrar()

        runner = CliRunner()
        with patch("headroom.mcp_registry.get_all_registrars", return_value=[registrar]):
            result = runner.invoke(main, ["mcp", "uninstall"])

        assert result.exit_code == 0
        assert registrar.removed == ["headroom"]

    def test_uninstall_no_config_file(self, mock_claude_config_path):
        """Uninstall with no config file exits cleanly."""
        runner = CliRunner()
        with patch("headroom.mcp_registry.get_all_registrars", return_value=[]):
            result = runner.invoke(main, ["mcp", "uninstall"])

        assert result.exit_code == 0
        assert "nothing to uninstall" in result.output.lower()

    def test_uninstall_not_configured(self, mock_claude_config_path):
        """Uninstall when headroom not in config exits cleanly."""
        registrar = FakeRegistrar(configured=False)

        runner = CliRunner()
        with patch("headroom.mcp_registry.get_all_registrars", return_value=[registrar]):
            result = runner.invoke(main, ["mcp", "uninstall"])

        assert result.exit_code == 0
        assert "not configured" in result.output.lower()


class TestMCPStatusCommand:
    """Test 'headroom mcp status' command."""

    def test_status_not_configured(self, mock_claude_config_path):
        """Status shows not configured when no config."""
        runner = CliRunner()
        registrar = FakeRegistrar(configured=False)
        with patch("headroom.mcp_registry.get_all_registrars", return_value=[registrar]):
            result = runner.invoke(main, ["mcp", "status"])

        assert result.exit_code == 0
        assert "MCP SDK" in result.output
        # Should show not configured
        assert (
            "✗" in result.output
            or "Not configured" in result.output.lower()
            or "No config" in result.output
        )

    def test_status_configured(self, mock_claude_config_path, mock_mcp_available):
        """Status reports configured when a registrar has headroom."""
        registrar = FakeRegistrar(configured=True)

        runner = CliRunner()
        with patch("headroom.mcp_registry.get_all_registrars", return_value=[registrar]):
            result = runner.invoke(main, ["mcp", "status"])

        assert result.exit_code == 0
        assert "✓ Configured" in result.output


class TestMCPServeCommand:
    """Test 'headroom mcp serve' command."""

    def test_serve_help(self):
        """Serve command shows help."""
        runner = CliRunner()
        result = runner.invoke(main, ["mcp", "serve", "--help"])

        assert result.exit_code == 0
        assert "transport" in result.output
        assert "host" in result.output
        assert "port" in result.output
        assert "path" in result.output
        assert "proxy-url" in result.output
        assert "debug" in result.output

    def test_serve_defaults_to_stdio(self):
        """Serve command defaults to stdio transport."""
        fake_server = MagicMock()
        fake_server.run_stdio = AsyncMock()
        fake_server.run_streamable_http = AsyncMock()
        fake_server.cleanup = AsyncMock()

        runner = CliRunner()
        with patch("headroom.ccr.mcp_server.create_ccr_mcp_server", return_value=fake_server):
            result = runner.invoke(main, ["mcp", "serve"])

        assert result.exit_code == 0
        fake_server.run_stdio.assert_awaited_once()
        fake_server.run_streamable_http.assert_not_awaited()
        fake_server.cleanup.assert_awaited_once()

    def test_serve_http_transport_uses_http_settings(self):
        """Serve command uses HTTP transport when requested with mixed-case input."""
        fake_server = MagicMock()
        fake_server.run_stdio = AsyncMock()
        fake_server.run_streamable_http = AsyncMock()
        fake_server.cleanup = AsyncMock()

        runner = CliRunner()
        with patch("headroom.ccr.mcp_server.create_ccr_mcp_server", return_value=fake_server):
            result = runner.invoke(
                main,
                [
                    "mcp",
                    "serve",
                    "--transport",
                    "HTTP",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "9191",
                    "--path",
                    "/mcp",
                ],
            )

        assert result.exit_code == 0
        fake_server.run_stdio.assert_not_awaited()
        fake_server.run_streamable_http.assert_awaited_once_with(
            host="0.0.0.0",
            port=9191,
            path="/mcp",
            debug=False,
        )
        fake_server.cleanup.assert_awaited_once()


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP SDK not installed")
class TestMCPServerInitialization:
    """Test actual MCP server creation.

    These tests require the MCP SDK to be installed.
    """

    def test_mcp_server_can_be_created(self):
        """MCP server can be instantiated."""
        from headroom.ccr.mcp_server import create_ccr_mcp_server

        server = create_ccr_mcp_server()
        assert server is not None
        assert server.proxy_url == "http://127.0.0.1:8787"

    def test_mcp_server_with_custom_url(self):
        """MCP server accepts custom proxy URL."""
        from headroom.ccr.mcp_server import create_ccr_mcp_server

        server = create_ccr_mcp_server(proxy_url="http://custom:9000")
        assert server.proxy_url == "http://custom:9000"

    def test_mcp_server_has_correct_tool_name(self):
        """MCP server is configured for headroom_retrieve tool."""
        from headroom.ccr.mcp_server import create_ccr_mcp_server
        from headroom.ccr.tool_injection import CCR_TOOL_NAME

        server = create_ccr_mcp_server()

        # Verify the server was created with correct configuration
        assert server.server is not None
        assert server.server.name == "headroom"
        # The tool name should be headroom_retrieve
        assert CCR_TOOL_NAME == "headroom_retrieve"


#
# Tests for "install via claude CLI" used to live here, exercising
# subprocess.run patches against the old direct CLI invocation. Equivalent
# coverage now lives in tests/test_mcp_registry/test_claude_registrar.py
# using constructor injection (`claude_cli="/path/to/fake"`) and bounded
# subprocess.run mocks at the registrar boundary — no module-level patches.


class TestMCPUninstallWithRegistrars:
    """Test mcp_uninstall delegates to registrars."""

    def test_uninstall_reports_removed_registrar_server(self):
        registrar = FakeRegistrar(configured=True)

        runner = CliRunner()
        with patch("headroom.mcp_registry.get_all_registrars", return_value=[registrar]):
            result = runner.invoke(main, ["mcp", "uninstall"])

        assert result.exit_code == 0
        assert "Fake Agent" in result.output
        assert registrar.removed == ["headroom"]

    def test_uninstall_skips_unconfigured_registrar(self):
        registrar = FakeRegistrar(configured=False)

        runner = CliRunner()
        with patch("headroom.mcp_registry.get_all_registrars", return_value=[registrar]):
            result = runner.invoke(main, ["mcp", "uninstall"])

        assert result.exit_code == 0
        assert "nothing to uninstall" in result.output.lower()
        assert registrar.removed == []
