"""Integration tests for MCP CLI commands.

These are real tests that:
- Actually write/read config files
- Test actual CLI behavior
- Test MCP server initialization (when MCP SDK is available)
"""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.cli.mcp import (
    get_headroom_command,
    load_mcp_config,
    save_mcp_config,
)

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
    """Patch the MCP config path to use temp directory.

    Also mocks the claude CLI as absent so tests exercise the mcp.json
    fallback path rather than the `claude mcp add` path.
    """
    import shutil as _shutil

    config_path = temp_claude_dir / "mcp.json"
    # Capture the original function reference before patching so we don't recurse.
    _real_which = _shutil.which

    def which_no_claude(cmd):
        if cmd == "claude":
            return None
        return _real_which(cmd)

    with patch("headroom.cli.mcp.MCP_CONFIG_PATH", config_path):
        with patch("headroom.cli.mcp.CLAUDE_CONFIG_DIR", temp_claude_dir):
            with patch("headroom.cli.mcp.shutil.which", side_effect=which_no_claude):
                yield config_path


@pytest.fixture
def mock_mcp_available():
    """Mock MCP SDK as available for testing install/uninstall commands."""
    mock_mcp = MagicMock()
    with patch.dict(sys.modules, {"mcp": mock_mcp}):
        yield mock_mcp


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

    def test_uninstall_removes_headroom(self, mock_claude_config_path, mock_mcp_available):
        """Uninstall removes headroom from the legacy config file."""
        # Pre-populate the config directly rather than depending on
        # `mcp install` plumbing — keeps the test focused on uninstall.
        mock_claude_config_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "headroom": {"command": "headroom", "args": ["mcp", "serve"]},
                    }
                }
            )
        )

        runner = CliRunner()
        result = runner.invoke(main, ["mcp", "uninstall"])

        assert result.exit_code == 0
        assert "removed" in result.output.lower()

        config = json.loads(mock_claude_config_path.read_text())
        assert "headroom" not in config["mcpServers"]

    def test_uninstall_preserves_other_servers(self, mock_claude_config_path):
        """Uninstall preserves other MCP servers."""
        # Create config with headroom and another server
        config = {
            "mcpServers": {
                "headroom": {"command": "headroom", "args": ["mcp", "serve"]},
                "github": {"command": "github-mcp", "args": []},
            }
        }
        mock_claude_config_path.write_text(json.dumps(config))

        runner = CliRunner()
        result = runner.invoke(main, ["mcp", "uninstall"])

        assert result.exit_code == 0

        config = json.loads(mock_claude_config_path.read_text())
        assert "headroom" not in config["mcpServers"]
        assert "github" in config["mcpServers"]

    def test_uninstall_no_config_file(self, mock_claude_config_path):
        """Uninstall with no config file exits cleanly."""
        runner = CliRunner()
        result = runner.invoke(main, ["mcp", "uninstall"])

        assert result.exit_code == 0
        assert "nothing to uninstall" in result.output.lower()

    def test_uninstall_not_configured(self, mock_claude_config_path):
        """Uninstall when headroom not in config exits cleanly."""
        # Create config without headroom
        config = {"mcpServers": {"other": {"command": "other"}}}
        mock_claude_config_path.write_text(json.dumps(config))

        runner = CliRunner()
        result = runner.invoke(main, ["mcp", "uninstall"])

        assert result.exit_code == 0
        assert "not configured" in result.output.lower()


class TestMCPStatusCommand:
    """Test 'headroom mcp status' command."""

    def test_status_not_configured(self, mock_claude_config_path):
        """Status shows not configured when no config."""
        runner = CliRunner()
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
        """Status reports configured when the legacy config has headroom."""
        # Pre-populate the legacy mcp.json directly. mcp_status() reads
        # from MCP_CONFIG_PATH, which the fixture redirects here.
        mock_claude_config_path.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "headroom": {"command": "headroom", "args": ["mcp", "serve"]},
                    }
                }
            )
        )

        runner = CliRunner()
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
        assert "proxy-url" in result.output
        assert "debug" in result.output


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


class TestMCPUninstallWithClaudeCLI:
    """Test mcp_uninstall when the claude CLI is available."""

    def test_uninstall_calls_claude_mcp_remove(self):
        """Uninstall calls claude mcp remove when headroom is registered."""
        calls = []

        def capturing_run(cmd, **kwargs):
            calls.append(list(cmd))
            return MagicMock(returncode=0, stderr="")

        runner = CliRunner()
        with patch("headroom.cli.mcp.shutil.which", return_value="/usr/bin/claude"):
            with patch("headroom.cli.mcp.subprocess.run", side_effect=capturing_run):
                result = runner.invoke(main, ["mcp", "uninstall"])

        assert result.exit_code == 0
        assert "removed" in result.output.lower()
        subcommands = [c[2] for c in calls]
        assert "remove" in subcommands

    def test_uninstall_skips_remove_when_not_registered(self):
        """Uninstall does not call remove when headroom is not registered via claude CLI."""
        calls = []

        def capturing_run(cmd, **kwargs):
            calls.append(list(cmd))
            # mcp get returns non-zero → not registered
            if "get" in cmd:
                return MagicMock(returncode=1, stderr="")
            return MagicMock(returncode=0, stderr="")

        runner = CliRunner()
        with patch("headroom.cli.mcp.shutil.which", return_value="/usr/bin/claude"):
            with patch("headroom.cli.mcp.subprocess.run", side_effect=capturing_run):
                result = runner.invoke(main, ["mcp", "uninstall"])

        assert result.exit_code == 0
        subcommands = [c[2] for c in calls]
        assert "remove" not in subcommands
