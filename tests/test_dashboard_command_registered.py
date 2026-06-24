"""Regression test: headroom dashboard command must be registered.

Issue #1306: headroom v0.27.0 returned 'Error: No such command dashboard'
despite the command being defined in headroom/cli/proxy.py.
The fix was re-registering it in _register_commands(); this test prevents
the regression from reoccurring.
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from headroom.cli import main


def test_dashboard_command_registered() -> None:
    """headroom dashboard must exist as a top-level CLI command."""
    assert "dashboard" in main.commands, (
        "'headroom dashboard' is not registered. "
        "Check that proxy.py is imported in cli/main.py::_register_commands()."
    )


def test_dashboard_no_open_prints_url() -> None:
    """headroom dashboard --no-open must print the proxy URL without errors."""
    runner = CliRunner()
    result = runner.invoke(main, ["dashboard", "--no-open"])
    assert result.exit_code == 0, f"exit {result.exit_code}: {result.output}"
    assert "127.0.0.1" in result.output or "localhost" in result.output
    assert "/dashboard" in result.output


def test_dashboard_custom_port() -> None:
    """headroom dashboard --port 9999 --no-open must use the custom port."""
    runner = CliRunner()
    result = runner.invoke(main, ["dashboard", "--port", "9999", "--no-open"])
    assert result.exit_code == 0
    assert "9999" in result.output


def test_dashboard_help_mentions_proxy() -> None:
    """headroom dashboard --help must mention the proxy requirement."""
    runner = CliRunner()
    result = runner.invoke(main, ["dashboard", "--help"])
    assert result.exit_code == 0
    # Must tell users the proxy needs to be running
    assert "proxy" in result.output.lower()
