"""Regression tests for ``headroom unwrap copilot`` CLI command.

Covers registration, option wiring, and proxy-stop behaviour as requested
in the review of PR #1178 (GitHub issue #1172).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


class TestUnwrapCopilotCLI:
    """Tests that ``headroom unwrap copilot`` is registered and wired correctly."""

    def test_unwrap_copilot_registered(self, runner: CliRunner) -> None:
        """The command exists and exposes --port / --no-stop-proxy in --help."""
        result = runner.invoke(main, ["unwrap", "copilot", "--help"])
        assert result.exit_code == 0, result.output
        assert "--port" in result.output
        assert "--no-stop-proxy" in result.output

    def test_unwrap_copilot_default_calls_proxy_stop(self, runner: CliRunner) -> None:
        """With no flags, the proxy-stop helper is called with the default port."""
        with (
            patch(
                "headroom.cli.wrap._stop_local_proxy_for_unwrap", return_value="not_running"
            ) as mock_stop,
            patch("headroom.cli.wrap._echo_unwrap_proxy_stop_status") as mock_echo,
        ):
            result = runner.invoke(main, ["unwrap", "copilot"], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        mock_stop.assert_called_once_with(8787)
        mock_echo.assert_called_once_with("not_running", 8787)

    def test_unwrap_copilot_custom_port(self, runner: CliRunner) -> None:
        """--port is forwarded to the proxy-stop helper."""
        with (
            patch(
                "headroom.cli.wrap._stop_local_proxy_for_unwrap", return_value="not_running"
            ) as mock_stop,
            patch("headroom.cli.wrap._echo_unwrap_proxy_stop_status") as mock_echo,
        ):
            result = runner.invoke(
                main, ["unwrap", "copilot", "--port", "9999"], catch_exceptions=False
            )

        assert result.exit_code == 0, result.output
        mock_stop.assert_called_once_with(9999)
        mock_echo.assert_called_once_with("not_running", 9999)

    def test_unwrap_copilot_short_port_flag(self, runner: CliRunner) -> None:
        """-p also works as the short form of --port."""
        with (
            patch(
                "headroom.cli.wrap._stop_local_proxy_for_unwrap", return_value="not_running"
            ) as mock_stop,
            patch("headroom.cli.wrap._echo_unwrap_proxy_stop_status"),
        ):
            result = runner.invoke(
                main, ["unwrap", "copilot", "-p", "7777"], catch_exceptions=False
            )

        assert result.exit_code == 0, result.output
        mock_stop.assert_called_once_with(7777)

    def test_unwrap_copilot_no_stop_proxy_skips_stop(self, runner: CliRunner) -> None:
        """--no-stop-proxy suppresses the proxy-stop helper entirely."""
        with (
            patch("headroom.cli.wrap._stop_local_proxy_for_unwrap") as mock_stop,
            patch("headroom.cli.wrap._echo_unwrap_proxy_stop_status") as mock_echo,
        ):
            result = runner.invoke(
                main, ["unwrap", "copilot", "--no-stop-proxy"], catch_exceptions=False
            )

        assert result.exit_code == 0, result.output
        mock_stop.assert_not_called()
        mock_echo.assert_not_called()

    def test_unwrap_copilot_output_contains_header(self, runner: CliRunner) -> None:
        """The output includes the visual header banner."""
        with (
            patch("headroom.cli.wrap._stop_local_proxy_for_unwrap", return_value="not_running"),
            patch("headroom.cli.wrap._echo_unwrap_proxy_stop_status"),
        ):
            result = runner.invoke(main, ["unwrap", "copilot"], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        assert "HEADROOM UNWRAP: COPILOT" in result.output

    def test_unwrap_copilot_output_mentions_no_filesystem_state(self, runner: CliRunner) -> None:
        """The output explains that the copilot wrap is process-scoped."""
        with (
            patch("headroom.cli.wrap._stop_local_proxy_for_unwrap", return_value="not_running"),
            patch("headroom.cli.wrap._echo_unwrap_proxy_stop_status"),
        ):
            result = runner.invoke(main, ["unwrap", "copilot"], catch_exceptions=False)

        assert result.exit_code == 0, result.output
        assert "process-scoped" in result.output.lower()
        assert "no filesystem state" in result.output.lower()
