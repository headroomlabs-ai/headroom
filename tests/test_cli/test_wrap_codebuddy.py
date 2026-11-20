"""Tests for CodeBuddy wrap CLI commands."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _fake_proxy():
    p = MagicMock()
    p.proxy_base_url = "http://127.0.0.1:8787/v2"
    p.pid = 12345
    return p


def _mock_run_success():
    return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")


def test_wrap_codebuddy_prepare_only(runner: CliRunner) -> None:
    with (
        patch("headroom.cli.wrap._ensure_proxy"),
        patch("headroom.cli.wrap._setup_headroom_mcp"),
    ):
        result = runner.invoke(
            main,
            ["wrap", "codebuddy", "--prepare-only", "--no-mcp", "--no-proxy"],
        )
    assert result.exit_code == 0, result.output


def test_wrap_codebuddy_rejects_retired_context_tool_flag(runner: CliRunner) -> None:
    result = runner.invoke(main, ["wrap", "codebuddy", "--no-context-tool"])

    assert result.exit_code != 0
    assert "CLI context tools (rtk, lean-ctx) have been removed" in result.output


def test_wrap_codebuddy_registers_mcp_and_launches_with_v2_base_url(
    runner: CliRunner,
) -> None:
    launched_env: dict[str, str] = {}

    def run_codebuddy(*args, **kwargs):
        launched_env.update(kwargs["env"])
        return _mock_run_success()

    with (
        patch("headroom.cli.wrap._ensure_proxy", return_value=(_fake_proxy(), 8787)) as ensure,
        patch("headroom.cli.wrap._setup_headroom_mcp") as register_mcp,
        patch("headroom.cli.wrap._setup_serena_mcp") as register_serena,
        patch("headroom.cli.wrap.shutil.which", return_value="/usr/local/bin/codebuddy"),
        patch("subprocess.run", side_effect=run_codebuddy),
    ):
        result = runner.invoke(main, ["wrap", "codebuddy", "--resume", "session-1"])

    assert result.exit_code == 0, result.output
    ensure.assert_called_once_with(
        8787,
        False,
        learn=False,
        memory=False,
        agent_type="codebuddy",
        backend="codebuddy",
    )
    register_mcp.assert_called_once()
    register_serena.assert_called_once()
    assert launched_env["CODEBUDDY_BASE_URL"] == "http://127.0.0.1:8787/v2"


def test_wrap_codebuddy_no_mcp(runner: CliRunner) -> None:
    fake_proxy = _fake_proxy()

    with (
        patch("headroom.cli.wrap._ensure_proxy", return_value=(fake_proxy, 8787)),
        patch("headroom.cli.wrap._setup_headroom_mcp") as register_mcp,
        patch(
            "headroom.cli.wrap._codebuddy_proxy_base_url", return_value="http://127.0.0.1:8787/v2"
        ),
        patch("headroom.cli.wrap.shutil.which", return_value="/usr/local/bin/codebuddy"),
        patch("subprocess.run", return_value=_mock_run_success()),
    ):
        result = runner.invoke(
            main,
            ["wrap", "codebuddy", "--no-mcp"],
        )
    assert result.exit_code == 0, result.output
    register_mcp.assert_not_called()


def test_wrap_codebuddy_no_serena(runner: CliRunner) -> None:
    fake_proxy = _fake_proxy()

    with (
        patch("headroom.cli.wrap._ensure_proxy", return_value=(fake_proxy, 8787)),
        patch("headroom.cli.wrap._setup_headroom_mcp"),
        patch("headroom.cli.wrap._setup_serena_mcp") as register_serena,
        patch(
            "headroom.cli.wrap._codebuddy_proxy_base_url", return_value="http://127.0.0.1:8787/v2"
        ),
        patch("headroom.cli.wrap.shutil.which", return_value="/usr/local/bin/codebuddy"),
        patch("subprocess.run", return_value=_mock_run_success()),
    ):
        result = runner.invoke(
            main,
            ["wrap", "codebuddy", "--no-mcp", "--no-serena"],
        )
    assert result.exit_code == 0, result.output
    register_serena.assert_not_called()
