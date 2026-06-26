"""Tests for `headroom wrap grok` and `headroom unwrap grok`."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_wrap_help_lists_grok(runner: CliRunner) -> None:
    result = runner.invoke(main, ["wrap", "--help"])

    assert result.exit_code == 0, result.output
    assert "headroom wrap grok" in result.output


def test_unwrap_help_lists_grok(runner: CliRunner) -> None:
    result = runner.invoke(main, ["unwrap", "--help"])

    assert result.exit_code == 0, result.output
    assert "grok" in result.output


def test_wrap_grok_uses_verified_launch_contract(runner: CliRunner) -> None:
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs: object) -> None:
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="grok"),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            ["wrap", "grok", "--no-rtk", "--", "--model", "grok-beta"],
        )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["GROK_PROXY_URL"] == "http://127.0.0.1:8787/v1"
    assert captured["tool_label"] == "GROK"
    assert captured["agent_type"] == "grok"
    assert captured["args"] == ("--model", "grok-beta")
    assert captured["env_vars_display"] == ["GROK_PROXY_URL=http://127.0.0.1:8787/v1"]


def test_wrap_grok_prepare_only_skips_binary_lookup(runner: CliRunner) -> None:
    with (
        patch("headroom.cli.wrap.shutil.which") as which_mock,
        patch("headroom.cli.wrap._launch_tool") as launch_mock,
    ):
        result = runner.invoke(main, ["wrap", "grok", "--prepare-only", "--no-rtk"])

    assert result.exit_code == 0, result.output
    which_mock.assert_not_called()
    launch_mock.assert_not_called()


def test_wrap_grok_fails_when_binary_missing(runner: CliRunner) -> None:
    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        result = runner.invoke(main, ["wrap", "grok", "--no-rtk"])

    assert result.exit_code == 1
    assert "Error: 'grok' not found in PATH." in result.output
    assert "Install Grok Build CLI: https://docs.x.ai/build/overview" in result.output


def test_unwrap_grok_is_env_only_and_stops_proxy(runner: CliRunner) -> None:
    with patch(
        "headroom.cli.wrap._stop_local_proxy_for_unwrap", return_value="stopped"
    ) as stop_proxy:
        result = runner.invoke(main, ["unwrap", "grok", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert "Nothing to undo for `grok`; no durable wrap state is written." in result.output
    assert "Stopped local Headroom proxy on port 9999" in result.output
    stop_proxy.assert_called_once_with(9999)
