"""Tests for `headroom wrap grok` command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_wrap_grok_sets_proxy_env(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="grok"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "grok", "-p", "fix the bug"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["GROK_CLI_CHAT_PROXY_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert captured["tool_label"] == "GROK"
    assert captured["agent_type"] == "grok"
    assert captured["args"] == ("-p", "fix the bug")


def test_wrap_grok_keeps_headroom_port_long_option(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="grok"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "grok", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert captured["port"] == 9999
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["GROK_CLI_CHAT_PROXY_BASE_URL"] == "http://127.0.0.1:9999/v1"


def test_wrap_grok_forwards_no_proxy(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="grok"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "grok", "--no-proxy"])

    assert result.exit_code == 0, result.output
    assert captured["no_proxy"] is True


def test_wrap_grok_missing_binary(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        result = runner.invoke(main, ["wrap", "grok"])

    assert result.exit_code == 1
    assert "grok" in result.output.lower()
