"""Tests for `headroom wrap grok` command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.providers.grok import PROXY_ENV_KEY


def _expected_project_prefix() -> str:
    return f"/p/{quote(Path.cwd().name, safe='')}"


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
        with patch("headroom.cli.wrap._setup_headroom_mcp"):
            with patch("headroom.cli.wrap._setup_coding_compressor"):
                with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
                    result = runner.invoke(main, ["wrap", "grok", "--no-mcp", "--", "-p", "hello"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env[PROXY_ENV_KEY] == f"http://127.0.0.1:8787{_expected_project_prefix()}/v1"
    assert "GROK_CLI_CHAT_PROXY_BASE_URL" not in env
    assert captured["tool_label"] == "GROK"
    assert captured["agent_type"] == "grok"
    assert captured["args"] == ("-p", "hello")


def test_wrap_grok_missing_binary_exits(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        with patch("headroom.cli.wrap._setup_headroom_mcp"):
            with patch("headroom.cli.wrap._setup_coding_compressor"):
                result = runner.invoke(main, ["wrap", "grok", "--no-mcp"])

    assert result.exit_code == 1
    assert "grok" in result.output.lower()
