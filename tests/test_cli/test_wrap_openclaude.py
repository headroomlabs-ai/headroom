"""Tests for `headroom wrap openclaude` command (issue #1411)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import pytest
from click.testing import CliRunner

from headroom.cli.main import main


def _expected_project_prefix() -> str:
    return f"/p/{quote(Path.cwd().name, safe='')}"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_wrap_openclaude_routes_proxy_envs(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="openclaude"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(
                main,
                ["wrap", "openclaude", "--no-rtk", "--", "--model", "gpt-4o"],
            )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    prefix = _expected_project_prefix()
    assert env["OPENAI_API_BASE"] == f"http://127.0.0.1:8787{prefix}/v1"
    assert env["ANTHROPIC_BASE_URL"] == f"http://127.0.0.1:8787{prefix}"
    assert captured["tool_label"] == "OPENCLAUDE"
    assert captured["agent_type"] == "openclaude"
    assert captured["args"] == ("--model", "gpt-4o")


def test_wrap_openclaude_missing_binary_errors(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        result = runner.invoke(main, ["wrap", "openclaude", "--no-rtk"])
    assert result.exit_code == 1
    assert "openclaude" in result.output.lower()
