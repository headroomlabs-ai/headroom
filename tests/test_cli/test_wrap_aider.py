"""Tests for `headroom wrap aider` command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote

import pytest
from click.testing import CliRunner

from headroom.cli.main import main


def _expected_project_prefix() -> str:
    """The /p/<name> prefix the wrap now embeds (launch-directory basename)."""
    return f"/p/{quote(Path.cwd().name, safe='')}"


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_wrap_aider_sets_provider_envs(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="aider"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "aider", "--no-rtk", "--", "--model", "gpt-4o"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["OPENAI_API_BASE"] == f"http://127.0.0.1:8787{_expected_project_prefix()}/v1"
    assert env["ANTHROPIC_BASE_URL"] == f"http://127.0.0.1:8787{_expected_project_prefix()}"
    assert captured["tool_label"] == "AIDER"
    assert captured["agent_type"] == "aider"
    assert captured["args"] == ("--model", "gpt-4o")
