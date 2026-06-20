"""Tests for `headroom wrap antigravity` command."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_wrap_antigravity_launch(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Antigravity launches with correct configuration."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="agy"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(
                main, ["wrap", "antigravity", "--port", "9000", "--", "--model", "gemini-2.5-pro"]
            )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9000/v1"
    assert env["OPENAI_API_BASE"] == "http://127.0.0.1:9000/v1"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9000"
    assert env["GEMINI_BASE_URL"] == "http://127.0.0.1:9000/v1beta"
    assert env["GOOGLE_GENAI_API_BASE"] == "http://127.0.0.1:9000/v1beta"
    assert env["AGY_BASE_URL"] == "http://127.0.0.1:9000/v1beta"
    assert env["ANTIGRAVITY_BASE_URL"] == "http://127.0.0.1:9000/v1beta"
    assert captured["tool_label"] == "ANTIGRAVITY"
    assert captured["agent_type"] == "antigravity"
    assert captured["args"] == ("--model", "gemini-2.5-pro")


def test_wrap_antigravity_not_found(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Error message when agy binary is not found."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod.shutil, "which", return_value=None):
        result = runner.invoke(main, ["wrap", "antigravity"])

    assert result.exit_code == 1
    assert "Error: 'agy' not found in PATH" in result.output
    assert "Install Antigravity: https://antigravity.google/docs" in result.output


def test_wrap_antigravity_no_proxy(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-proxy flag prevents proxy startup."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="agy"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "antigravity", "--no-proxy"])

    assert result.exit_code == 0, result.output
    assert captured["no_proxy"] is True


def test_wrap_antigravity_learn_memory(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--learn and --memory flags are passed to _launch_tool."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="agy"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "antigravity", "--learn", "--memory"])

    assert result.exit_code == 0, result.output
    assert captured["learn"] is True
    assert captured["memory"] is True
