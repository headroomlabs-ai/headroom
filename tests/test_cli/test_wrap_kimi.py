"""Tests for `headroom wrap kimi` command."""

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


def test_wrap_kimi_launch(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kimi launches with correct configuration."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="kimi"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                result = runner.invoke(
                    main, ["wrap", "kimi", "--port", "9000", "--", "-m", "kimi-for-coding"]
                )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert captured["tool_label"] == "KIMI"
    assert captured["agent_type"] == "kimi"
    assert captured["args"] == ("-m", "kimi-for-coding")
    assert captured["openai_api_url"] == "https://api.kimi.com/coding/v1"
    assert env["KIMI_BASE_URL"] == "http://127.0.0.1:9000/v1"


def test_wrap_kimi_with_project_name(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Project name is encoded in KIMI_BASE_URL when run from a project directory."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="kimi"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "kimi", "--port", "7000"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert env["KIMI_BASE_URL"] == "http://127.0.0.1:7000/p/my-project/v1"


def test_wrap_kimi_cli_fallback(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Falls back to the `kimi-cli` binary when `kimi` is not on PATH."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    def fake_which(name: str) -> str | None:
        return "/usr/local/bin/kimi-cli" if name == "kimi-cli" else None

    with patch.object(wrap_mod.shutil, "which", side_effect=fake_which):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                result = runner.invoke(main, ["wrap", "kimi"])

    assert result.exit_code == 0, result.output
    assert captured["binary"] == "/usr/local/bin/kimi-cli"


def test_wrap_kimi_not_found(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Error message when neither kimi nor kimi-cli is found."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod.shutil, "which", return_value=None):
        result = runner.invoke(main, ["wrap", "kimi"])

    assert result.exit_code == 1
    assert "Error: 'kimi' (or 'kimi-cli') not found in PATH" in result.output
    assert "https://github.com/MoonshotAI/kimi-cli" in result.output


def test_wrap_kimi_custom_port(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom --port is passed to _launch_tool and appears in KIMI_BASE_URL."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="kimi"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                result = runner.invoke(main, ["wrap", "kimi", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert captured["port"] == 9999
    assert captured["env"]["KIMI_BASE_URL"] == "http://127.0.0.1:9999/v1"


def test_wrap_kimi_custom_api_url(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--kimi-api-url overrides the upstream endpoint passed to _launch_tool."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, Any] = {}

    def fake_launch_tool(**kwargs: Any) -> None:  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="kimi"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                result = runner.invoke(
                    main,
                    ["wrap", "kimi", "--kimi-api-url", "https://api.moonshot.ai/v1"],
                )

    assert result.exit_code == 0, result.output
    assert captured["openai_api_url"] == "https://api.moonshot.ai/v1"


def test_wrap_kimi_no_proxy(
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

    with patch.object(wrap_mod.shutil, "which", return_value="kimi"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                result = runner.invoke(main, ["wrap", "kimi", "--no-proxy"])

    assert result.exit_code == 0, result.output
    assert captured["no_proxy"] is True


def test_wrap_kimi_learn_memory(
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

    with patch.object(wrap_mod.shutil, "which", return_value="kimi"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_project_name_from_cwd", return_value=None):
                result = runner.invoke(main, ["wrap", "kimi", "--learn", "--memory"])

    assert result.exit_code == 0, result.output
    assert captured["learn"] is True
    assert captured["memory"] is True
