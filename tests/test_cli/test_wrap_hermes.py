"""Tests for `headroom wrap hermes` command."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_wrap_hermes_sets_provider_envs(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENAI_BASE_URL and ANTHROPIC_BASE_URL are set on launch."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="hermes"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(
                wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")
            ):
                result = runner.invoke(
                    main, ["wrap", "hermes", "--port", "9000", "--", "session"]
                )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9000/v1"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9000"
    assert captured["tool_label"] == "HERMES"
    assert captured["agent_type"] == "hermes"
    assert captured["args"] == ("session",)


def test_wrap_hermes_missing_binary_errors_clearly(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the hermes binary is missing, fail with a clear error."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod.shutil, "which", return_value=None):
        with patch.object(
            wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")
        ):
            result = runner.invoke(main, ["wrap", "hermes"])

    assert result.exit_code == 1
    assert "'hermes' not found in PATH" in result.output
    assert "https://hermes-agent.nousresearch.com/" in result.output


def test_wrap_hermes_prepare_only_exits_cleanly(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--prepare-only must not attempt to launch Hermes."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod, "_launch_tool") as launch_tool:
        with patch.object(
            wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")
        ):
            result = runner.invoke(main, ["wrap", "hermes", "--prepare-only"])

    assert result.exit_code == 0, result.output
    launch_tool.assert_not_called()
