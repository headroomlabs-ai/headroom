"""Tests for `headroom wrap openhands` command (PR-G1, Phase G)."""

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


def test_wrap_openhands_prepare_only_succeeds_unpatched(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`wrap openhands --prepare-only` must succeed with nothing patched.

    The subcommand used to hard-fail by default because a CLI context tool was
    a required dependency of the prelude. Nothing is fetched or written now, so
    the bare invocation has to exit 0 on its own.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    result = runner.invoke(main, ["wrap", "openhands", "--prepare-only"])

    assert result.exit_code == 0, result.output


def test_wrap_openhands_sets_provider_envs(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENAI_BASE_URL, ANTHROPIC_BASE_URL, LLM_BASE_URL are set on launch."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="openhands"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(
                main, ["wrap", "openhands", "--port", "9000", "--", "--task", "demo"]
            )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9000/v1"
    assert env["OPENAI_API_BASE"] == "http://127.0.0.1:9000/v1"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9000"
    assert env["LLM_BASE_URL"] == "http://127.0.0.1:9000/v1"
    assert captured["tool_label"] == "OPENHANDS"
    assert captured["agent_type"] == "openhands"
    assert captured["args"] == ("--task", "demo")


def test_wrap_openhands_missing_binary_errors_clearly(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the openhands binary is missing the command must fail with a clear error."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod.shutil, "which", return_value=None):
        result = runner.invoke(main, ["wrap", "openhands"])

    assert result.exit_code == 1
    assert "'openhands' not found in PATH" in result.output
