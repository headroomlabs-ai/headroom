"""Tests for `headroom wrap goose` command (PR-G1, Phase G).

Covers goose-specific behavior: the OPENAI/ANTHROPIC env-var fan-out for
the child binary launch, and the goose-binary-not-found error path.
"""

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


def test_wrap_goose_sets_provider_envs(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENAI_BASE_URL, OPENAI_API_BASE, ANTHROPIC_BASE_URL, ANTHROPIC_HOST are set.

    Goose's Anthropic provider reads ANTHROPIC_HOST rather than
    ANTHROPIC_BASE_URL, so the value is asserted rather than mere presence.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="goose"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "goose", "--port", "9000", "--", "session"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9000/v1"
    assert env["OPENAI_API_BASE"] == "http://127.0.0.1:9000/v1"
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9000"
    assert env["ANTHROPIC_HOST"] == "http://127.0.0.1:9000"
    assert captured["tool_label"] == "GOOSE"
    assert captured["agent_type"] == "goose"
    assert captured["args"] == ("session",)


def test_wrap_goose_missing_binary_errors_clearly(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the goose binary is missing the command must fail with a clear error."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod.shutil, "which", return_value=None):
        result = runner.invoke(main, ["wrap", "goose"])

    assert result.exit_code == 1
    assert "'goose' not found in PATH" in result.output
