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
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
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


def test_wrap_openhands_injects_rtk_via_env_var(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENHANDS_INSTRUCTIONS env var must contain the RTK block at launch."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    monkeypatch.delenv("OPENHANDS_INSTRUCTIONS", raising=False)

    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="openhands"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "openhands"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    instructions = env.get("OPENHANDS_INSTRUCTIONS", "")
    assert wrap_mod._RTK_MARKER in instructions
    assert "RTK (Rust Token Killer)" in instructions


def test_wrap_openhands_preserves_existing_openhands_instructions(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-existing OPENHANDS_INSTRUCTIONS env content is preserved, rtk is appended."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    monkeypatch.setenv("OPENHANDS_INSTRUCTIONS", "Prefer typed Python.")

    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="openhands"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "openhands"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    instructions = env.get("OPENHANDS_INSTRUCTIONS", "")
    assert "Prefer typed Python." in instructions
    assert wrap_mod._RTK_MARKER in instructions


def test_wrap_openhands_idempotent_already_injected(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If OPENHANDS_INSTRUCTIONS already contains the marker, do not re-append."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    pre_existing = "Prefer typed Python.\n\n" + wrap_mod.RTK_INSTRUCTIONS_BLOCK
    monkeypatch.setenv("OPENHANDS_INSTRUCTIONS", pre_existing)

    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="openhands"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "openhands"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    instructions = env.get("OPENHANDS_INSTRUCTIONS", "")
    assert instructions.count(wrap_mod._RTK_MARKER) == 1


def test_wrap_openhands_missing_binary_errors_clearly(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the openhands binary is missing the command must fail with a clear error."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod.shutil, "which", return_value=None):
        with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
            result = runner.invoke(main, ["wrap", "openhands"])

    assert result.exit_code == 1
    assert "'openhands' not found in PATH" in result.output


def test_wrap_openhands_no_context_tool_does_not_inject(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-context-tool must skip OPENHANDS_INSTRUCTIONS injection."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENHANDS_INSTRUCTIONS", raising=False)

    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="openhands"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_ensure_rtk_binary") as ensure:
                result = runner.invoke(main, ["wrap", "openhands", "--no-context-tool"])

    assert result.exit_code == 0, result.output
    ensure.assert_not_called()
    env = captured["env"]
    assert isinstance(env, dict)
    assert "OPENHANDS_INSTRUCTIONS" not in env or env["OPENHANDS_INSTRUCTIONS"] == ""


# ---------------------------------------------------------------------------
# M3: rtk install failure must fail loudly — no silent fallback to env
# injection without rtk on disk.
# ---------------------------------------------------------------------------


def test_wrap_openhands_rtk_install_failure_aborts_loudly(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If rtk install fails, command must exit non-zero with a clear error."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    monkeypatch.delenv("OPENHANDS_INSTRUCTIONS", raising=False)

    launch_called: list[bool] = []

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        launch_called.append(True)

    with patch.object(wrap_mod.shutil, "which", return_value="openhands"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=None):
                result = runner.invoke(main, ["wrap", "openhands"])

    assert result.exit_code == 1
    assert "rtk install failed" in result.output
    assert "--no-context-tool" in result.output
    # _launch_tool must NOT have been invoked when rtk install fails.
    assert launch_called == []


def test_wrap_openhands_rtk_install_failure_with_no_context_tool_still_launches(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-context-tool bypasses rtk entirely — should still launch."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENHANDS_INSTRUCTIONS", raising=False)

    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="openhands"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=None) as ensure:
                result = runner.invoke(main, ["wrap", "openhands", "--no-context-tool"])

    assert result.exit_code == 0, result.output
    # rtk should never have been queried.
    ensure.assert_not_called()
    env = captured["env"]
    assert "OPENHANDS_INSTRUCTIONS" not in env or env["OPENHANDS_INSTRUCTIONS"] == ""


# ---------------------------------------------------------------------------
# M4: Ctrl-C during prelude emits a clear "no on-disk changes" message.
# openhands never writes to disk (env-var injection only).
# ---------------------------------------------------------------------------


def test_wrap_openhands_keyboardinterrupt_during_prelude_emits_clear_message(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C during the prelude must signal cleanly with no on-disk artifact."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    monkeypatch.delenv("OPENHANDS_INSTRUCTIONS", raising=False)

    with patch.object(wrap_mod, "_ensure_rtk_binary", side_effect=KeyboardInterrupt):
        result = runner.invoke(main, ["wrap", "openhands"])

    assert result.exit_code == 130
    assert "interrupted" in result.output.lower()
    assert "idempotent" in result.output.lower()
