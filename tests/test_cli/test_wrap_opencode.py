"""Tests for `headroom wrap opencode` and `headroom unwrap opencode`."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _set_test_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)


# ---------------------------------------------------------------------------
# Wrap opencode
# ---------------------------------------------------------------------------


def test_wrap_opencode_sets_config_content_env(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENCODE_CONFIG_CONTENT env var is set with the headroom provider."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://deepseek.example/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example")

    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(
                    main,
                    ["wrap", "opencode", "--port", "9000", "--no-mcp", "--", "--model", "gpt-4o"],
                )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert "OPENCODE_CONFIG_CONTENT" in env
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert config["provider"]["headroom"]["npm"] == "@ai-sdk/openai-compatible"
    assert config["provider"]["headroom"]["options"]["baseURL"] == "http://127.0.0.1:9000/v1"
    assert "model" not in config  # headroom provider is a transparent pass-through
    assert captured["tool_label"] == "OPENCODE"
    assert captured["agent_type"] == "opencode"
    assert captured["args"] == ("--model", "gpt-4o")


def test_wrap_opencode_does_not_add_base_url_env_vars(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENAI_BASE_URL and ANTHROPIC_BASE_URL are left to OpenCode providers."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://deepseek.example/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example")

    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--no-mcp"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["OPENAI_BASE_URL"] == "https://deepseek.example/v1"
    assert env["ANTHROPIC_BASE_URL"] == "https://anthropic.example"


def test_wrap_opencode_missing_binary_errors_clearly(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the opencode binary is missing the command must fail with a clear error."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)

    with patch.object(wrap_mod.shutil, "which", return_value=None):
        with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
            result = runner.invoke(main, ["wrap", "opencode"])

    assert result.exit_code == 1
    assert "'opencode' not found in PATH" in result.output


def test_wrap_opencode_prepare_only_injects_config(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`wrap opencode --prepare-only` writes the provider config to opencode.json."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
            result = runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--prepare-only"])

    assert result.exit_code == 0, result.output
    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    assert config_file.exists()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    assert config["provider"]["headroom"]["options"]["baseURL"] == "http://127.0.0.1:9000/v1"


def test_wrap_opencode_prepare_only_registers_serena_with_agent_context(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
            result = runner.invoke(main, ["wrap", "opencode", "--prepare-only"])

    assert result.exit_code == 0, result.output
    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config = json.loads(config_file.read_text())
    serena_command = config["mcp"]["serena"]["command"]
    assert serena_command[serena_command.index("--context") + 1] == "agent"


def test_wrap_opencode_no_mcp_skips_mcp_injection(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--no-mcp` skips MCP server injection."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--no-mcp"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert "mcp" not in config
    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    persisted_config = json.loads(config_file.read_text())
    assert "headroom" not in persisted_config.get("mcp", {})


def test_wrap_opencode_injects_mcp_by_default(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MCP is included in OPENCODE_CONFIG_CONTENT by default."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "opencode", "--port", "9000"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])
    assert "mcp" in config
    assert config["mcp"]["headroom"] == {
        "type": "local",
        "command": ["headroom", "mcp", "serve"],
        "enabled": True,
        "environment": {"HEADROOM_PROXY_URL": "http://127.0.0.1:9000"},
    }


def test_wrap_opencode_injects_rtk_into_agents_md(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RTK instructions are injected into global and project AGENTS.md."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--no-mcp"])

    assert result.exit_code == 0, result.output
    global_agents = tmp_path / ".config" / "opencode" / "AGENTS.md"
    project_agents = tmp_path / "AGENTS.md"
    assert global_agents.exists(), "Global AGENTS.md should be created"
    assert project_agents.exists(), "Project AGENTS.md should be created"
    assert wrap_mod._RTK_MARKER in global_agents.read_text(encoding="utf-8")
    assert wrap_mod._RTK_MARKER in project_agents.read_text(encoding="utf-8")


def test_wrap_opencode_idempotent_no_duplicate_block(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running wrap twice must not duplicate the RTK block in AGENTS.md."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--no-mcp"])
                runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--no-mcp"])

    project_agents = tmp_path / "AGENTS.md"
    content = project_agents.read_text(encoding="utf-8")
    assert content.count(wrap_mod._RTK_MARKER) == 1


# ---------------------------------------------------------------------------
# Unwrap opencode
# ---------------------------------------------------------------------------


def test_unwrap_opencode_restores_from_backup(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unwrap restores the pre-wrap backup and removes it."""
    monkeypatch.chdir(tmp_path)
    _set_test_home(monkeypatch, tmp_path)

    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    backup_file = config_file.with_suffix(".json.headroom-backup")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    original = '{"model": "openai/gpt-4o"}'
    config_file.write_text(original)
    backup_file.write_text(original)

    with patch.object(wrap_mod, "_stop_local_proxy_for_unwrap", return_value="stopped"):
        result = runner.invoke(main, ["unwrap", "opencode"])

    assert result.exit_code == 0, result.output
    assert "Restored prior" in result.output
    assert not backup_file.exists()
    assert config_file.read_text(encoding="utf-8") == original


def test_unwrap_opencode_strips_blocks_when_no_backup(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unwrap strips Headroom blocks when no backup exists."""
    monkeypatch.chdir(tmp_path)
    _set_test_home(monkeypatch, tmp_path)

    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    user_content = '{"model": "openai/gpt-4o"}'
    wrapped_content = (
        wrap_mod._PROVIDER_MARKER_START
        + '\n"provider": {},\n'
        + wrap_mod._PROVIDER_MARKER_END
        + "\n"
        + user_content
    )
    config_file.write_text(wrapped_content)

    with patch.object(wrap_mod, "_stop_local_proxy_for_unwrap", return_value="stopped"):
        result = runner.invoke(main, ["unwrap", "opencode"])

    assert result.exit_code == 0, result.output
    assert "Removed Headroom block" in result.output
    assert user_content in config_file.read_text(encoding="utf-8")
    assert wrap_mod._PROVIDER_MARKER_START not in config_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Edge cases — wrap
# ---------------------------------------------------------------------------


def test_wrap_opencode_preserves_existing_user_providers(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap merges headroom provider without disturbing user's existing providers."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text('{"provider": {"openai": {"models": {"gpt-4o": {"name": "GPT-4o"}}}}}')

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--no-mcp"])

    assert result.exit_code == 0, result.output
    config = json.loads(config_file.read_text(encoding="utf-8"))
    assert "headroom" in config["provider"], "headroom provider not injected"
    assert "openai" in config["provider"], "user's openai provider was removed"


def test_wrap_opencode_port_change_updates_existing_config(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrapping with a different port updates the baseURL in opencode.json."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--no-mcp"])
                runner.invoke(main, ["wrap", "opencode", "--port", "9001", "--no-mcp"])

    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    assert config["provider"]["headroom"]["options"]["baseURL"] == "http://127.0.0.1:9001/v1"


def test_wrap_opencode_handles_malformed_config_file(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap handles a malformed opencode.json by backing it up before overwriting."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    malformed = '{"model": "gpt-4o",}'  # trailing comma
    config_file.write_text(malformed)
    backup_file = config_file.with_suffix(".json.headroom-backup")

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--no-mcp"])

    assert result.exit_code == 0, result.output
    assert backup_file.exists(), "backup must be created before overwriting"
    assert backup_file.read_text(encoding="utf-8") == malformed, (
        "backup must preserve original byte-for-byte"
    )
    # The config file is now valid JSON with headroom provider.
    config = json.loads(config_file.read_text(encoding="utf-8"))
    assert "headroom" in config.get("provider", {})


def test_wrap_opencode_handles_empty_config_file(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap handles an empty opencode.json file gracefully."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text("")

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--no-mcp"])

    assert result.exit_code == 0, result.output
    config = json.loads(config_file.read_text(encoding="utf-8"))
    assert config["provider"]["headroom"]["options"]["baseURL"] == "http://127.0.0.1:9000/v1"


def test_wrap_opencode_handles_config_dir_missing(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap creates the config directory when it doesn't exist."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    config_dir = tmp_path / ".config" / "opencode"
    assert not config_dir.exists()

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--no-mcp"])

    assert result.exit_code == 0, result.output
    assert config_dir.exists()
    assert (config_dir / "opencode.json").exists()


def test_wrap_opencode_rtk_preserves_existing_agents_md(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RTK injection appends to AGENTS.md without removing existing content."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    existing_content = "# My custom rules\nUse spaces, not tabs."
    (tmp_path / "AGENTS.md").write_text(existing_content)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--no-mcp"])

    assert result.exit_code == 0, result.output
    content = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert existing_content in content
    assert wrap_mod._RTK_MARKER in content


def test_wrap_opencode_no_rtk_leaves_agents_md_untouched(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--no-rtk` flag leaves existing AGENTS.md untouched."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    existing_content = "# My custom rules\nUse spaces, not tabs."
    (tmp_path / "AGENTS.md").write_text(existing_content)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(
                    main, ["wrap", "opencode", "--port", "9000", "--no-rtk", "--no-mcp"]
                )

    assert result.exit_code == 0, result.output
    content = (tmp_path / "AGENTS.md").read_text(encoding="utf-8")
    assert content == existing_content, "--no-rtk modified AGENTS.md"
    assert wrap_mod._RTK_MARKER not in content


def test_wrap_opencode_respects_opencode_config_env(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENCODE_CONFIG env var overrides the default config path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    custom_config = tmp_path / "custom" / "config.json"
    monkeypatch.setenv("OPENCODE_CONFIG", str(custom_config))

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--no-mcp"])

    assert result.exit_code == 0, result.output
    assert custom_config.exists()
    default_config = tmp_path / ".config" / "opencode" / "opencode.json"
    assert not default_config.exists(), (
        "default config should not be created when OPENCODE_CONFIG is set"
    )


def test_wrap_opencode_headroom_project_from_cwd(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HEADROOM_PROJECT is set based on the current working directory name."""
    project_dir = tmp_path / "my-project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    monkeypatch.delenv("HEADROOM_PROJECT", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--no-mcp"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert env.get("HEADROOM_PROJECT") == "my-project"


def test_wrap_opencode_respects_existing_headroom_project(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User-set HEADROOM_PROJECT env var is preserved, not overridden."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)
    monkeypatch.setenv("HEADROOM_PROJECT", "user-set-value")

    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--no-mcp"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert env["HEADROOM_PROJECT"] == "user-set-value"


def test_wrap_opencode_config_merges_existing_model(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrap preserves the user's existing model selection."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text('{"model": "openai/gpt-4o"}')

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--no-mcp"])

    assert result.exit_code == 0, result.output
    config = json.loads(config_file.read_text(encoding="utf-8"))
    assert config["model"] == "openai/gpt-4o"
    assert config["provider"]["headroom"]["npm"] == "@ai-sdk/openai-compatible"


# ---------------------------------------------------------------------------
# Edge cases — unwrap
# ---------------------------------------------------------------------------


def test_unwrap_opencode_removes_config_when_only_headroom_content(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unwrap removes the config file entirely when it contained only Headroom content."""
    monkeypatch.chdir(tmp_path)
    _set_test_home(monkeypatch, tmp_path)

    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    wrapped_content = (
        wrap_mod._PROVIDER_MARKER_START + '\n"provider": {},\n' + wrap_mod._PROVIDER_MARKER_END
    )
    config_file.write_text(wrapped_content)

    with patch.object(wrap_mod, "_stop_local_proxy_for_unwrap", return_value="stopped"):
        result = runner.invoke(main, ["unwrap", "opencode"])

    assert result.exit_code == 0, result.output
    assert "Removed" in result.output
    assert not config_file.exists()


def test_unwrap_opencode_noop_when_config_missing(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unwrap is a safe no-op when the config file doesn't exist."""
    monkeypatch.chdir(tmp_path)
    _set_test_home(monkeypatch, tmp_path)

    with patch.object(wrap_mod, "_stop_local_proxy_for_unwrap", return_value="stopped"):
        result = runner.invoke(main, ["unwrap", "opencode"])

    assert result.exit_code == 0, result.output
    assert "does not exist" in result.output


def test_unwrap_opencode_noop_when_no_headroom_markers(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unwrap is a safe no-op when the config has no Headroom markers."""
    monkeypatch.chdir(tmp_path)
    _set_test_home(monkeypatch, tmp_path)

    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text('{"model": "openai/gpt-4o"}')

    with patch.object(wrap_mod, "_stop_local_proxy_for_unwrap", return_value="stopped"):
        result = runner.invoke(main, ["unwrap", "opencode"])

    assert result.exit_code == 0, result.output
    assert "no Headroom wrap markers" in result.output
    assert config_file.read_text(encoding="utf-8").strip() == '{"model": "openai/gpt-4o"}'


def test_wrap_unwrap_rewrap_is_idempotent(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full wrap-unwrap-rewrap cycle produces consistent results."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    user_config = '{"model": "openai/gpt-4o", "provider": {"openai": {}}}'
    config_file.write_text(user_config)

    # First wrap
    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--no-mcp"])

    # Unwrap
    with patch.object(wrap_mod, "_stop_local_proxy_for_unwrap", return_value="stopped"):
        runner.invoke(main, ["unwrap", "opencode"])

    # After unwrap, file should match original
    after_unwrap = json.loads(config_file.read_text(encoding="utf-8"))
    assert after_unwrap["model"] == "openai/gpt-4o"
    assert "headroom" not in after_unwrap.get("provider", {})

    # Re-wrap
    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                runner.invoke(main, ["wrap", "opencode", "--port", "9001", "--no-mcp"])

    # After re-wrap, headroom should be back, model unchanged
    after_rewrap = json.loads(config_file.read_text(encoding="utf-8"))
    assert after_rewrap["model"] == "openai/gpt-4o"
    assert "headroom" in after_rewrap.get("provider", {})


def test_unwrap_opencode_restores_backup_and_removes_it(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unwrap removes the backup file after successful restore."""
    monkeypatch.chdir(tmp_path)
    _set_test_home(monkeypatch, tmp_path)

    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    backup_file = config_file.with_suffix(".json.headroom-backup")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    original = '{"model": "openai/gpt-4o"}'
    config_file.write_text(original)
    backup_file.write_text(original)

    with patch.object(wrap_mod, "_stop_local_proxy_for_unwrap", return_value="stopped"):
        result = runner.invoke(main, ["unwrap", "opencode"])

    assert result.exit_code == 0, result.output
    assert "Restored prior" in result.output
    assert not backup_file.exists(), "backup file was not cleaned up after restore"


def test_wrap_opencode_no_arguments_is_valid(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`headroom wrap opencode` with no additional arguments is a valid command."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=fake_launch_tool):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "opencode", "--no-mcp"])

    assert result.exit_code == 0, result.output
    assert captured["tool_label"] == "OPENCODE"
    assert captured["args"] == ()


def test_wrap_opencode_with_memory_flag(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--memory flag is accepted and does not crash."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(
                    main, ["wrap", "opencode", "--port", "9000", "--memory", "--no-mcp"]
                )

    assert result.exit_code == 0, result.output


def test_wrap_opencode_with_backend_and_anyllm_provider(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--backend and --anyllm-provider flags are accepted."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(
                    main,
                    [
                        "wrap",
                        "opencode",
                        "--port",
                        "9000",
                        "--backend",
                        "anyllm",
                        "--anyllm-provider",
                        "groq",
                        "--no-mcp",
                    ],
                )

    assert result.exit_code == 0, result.output


def test_wrap_opencode_with_no_proxy(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--no-proxy flag skips proxy startup but still configures the tool."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(
                    main, ["wrap", "opencode", "--port", "9000", "--no-proxy", "--no-mcp"]
                )

    assert result.exit_code == 0, result.output


def test_wrap_opencode_with_verbose_flag(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--verbose flag does not crash."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    _set_test_home(monkeypatch, tmp_path)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(
                    main, ["wrap", "opencode", "--port", "9000", "--verbose", "--no-mcp"]
                )

    assert result.exit_code == 0, result.output


def test_wrap_opencode_respects_opencode_home_env(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENCODE_HOME env var controls where AGENTS.md is written."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HEADROOM_CONTEXT_TOOL", raising=False)
    custom_home = str(tmp_path / "custom-opencode-home")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OPENCODE_HOME", custom_home)

    with patch.object(wrap_mod.shutil, "which", return_value="opencode"):
        with patch.object(wrap_mod, "_launch_tool", side_effect=SystemExit(0)):
            with patch.object(wrap_mod, "_ensure_rtk_binary", return_value=Path("/tmp/rtk")):
                result = runner.invoke(main, ["wrap", "opencode", "--port", "9000", "--no-mcp"])

    assert result.exit_code == 0, result.output
    agents_md = Path(custom_home) / "AGENTS.md"
    assert agents_md.exists()


# ---------------------------------------------------------------------------
# Regression: unwrap must preserve non-ASCII UTF-8 user content (#1126)
# ---------------------------------------------------------------------------


def test_unwrap_opencode_preserves_utf8_user_content(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unwrap strips Headroom blocks but preserves non-ASCII UTF-8 user content (#1126)."""
    monkeypatch.chdir(tmp_path)
    _set_test_home(monkeypatch, tmp_path)

    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)

    # User content with smart quotes and em dashes (non-ASCII UTF-8)
    user_config = {
        "model": "openai/gpt-4o",
        "description": "“smart quotes” and an em dash — here",
    }
    user_json = json.dumps(user_config, ensure_ascii=False)

    wrapped_content = (
        wrap_mod._PROVIDER_MARKER_START
        + '\n"provider": {},\n'
        + wrap_mod._PROVIDER_MARKER_END
        + "\n"
        + user_json
    )
    config_file.write_text(wrapped_content, encoding="utf-8")

    # Mock out OpencodeRegistrar to avoid its own bare-open encoding issue
    # (pre-existing; outside this PR's scope).
    fake_registrar = type("FakeRegistrar", (), {"detect": lambda self: False})()
    with patch.object(wrap_mod, "_stop_local_proxy_for_unwrap", return_value="stopped"):
        with patch("headroom.mcp_registry.OpencodeRegistrar", return_value=fake_registrar):
            result = runner.invoke(main, ["unwrap", "opencode"])

    assert result.exit_code == 0, result.output
    assert "Removed Headroom block" in result.output
    content = config_file.read_text(encoding="utf-8")
    assert "“smart quotes”" in content
    assert "—" in content
    assert wrap_mod._PROVIDER_MARKER_START not in content
