from __future__ import annotations

import importlib
import json
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

import click
import pytest
from click.testing import CliRunner


def _load_init_module(monkeypatch):
    monkeypatch.delitem(sys.modules, "headroom.cli.init", raising=False)
    monkeypatch.delitem(sys.modules, "headroom.cli.main", raising=False)
    fake_main_module = types.ModuleType("headroom.cli.main")

    @click.group()
    def fake_main() -> None:
        pass

    fake_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "headroom.cli.main", fake_main_module)
    importlib.invalidate_caches()
    init_cli = importlib.import_module("headroom.cli.init")
    monkeypatch.delitem(sys.modules, "headroom.cli.init", raising=False)
    return init_cli, fake_main


def test_init_auto_detects_targets(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()
    captured: dict[str, object] = {}

    monkeypatch.setattr(init_cli, "detect_init_targets", lambda global_scope: ["claude", "codex"])
    monkeypatch.setattr(init_cli, "_run_init_targets", lambda **kwargs: captured.update(kwargs))

    result = runner.invoke(fake_main, ["init", "-g"])

    assert result.exit_code == 0, result.output
    assert captured["targets"] == ["claude", "codex"]
    assert captured["global_scope"] is True


def test_init_fails_when_auto_detection_empty(monkeypatch) -> None:
    """Bare ``headroom init`` with no agents on PATH prints a guided error.

    Regression guard for issue #245: the error must list every target that
    was probed, confirm that -g / --global is a valid flag, and show the
    explicit per-target invocation so the user knows how to proceed.
    """

    init_cli, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()
    monkeypatch.setattr(init_cli.shutil, "which", lambda name: None)

    result = runner.invoke(fake_main, ["init", "-g"])

    assert result.exit_code != 0
    assert "No supported user-scope agents were found on PATH" in result.output
    assert "probed the following agents" in result.output
    # Every in-scope target is listed with its lookup status.
    for target in ("claude", "codebuddy", "codex", "copilot", "openclaw"):
        assert target in result.output
    # The user is told that -g is still valid and given a concrete next step.
    assert "-g" in result.output
    assert "headroom init -g claude" in result.output


def test_format_empty_detection_error_local_scope(monkeypatch) -> None:
    """Local-scope variant of the guided error only lists local-scope agents."""

    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli.shutil, "which", lambda name: None)

    message = init_cli._format_empty_detection_error(global_scope=False)

    assert "local-scope agents" in message
    assert "claude" in message and "codebuddy" in message and "codex" in message
    # Copilot / openclaw are global-only; must not be suggested for local.
    assert "headroom init copilot" not in message
    assert "headroom init openclaw" not in message
    assert "headroom init claude" in message
    assert "headroom init codex" in message


def test_format_empty_detection_error_reports_found_paths(monkeypatch, tmp_path) -> None:
    """When a binary IS present, the error still surfaces its path for debugging."""

    init_cli, _ = _load_init_module(monkeypatch)
    fake_claude = tmp_path / "claude"
    fake_claude.write_text("")
    monkeypatch.setattr(
        init_cli.shutil,
        "which",
        lambda name: str(fake_claude) if name == "claude" else None,
    )

    message = init_cli._format_empty_detection_error(global_scope=True)

    assert f"claude: found at {fake_claude}" in message
    assert "codex: not found" in message


def test_init_verbose_enables_debug_logging_on_stderr(monkeypatch) -> None:
    """``headroom init -v`` should emit diagnostic lines to stderr.

    Different Click 8.x versions expose stderr on ``CliRunner`` results
    differently (``mix_stderr`` was removed in 8.2, and ``result.stderr``
    appeared around the same time). To stay compatible with any Click 8.x
    the repo targets, the test reads ``result.stderr`` when the attribute
    exists AND contains data, otherwise falls back to ``result.output``
    (which is the combined stream when stderr isn't captured separately).
    """

    init_cli, fake_main = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli.shutil, "which", lambda name: None)
    runner = CliRunner()

    result = runner.invoke(fake_main, ["init", "-v", "-g"])

    # Newer Click: stderr captured separately.
    stderr = getattr(result, "stderr", None) or ""
    if not stderr:
        # Older Click: everything in result.output.
        stderr = result.output

    assert result.exit_code != 0, f"output: {result.output!r}"
    assert "[headroom init]" in stderr
    assert "detect_init_targets" in stderr
    assert "global_scope=True" in stderr
    for target in ("claude", "codebuddy", "codex", "copilot", "openclaw"):
        assert target in stderr


def test_init_verbose_is_idempotent(monkeypatch) -> None:
    """Calling _enable_verbose_logging repeatedly keeps one handler attached."""

    init_cli, _ = _load_init_module(monkeypatch)
    # Clear any prior handler state on the dedicated init logger.
    init_cli.logger.handlers.clear()
    if hasattr(init_cli.logger, init_cli._VERBOSE_HANDLER_ATTR):
        delattr(init_cli.logger, init_cli._VERBOSE_HANDLER_ATTR)

    init_cli._enable_verbose_logging()
    init_cli._enable_verbose_logging()
    init_cli._enable_verbose_logging()

    assert len(init_cli.logger.handlers) == 1


def test_init_copilot_requires_global(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()
    monkeypatch.setattr(init_cli, "_ensure_runtime_manifest", lambda **kwargs: "init-local-test")

    result = runner.invoke(fake_main, ["init", "copilot"])

    assert result.exit_code != 0
    assert "requires -g" in result.output


def test_init_claude_local_writes_settings_and_installs_marketplace(
    monkeypatch, tmp_path: Path
) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    marketplace_calls: list[str] = []
    monkeypatch.setattr(init_cli, "_ensure_runtime_manifest", lambda **kwargs: "init-local-demo")
    monkeypatch.setattr(
        init_cli,
        "_install_claude_marketplace",
        lambda scope: marketplace_calls.append(scope),
    )

    result = runner.invoke(fake_main, ["init", "claude"])

    assert result.exit_code == 0, result.output
    settings_path = tmp_path / ".claude" / "settings.local.json"
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert marketplace_calls == ["local"]
    assert any(
        "--profile init-local-demo" in hook["command"] and "init hook ensure" in hook["command"]
        for entry in payload["hooks"]["SessionStart"]
        for hook in entry["hooks"]
    )


def test_init_codex_merges_feature_flag_into_existing_table(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("[features]\nshell_tool = true\n", encoding="utf-8")

    init_cli._init_codex(global_scope=False, profile="init-local-demo", port=9000)

    content = config_path.read_text(encoding="utf-8")
    assert 'base_url = "http://127.0.0.1:9000/v1"' in content
    assert content.count("[features]") == 1
    assert "hooks = true" in content
    assert 'env_key = "OPENAI_API_KEY"' not in content
    hooks = json.loads((tmp_path / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    assert "--profile init-local-demo" in hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    assert "init hook ensure" in hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]


def test_init_codex_creates_hooks_feature_flag_on_first_init(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.chdir(tmp_path)

    init_cli._init_codex(global_scope=False, profile="init-local-demo", port=9000)

    content = (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert parsed["model_provider"] == "headroom"
    assert parsed["features"]["hooks"] is True
    assert "codex_hooks" not in content


def test_init_claude_uses_custom_port(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_cli, "_install_claude_marketplace", lambda scope: None)

    init_cli._init_claude(global_scope=False, profile="init-local-demo", port=9011)

    payload = json.loads((tmp_path / ".claude" / "settings.local.json").read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9011"


def test_init_copilot_global_writes_hooks_and_env(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    captured_env: dict[str, str] = {}
    monkeypatch.setattr(init_cli, "_copilot_config_path", lambda: tmp_path / "copilot-config.json")
    monkeypatch.setattr(init_cli, "_apply_user_env", lambda values: captured_env.update(values))
    monkeypatch.setattr(init_cli, "_install_copilot_marketplace", lambda: None)

    init_cli._init_copilot(global_scope=True, profile="init-user", port=9005, backend="openai")

    payload = json.loads((tmp_path / "copilot-config.json").read_text(encoding="utf-8"))
    assert "SessionStart" in payload["hooks"]
    assert "PreToolUse" in payload["hooks"]
    assert "--profile init-user" in payload["hooks"]["SessionStart"][0]["command"]
    assert captured_env == {
        "COPILOT_PROVIDER_TYPE": "openai",
        "COPILOT_PROVIDER_BASE_URL": "http://127.0.0.1:9005/v1",
        "COPILOT_PROVIDER_WIRE_API": "completions",
    }


def test_init_hook_ensure_prefers_local_profile(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    ensured: list[str] = []

    def fake_load(profile: str):
        return object() if profile == "init-repo-12345678" else None

    monkeypatch.setattr(init_cli, "_local_profile", lambda cwd=None: "init-repo-12345678")
    monkeypatch.setattr(init_cli, "load_manifest", fake_load)
    monkeypatch.setattr(
        init_cli, "_ensure_profile_running", lambda profile: ensured.append(profile)
    )

    runner = CliRunner()
    result = runner.invoke(fake_main, ["init", "hook", "ensure"])

    assert result.exit_code == 0, result.output
    assert ensured == ["init-repo-12345678"]


def test_init_openclaw_requires_global(monkeypatch) -> None:
    _, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(fake_main, ["init", "openclaw"])

    assert result.exit_code != 0
    assert "requires -g" in result.output


def test_init_openclaw_delegates_to_wrap(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    calls: list[list[str]] = []

    class _Result:
        returncode = 0

    monkeypatch.setattr(init_cli, "resolve_headroom_command", lambda: ["headroom"])
    monkeypatch.setattr(
        init_cli.subprocess,
        "run",
        lambda cmd: calls.append(cmd) or _Result(),
    )

    init_cli._init_openclaw(global_scope=True, port=9999)

    assert calls == [["headroom", "wrap", "openclaw", "--proxy-port", "9999"]]


def test_detect_init_targets_respects_scope(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(
        init_cli.shutil,
        "which",
        lambda name: (
            name if name in {"claude", "codebuddy", "copilot", "codex", "openclaw"} else None
        ),
    )

    assert init_cli.detect_init_targets(False) == ["claude", "codebuddy", "codex"]
    assert init_cli.detect_init_targets(True) == [
        "claude",
        "codebuddy",
        "copilot",
        "codex",
        "openclaw",
    ]


def test_marketplace_source_prefers_env_override(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setenv("HEADROOM_MARKETPLACE_SOURCE", "custom/source")

    assert init_cli._marketplace_source() == "custom/source"


def test_run_checked_treats_existing_install_as_success(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)

    class _Result:
        returncode = 1
        stderr = "plugin already exists"
        stdout = ""

    monkeypatch.setattr(init_cli.subprocess, "run", lambda *args, **kwargs: _Result())

    init_cli._run_checked(["claude", "plugin", "install"], action="claude plugin install")


def test_command_string_and_matcher_on_windows(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(init_cli.subprocess, "list2cmdline", lambda parts: "joined-command")

    assert init_cli._command_string(["headroom", "init"]) == "joined-command"
    assert init_cli._powershell_matcher() == "Bash|PowerShell"


def test_command_string_normalizes_backslashes_on_windows(monkeypatch) -> None:
    """Backslash paths must become forward slashes so Git Bash hooks work (#724)."""
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli, "os", SimpleNamespace(name="nt"))

    result = init_cli._command_string(
        ["C:\\Users\\user\\.local\\bin\\headroom.exe", "init", "hook", "ensure"]
    )
    assert "\\" not in result
    assert "C:/Users/user/.local/bin/headroom.exe" in result


def test_command_string_quotes_spaces_after_normalization(monkeypatch) -> None:
    """Paths with spaces must stay properly quoted after backslash normalization (#724)."""
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli, "os", SimpleNamespace(name="nt"))

    result = init_cli._command_string(
        ["C:\\Program Files\\headroom\\headroom.exe", "init", "hook", "ensure"]
    )
    assert "\\" not in result
    assert '"C:/Program Files/headroom/headroom.exe"' in result


def test_json_file_handles_missing_empty_and_non_mapping(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    missing = tmp_path / "missing.json"
    empty = tmp_path / "empty.json"
    array_payload = tmp_path / "payload.json"
    empty.write_text("   \n", encoding="utf-8")
    array_payload.write_text('["value"]\n', encoding="utf-8")

    assert init_cli._json_file(missing) == {}
    assert init_cli._json_file(empty) == {}
    assert init_cli._json_file(array_payload) == {}


def test_ensure_claude_hooks_rewrites_existing_entries(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "env": {"KEEP": "1"},
                "hooks": {
                    "SessionStart": [
                        "not-a-dict",
                        {"hooks": "not-a-list"},
                        {
                            "matcher": "startup|resume",
                            "hooks": [{"type": "command", "command": "echo keep-me"}],
                        },
                        {
                            "matcher": "startup|resume",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "headroom init hook ensure --marker headroom-init-claude",
                                }
                            ],
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(init_cli, "_hook_command", lambda *parts: "headroom init hook ensure")

    init_cli._ensure_claude_hooks(settings_path, "init-local-demo", 9001)

    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["env"] == {
        "KEEP": "1",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9001",
        "ENABLE_TOOL_SEARCH": "true",
    }
    session_entries = payload["hooks"]["SessionStart"]
    assert session_entries[0] == "not-a-dict"
    assert session_entries[1] == {"hooks": "not-a-list"}
    assert session_entries[2]["hooks"][0]["command"] == "echo keep-me"
    assert session_entries[-1]["hooks"][0]["command"].endswith("--marker headroom-init-claude")


def test_ensure_copilot_hooks_replaces_existing_marker(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    config_path = tmp_path / "copilot.json"
    config_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {"type": "command", "command": "echo keep"},
                        {
                            "type": "command",
                            "command": "headroom init hook ensure --marker headroom-init-copilot",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(init_cli, "_hook_command", lambda *parts: "headroom init hook ensure")

    init_cli._ensure_copilot_hooks(config_path, "init-user")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    commands = [entry["command"] for entry in payload["hooks"]["SessionStart"]]
    assert commands == ["echo keep", "headroom init hook ensure --marker headroom-init-copilot"]


def test_replace_marker_block_replaces_existing_block(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    content = "before\n# start\nold\n# end\nafter\n"

    replaced = init_cli._replace_marker_block(content, "# start", "# end", "# start\nnew\n# end")

    assert replaced == "before\n\nafter\n\n# start\nnew\n# end\n"


def test_ensure_codex_provider_replaces_existing_marker(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text(
        f"prefix\n{init_cli._CODEX_PROVIDER_MARKER_START}\nold = true\n{init_cli._CODEX_PROVIDER_MARKER_END}\n",
        encoding="utf-8",
    )

    init_cli._ensure_codex_provider(path, 9100)

    content = path.read_text(encoding="utf-8")
    assert content.count(init_cli._CODEX_PROVIDER_MARKER_START) == 1
    assert 'base_url = "http://127.0.0.1:9100/v1"' in content
    assert "old = true" not in content
    assert 'env_key = "OPENAI_API_KEY"' not in content


def test_ensure_codex_provider_keeps_root_keys_above_existing_table(
    monkeypatch, tmp_path: Path
) -> None:
    """#260: a config ending in a table must not capture the provider root keys.

    Appending the block after a trailing [features] table scoped model_provider
    under it, so Codex refused to start with
    'invalid type: string "headroom", expected a boolean in features'.
    """
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text("[features]\nhooks = true\n", encoding="utf-8")

    init_cli._ensure_codex_provider(path, 8787)

    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    # model_provider belongs at the document root, not under [features].
    assert parsed["model_provider"] == "headroom"
    assert "model_provider" not in parsed["features"]
    assert "openai_base_url" not in parsed["features"]
    # The user's existing table is preserved.
    assert parsed["features"]["hooks"] is True
    assert parsed["model_providers"]["headroom"]["base_url"] == "http://127.0.0.1:8787/v1"


def test_ensure_codex_provider_replaces_existing_model_provider(
    monkeypatch, tmp_path: Path
) -> None:
    """A pre-existing root model_provider is replaced, never duplicated (#260).

    A second top-level model_provider key would be invalid TOML; init owns it.
    """
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text('model_provider = "openai"\n[features]\nhooks = true\n', encoding="utf-8")

    init_cli._ensure_codex_provider(path, 8787)

    parsed = tomllib.loads(path.read_text(encoding="utf-8"))  # raises on a duplicate key
    assert parsed["model_provider"] == "headroom"
    assert parsed["features"]["hooks"] is True


def test_ensure_codex_provider_emits_requires_openai_auth_for_chatgpt(
    monkeypatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    (tmp_path / "auth.json").write_text('{"auth_mode": "chatgpt"}', encoding="utf-8")

    init_cli._ensure_codex_provider(path, 8787)

    assert "requires_openai_auth = true" in path.read_text(encoding="utf-8")


def test_ensure_codex_provider_omits_requires_openai_auth_for_api_key(
    monkeypatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    (tmp_path / "auth.json").write_text('{"auth_mode": "apikey"}', encoding="utf-8")

    init_cli._ensure_codex_provider(path, 8787)

    assert "requires_openai_auth" not in path.read_text(encoding="utf-8")


def test_ensure_codex_feature_flag_replaces_existing_marker(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text(
        f"[features]\n{init_cli._CODEX_FEATURE_MARKER_START}\ncodex_hooks = false\n{init_cli._CODEX_FEATURE_MARKER_END}\n",
        encoding="utf-8",
    )

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    assert content.count(init_cli._CODEX_FEATURE_MARKER_START) == 1
    assert "hooks = true" in content
    assert "codex_hooks" not in content


def test_ensure_codex_feature_flag_replaces_marker_inside_features_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text(
        "[features]\n"
        f"{init_cli._CODEX_FEATURE_MARKER_START}\n"
        "hooks = true\n"
        f"{init_cli._CODEX_FEATURE_MARKER_END}\n"
        "\n[tools]\nhooks = false\n",
        encoding="utf-8",
    )

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert parsed["features"]["hooks"] is True
    assert parsed["tools"]["hooks"] is False
    assert content.count(init_cli._CODEX_FEATURE_MARKER_START) == 1


def test_ensure_codex_feature_flag_migrates_legacy_codex_hooks_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text("[features]\ncodex_hooks = true\nshell_tool = true\n", encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert parsed["features"]["hooks"] is True
    assert parsed["features"]["shell_tool"] is True
    assert "codex_hooks" not in parsed["features"]
    assert content.count("hooks = true") == 1
    assert content.count(init_cli._CODEX_FEATURE_MARKER_START) == 1


def test_ensure_codex_feature_flag_migrates_dotted_legacy_codex_hooks_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text("features.codex_hooks = true\nfeatures.shell_tool = true\n", encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert parsed["features"]["hooks"] is True
    assert parsed["features"]["shell_tool"] is True
    assert "codex_hooks" not in parsed["features"]
    assert "features.hooks = true" in content


def test_ensure_codex_feature_flag_migrates_when_both_keys_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    # A config that carried both the legacy and the correct key must not produce
    # a duplicate `hooks` key (which Codex would reject as invalid TOML).
    path.write_text("[features]\ncodex_hooks = true\nhooks = false\n", encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert "codex_hooks" not in parsed["features"]
    # The user's explicit `hooks` value is respected; only the legacy key is removed.
    assert parsed["features"]["hooks"] is False


def test_ensure_codex_feature_flag_migrates_when_keys_reversed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text("[features]\nhooks = false\ncodex_hooks = true\n", encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert "codex_hooks" not in parsed["features"]
    assert parsed["features"]["hooks"] is False


def test_ensure_codex_feature_flag_ignores_hooks_outside_features(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text(
        "[features]\nshell_tool = true\n\n[some_other_table]\nhooks = true\n",
        encoding="utf-8",
    )

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert parsed["features"]["hooks"] is True
    assert parsed["features"]["shell_tool"] is True
    assert parsed["some_other_table"]["hooks"] is True
    assert content.count(init_cli._CODEX_FEATURE_MARKER_START) == 1


def test_ensure_codex_feature_flag_ignores_hooks_after_commented_table_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text(
        "[features]\nshell_tool = true\n\n[some_other_table] # comment\nhooks = true\n",
        encoding="utf-8",
    )

    init_cli._ensure_codex_feature_flag(path)

    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert parsed["features"]["hooks"] is True
    assert parsed["features"]["shell_tool"] is True
    assert parsed["some_other_table"]["hooks"] is True


def test_ensure_codex_feature_flag_respects_commented_features_header_and_quoted_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text('[features] # comment\n"hooks" = false\n', encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert parsed["features"]["hooks"] is False
    assert init_cli._CODEX_FEATURE_MARKER_START not in content


def test_ensure_codex_feature_flag_migrates_quoted_legacy_codex_hooks_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text('[features]\n"codex_hooks" = true\n', encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert parsed["features"]["hooks"] is True
    assert "codex_hooks" not in parsed["features"]


def test_ensure_codex_feature_flag_respects_root_dotted_feature_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text("features.hooks = false\n", encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert parsed["features"]["hooks"] is False
    assert init_cli._CODEX_FEATURE_MARKER_START not in content


def test_ensure_codex_feature_flag_preserves_legacy_key_outside_features(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text("[some_other_table]\ncodex_hooks = true\n", encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert parsed["features"]["hooks"] is True
    assert parsed["some_other_table"]["codex_hooks"] is True


def test_ensure_codex_feature_flag_drops_legacy_key_outside_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text(
        "[features]\n"
        "codex_hooks = true\n"
        f"{init_cli._CODEX_FEATURE_MARKER_START}\n"
        "hooks = true\n"
        f"{init_cli._CODEX_FEATURE_MARKER_END}\n",
        encoding="utf-8",
    )

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert "codex_hooks" not in parsed["features"]
    assert parsed["features"]["hooks"] is True
    assert content.count("hooks = true") == 1


def test_ensure_codex_feature_flag_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text("[features]\ncodex_hooks = true\n", encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)
    first = path.read_text(encoding="utf-8")
    init_cli._ensure_codex_feature_flag(path)
    second = path.read_text(encoding="utf-8")

    assert first == second
    parsed = tomllib.loads(second)
    assert parsed["features"]["hooks"] is True
    assert second.count("hooks = true") == 1


def test_ensure_codex_feature_flag_creates_features_section_when_missing(
    monkeypatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text('model = "gpt-5"\n', encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    assert "[features]" in content
    assert "hooks = true" in content


def test_manifest_changed_detects_differences(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    existing = SimpleNamespace(
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory_enabled=False,
    )

    assert not init_cli._manifest_changed(
        existing,
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )
    assert init_cli._manifest_changed(
        existing,
        port=9000,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )


def test_ensure_runtime_manifest_merges_targets_and_stops_changed_runtime(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    existing = SimpleNamespace(
        targets=["claude"],
        mutations=["mutation"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory_enabled=False,
    )
    saved: list[object] = []
    stopped: list[object] = []
    built = SimpleNamespace(supervisor_kind="", artifacts=[], mutations=[], targets=[])

    monkeypatch.setattr(init_cli, "_runtime_profile", lambda global_scope, cwd=None: "init-user")
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: existing)
    monkeypatch.setattr(
        init_cli,
        "build_manifest",
        lambda **kwargs: built.__dict__.update(kwargs) or built,
    )
    monkeypatch.setattr(init_cli, "save_manifest", lambda manifest: saved.append(manifest))
    monkeypatch.setattr(init_cli, "stop_runtime", lambda manifest: stopped.append(manifest))

    profile = init_cli._ensure_runtime_manifest(
        global_scope=True,
        targets=["codex"],
        port=9001,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )

    assert profile == "init-user"
    assert stopped == [existing]
    assert saved == [built]
    assert built.targets == ["claude", "codex"]
    assert built.mutations == ["mutation"]
    assert built.supervisor_kind == init_cli.SupervisorKind.NONE.value
    assert built.artifacts == []


def test_ensure_runtime_manifest_ignores_stop_runtime_errors(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    existing = SimpleNamespace(
        targets=[],
        mutations=[],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory_enabled=False,
    )
    saved: list[object] = []
    built = SimpleNamespace(supervisor_kind="", artifacts=[], mutations=[], targets=[])

    monkeypatch.setattr(init_cli, "_runtime_profile", lambda global_scope, cwd=None: "init-user")
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: existing)
    monkeypatch.setattr(
        init_cli,
        "build_manifest",
        lambda **kwargs: built.__dict__.update(kwargs) or built,
    )
    monkeypatch.setattr(init_cli, "save_manifest", lambda manifest: saved.append(manifest))
    monkeypatch.setattr(
        init_cli, "stop_runtime", lambda manifest: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    init_cli._ensure_runtime_manifest(
        global_scope=True,
        targets=["claude"],
        port=9001,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )

    assert saved == [built]


def test_apply_user_env_routes_by_platform(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = SimpleNamespace(base_env={"OLD": "1"}, tool_envs={})
    windows_calls: list[object] = []
    unix_calls: list[object] = []
    monkeypatch.setattr(init_cli, "_env_manifest", lambda values: manifest)
    monkeypatch.setattr(
        init_cli, "_apply_windows_env_scope", lambda value: windows_calls.append(value)
    )
    monkeypatch.setattr(init_cli, "_apply_unix_env_scope", lambda value: unix_calls.append(value))

    monkeypatch.setattr(init_cli, "os", SimpleNamespace(name="nt"))
    init_cli._apply_user_env({"COPILOT_PROVIDER_TYPE": "openai"})
    monkeypatch.setattr(init_cli, "os", SimpleNamespace(name="posix"))
    init_cli._apply_user_env({"COPILOT_PROVIDER_TYPE": "anthropic"})

    assert manifest.base_env == {}
    assert manifest.tool_envs == {"copilot": {"COPILOT_PROVIDER_TYPE": "anthropic"}}
    assert windows_calls == [manifest]
    assert unix_calls == [manifest]


def test_resolve_copilot_env_supports_anthropic(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)

    assert init_cli._resolve_copilot_env(9010, "anthropic") == {
        "COPILOT_PROVIDER_TYPE": "anthropic",
        "COPILOT_PROVIDER_BASE_URL": "http://127.0.0.1:9010",
    }


def test_marketplace_source_prefers_repo_checkout(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.delenv("HEADROOM_MARKETPLACE_SOURCE", raising=False)

    assert init_cli._marketplace_source() == str(Path(init_cli.__file__).resolve().parents[2])


def test_run_checked_raises_on_failure(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)

    class _Result:
        returncode = 2
        stderr = "bad stderr"
        stdout = "bad stdout"

    monkeypatch.setattr(init_cli.subprocess, "run", lambda *args, **kwargs: _Result())

    with pytest.raises(
        click.ClickException, match="claude plugin install failed: bad stderr\nbad stdout"
    ):
        init_cli._run_checked(["claude", "plugin", "install"], action="claude plugin install")


def test_install_claude_marketplace_errors_without_binary(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli.shutil, "which", lambda name: None)

    with pytest.raises(click.ClickException, match="'claude' not found"):
        init_cli._install_claude_marketplace("local")


def test_install_claude_marketplace_runs_expected_commands(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    calls: list[tuple[list[str], str]] = []
    monkeypatch.setattr(init_cli.shutil, "which", lambda name: "claude")
    monkeypatch.setattr(init_cli, "_marketplace_source", lambda: "repo/source")
    monkeypatch.setattr(
        init_cli, "_run_checked", lambda command, action: calls.append((command, action))
    )

    init_cli._install_claude_marketplace("user")

    assert calls == [
        (["claude", "plugin", "marketplace", "add", "repo/source"], "claude marketplace add"),
        (
            ["claude", "plugin", "install", "headroom@headroom-marketplace", "--scope", "user"],
            "claude plugin install",
        ),
    ]


def test_install_copilot_marketplace_handles_missing_binary(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli.shutil, "which", lambda name: None)

    with pytest.raises(click.ClickException, match="'copilot' not found"):
        init_cli._install_copilot_marketplace()


def test_install_copilot_marketplace_runs_expected_commands(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    calls: list[tuple[list[str], str]] = []
    monkeypatch.setattr(init_cli.shutil, "which", lambda name: "copilot")
    monkeypatch.setattr(init_cli, "_marketplace_source", lambda: "repo/source")
    monkeypatch.setattr(
        init_cli, "_run_checked", lambda command, action: calls.append((command, action))
    )

    init_cli._install_copilot_marketplace()

    assert calls == [
        (["copilot", "plugin", "marketplace", "add", "repo/source"], "copilot marketplace add"),
        (
            ["copilot", "plugin", "install", "headroom@headroom-marketplace"],
            "copilot plugin install",
        ),
    ]


def test_ensure_profile_running_covers_runtime_modes(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    docker_manifest = SimpleNamespace(
        preset=init_cli.InstallPreset.PERSISTENT_DOCKER.value,
        supervisor_kind=init_cli.SupervisorKind.NONE.value,
        profile="docker-profile",
    )
    service_manifest = SimpleNamespace(
        preset=init_cli.InstallPreset.PERSISTENT_TASK.value,
        supervisor_kind=init_cli.SupervisorKind.SERVICE.value,
        profile="service-profile",
    )
    task_manifest = SimpleNamespace(
        preset=init_cli.InstallPreset.PERSISTENT_TASK.value,
        supervisor_kind=init_cli.SupervisorKind.NONE.value,
        profile="task-profile",
    )
    manifests = {
        "docker-profile": docker_manifest,
        "service-profile": service_manifest,
        "task-profile": task_manifest,
    }
    docker_calls: list[object] = []
    service_calls: list[object] = []
    detached_calls: list[str] = []
    wait_calls: list[tuple[str, int]] = []

    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifests.get(profile))
    monkeypatch.setattr(init_cli, "runtime_status", lambda manifest: "stopped")

    @contextmanager
    def fake_start_lock(profile: str):
        yield True

    monkeypatch.setattr(init_cli, "acquire_runtime_start_lock", fake_start_lock)

    def fake_wait_ready(manifest, timeout_seconds: int) -> bool:
        wait_calls.append((manifest.profile, timeout_seconds))
        return False

    monkeypatch.setattr(init_cli, "wait_ready", fake_wait_ready)
    monkeypatch.setattr(
        init_cli, "start_persistent_docker", lambda manifest: docker_calls.append(manifest)
    )
    monkeypatch.setattr(
        init_cli, "start_supervisor", lambda manifest: service_calls.append(manifest)
    )
    monkeypatch.setattr(
        init_cli,
        "start_detached_agent",
        lambda profile: detached_calls.append(profile),
    )

    init_cli._ensure_profile_running("missing")
    init_cli._ensure_profile_running("docker-profile")
    init_cli._ensure_profile_running("service-profile")
    init_cli._ensure_profile_running("task-profile")

    assert docker_calls == [docker_manifest]
    assert service_calls == [service_manifest]
    assert detached_calls == ["task-profile"]
    assert ("docker-profile", 1) in wait_calls
    assert ("docker-profile", 45) in wait_calls


def test_ensure_profile_running_suppresses_hook_recovery_output(monkeypatch, capfd) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = SimpleNamespace(
        preset=init_cli.InstallPreset.PERSISTENT_TASK.value,
        supervisor_kind=init_cli.SupervisorKind.SERVICE.value,
        profile="service-profile",
    )

    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli, "wait_ready", lambda manifest, timeout_seconds: False)

    def noisy_start_supervisor(manifest) -> None:
        print("python stdout")
        print("python stderr", file=sys.stderr)
        os.write(1, b"fd stdout\n")
        os.write(2, b"fd stderr\n")
        raise RuntimeError("not permitted")

    monkeypatch.setattr(init_cli, "start_supervisor", noisy_start_supervisor)

    init_cli._ensure_profile_running("service-profile")

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_ensure_profile_running_returns_when_ready_or_on_exception(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = SimpleNamespace(
        preset=init_cli.InstallPreset.PERSISTENT_TASK.value,
        supervisor_kind=init_cli.SupervisorKind.NONE.value,
        profile="task-profile",
    )
    detached_calls: list[str] = []
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli, "wait_ready", lambda manifest, timeout_seconds: True)
    monkeypatch.setattr(
        init_cli,
        "start_detached_agent",
        lambda profile: detached_calls.append(profile),
    )

    init_cli._ensure_profile_running("task-profile")
    assert detached_calls == []

    @contextmanager
    def fake_start_lock(profile: str):
        yield True

    monkeypatch.setattr(init_cli, "acquire_runtime_start_lock", fake_start_lock)
    monkeypatch.setattr(init_cli, "runtime_status", lambda manifest: "stopped")
    monkeypatch.setattr(init_cli, "wait_ready", lambda manifest, timeout_seconds: False)
    monkeypatch.setattr(
        init_cli,
        "start_detached_agent",
        lambda profile: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    init_cli._ensure_profile_running("task-profile")


def test_ensure_profile_running_skips_spawn_when_start_lock_is_held(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = SimpleNamespace(
        preset=init_cli.InstallPreset.PERSISTENT_TASK.value,
        supervisor_kind=init_cli.SupervisorKind.NONE.value,
        profile="task-profile",
    )
    detached_calls: list[str] = []

    @contextmanager
    def fake_start_lock(profile: str):
        yield False

    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli, "wait_ready", lambda manifest, timeout_seconds: False)
    monkeypatch.setattr(init_cli, "acquire_runtime_start_lock", fake_start_lock)
    monkeypatch.setattr(
        init_cli,
        "start_detached_agent",
        lambda profile: detached_calls.append(profile),
    )

    init_cli._ensure_profile_running("task-profile")

    assert detached_calls == []


def test_ensure_profile_running_does_not_spawn_again_during_slow_startup(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = SimpleNamespace(
        preset=init_cli.InstallPreset.PERSISTENT_TASK.value,
        supervisor_kind=init_cli.SupervisorKind.NONE.value,
        profile="task-profile",
    )
    detached_calls: list[str] = []
    wait_calls: list[int] = []
    stop_calls: list[object] = []

    @contextmanager
    def fake_start_lock(profile: str):
        yield True

    def fake_wait_ready(manifest, timeout_seconds: int) -> bool:
        wait_calls.append(timeout_seconds)
        return bool(detached_calls and timeout_seconds == init_cli._STARTUP_READY_TIMEOUT_SECONDS)

    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli, "wait_ready", fake_wait_ready)
    monkeypatch.setattr(init_cli, "acquire_runtime_start_lock", fake_start_lock)
    monkeypatch.setattr(
        init_cli,
        "runtime_status",
        lambda manifest: "running" if detached_calls else "stopped",
    )
    monkeypatch.setattr(
        init_cli,
        "start_detached_agent",
        lambda profile: detached_calls.append(profile),
    )
    monkeypatch.setattr(init_cli, "stop_runtime", lambda manifest: stop_calls.append(manifest))

    init_cli._ensure_profile_running("task-profile")
    init_cli._ensure_profile_running("task-profile")

    assert detached_calls == ["task-profile"]
    assert init_cli._STARTUP_READY_TIMEOUT_SECONDS in wait_calls
    assert stop_calls == []


def test_init_codex_windows_warns_about_upstream_hook_limitation(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    messages: list[str] = []
    monkeypatch.setattr(init_cli, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(init_cli, "_codex_scope_path", lambda global_scope: Path("config.toml"))
    monkeypatch.setattr(init_cli, "_codex_hooks_path", lambda global_scope: Path("hooks.json"))
    monkeypatch.setattr(init_cli, "_ensure_codex_provider", lambda path, port: None)
    monkeypatch.setattr(init_cli, "_ensure_codex_feature_flag", lambda path: None)
    monkeypatch.setattr(init_cli, "_ensure_codex_hooks", lambda path, profile: None)
    monkeypatch.setattr(init_cli.click, "echo", lambda message: messages.append(message))

    init_cli._init_codex(global_scope=True, profile="init-user", port=9000)

    assert any("disabled upstream on Windows" in message for message in messages)


def test_init_openclaw_propagates_nonzero_exit(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)

    class _Result:
        returncode = 9

    monkeypatch.setattr(init_cli, "resolve_headroom_command", lambda: ["headroom"])
    monkeypatch.setattr(init_cli.subprocess, "run", lambda command: _Result())

    with pytest.raises(SystemExit) as exc:
        init_cli._init_openclaw(global_scope=True, port=9999)

    assert exc.value.code == 9


def test_run_init_targets_dispatches_supported_targets(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    calls: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(init_cli, "_ensure_runtime_manifest", lambda **kwargs: "init-profile")
    monkeypatch.setattr(
        init_cli,
        "_init_claude",
        lambda **kwargs: calls.append(
            ("claude", (kwargs["global_scope"], kwargs["profile"], kwargs["port"]))
        ),
    )
    monkeypatch.setattr(
        init_cli,
        "_init_codebuddy",
        lambda **kwargs: calls.append(
            ("codebuddy", (kwargs["global_scope"], kwargs["profile"], kwargs["port"]))
        ),
    )
    monkeypatch.setattr(
        init_cli,
        "_init_copilot",
        lambda **kwargs: calls.append(
            ("copilot", (kwargs["global_scope"], kwargs["profile"], kwargs["port"]))
        ),
    )
    monkeypatch.setattr(
        init_cli,
        "_init_codex",
        lambda **kwargs: calls.append(
            ("codex", (kwargs["global_scope"], kwargs["profile"], kwargs["port"]))
        ),
    )
    monkeypatch.setattr(
        init_cli,
        "_init_openclaw",
        lambda **kwargs: calls.append(("openclaw", (kwargs["global_scope"], kwargs["port"]))),
    )

    init_cli._run_init_targets(
        targets=["claude", "codebuddy", "copilot", "codex", "openclaw"],
        global_scope=True,
        port=9000,
        backend="openai",
        anyllm_provider="provider",
        region="us-east-1",
        memory=True,
    )

    assert calls == [
        ("claude", (True, "init-profile", 9000)),
        ("codebuddy", (True, "init-profile", 9000)),
        ("copilot", (True, "init-profile", 9000)),
        ("codex", (True, "init-profile", 9000)),
        ("openclaw", (True, 9000)),
    ]


def test_init_subcommand_uses_group_options(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()
    captured: dict[str, object] = {}
    monkeypatch.setattr(init_cli, "_run_init_targets", lambda **kwargs: captured.update(kwargs))

    result = runner.invoke(
        fake_main,
        ["init", "-g", "--port", "9007", "--backend", "openai", "--memory", "claude"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "targets": ["claude"],
        "global_scope": True,
        "port": 9007,
        "backend": "openai",
        "anyllm_provider": None,
        "region": None,
        "memory": True,
    }


def test_init_hook_ensure_prefers_global_when_local_missing(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    ensured: list[str] = []
    monkeypatch.setattr(init_cli, "_local_profile", lambda cwd=None: "init-repo-12345678")
    monkeypatch.setattr(
        init_cli,
        "load_manifest",
        lambda profile: object() if profile == init_cli._GLOBAL_PROFILE else None,
    )
    monkeypatch.setattr(
        init_cli, "_ensure_profile_running", lambda profile: ensured.append(profile)
    )

    runner = CliRunner()
    result = runner.invoke(fake_main, ["init", "hook", "ensure"])

    assert result.exit_code == 0, result.output
    assert ensured == [init_cli._GLOBAL_PROFILE]


def test_init_hook_ensure_uses_explicit_profile(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    ensured: list[str] = []
    monkeypatch.setattr(
        init_cli, "_ensure_profile_running", lambda profile: ensured.append(profile)
    )

    runner = CliRunner()
    result = runner.invoke(fake_main, ["init", "hook", "ensure", "--profile", "init-explicit"])

    assert result.exit_code == 0, result.output
    assert ensured == ["init-explicit"]


# ---------------------------------------------------------------------------
# Bug 3 (#406): _ensure_codex_provider must inject openai_base_url
# ---------------------------------------------------------------------------


def test_init_codex_writes_openai_base_url(monkeypatch, tmp_path: Path) -> None:
    """_ensure_codex_provider must write openai_base_url at the top level so that
    subscription (ChatGPT plan) users are routed through headroom even when the
    init entry point is used instead of wrap."""
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / ".codex" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)

    init_cli._ensure_codex_provider(path, 8787)

    content = path.read_text(encoding="utf-8")
    assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in content, (
        f"openai_base_url missing from init codex config:\n{content}"
    )
    # Must NOT appear inside a [section] block.
    lines = content.splitlines()
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_section = True
        if in_section and stripped.startswith("openai_base_url"):
            raise AssertionError(
                f"openai_base_url appeared inside a section block in init output:\n{content}"
            )
    # Bug 3 regression guard.
    assert "requires_openai_auth" not in content, (
        f"requires_openai_auth must not appear in init codex config:\n{content}"
    )


def test_init_codex_provider_retags_existing_threads(monkeypatch, tmp_path: Path) -> None:
    """`headroom init` injects `model_provider = "headroom"` for Codex, which
    Codex Desktop filters its history menu by. Without retagging, existing native
    `openai` threads vanish from the sidebar/search (#961). `_ensure_codex_provider`
    must retag existing threads openai->headroom so the history stays visible —
    the same reconciliation the install and wrap paths already perform."""
    import sqlite3

    init_cli, _ = _load_init_module(monkeypatch)

    codex_home = tmp_path / ".codex"
    config_path = codex_home / "config.toml"
    # Codex Desktop reads <codex_home>/sqlite/state_5.sqlite.
    db = codex_home / "sqlite" / "state_5.sqlite"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, model_provider TEXT NOT NULL)")
        conn.executemany(
            "INSERT INTO threads (id, model_provider) VALUES (?, ?)",
            [("t1", "openai"), ("t2", "openai"), ("t3", "anthropic")],
        )
        conn.commit()
    finally:
        conn.close()

    init_cli._ensure_codex_provider(config_path, 8787)

    conn = sqlite3.connect(str(db))
    try:
        counts = dict(
            conn.execute("SELECT model_provider, COUNT(*) FROM threads GROUP BY model_provider")
        )
    finally:
        conn.close()
    # Native threads now live under the active headroom provider (stay visible);
    # third-party providers are left untouched.
    assert counts.get("headroom") == 2, f"existing openai threads not retagged: {counts}"
    assert counts.get("openai", 0) == 0, f"openai threads still hidden: {counts}"
    assert counts.get("anthropic") == 1, f"third-party provider must be left alone: {counts}"


def test_init_codex_strip_removes_openai_base_url(monkeypatch, tmp_path: Path) -> None:
    """_strip_codex_init_block must remove both the managed block and any orphaned
    openai_base_url lines left by a crashed or partial init."""
    init_cli, _ = _load_init_module(monkeypatch)

    # Normal install-then-strip cycle.
    path = tmp_path / "config.toml"
    path.write_text('model = "gpt-4o"\n', encoding="utf-8")
    init_cli._ensure_codex_provider(path, 8787)
    assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in path.read_text(encoding="utf-8")

    stripped = init_cli._strip_codex_init_block(path.read_text(encoding="utf-8"))
    assert "openai_base_url" not in stripped, (
        f"_strip_codex_init_block must remove openai_base_url after install:\n{stripped}"
    )
    assert "requires_openai_auth" not in stripped
    assert 'model = "gpt-4o"' in stripped

    # Orphan-cleanup path: openai_base_url left outside marker block.
    orphan_content = (
        'model = "gpt-4o"\n'
        'openai_base_url = "http://127.0.0.1:8787/v1"\n'
        'model_provider = "headroom"\n'
    )
    orphan_stripped = init_cli._strip_codex_init_block(orphan_content)
    assert "openai_base_url" not in orphan_stripped, (
        f"_strip_codex_init_block must remove orphaned openai_base_url:\n{orphan_stripped}"
    )
    assert 'model = "gpt-4o"' in orphan_stripped


# ---------------------------------------------------------------------------
# CodeBuddy hooks tests
# ---------------------------------------------------------------------------


def test_init_codebuddy_hooks_sets_env_with_v2(monkeypatch, tmp_path: Path) -> None:
    """_ensure_codebuddy_hooks must set CODEBUDDY_BASE_URL with /v2 prefix."""
    init_cli, _ = _load_init_module(monkeypatch)
    settings = tmp_path / ".codebuddy" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text("{}", encoding="utf-8")

    init_cli._ensure_codebuddy_hooks(settings, "test-profile", 8787)

    data = json.loads(settings.read_text(encoding="utf-8"))
    assert data["env"]["CODEBUDDY_BASE_URL"] == "http://127.0.0.1:8787/v2"


def test_init_codebuddy_hooks_adds_session_start_hook(monkeypatch, tmp_path: Path) -> None:
    """_ensure_codebuddy_hooks must add SessionStart and PreToolUse hooks."""
    init_cli, _ = _load_init_module(monkeypatch)
    settings = tmp_path / ".codebuddy" / "settings.json"
    settings.parent.mkdir(parents=True, exist_ok=True)
    settings.write_text("{}", encoding="utf-8")

    init_cli._ensure_codebuddy_hooks(settings, "test-profile", 8787)

    data = json.loads(settings.read_text(encoding="utf-8"))
    assert "hooks" in data
    assert "SessionStart" in data["hooks"]
    assert "PreToolUse" in data["hooks"]

    # Verify hooks contain the headroom marker
    session_hooks = data["hooks"]["SessionStart"]
    assert any(
        isinstance(entry, dict)
        and isinstance(entry.get("hooks"), list)
        and any(
            isinstance(h, dict) and "headroom-init-codebuddy" in str(h.get("command", ""))
            for h in entry["hooks"]
        )
        for entry in session_hooks
    )
