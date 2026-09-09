"""Tests for ``headroom init`` (Claude Code only)."""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

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

    monkeypatch.setattr(init_cli, "detect_init_targets", lambda global_scope: ["claude"])
    monkeypatch.setattr(init_cli, "_run_init_targets", lambda **kwargs: captured.update(kwargs))

    result = runner.invoke(fake_main, ["init", "-g"])

    assert result.exit_code == 0, result.output
    assert captured["targets"] == ["claude"]
    assert captured["global_scope"] is True


def test_init_fails_when_auto_detection_empty(monkeypatch) -> None:
    """Bare ``headroom init`` with no agents on PATH prints a guided error."""

    init_cli, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()
    monkeypatch.setattr(init_cli.shutil, "which", lambda name: None)

    result = runner.invoke(fake_main, ["init", "-g"])

    assert result.exit_code != 0
    assert "No supported user-scope agents were found on PATH" in result.output
    assert "probed the following agents" in result.output
    assert "claude: not found" in result.output
    assert "-g" in result.output
    assert "headroom init -g claude" in result.output


def test_format_empty_detection_error_reports_found_paths(monkeypatch, tmp_path) -> None:
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


def test_init_verbose_enables_debug_logging_on_stderr(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli.shutil, "which", lambda name: None)
    runner = CliRunner()

    result = runner.invoke(fake_main, ["init", "-v", "-g"])

    stderr = getattr(result, "stderr", None) or ""
    if not stderr:
        stderr = result.output

    assert result.exit_code != 0, f"output: {result.output!r}"
    assert "[headroom init]" in stderr
    assert "detect_init_targets" in stderr
    assert "global_scope=True" in stderr
    assert "claude" in stderr


def test_init_verbose_is_idempotent(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    init_cli.logger.handlers.clear()
    if hasattr(init_cli.logger, init_cli._VERBOSE_HANDLER_ATTR):
        delattr(init_cli.logger, init_cli._VERBOSE_HANDLER_ATTR)

    init_cli._enable_verbose_logging()
    init_cli._enable_verbose_logging()
    init_cli._enable_verbose_logging()

    assert len(init_cli.logger.handlers) == 1


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


def test_init_claude_uses_custom_port(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_cli, "_install_claude_marketplace", lambda scope: None)

    init_cli._init_claude(global_scope=False, profile="init-local-demo", port=9011)

    payload = json.loads((tmp_path / ".claude" / "settings.local.json").read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9011"


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


def test_detect_init_targets_only_knows_claude(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli.shutil, "which", lambda name: f"/bin/{name}")

    assert init_cli.detect_init_targets(global_scope=True) == ["claude"]
    assert init_cli.detect_init_targets(global_scope=False) == ["claude"]


def test_marketplace_source_prefers_env_override(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setenv("HEADROOM_MARKETPLACE_SOURCE", "custom/source")

    assert init_cli._marketplace_source() == "custom/source"


def test_marketplace_source_prefers_repo_checkout(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.delenv("HEADROOM_MARKETPLACE_SOURCE", raising=False)

    assert init_cli._marketplace_source() == str(Path(init_cli.__file__).resolve().parents[2])


def test_run_checked_treats_existing_install_as_success(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)

    class _Result:
        returncode = 1
        stderr = "plugin already exists"
        stdout = ""

    monkeypatch.setattr(init_cli.subprocess, "run", lambda *args, **kwargs: _Result())

    init_cli._run_checked(["claude", "plugin", "install"], action="claude plugin install")


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


def test_command_string_and_matcher_on_windows(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(init_cli.subprocess, "list2cmdline", lambda parts: "joined-command")

    assert init_cli._command_string(["headroom", "init"]) == "joined-command"
    assert init_cli._powershell_matcher() == "Bash|PowerShell"


def test_command_string_normalizes_backslashes_on_windows(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli, "os", SimpleNamespace(name="nt"))

    result = init_cli._command_string(
        ["C:\\Users\\user\\.local\\bin\\headroom.exe", "init", "hook", "ensure"]
    )
    assert "\\" not in result
    assert "C:/Users/user/.local/bin/headroom.exe" in result


def test_command_string_quotes_spaces_after_normalization(monkeypatch) -> None:
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


def test_json_file_rejects_malformed_json(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    malformed = tmp_path / "settings.json"
    malformed.write_text('{"env": {"A": "B",}}\n', encoding="utf-8")

    with pytest.raises(click.ClickException, match="invalid JSON"):
        init_cli._json_file(malformed)

    assert malformed.read_text(encoding="utf-8") == '{"env": {"A": "B",}}\n'


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


def test_manifest_changed_detects_differences(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    existing = SimpleNamespace(port=8787, memory_enabled=False)

    assert not init_cli._manifest_changed(existing, port=8787, memory=False)
    assert init_cli._manifest_changed(existing, port=9000, memory=False)
    assert init_cli._manifest_changed(existing, port=8787, memory=True)


def test_ensure_runtime_manifest_merges_targets_and_stops_changed_runtime(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    existing = SimpleNamespace(
        targets=["claude"],
        mutations=["mutation"],
        port=8787,
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
        targets=["claude"],
        port=9001,
        memory=False,
    )

    assert profile == "init-user"
    assert stopped == [existing]
    assert saved == [built]
    assert built.targets == ["claude"]
    assert built.mutations == ["mutation"]
    assert built.backend == "anthropic"
    assert built.telemetry_enabled is False
    assert built.supervisor_kind == init_cli.SupervisorKind.NONE.value
    assert built.artifacts == []


def test_ensure_runtime_manifest_ignores_stop_runtime_errors(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    existing = SimpleNamespace(targets=[], mutations=[], port=8787, memory_enabled=False)
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

    init_cli._ensure_runtime_manifest(global_scope=True, targets=["claude"], port=9001, memory=False)

    assert saved == [built]


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


def test_run_init_targets_dispatches_claude(monkeypatch) -> None:
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
    monkeypatch.setattr(init_cli, "_install_headroom_mcp_for_targets", lambda **kwargs: None)

    init_cli._run_init_targets(targets=["claude"], global_scope=True, port=9000, memory=True)

    assert calls == [("claude", (True, "init-profile", 9000))]


def test_init_subcommand_uses_group_options(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()
    captured: dict[str, object] = {}
    monkeypatch.setattr(init_cli, "_run_init_targets", lambda **kwargs: captured.update(kwargs))

    result = runner.invoke(fake_main, ["init", "-g", "--port", "9007", "--memory", "claude"])

    assert result.exit_code == 0, result.output
    assert captured == {
        "targets": ["claude"],
        "global_scope": True,
        "port": 9007,
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
