"""Claude wrap Vertex upstream handoff tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main
from headroom.providers.registry import DEFAULT_VERTEX_API_URL


class _Completed:
    returncode = 0


class _FakeProxyProcess:
    returncode = None

    def poll(self) -> None:
        return None

    def kill(self) -> None:
        return None


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _clear_claude_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_VERTEX_BASE_URL",
        "ANTHROPIC_FOUNDRY_BASE_URL",
        "ANTHROPIC_FOUNDRY_RESOURCE",
        "HEADROOM_CLAUDE_PROJECT_SETTINGS",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "VERTEX_TARGET_API_URL",
        # Issue #1779: these put Claude Code on a non-subscription auth path, so
        # the Remote Control gate warning must not fire. Clear them so the
        # plain-mode RC-warning assertion is deterministic regardless of the
        # ambient environment the test runs in.
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        # The RC sibling note reflects the resolved ENABLE_TOOL_SEARCH mode;
        # clear any ambient value so the default-session assertions hold.
        "ENABLE_TOOL_SEARCH",
    ):
        monkeypatch.delenv(key, raising=False)


def _invoke_wrap_claude(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    *,
    env: dict[str, str],
    extra_args: tuple[str, ...] = (),
) -> tuple[dict[str, Any], str]:
    captured: dict[str, Any] = {}

    _clear_claude_mode_env(monkeypatch)
    monkeypatch.setattr(wrap_mod.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr(wrap_mod, "_register_proxy_client", lambda _port: None)
    monkeypatch.setattr(wrap_mod, "_make_cleanup", lambda _holder, _port: lambda: None)
    monkeypatch.setattr(wrap_mod.signal, "signal", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wrap_mod, "_push_runtime_env", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wrap_mod, "_setup_coding_compressor", lambda *_args, **_kwargs: None)

    def fake_write_base_url(*args: object, **kwargs: object) -> None:
        captured["write_base_url_args"] = args
        captured["write_base_url_kwargs"] = kwargs

    monkeypatch.setattr(wrap_mod, "_write_claude_wrap_base_url", fake_write_base_url)
    monkeypatch.setattr(wrap_mod, "_restore_claude_wrap_base_url", lambda *_args, **_kwargs: None)

    def fake_write_tool_search(value: str, **kwargs: object) -> None:
        captured["write_tool_search_value"] = value
        captured["write_tool_search_kwargs"] = kwargs

    monkeypatch.setattr(wrap_mod, "_write_claude_wrap_tool_search", fake_write_tool_search)
    monkeypatch.setattr(wrap_mod, "_restore_claude_wrap_tool_search", lambda *_a, **_k: None)
    monkeypatch.setattr(wrap_mod, "_print_telemetry_notice", lambda: None)

    def fake_ensure_proxy(*args: object, **kwargs: object) -> tuple[None, int]:
        captured["ensure_args"] = args
        captured["ensure_kwargs"] = kwargs
        return None, args[0] if args else 8787

    def fake_run(cmd: list[str], *, env: dict[str, str]) -> _Completed:
        captured["child_cmd"] = cmd
        captured["child_env"] = env
        return _Completed()

    monkeypatch.setattr(wrap_mod, "_ensure_proxy", fake_ensure_proxy)
    # Issue #1779: pin the detected Claude Code version to the gated release so
    # the plain-mode RC warning is deterministic without shelling out to a real
    # `claude --version` (which would otherwise hit the child-launch fake_run).
    monkeypatch.setattr(wrap_mod, "detect_claude_code_version", lambda *_a, **_k: (2, 1, 196))
    monkeypatch.setattr(wrap_mod.subprocess, "run", fake_run)

    result = runner.invoke(
        main,
        [
            "wrap",
            "claude",
            "--no-mcp",
            "--no-tokensave",
            "--no-serena",
            *extra_args,
        ],
        env=env,
    )

    assert result.exit_code == 0, result.output
    return captured, result.output


def test_wrap_claude_plain_mode_warns_about_remote_control_gate(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured, output = _invoke_wrap_claude(runner, monkeypatch, env={})

    assert captured["child_cmd"] == ["/usr/bin/claude"]
    assert "Remote Control" in output
    assert "wrapped Claude session's ANTHROPIC_BASE_URL" in output
    # Issue #1779: the warning is accurate (deterministic, names /rc) and
    # co-reports the sibling base-URL gates (#746 / #1158).
    assert "2.1.196" in output
    assert "/rc" in output
    assert "may hide" not in output
    assert "#746" in output and "#1158" in output


def test_wrap_claude_plain_mode_api_key_auth_skips_remote_control_warning(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Issue #1779: an API-key (PAYG) session never had Remote Control, so the
    # gate warning must not fire even in plain proxy mode.
    _captured, output = _invoke_wrap_claude(
        runner, monkeypatch, env={"ANTHROPIC_API_KEY": "sk-ant-api-xxx"}
    )
    assert "Remote Control" not in output


def test_wrap_claude_rejects_conflicting_auth_before_proxy_mutation(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    user_settings = tmp_path / "user-settings.json"
    user_settings.write_text('{"env":{"ANTHROPIC_AUTH_TOKEN":"token-value"}}', encoding="utf-8")
    monkeypatch.setattr(wrap_mod, "claude_user_settings_path", lambda: user_settings)
    monkeypatch.setattr(wrap_mod.shutil, "which", lambda _name: "/usr/bin/claude")
    proxy_calls: list[int] = []
    monkeypatch.setattr(wrap_mod, "_register_proxy_client", lambda port: proxy_calls.append(port))

    result = runner.invoke(
        main,
        ["wrap", "claude", "--no-mcp", "--no-tokensave", "--no-serena"],
        env={"ANTHROPIC_API_KEY": "api-value"},
    )

    assert result.exit_code != 0
    assert "both ANTHROPIC_API_KEY" in result.output
    assert "shell environment" in result.output
    assert str(user_settings) in result.output
    assert "api-value" not in result.output
    assert "token-value" not in result.output
    assert proxy_calls == []


def test_wrap_claude_includes_shared_project_settings_in_auth_precedence(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    user_settings = tmp_path / "user-settings.json"
    user_settings.write_text("{}", encoding="utf-8")
    project_dir = tmp_path / ".claude"
    project_dir.mkdir()
    shared_settings = project_dir / "settings.json"
    shared_settings.write_text('{"env":{"ANTHROPIC_AUTH_TOKEN":"token-value"}}', encoding="utf-8")
    monkeypatch.setattr(wrap_mod, "claude_user_settings_path", lambda: user_settings)
    monkeypatch.setattr(wrap_mod.shutil, "which", lambda _name: "/usr/bin/claude")

    result = runner.invoke(
        main,
        ["wrap", "claude", "--no-mcp", "--no-tokensave", "--no-serena"],
        env={"ANTHROPIC_API_KEY": "api-value"},
    )

    assert result.exit_code != 0
    assert str(shared_settings) in result.output
    assert "api-value" not in result.output
    assert "token-value" not in result.output


def test_wrap_claude_sibling_note_accurate_under_1m_and_tool_search_optouts(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Issue #1779 accuracy under opt-ins: with --1m the note must not advise
    # adding --1m again, and with --tool-search false it must not claim
    # deferral is kept on — nor may the #746 banner line say "kept on".
    _captured, output = _invoke_wrap_claude(
        runner,
        monkeypatch,
        env={},
        extra_args=("--1m", "--tool-search", "false"),
    )
    assert "already restored via --1m" in output
    assert "restore with `headroom wrap claude --1m`" not in output
    assert "OFF for this session" in output
    assert "DISABLED per your setting" in output
    assert "kept on" not in output


def test_wrap_claude_1m_adds_suffix_to_passthrough_model_flag(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    # #2915: Claude Code's --model CLI flag outranks ANTHROPIC_MODEL, so with
    # both --1m and an explicit --model the env-var [1m] suffix is shadowed and
    # the window silently caps at 200k. The wrapper must add the suffix to the
    # pass-through flag so the 1M window actually activates.
    captured, output = _invoke_wrap_claude(
        runner,
        monkeypatch,
        env={},
        extra_args=("--1m", "--model", "opusplan"),
    )
    assert captured["child_cmd"] == ["/usr/bin/claude", "--model", "opusplan[1m]"]
    # The banner reports what actually takes effect, not the shadowed env value.
    assert "--model opusplan[1m]" in output


def test_wrap_claude_1m_adds_suffix_to_equals_model_flag(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured, _output = _invoke_wrap_claude(
        runner,
        monkeypatch,
        env={},
        extra_args=("--1m", "--model=opusplan"),
    )
    assert captured["child_cmd"] == ["/usr/bin/claude", "--model=opusplan[1m]"]


def test_wrap_claude_1m_without_model_flag_still_uses_env(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    # No pass-through --model: ANTHROPIC_MODEL carries the suffix as before, and
    # the launched command is untouched.
    captured, _output = _invoke_wrap_claude(runner, monkeypatch, env={}, extra_args=("--1m",))
    assert captured["child_cmd"] == ["/usr/bin/claude"]
    assert captured["child_env"]["ANTHROPIC_MODEL"].endswith("[1m]")


def test_wrap_claude_tool_search_banner_line_still_accurate_when_active(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Default session: deferral is on, and both the #746 banner line and the
    # RC sibling note say so.
    _captured, output = _invoke_wrap_claude(runner, monkeypatch, env={})
    assert "on-demand tool loading kept on" in output
    assert "keeps it on for this session" in output
    assert "DISABLED per your setting" not in output
    assert "write_tool_search_value" not in _captured


def test_wrap_claude_foundry_keeps_disabled_tool_search_process_local_by_default(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured, output = _invoke_wrap_claude(
        runner,
        monkeypatch,
        env={
            "CLAUDE_CODE_USE_FOUNDRY": "1",
            "ANTHROPIC_FOUNDRY_BASE_URL": "https://tenant.services.ai.azure.com/anthropic",
        },
    )

    assert captured["child_env"]["ENABLE_TOOL_SEARCH"] == "false"
    assert "write_tool_search_value" not in captured
    assert "on-demand tool loading DISABLED" in output


def test_wrap_claude_foundry_project_settings_opt_in_persists_disabled_tool_search(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured, _output = _invoke_wrap_claude(
        runner,
        monkeypatch,
        env={
            "CLAUDE_CODE_USE_FOUNDRY": "1",
            "ANTHROPIC_FOUNDRY_BASE_URL": "https://tenant.services.ai.azure.com/anthropic",
        },
        extra_args=("--project-settings",),
    )

    assert captured["write_tool_search_value"] == "false"


def test_wrap_claude_vertex_passes_custom_base_url_to_proxy_before_child_redirect(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_vertex_url = "https://vertex-gateway.internal/custom/v1"

    captured, output = _invoke_wrap_claude(
        runner,
        monkeypatch,
        env={
            "CLAUDE_CODE_USE_VERTEX": "1",
            "ANTHROPIC_VERTEX_BASE_URL": custom_vertex_url,
        },
    )

    # Issue #1779: Vertex sessions authenticate with cloud IAM and never had
    # Remote Control — the RC gate warning must not fire in this mode.
    assert "Remote Control" not in output

    ensure_kwargs = captured["ensure_kwargs"]
    child_env = captured["child_env"]
    assert ensure_kwargs["vertex_api_url"] == custom_vertex_url
    assert ensure_kwargs["clear_vertex_api_url"] is False
    assert ensure_kwargs["anthropic_api_url"] is None
    assert child_env["ANTHROPIC_VERTEX_BASE_URL"] == "http://127.0.0.1:8787"
    assert "write_base_url_kwargs" not in captured


def test_wrap_claude_vertex_target_env_beats_anthropic_vertex_base_url(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured, _output = _invoke_wrap_claude(
        runner,
        monkeypatch,
        env={
            "CLAUDE_CODE_USE_VERTEX": "1",
            "ANTHROPIC_VERTEX_BASE_URL": "https://client-gateway.example.com/vertex/v1",
            "VERTEX_TARGET_API_URL": "https://proxy-gateway.example.com/vertex/v1",
        },
    )

    ensure_kwargs = captured["ensure_kwargs"]
    child_env = captured["child_env"]
    assert ensure_kwargs["vertex_api_url"] == "https://proxy-gateway.example.com/vertex/v1"
    assert ensure_kwargs["clear_vertex_api_url"] is False
    assert child_env["ANTHROPIC_VERTEX_BASE_URL"] == "http://127.0.0.1:8787"


@pytest.mark.parametrize(
    "env",
    [
        {"CLAUDE_CODE_USE_VERTEX": "1"},
        {
            "CLAUDE_CODE_USE_VERTEX": "1",
            "ANTHROPIC_VERTEX_BASE_URL": DEFAULT_VERTEX_API_URL,
        },
        {
            "CLAUDE_CODE_USE_VERTEX": "1",
            "ANTHROPIC_VERTEX_BASE_URL": "http://127.0.0.1:8787",
        },
        {
            "CLAUDE_CODE_USE_VERTEX": "1",
            "VERTEX_TARGET_API_URL": "http://127.0.0.1:8787",
        },
    ],
)
def test_wrap_claude_vertex_default_or_absent_base_url_does_not_force_vertex_target(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    captured, _output = _invoke_wrap_claude(runner, monkeypatch, env=env)

    ensure_kwargs = captured["ensure_kwargs"]
    child_env = captured["child_env"]
    assert ensure_kwargs["vertex_api_url"] is None
    assert ensure_kwargs["clear_vertex_api_url"] is True
    assert child_env["ANTHROPIC_VERTEX_BASE_URL"] == "http://127.0.0.1:8787"


def test_wrap_claude_foundry_proxy_env_uses_process_environment_only_by_default(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundry_url = "https://my-resource.services.ai.azure.com/anthropic"

    captured, output = _invoke_wrap_claude(
        runner,
        monkeypatch,
        env={
            "CLAUDE_CODE_USE_FOUNDRY": "1",
            "ANTHROPIC_FOUNDRY_BASE_URL": foundry_url,
        },
    )

    # Issue #1779: Foundry sessions authenticate with Azure credentials and
    # never had Remote Control — the RC gate warning must not fire.
    assert "Remote Control" not in output

    ensure_kwargs = captured["ensure_kwargs"]
    child_env = captured["child_env"]
    assert ensure_kwargs["anthropic_api_url"] == foundry_url
    assert ensure_kwargs["vertex_api_url"] is None
    assert child_env["ANTHROPIC_FOUNDRY_BASE_URL"] == "http://127.0.0.1:8787/anthropic"
    assert "write_base_url_kwargs" not in captured


def test_write_vertex_mode_sets_vertex_key(tmp_path: Path) -> None:
    path = tmp_path / ".claude" / "settings.local.json"

    previous = wrap_mod._write_claude_wrap_base_url(
        "http://127.0.0.1:8787",
        vertex_mode=True,
        settings_path=path,
    )

    assert previous is None
    payload = path.read_text(encoding="utf-8")
    assert '"ANTHROPIC_VERTEX_BASE_URL": "http://127.0.0.1:8787"' in payload
    assert "ANTHROPIC_BASE_URL" not in payload


def test_restore_vertex_mode_restores_previous_vertex_key(tmp_path: Path) -> None:
    path = tmp_path / ".claude" / "settings.local.json"
    wrap_mod._write_claude_wrap_base_url(
        "http://127.0.0.1:8787",
        vertex_mode=True,
        settings_path=path,
    )

    wrap_mod._restore_claude_wrap_base_url(
        "https://existing-gateway.example.com/vertex/v1",
        vertex_mode=True,
        settings_path=path,
    )

    payload = path.read_text(encoding="utf-8")
    assert (
        '"ANTHROPIC_VERTEX_BASE_URL": "https://existing-gateway.example.com/vertex/v1"' in payload
    )


def test_start_proxy_sets_vertex_target_env_for_proxy_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_proc = _FakeProxyProcess()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(wrap_mod, "_get_log_path", lambda: tmp_path / "proxy.log")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: True)
    monkeypatch.setattr(wrap_mod.time, "sleep", lambda _seconds: None)

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakeProxyProcess:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return fake_proc

    monkeypatch.setattr(wrap_mod.subprocess, "Popen", fake_popen)

    proc = wrap_mod._start_proxy(
        8787,
        agent_type="claude",
        vertex_api_url="https://vertex-gateway.internal/custom",
    )

    assert proc is fake_proc
    assert captured["cmd"][-2:] == [
        "--vertex-api-url",
        "https://vertex-gateway.internal/custom",
    ]
    proxy_env = captured["kwargs"]["env"]
    assert proxy_env["VERTEX_TARGET_API_URL"] == "https://vertex-gateway.internal/custom"


def test_start_proxy_clears_inherited_vertex_target_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_proc = _FakeProxyProcess()
    captured: dict[str, Any] = {}

    monkeypatch.setenv("VERTEX_TARGET_API_URL", "http://127.0.0.1:8787")
    monkeypatch.setattr(wrap_mod, "_get_log_path", lambda: tmp_path / "proxy.log")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: True)
    monkeypatch.setattr(wrap_mod.time, "sleep", lambda _seconds: None)

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakeProxyProcess:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return fake_proc

    monkeypatch.setattr(wrap_mod.subprocess, "Popen", fake_popen)

    proc = wrap_mod._start_proxy(8787, agent_type="claude", clear_vertex_api_url=True)

    assert proc is fake_proc
    assert "--vertex-api-url" not in captured["cmd"]
    proxy_env = captured["kwargs"]["env"]
    assert "VERTEX_TARGET_API_URL" not in proxy_env


def test_start_proxy_sets_pythonsafepath_to_avoid_cwd_shadow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`python -m headroom.cli` prepends the launch cwd to sys.path, so running
    wrap from a directory that contains a `headroom/` folder (a clone of this
    repo) shadows the installed wheel with the raw source tree, which has no
    compiled `headroom._core`, and the proxy dies importing it (#2793). The
    subprocess env must set PYTHONSAFEPATH=1 to disable that cwd prepend."""
    fake_proc = _FakeProxyProcess()
    captured: dict[str, Any] = {}

    monkeypatch.setattr(wrap_mod, "_get_log_path", lambda: tmp_path / "proxy.log")
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: True)
    monkeypatch.setattr(wrap_mod.time, "sleep", lambda _seconds: None)

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakeProxyProcess:
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return fake_proc

    monkeypatch.setattr(wrap_mod.subprocess, "Popen", fake_popen)

    proc = wrap_mod._start_proxy(8787, agent_type="claude")

    assert proc is fake_proc
    assert captured["kwargs"]["env"]["PYTHONSAFEPATH"] == "1"
    # Still launched as a module of the installed package.
    assert captured["cmd"][:4] == [wrap_mod.sys.executable, "-m", "headroom.cli", "proxy"]


def test_ensure_proxy_restarts_idle_proxy_for_vertex_api_url_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    health = {
        "version": wrap_mod._HEADROOM_VERSION,
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {
            "pid": "12345",
            "memory": False,
            "learn": False,
            "code_graph": False,
            "vertex_api_url": "https://old-gateway.example.com/vertex/v1",
        },
    }

    monkeypatch.setattr(wrap_mod, "_find_persistent_manifest", lambda _port: None)
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: len(calls) == 0)
    monkeypatch.setattr(wrap_mod, "_query_proxy_health", lambda _port: health)
    monkeypatch.setattr(wrap_mod, "_port_bind_error", lambda _port: None)
    monkeypatch.setattr(wrap_mod, "_live_proxy_clients", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        wrap_mod,
        "_kill_proxy_by_pid",
        lambda pid, port: calls.append(("kill", pid, port)) or True,
    )
    monkeypatch.setattr(
        wrap_mod,
        "_start_proxy",
        lambda *args, **kwargs: calls.append(("start", args, kwargs)),
    )

    proc, actual_port = wrap_mod._ensure_proxy(
        8787,
        False,
        vertex_api_url="https://new-gateway.example.com/vertex/v1",
    )

    assert proc is None
    assert actual_port == 8787
    assert calls[0] == ("kill", 12345, 8787)
    assert calls[1][0] == "start"
    assert calls[1][2]["vertex_api_url"] == "https://new-gateway.example.com/vertex/v1"


def test_ensure_proxy_restarts_idle_proxy_to_clear_vertex_api_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    health = {
        "version": wrap_mod._HEADROOM_VERSION,
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {
            "pid": "12345",
            "memory": False,
            "learn": False,
            "code_graph": False,
            "vertex_api_url": "https://old-gateway.example.com/vertex/v1",
        },
    }

    monkeypatch.setattr(wrap_mod, "_find_persistent_manifest", lambda _port: None)
    monkeypatch.setattr(wrap_mod, "_check_proxy", lambda _port: len(calls) == 0)
    monkeypatch.setattr(wrap_mod, "_query_proxy_health", lambda _port: health)
    monkeypatch.setattr(wrap_mod, "_port_bind_error", lambda _port: None)
    monkeypatch.setattr(wrap_mod, "_live_proxy_clients", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        wrap_mod,
        "_kill_proxy_by_pid",
        lambda pid, port: calls.append(("kill", pid, port)) or True,
    )
    monkeypatch.setattr(
        wrap_mod,
        "_start_proxy",
        lambda *args, **kwargs: calls.append(("start", args, kwargs)),
    )

    proc, actual_port = wrap_mod._ensure_proxy(8787, False, clear_vertex_api_url=True)

    assert proc is None
    assert actual_port == 8787
    assert calls[0] == ("kill", 12345, 8787)
    assert calls[1][0] == "start"
    assert calls[1][2]["vertex_api_url"] is None
    assert calls[1][2]["clear_vertex_api_url"] is True
