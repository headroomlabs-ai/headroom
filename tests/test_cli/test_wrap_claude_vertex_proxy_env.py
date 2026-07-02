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
        "ANTHROPIC_VERTEX_BASE_URL",
        "ANTHROPIC_FOUNDRY_BASE_URL",
        "ANTHROPIC_FOUNDRY_RESOURCE",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "VERTEX_TARGET_API_URL",
    ):
        monkeypatch.delenv(key, raising=False)


def _invoke_wrap_claude(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
    *,
    env: dict[str, str],
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
    monkeypatch.setattr(wrap_mod.subprocess, "run", fake_run)

    result = runner.invoke(
        main,
        [
            "wrap",
            "claude",
            "--no-context-tool",
            "--no-mcp",
            "--no-tokensave",
            "--no-serena",
        ],
        env=env,
    )

    assert result.exit_code == 0, result.output
    return captured, result.output


def test_wrap_claude_vertex_passes_custom_base_url_to_proxy_before_child_redirect(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_vertex_url = "https://vertex-gateway.internal/custom/v1"

    captured, _output = _invoke_wrap_claude(
        runner,
        monkeypatch,
        env={
            "CLAUDE_CODE_USE_VERTEX": "1",
            "ANTHROPIC_VERTEX_BASE_URL": custom_vertex_url,
        },
    )

    ensure_kwargs = captured["ensure_kwargs"]
    child_env = captured["child_env"]
    write_kwargs = captured["write_base_url_kwargs"]
    assert ensure_kwargs["vertex_api_url"] == custom_vertex_url
    assert ensure_kwargs["clear_vertex_api_url"] is False
    assert ensure_kwargs["anthropic_api_url"] is None
    assert child_env["ANTHROPIC_VERTEX_BASE_URL"] == "http://127.0.0.1:8787"
    assert write_kwargs["vertex_mode"] is True
    assert write_kwargs["foundry_mode"] is False


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


def test_wrap_claude_foundry_proxy_env_behavior_is_unchanged(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    foundry_url = "https://my-resource.services.ai.azure.com/anthropic"

    captured, _output = _invoke_wrap_claude(
        runner,
        monkeypatch,
        env={
            "CLAUDE_CODE_USE_FOUNDRY": "1",
            "ANTHROPIC_FOUNDRY_BASE_URL": foundry_url,
        },
    )

    ensure_kwargs = captured["ensure_kwargs"]
    child_env = captured["child_env"]
    assert ensure_kwargs["anthropic_api_url"] == foundry_url
    assert ensure_kwargs["vertex_api_url"] is None
    assert child_env["ANTHROPIC_FOUNDRY_BASE_URL"] == "http://127.0.0.1:8787/anthropic"
    assert captured["write_base_url_kwargs"]["foundry_mode"] is True
    assert captured["write_base_url_kwargs"]["vertex_mode"] is False


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
