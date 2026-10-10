"""Tests for ``headroom wrap droid`` (Factory Droid).

The command points Droid's ``FACTORY_API_BASE_URL`` at the local proxy and
starts (or reuses) a proxy whose ``factory_api_url`` is the real Factory
upstream. Nothing durable is written. Every proxy/launch seam is patched, so
no proxy starts and no Droid binary runs.
"""

from __future__ import annotations

import subprocess
from typing import Any

import click
import pytest
from click.testing import CliRunner

import headroom.cli.wrap as wrap_cli
from headroom.cli.main import main

FACTORY = "https://api.factory.ai"
OTHER_FACTORY = "https://eu.factory.example"


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / ".headroom"))
    for name in (
        "FACTORY_API_BASE_URL",
        "FACTORY_TARGET_API_URL",
        "ANTHROPIC_TARGET_API_URL",
        "OPENAI_TARGET_API_URL",
        "GEMINI_TARGET_API_URL",
        "CLOUDCODE_TARGET_API_URL",
        "VERTEX_TARGET_API_URL",
        "AUGMENT_TARGET_API_URL",
        "HEADROOM_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(wrap_cli, "_live_proxy_clients", lambda *a, **kw: [])
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: None)


def _boom(message: str):
    def _raise(*args: Any, **kwargs: Any):
        raise AssertionError(message)

    return _raise


def _invoke(args: list[str], env: dict[str, str] | None = None):
    return CliRunner().invoke(main, ["wrap", "droid", *args], env=env)


# ── command: env, precedence, rejection, missing binary ───────────────


def test_launch_env_points_droid_at_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(wrap_cli.shutil, "which", lambda name: "/usr/bin/droid")
    monkeypatch.setattr(wrap_cli, "_launch_tool", lambda **kw: captured.update(kw))

    result = _invoke(["--port", "8787", "--", "exec", "hi"])

    assert result.exit_code == 0, result.output
    assert captured["binary"] == "/usr/bin/droid"
    assert captured["args"] == ("exec", "hi")
    assert captured["env"]["FACTORY_API_BASE_URL"] == "http://127.0.0.1:8787"
    assert captured["factory_api_url"] == FACTORY
    assert captured["agent_type"] == "droid"
    assert captured["tool_label"] == "DROID"


def test_fallback_port_rewrites_child_env(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    ensure_kwargs: dict[str, Any] = {}

    def _ensure(port: int, no_proxy: bool, **kwargs: Any):
        ensure_kwargs.update(kwargs)
        return None, 8799

    def _run(cmd: list[str], env: dict[str, str]):
        captured["cmd"] = cmd
        captured["env"] = env
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(wrap_cli.shutil, "which", lambda name: "/usr/bin/droid")
    monkeypatch.setattr(wrap_cli, "_ensure_proxy", _ensure)
    monkeypatch.setattr(wrap_cli, "_push_runtime_env", lambda *a, **kw: None)
    monkeypatch.setattr(wrap_cli.subprocess, "run", _run)

    result = _invoke(["--port", "8787"])

    assert result.exit_code == 0, result.output
    assert captured["env"]["FACTORY_API_BASE_URL"] == "http://127.0.0.1:8799"
    assert ensure_kwargs["factory_api_url"] == FACTORY


@pytest.mark.parametrize(
    ("args", "env", "expected"),
    [
        ([], {}, FACTORY),
        ([], {"FACTORY_API_BASE_URL": "https://EU.factory.example/"}, OTHER_FACTORY),
        (
            ["--factory-api-url", "https://flag.factory.example"],
            {"FACTORY_API_BASE_URL": OTHER_FACTORY},
            "https://flag.factory.example",
        ),
    ],
)
def test_upstream_precedence(args: list[str], env: dict[str, str], expected: str) -> None:
    result = _invoke([*args, "--port", "8787", "--prepare-only"], env=env)

    assert result.exit_code == 0, result.output
    assert "FACTORY_API_BASE_URL=http://127.0.0.1:8787" in result.output
    assert f"upstream={expected}" in result.output


@pytest.mark.parametrize(
    ("args", "env"),
    [
        (["--factory-api-url", "http://127.0.0.1:8787"], {}),
        ([], {"FACTORY_API_BASE_URL": "http://127.0.0.1:8787"}),
        (["--factory-api-url", "http://localhost:9000"], {}),
        (["--factory-api-url", "https://user:pw@api.factory.ai"], {}),
        (["--factory-api-url", "https://api.factory.ai?x=1"], {}),
        (["--factory-api-url", "https://api.factory.ai#frag"], {}),
    ],
)
def test_rejected_upstreams_never_launch(
    monkeypatch: pytest.MonkeyPatch, args: list[str], env: dict[str, str]
) -> None:
    monkeypatch.setattr(wrap_cli.shutil, "which", lambda name: "/usr/bin/droid")
    monkeypatch.setattr(wrap_cli, "_launch_tool", _boom("must not launch"))
    monkeypatch.setattr(wrap_cli, "_ensure_proxy", _boom("must not start a proxy"))

    result = _invoke([*args, "--port", "8787"], env=env)

    assert result.exit_code != 0
    assert "FACTORY_API_BASE_URL" in result.output


def test_missing_binary_exits_before_proxy_work(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wrap_cli.shutil, "which", lambda name: None)
    monkeypatch.setattr(wrap_cli, "_ensure_proxy", _boom("must not ensure a proxy"))
    monkeypatch.setattr(wrap_cli, "_start_proxy", _boom("must not start a proxy"))

    result = _invoke(["--port", "8787"])

    assert result.exit_code == 1
    assert "'droid' not found" in result.output


# ── proxy routing plumbing ────────────────────────────────────────────


def test_requested_routing_uses_explicit_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = {
        "backend": None,
        "openai_api_url": None,
        "anthropic_api_url": None,
        "vertex_api_url": None,
        "clear_vertex_api_url": False,
    }
    _, urls = wrap_cli._effective_requested_proxy_routing(**kwargs, factory_api_url=FACTORY)
    assert urls["factory_api_url"] == FACTORY

    monkeypatch.setenv("FACTORY_TARGET_API_URL", OTHER_FACTORY)
    _, urls = wrap_cli._effective_requested_proxy_routing(**kwargs)
    assert urls["factory_api_url"] == OTHER_FACTORY
    _, urls = wrap_cli._effective_requested_proxy_routing(**kwargs, factory_api_url=FACTORY)
    assert urls["factory_api_url"] == FACTORY


def test_start_proxy_forwards_factory_flag_and_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    captured: dict[str, Any] = {}

    class _Proc:
        returncode = None

        def poll(self):
            return None

    def _popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs["env"]
        return _Proc()

    monkeypatch.setattr(wrap_cli, "_get_log_path", lambda port=None: tmp_path / "proxy.log")
    monkeypatch.setattr(
        wrap_cli, "_get_proxy_stdio_log_path", lambda port=None: tmp_path / "stdio.log"
    )
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli.subprocess, "Popen", _popen)

    wrap_cli._start_proxy(8787, agent_type="droid", factory_api_url=FACTORY)

    cmd = captured["cmd"]
    assert cmd[cmd.index("--factory-api-url") + 1] == FACTORY
    assert captured["env"]["FACTORY_TARGET_API_URL"] == FACTORY


def _health(config: dict[str, Any], *, sessions: int = 0) -> dict[str, Any]:
    return {
        "service": "headroom-proxy",
        "version": wrap_cli._HEADROOM_VERSION,
        "runtime": {"websocket_sessions": {"active_sessions": sessions, "active_relay_tasks": 0}},
        "config": {"pid": 4242, "backend": "anthropic", **config},
    }


def test_idle_non_factory_proxy_is_restarted_in_factory_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: _health({}))
    monkeypatch.setattr(
        wrap_cli, "_kill_proxy_by_pid", lambda pid, port: calls.append(("kill", pid)) or True
    )
    monkeypatch.setattr(wrap_cli, "_find_available_port", lambda start, **kw: start)
    monkeypatch.setattr(
        wrap_cli, "_start_proxy", lambda port, **kw: calls.append(("start", port, kw))
    )

    assert wrap_cli._ensure_proxy(8787, False, factory_api_url=FACTORY) == (None, 8787)
    assert calls[0] == ("kill", 4242)
    assert calls[1][0] == "start"
    assert calls[1][1] == 8787
    assert calls[1][2]["factory_api_url"] == FACTORY


def test_attached_non_factory_proxy_pushes_droid_to_dedicated_port(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: _health({}))
    monkeypatch.setattr(wrap_cli, "_live_proxy_clients", lambda *a, **kw: [{"pid": 1}])
    monkeypatch.setattr(wrap_cli, "_kill_proxy_by_pid", _boom("attached proxy must survive"))
    monkeypatch.setattr(
        wrap_cli,
        "_find_available_port",
        lambda start, **kw: calls.append(("find", start)) or 8799,
    )
    monkeypatch.setattr(
        wrap_cli, "_start_proxy", lambda port, **kw: calls.append(("start", port, kw))
    )

    assert wrap_cli._ensure_proxy(8787, False, factory_api_url=FACTORY) == (None, 8799)
    assert calls[0] == ("find", 8787)
    assert calls[1][1] == 8799
    assert calls[1][2]["factory_api_url"] == FACTORY


def test_matching_factory_proxy_is_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(
        wrap_cli,
        "_query_proxy_health",
        lambda port: _health({"factory_api_url": f"{FACTORY}/"}),
    )
    monkeypatch.setattr(wrap_cli, "_kill_proxy_by_pid", _boom("must reuse"))
    monkeypatch.setattr(wrap_cli, "_find_available_port", _boom("must reuse"))
    monkeypatch.setattr(wrap_cli, "_start_proxy", _boom("must reuse"))

    assert wrap_cli._ensure_proxy(8787, False, factory_api_url=FACTORY) == (None, 8787)


# ── --no-proxy ────────────────────────────────────────────────────────


_NO_PROXY_REFUSALS = {
    "no-listener": (False, None),
    "foreign-service": (
        True,
        {"service": "something-else", "config": {"factory_api_url": FACTORY}},
    ),
    "config-less": (True, {"service": "headroom-proxy"}),
    "non-factory": (True, _health({})),
    "other-factory": (True, _health({"factory_api_url": OTHER_FACTORY})),
}


@pytest.mark.parametrize("case", sorted(_NO_PROXY_REFUSALS))
def test_no_proxy_refuses_unverified_listener(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    listening, payload = _NO_PROXY_REFUSALS[case]
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: listening)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: payload)
    monkeypatch.setattr(wrap_cli, "_query_proxy_config", lambda port: None)

    with pytest.raises(click.ClickException, match="--no-proxy requires a Headroom proxy"):
        wrap_cli._ensure_proxy(8787, True, factory_api_url=FACTORY)


@pytest.mark.parametrize("case", sorted(_NO_PROXY_REFUSALS))
def test_no_proxy_refusal_never_launches_droid(monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    listening, payload = _NO_PROXY_REFUSALS[case]
    monkeypatch.setattr(wrap_cli.shutil, "which", lambda name: "/usr/bin/droid")
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: listening)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: payload)
    monkeypatch.setattr(wrap_cli, "_query_proxy_config", lambda port: None)
    monkeypatch.setattr(wrap_cli.subprocess, "run", _boom("droid must not launch"))

    result = _invoke(["--port", "8787", "--no-proxy"])

    assert result.exit_code != 0
    assert "--no-proxy requires a Headroom proxy" in result.output


def test_no_proxy_matching_factory_launches(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _run(cmd: list[str], env: dict[str, str]):
        captured["env"] = env
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(wrap_cli.shutil, "which", lambda name: "/usr/bin/droid")
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(
        wrap_cli, "_query_proxy_health", lambda port: _health({"factory_api_url": FACTORY})
    )
    monkeypatch.setattr(wrap_cli, "_push_runtime_env", lambda *a, **kw: None)
    monkeypatch.setattr(wrap_cli, "_start_proxy", _boom("--no-proxy must not start a proxy"))
    monkeypatch.setattr(wrap_cli.subprocess, "run", _run)

    result = _invoke(["--port", "8787", "--no-proxy"])

    assert result.exit_code == 0, result.output
    assert captured["env"]["FACTORY_API_BASE_URL"] == "http://127.0.0.1:8787"
