from __future__ import annotations

import click

import headroom.cli.wrap as wrap_cli


class _Manifest:
    profile = "default"
    preset = "persistent-service"
    supervisor_kind = "service"
    health_url = "http://127.0.0.1:8787/readyz"


def test_ensure_proxy_recovers_matching_persistent_deployment(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: False)
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)
    monkeypatch.setattr(
        "headroom.install.supervisors.start_supervisor",
        lambda manifest: calls.append(f"start:{manifest.profile}"),
    )
    monkeypatch.setattr(
        "headroom.install.runtime.wait_ready", lambda manifest, timeout_seconds=45: True
    )
    monkeypatch.setattr(
        wrap_cli,
        "_start_proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("ephemeral proxy should not start")
        ),
    )

    result = wrap_cli._ensure_proxy(8787, False)

    assert result is None
    assert calls == ["start:default"]


def test_ensure_proxy_recovers_persistent_deployment_when_socket_is_bound(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)
    monkeypatch.setattr(
        "headroom.install.supervisors.start_supervisor",
        lambda manifest: calls.append(f"start:{manifest.profile}"),
    )
    monkeypatch.setattr(
        "headroom.install.runtime.wait_ready", lambda manifest, timeout_seconds=45: True
    )

    result = wrap_cli._ensure_proxy(8787, False)

    assert result is None
    assert calls == ["start:default"]


def test_ensure_proxy_rejects_unhealthy_persistent_deployment(monkeypatch) -> None:
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)
    monkeypatch.setattr(wrap_cli, "_recover_persistent_proxy", lambda port: False)

    try:
        wrap_cli._ensure_proxy(8787, False)
    except click.ClickException as exc:
        assert "is not healthy" in str(exc)
    else:
        raise AssertionError("expected unhealthy persistent deployment to raise")


def test_ensure_proxy_falls_back_when_persistent_manifest_is_stale(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: False)
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)
    monkeypatch.setattr(wrap_cli, "_recover_persistent_proxy", lambda port: False)
    monkeypatch.setattr(wrap_cli, "_start_proxy", lambda *args, **kwargs: calls.append("start"))

    result = wrap_cli._ensure_proxy(8787, False)

    assert result is None
    assert calls == ["start"]


def test_ensure_proxy_restarts_idle_stale_persistent_deployment(monkeypatch) -> None:
    calls: list[str] = []
    health = {
        "version": "0.0.1",
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {"pid": 12345},
    }

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    monkeypatch.setattr(
        wrap_cli,
        "_restart_persistent_proxy",
        lambda manifest, port: calls.append(f"restart:{manifest.profile}:{port}") or True,
    )
    monkeypatch.setattr(
        wrap_cli,
        "_start_proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("ephemeral proxy should not start")
        ),
    )

    result = wrap_cli._ensure_proxy(8787, False)

    assert result is None
    assert calls == ["restart:default:8787"]


def test_ensure_proxy_leaves_active_stale_persistent_deployment_running(monkeypatch) -> None:
    health = {
        "version": "0.0.1",
        "runtime": {"websocket_sessions": {"active_sessions": 1, "active_relay_tasks": 2}},
        "config": {"pid": 12345},
    }

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    monkeypatch.setattr(
        wrap_cli,
        "_restart_persistent_proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("active deployment should not restart")
        ),
    )

    result = wrap_cli._ensure_proxy(8787, False)

    assert result is None


def test_find_persistent_manifest_prefers_default_profile(monkeypatch) -> None:
    class DefaultManifest:
        profile = "default"
        port = 8787

    class OtherManifest:
        profile = "custom"
        port = 8787

    monkeypatch.setattr(
        "headroom.install.state.list_manifests",
        lambda: [OtherManifest(), DefaultManifest()],
    )

    manifest = wrap_cli._find_persistent_manifest(8787)

    assert manifest.profile == "default"


def test_recover_persistent_proxy_reuses_healthy_deployment(monkeypatch) -> None:
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: True)

    assert wrap_cli._recover_persistent_proxy(8787) is True


def test_recover_persistent_proxy_warns_for_task_deployment(monkeypatch) -> None:
    class TaskManifest(_Manifest):
        supervisor_kind = "task"

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: TaskManifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)

    assert wrap_cli._recover_persistent_proxy(8787) is False


def test_ensure_proxy_restarts_idle_stale_ephemeral_proxy(monkeypatch) -> None:
    calls: list[object] = []
    health = {
        "version": "0.0.1",
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {"pid": "12345", "memory": False, "learn": False, "code_graph": False},
    }

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: None)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: len(calls) == 0)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    monkeypatch.setattr(
        wrap_cli,
        "_kill_proxy_by_pid",
        lambda pid, port: calls.append(("kill", pid, port)) or True,
    )
    monkeypatch.setattr(
        wrap_cli,
        "_start_proxy",
        lambda *args, **kwargs: calls.append(("start", args, kwargs)),
    )

    result = wrap_cli._ensure_proxy(8787, False)

    assert result is None
    assert calls[0] == ("kill", 12345, 8787)
    assert calls[1][0] == "start"


def test_ensure_proxy_restarts_ephemeral_proxy_for_openai_api_url_mismatch(monkeypatch) -> None:
    calls: list[object] = []
    health = {
        "version": wrap_cli._HEADROOM_VERSION,
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {
            "pid": "12345",
            "memory": False,
            "learn": False,
            "code_graph": False,
            "openai_api_url": "https://api.githubcopilot.com",
        },
    }

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: None)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: len(calls) == 0)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    monkeypatch.setattr(
        wrap_cli,
        "_kill_proxy_by_pid",
        lambda pid, port: calls.append(("kill", pid, port)) or True,
    )
    monkeypatch.setattr(
        wrap_cli,
        "_start_proxy",
        lambda *args, **kwargs: calls.append(("start", args, kwargs)),
    )

    result = wrap_cli._ensure_proxy(
        8787,
        False,
        openai_api_url="https://api.individual.githubcopilot.com",
    )

    assert result is None
    assert calls[0] == ("kill", 12345, 8787)
    assert calls[1][0] == "start"
    assert calls[1][2]["openai_api_url"] == "https://api.individual.githubcopilot.com"


def test_ensure_proxy_leaves_active_stale_ephemeral_proxy_running(monkeypatch) -> None:
    health = {
        "version": "0.0.1",
        "runtime": {"websocket_sessions": {"active_sessions": 2, "active_relay_tasks": 2}},
        "config": {"pid": "12345", "memory": False, "learn": False, "code_graph": False},
    }

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: None)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    monkeypatch.setattr(
        wrap_cli,
        "_kill_proxy_by_pid",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("active proxy should not be killed")
        ),
    )
    monkeypatch.setattr(
        wrap_cli,
        "_start_proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("replacement proxy should not start")
        ),
    )

    result = wrap_cli._ensure_proxy(8787, False)

    assert result is None
