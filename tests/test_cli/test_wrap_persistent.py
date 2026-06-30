from __future__ import annotations

import click
import pytest

import headroom.cli.wrap as wrap_cli


@pytest.fixture(autouse=True)
def _no_attached_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default: no other wrap clients attached, so restart paths are hermetic.

    The ephemeral restart guards consult ``_live_proxy_clients``; without this, a
    real ``headroom wrap`` session on the dev's machine could make these tests
    flaky. Individual tests override this to simulate attached wrappers.
    """
    monkeypatch.setattr(wrap_cli, "_live_proxy_clients", lambda *a, **kw: [])


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
    monkeypatch.setattr(wrap_cli, "_port_bind_error", lambda port: None)
    monkeypatch.setattr(wrap_cli, "_start_proxy", lambda *args, **kwargs: calls.append("start"))

    result = wrap_cli._ensure_proxy(8787, False)

    assert result is None
    assert calls == ["start"]


def test_ensure_proxy_reports_unbindable_port_before_starting_subprocess(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: False)
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: None)
    monkeypatch.setattr(
        wrap_cli,
        "_port_bind_error",
        lambda port: PermissionError(10013, "access denied by OS port reservation"),
    )
    monkeypatch.setattr(wrap_cli, "_start_proxy", lambda *args, **kwargs: calls.append("start"))

    try:
        wrap_cli._ensure_proxy(8787, False, agent_type="cursor")
    except click.ClickException as exc:
        message = str(exc)
    else:
        raise AssertionError("expected unbindable port to raise before starting proxy")

    assert "Port 8787 is unavailable" in message
    assert "Windows" in message
    assert "headroom wrap cursor --port 8788" in message
    assert calls == []


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


def test_ensure_proxy_defers_persistent_restart_when_http_wrapper_attached(
    monkeypatch,
) -> None:
    """A stale persistent proxy is left running while marker-tracked HTTP
    wrappers are attached, even when WebSocket session count is zero."""
    health = {
        "version": "0.0.1",
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {"pid": 12345},
    }

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    monkeypatch.setattr(wrap_cli, "_live_proxy_clients", lambda *a, **kw: [999])
    monkeypatch.setattr(
        wrap_cli,
        "_restart_persistent_proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("attached persistent proxy should not restart")
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
    monkeypatch.setattr(wrap_cli, "_port_bind_error", lambda port: None)
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
    monkeypatch.setattr(wrap_cli, "_port_bind_error", lambda port: None)
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


def test_ensure_proxy_reuses_agent_proxy_without_savings_profile(monkeypatch) -> None:
    health = {
        "version": wrap_cli._HEADROOM_VERSION,
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {"pid": "12345", "memory": False, "learn": False, "code_graph": False},
    }

    monkeypatch.delenv("HEADROOM_SAVINGS_PROFILE", raising=False)
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: None)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    monkeypatch.setattr(
        wrap_cli,
        "_kill_proxy_by_pid",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("default agent proxy should not restart for savings profile")
        ),
    )
    monkeypatch.setattr(
        wrap_cli,
        "_start_proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("replacement proxy should not start")
        ),
    )

    result = wrap_cli._ensure_proxy(8787, False, agent_type="codex")

    assert result is None


def test_ensure_proxy_restarts_for_explicit_agent_savings_profile(monkeypatch) -> None:
    calls: list[object] = []
    health = {
        "version": wrap_cli._HEADROOM_VERSION,
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {"pid": "12345", "memory": False, "learn": False, "code_graph": False},
    }

    monkeypatch.setenv("HEADROOM_SAVINGS_PROFILE", "agent-90")
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: None)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: len(calls) == 0)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    monkeypatch.setattr(wrap_cli, "_port_bind_error", lambda port: None)
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

    result = wrap_cli._ensure_proxy(8787, False, agent_type="codex")

    assert result is None
    assert calls[0] == ("kill", 12345, 8787)
    assert calls[1][0] == "start"


def test_ensure_proxy_reuses_agent_proxy_with_savings_profile(monkeypatch) -> None:
    health = {
        "version": wrap_cli._HEADROOM_VERSION,
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {
            "pid": "12345",
            "memory": False,
            "learn": False,
            "code_graph": False,
            "savings_profile": "agent-90",
            "target_ratio": 0.10,
            "compress_user_messages": True,
            "compress_system_messages": True,
            "protect_recent": 2,
            "protect_analysis_context": True,
            "min_tokens_to_crush": 120,
            "max_items_after_crush": 8,
            "smart_crusher_with_compaction": False,
            "accuracy_guard": "strict",
        },
    }

    monkeypatch.setenv("HEADROOM_SAVINGS_PROFILE", "agent-90")
    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: None)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    monkeypatch.setattr(
        wrap_cli,
        "_kill_proxy_by_pid",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("configured proxy should not restart")
        ),
    )
    monkeypatch.setattr(
        wrap_cli,
        "_start_proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("replacement proxy should not start")
        ),
    )

    result = wrap_cli._ensure_proxy(8787, False, agent_type="cursor")

    assert result is None


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


def test_ensure_proxy_defers_version_restart_when_http_wrapper_attached(monkeypatch) -> None:
    """A stale-version proxy is NOT restarted while a marker-tracked HTTP
    wrapper is attached, even though the WebSocket session count is zero."""
    health = {
        "version": "0.0.1",  # stale → version restart wanted
        # No WebSocket relay sessions — the gap that let the old code kill it.
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {"pid": "12345", "memory": False, "learn": False, "code_graph": False},
    }

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: None)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    # Another HTTP wrapper (PID 999) is attached per the marker registry.
    monkeypatch.setattr(wrap_cli, "_live_proxy_clients", lambda *a, **kw: [999])
    monkeypatch.setattr(
        wrap_cli,
        "_kill_proxy_by_pid",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("attached proxy must not be killed for a version restart")
        ),
    )
    monkeypatch.setattr(
        wrap_cli,
        "_start_proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("replacement proxy must not start")
        ),
    )

    result = wrap_cli._ensure_proxy(8787, False)

    assert result is None


def test_ensure_proxy_defers_flag_restart_when_other_wrapper_attached(monkeypatch) -> None:
    """Requesting --memory must not restart the proxy out from under another
    attached wrapper; reuse the running proxy as-is instead."""
    health = {
        "version": wrap_cli._HEADROOM_VERSION,  # same version → no version restart
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        # Running proxy lacks `memory`; this session asks for it.
        "config": {"pid": "12345", "memory": False, "learn": False, "code_graph": False},
    }

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: None)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    monkeypatch.setattr(wrap_cli, "_live_proxy_clients", lambda *a, **kw: [999])
    monkeypatch.setattr(
        wrap_cli,
        "_kill_proxy_by_pid",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("attached proxy must not be killed to add flags")
        ),
    )
    monkeypatch.setattr(
        wrap_cli,
        "_start_proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("replacement proxy must not start")
        ),
    )

    result = wrap_cli._ensure_proxy(8787, False, memory=True)

    assert result is None


def test_ensure_proxy_restarts_for_flags_when_no_other_wrapper(monkeypatch) -> None:
    """Control: with no other wrapper attached, a missing-flag restart still
    happens — the guard must not block the single-client upgrade path."""
    calls: list[object] = []
    health = {
        "version": wrap_cli._HEADROOM_VERSION,
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {"pid": "12345", "memory": False, "learn": False, "code_graph": False},
    }

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: None)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: len(calls) == 0)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    monkeypatch.setattr(wrap_cli, "_live_proxy_clients", lambda *a, **kw: [])
    monkeypatch.setattr(wrap_cli, "_port_bind_error", lambda port: None)
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

    result = wrap_cli._ensure_proxy(8787, False, memory=True)

    assert result is None
    assert calls[0] == ("kill", 12345, 8787)
    assert calls[1][0] == "start"


def test_ensure_proxy_restarts_persistent_deployment_for_feature_mismatch(monkeypatch) -> None:
    """Persistent deployment should restart when requested features differ from running config."""
    calls: list[object] = []
    health = {
        "version": wrap_cli._HEADROOM_VERSION,
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {
            "pid": 12345,
            "memory": False,
            "learn": False,
            "code_graph": False,
            "openai_api_url": None,
        },
    }

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    # Persistent proxy is running, so _check_proxy returns True
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_port_bind_error", lambda port: None)
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

    # Request openai_api_url that differs from running config (None)
    result = wrap_cli._ensure_proxy(
        8787,
        False,
        openai_api_url="https://api.githubcopilot.com",
    )

    assert result is None
    # Proxy should be killed and restarted due to openai_api_url mismatch
    assert calls[0] == ("kill", 12345, 8787)
    assert calls[1][0] == "start"
    assert calls[1][2]["openai_api_url"] == "https://api.githubcopilot.com"


def test_ensure_proxy_restarts_persistent_deployment_for_memory_mismatch(monkeypatch) -> None:
    """Persistent deployment should restart when memory is requested but not enabled."""
    calls: list[object] = []
    health = {
        "version": wrap_cli._HEADROOM_VERSION,
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {
            "pid": 12345,
            "memory": False,
            "learn": False,
            "code_graph": False,
            "openai_api_url": None,
        },
    }

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    # Persistent proxy is running, so _check_proxy returns True
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_port_bind_error", lambda port: None)
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

    # Request memory that differs from running config (False)
    result = wrap_cli._ensure_proxy(8787, False, memory=True)

    assert result is None
    # Proxy should be killed and restarted due to memory mismatch
    assert calls[0] == ("kill", 12345, 8787)
    assert calls[1][0] == "start"


def test_ensure_proxy_restarts_recovered_persistent_for_openai_api_url_mismatch(
    monkeypatch,
) -> None:
    calls: list[object] = []
    health = {
        "version": wrap_cli._HEADROOM_VERSION,
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {
            "pid": 12345,
            "memory": False,
            "learn": False,
            "code_graph": False,
            "openai_api_url": None,
        },
    }

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)
    monkeypatch.setattr(wrap_cli, "_recover_persistent_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    monkeypatch.setattr(
        wrap_cli,
        "_restart_persistent_proxy",
        lambda manifest, port: calls.append(("restart", manifest.profile, port)) or True,
    )
    monkeypatch.setattr(
        wrap_cli,
        "_start_proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("ephemeral proxy should not start")
        ),
    )

    result = wrap_cli._ensure_proxy(
        8787,
        False,
        openai_api_url="https://api.business.githubcopilot.com",
    )

    assert result is None
    assert calls == [("restart", "default", 8787)]


def test_ensure_proxy_restarts_recovered_persistent_when_config_unavailable(monkeypatch) -> None:
    calls: list[object] = []

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)
    monkeypatch.setattr(wrap_cli, "_recover_persistent_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: {"version": "x"})
    monkeypatch.setattr(wrap_cli, "_query_proxy_config", lambda port: None)
    monkeypatch.setattr(
        wrap_cli,
        "_restart_persistent_proxy",
        lambda manifest, port: calls.append(("restart", manifest.profile, port)) or True,
    )
    monkeypatch.setattr(
        wrap_cli,
        "_start_proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("ephemeral proxy should not start")
        ),
    )

    result = wrap_cli._ensure_proxy(
        8787,
        False,
        openai_api_url="https://api.business.githubcopilot.com",
    )

    assert result is None
    assert calls == [("restart", "default", 8787)]


def test_ensure_proxy_reuses_persistent_deployment_when_features_match(monkeypatch) -> None:
    """Persistent deployment should be reused when all requested features match."""
    health = {
        "version": wrap_cli._HEADROOM_VERSION,
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {
            "pid": 12345,
            "memory": True,
            "learn": False,
            "code_graph": False,
            "openai_api_url": "https://api.githubcopilot.com",
        },
    }

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    monkeypatch.setattr(
        wrap_cli,
        "_restart_persistent_proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("should not restart when features match")
        ),
    )
    monkeypatch.setattr(
        wrap_cli,
        "_start_proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("should not start ephemeral proxy when features match")
        ),
    )

    # Request same features as running config
    result = wrap_cli._ensure_proxy(
        8787,
        False,
        memory=True,
        openai_api_url="https://api.githubcopilot.com",
    )

    assert result is None


def test_ensure_proxy_recovered_persistent_deployment_checks_feature_mismatch(monkeypatch) -> None:
    """Recovered persistent deployments must still restart on feature mismatch.

    Regression guard for the recover path: when wrap requests a different
    openai_api_url (Copilot subscription), do not early-return right after
    recover; run the shared mismatch checks and restart if needed.
    """

    calls: list[object] = []
    health = {
        "version": wrap_cli._HEADROOM_VERSION,
        "runtime": {"websocket_sessions": {"active_sessions": 0, "active_relay_tasks": 0}},
        "config": {
            "pid": "12345",
            "memory": False,
            "learn": False,
            "code_graph": False,
            "openai_api_url": None,
        },
    }

    monkeypatch.setattr(wrap_cli, "_find_persistent_manifest", lambda port: _Manifest())
    monkeypatch.setattr("headroom.install.health.probe_ready", lambda url: False)
    monkeypatch.setattr(wrap_cli, "_recover_persistent_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap_cli, "_query_proxy_health", lambda port: health)
    monkeypatch.setattr(
        wrap_cli,
        "_restart_persistent_proxy",
        lambda manifest, port: calls.append(("restart", manifest.profile, port)) or True,
    )
    monkeypatch.setattr(
        wrap_cli,
        "_start_proxy",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("ephemeral proxy should not start")
        ),
    )

    result = wrap_cli._ensure_proxy(
        8787,
        False,
        openai_api_url="https://api.githubcopilot.com",
    )

    assert result is None
    assert calls == [("restart", "default", 8787)]
