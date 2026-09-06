from __future__ import annotations

from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from headroom.cli import install as inst
from headroom.cli.main import main


def test_require_manifest_resolves_single_profile_when_default_missing(monkeypatch):
    """On an init'd machine (one profile, e.g. init-user), a bare lifecycle
    command whose --profile defaults to 'default' resolves to the single
    installed deployment instead of dead-ending (#2811)."""
    only = SimpleNamespace(profile="init-user")
    monkeypatch.delenv("HEADROOM_DEPLOYMENT_PROFILE", raising=False)
    monkeypatch.setattr(inst, "load_manifest", lambda profile: None)
    monkeypatch.setattr(inst, "list_manifests", lambda: [only])

    assert inst._require_manifest("default") is only


def test_require_manifest_honors_env_profile(monkeypatch):
    """An explicit HEADROOM_DEPLOYMENT_PROFILE (exported by the runtime) selects
    the target even when the requested profile is not installed."""
    target = SimpleNamespace(profile="init-user")
    monkeypatch.setenv("HEADROOM_DEPLOYMENT_PROFILE", "init-user")
    monkeypatch.setattr(
        inst, "load_manifest", lambda profile: target if profile == "init-user" else None
    )
    monkeypatch.setattr(inst, "list_manifests", lambda: [target])

    assert inst._require_manifest("default") is target


def test_require_manifest_lists_installed_profiles_when_ambiguous(monkeypatch):
    """With several installed profiles and no signal, the error names them and
    points at --profile instead of dead-ending on 'default'."""
    monkeypatch.delenv("HEADROOM_DEPLOYMENT_PROFILE", raising=False)
    monkeypatch.setattr(inst, "load_manifest", lambda profile: None)
    monkeypatch.setattr(
        inst,
        "list_manifests",
        lambda: [SimpleNamespace(profile="init-user"), SimpleNamespace(profile="ci")],
    )

    with pytest.raises(click.ClickException) as exc:
        inst._require_manifest("default")
    msg = str(exc.value)
    assert "ci" in msg and "init-user" in msg and "--profile" in msg


def _status_manifest(profile: str) -> SimpleNamespace:
    return SimpleNamespace(
        profile=profile,
        preset="persistent-task",
        runtime_kind="python",
        supervisor_kind="none",
        scope="user",
        port=8787,
        health_url="http://127.0.0.1:8787/readyz",
        backend="anthropic",
    )


def test_install_status_explicit_missing_profile_is_not_redirected_to_env(monkeypatch):
    """An explicit --profile must be honored or rejected verbatim, never
    redirected to HEADROOM_DEPLOYMENT_PROFILE or a lone installed deployment: a
    typo must fail even when the env profile exists (#2832 review). Only a
    CliRunner invocation exercises the default-vs-explicit distinction."""
    init_user = _status_manifest("init-user")
    monkeypatch.setenv("HEADROOM_DEPLOYMENT_PROFILE", "init-user")
    monkeypatch.setattr(inst, "load_manifest", lambda p: init_user if p == "init-user" else None)
    monkeypatch.setattr(inst, "list_manifests", lambda: [init_user])

    res = CliRunner().invoke(main, ["install", "status", "--profile", "typo"])

    assert res.exit_code != 0
    assert "typo" in res.output
    # The error names the installed profile, but the command never operated on it.
    assert "Preset:" not in res.output
    assert "Status:" not in res.output


def test_install_status_stale_env_profile_is_not_redirected_to_lone_manifest(monkeypatch):
    """A non-empty HEADROOM_DEPLOYMENT_PROFILE is an explicit selection: if it
    names a missing/stale profile the command must fail naming that profile, never
    silently redirect to a different lone installed deployment (#2832 review)."""
    init_user = _status_manifest("init-user")
    monkeypatch.setenv("HEADROOM_DEPLOYMENT_PROFILE", "missing")
    monkeypatch.setattr(inst, "load_manifest", lambda p: init_user if p == "init-user" else None)
    monkeypatch.setattr(inst, "list_manifests", lambda: [init_user])

    res = CliRunner().invoke(main, ["install", "status"])

    assert res.exit_code != 0
    assert "missing" in res.output
    # Never operated on the lone init-user deployment.
    assert "Preset:" not in res.output
    assert "Status:" not in res.output


def test_install_status_omitted_profile_resolves_env_deployment(monkeypatch):
    """With --profile omitted (Click default), HEADROOM_DEPLOYMENT_PROFILE selects
    the target so the documented bare command works on an init'd machine."""
    init_user = _status_manifest("init-user")
    monkeypatch.setenv("HEADROOM_DEPLOYMENT_PROFILE", "init-user")
    monkeypatch.setattr(inst, "load_manifest", lambda p: init_user if p == "init-user" else None)
    monkeypatch.setattr(inst, "list_manifests", lambda: [init_user])
    monkeypatch.setattr(inst, "probe_json", lambda url: None)
    monkeypatch.setattr(inst, "runtime_status", lambda m: "running")
    monkeypatch.setattr(inst, "probe_ready", lambda url: True)

    res = CliRunner().invoke(main, ["install", "status"])

    assert res.exit_code == 0, res.output
    assert "Profile:    init-user" in res.output


def test_install_apply_starts_service_supervisor(monkeypatch) -> None:
    runner = CliRunner()
    calls: list[str] = []

    class Manifest:
        profile = "default"
        preset = "persistent-service"
        runtime_kind = "python"
        supervisor_kind = "service"
        scope = "user"
        health_url = "http://127.0.0.1:8787/readyz"
        mutations = [object()]
        mutations = [object()]
        mutations = []
        targets = ["claude", "codex"]
        artifacts = []

    manifest = Manifest()

    monkeypatch.setattr("headroom.cli.install.build_manifest", lambda **_: manifest)
    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: None)
    monkeypatch.setattr(
        "headroom.cli.install.apply_mutations",
        lambda deployment: calls.append("apply") or [],
    )
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment: [])
    monkeypatch.setattr(
        "headroom.cli.install.save_manifest", lambda deployment: calls.append("save")
    )
    monkeypatch.setattr(
        "headroom.cli.install.start_supervisor", lambda deployment: calls.append("start_service")
    )
    monkeypatch.setattr(
        "headroom.cli.install.start_detached_agent", lambda profile: calls.append("start_agent")
    )
    monkeypatch.setattr(
        "headroom.cli.install.start_persistent_docker",
        lambda deployment: calls.append("start_docker"),
    )
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45: True
    )
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: False)
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda manifest: "stopped")

    result = runner.invoke(main, ["install", "apply"])

    assert result.exit_code == 0, result.output
    assert "Installed persistent deployment 'default'" in result.output
    assert "Targets: claude, codex" in result.output
    assert calls == ["save", "start_service", "apply", "save"]


def test_install_apply_announces_windows_service_fallback(monkeypatch) -> None:
    runner = CliRunner()
    calls: list[str] = []

    class Manifest:
        profile = "default"
        preset = "persistent-task"
        runtime_kind = "python"
        supervisor_kind = "task"
        scope = "user"
        health_url = "http://127.0.0.1:8787/readyz"
        mutations: list[object] = []
        targets: list[str] = []
        artifacts: list[object] = []

    monkeypatch.setattr("headroom.cli.install._is_windows", lambda: True)
    monkeypatch.setattr("headroom.cli.install.build_manifest", lambda **_: Manifest())
    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: None)
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.apply_mutations", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: False)
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda manifest: "stopped")
    monkeypatch.setattr(
        "headroom.cli.install.start_detached_agent", lambda profile: calls.append("start_agent")
    )
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45: True
    )

    result = runner.invoke(main, ["install", "apply", "--preset", "persistent-service"])

    assert result.exit_code == 0, result.output
    assert "Falling back to persistent-task with Task Scheduler" in result.output
    assert "sc.exe" not in result.output
    assert calls == ["start_agent"]


def test_install_apply_forwards_no_http2_to_build_manifest(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    class Manifest:
        profile = "default"
        preset = "persistent-service"
        runtime_kind = "python"
        supervisor_kind = "service"
        scope = "user"
        health_url = "http://127.0.0.1:8787/readyz"
        mutations = [object()]
        mutations = [object()]
        targets = ["claude"]
        mutations = []
        artifacts = []

    manifest = Manifest()

    def fake_build_manifest(**kwargs):
        captured.update(kwargs)
        return manifest

    monkeypatch.setattr("headroom.cli.install.build_manifest", fake_build_manifest)
    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: None)
    monkeypatch.setattr("headroom.cli.install.apply_mutations", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.start_supervisor", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.start_detached_agent", lambda profile: None)
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45: True
    )

    result = runner.invoke(main, ["install", "apply", "--no-http2"])

    assert result.exit_code == 0, result.output
    assert captured["no_http2"] is True


def _patch_apply_pipeline(monkeypatch, captured: dict[str, object]):
    """Stub out the apply side effects and capture ``build_manifest`` kwargs."""

    class Manifest:
        profile = "default"
        preset = "persistent-service"
        runtime_kind = "python"
        supervisor_kind = "service"
        scope = "user"
        health_url = "http://127.0.0.1:8787/readyz"
        targets = ["claude"]
        mutations: list = []
        artifacts: list = []

    def fake_build_manifest(**kwargs):
        captured.update(kwargs)
        return Manifest()

    monkeypatch.setattr("headroom.cli.install.build_manifest", fake_build_manifest)
    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: None)
    monkeypatch.setattr("headroom.cli.install.apply_mutations", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.start_supervisor", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.start_detached_agent", lambda profile: None)
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45: True
    )


def test_install_apply_honors_headroom_port_env(monkeypatch) -> None:
    """An explicit HEADROOM_PORT must reach build_manifest, like `proxy --port` honors it.

    Regression for #3072 bug 1: `install apply` ignored HEADROOM_PORT and always
    configured 8787 because the --port option had no envvar binding.
    """
    captured: dict[str, object] = {}
    _patch_apply_pipeline(monkeypatch, captured)
    monkeypatch.setenv("HEADROOM_PORT", "8788")

    result = CliRunner().invoke(main, ["install", "apply"])

    assert result.exit_code == 0, result.output
    assert captured["port"] == 8788


def test_install_apply_explicit_port_overrides_env(monkeypatch) -> None:
    """An explicit --port still wins over HEADROOM_PORT (Click precedence)."""
    captured: dict[str, object] = {}
    _patch_apply_pipeline(monkeypatch, captured)
    monkeypatch.setenv("HEADROOM_PORT", "8788")

    result = CliRunner().invoke(main, ["install", "apply", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert captured["port"] == 9999


def test_deploy_honors_headroom_port_env(monkeypatch) -> None:
    """`headroom deploy` must honor HEADROOM_PORT the same way (#3072 bug 1)."""
    captured: dict[str, object] = {}

    plan = SimpleNamespace(
        preset="persistent-service",
        runtime="python",
        reason="test",
        supervisor_kind="service",
        base_env={},
    )
    manifest = SimpleNamespace(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        port=0,
        health_url="http://127.0.0.1:8788/readyz",
        targets=["claude"],
    )

    def fake_build(**kwargs):
        captured.update(kwargs)
        return manifest

    monkeypatch.setattr(
        "headroom.cli.install._select_turnkey_plan", lambda prefer_docker=True: plan
    )
    monkeypatch.setattr("headroom.cli.install._build_deployment_manifest", fake_build)
    monkeypatch.setattr("headroom.cli.install._apply_manifest", lambda m: None)
    monkeypatch.setattr("headroom.cli.install._echo_installed", lambda m, prefix="": None)
    monkeypatch.setenv("HEADROOM_PORT", "8788")

    result = CliRunner().invoke(main, ["deploy"])

    assert result.exit_code == 0, result.output
    assert captured["port"] == 8788


def test_install_apply_help_lists_no_http2() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["install", "apply", "--help"])

    assert result.exit_code == 0, result.output
    assert "--no-http2" in result.output


def test_capture_passthrough_env_skips_empty_and_unrelated() -> None:
    from headroom.cli.install import _capture_passthrough_env

    captured = _capture_passthrough_env(
        {
            "ANTHROPIC_TARGET_API_URL": "https://gw.example/v1",
            "OPENAI_TARGET_API_URL": "",  # unset-equivalent, must be skipped
            "SOME_UNRELATED_VAR": "x",
        }
    )

    assert captured == {"ANTHROPIC_TARGET_API_URL": "https://gw.example/v1"}


def _apply_capturing_build_manifest(monkeypatch) -> dict[str, object]:
    """Stub install-apply side effects and return the captured build_manifest kwargs."""
    captured: dict[str, object] = {}

    class Manifest:
        profile = "default"
        preset = "persistent-service"
        runtime_kind = "python"
        supervisor_kind = "service"
        scope = "user"
        health_url = "http://127.0.0.1:8787/readyz"
        targets = ["claude"]
        mutations: list[object] = []
        artifacts: list[object] = []

    def fake_build_manifest(**kwargs):
        captured.update(kwargs)
        return Manifest()

    monkeypatch.setattr("headroom.cli.install.build_manifest", fake_build_manifest)
    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: None)
    monkeypatch.setattr("headroom.cli.install.apply_mutations", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.start_supervisor", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.start_detached_agent", lambda profile: None)
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45: True
    )
    return captured


def test_install_apply_captures_target_api_url_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_TARGET_API_URL", "https://gateway.internal/v1")
    captured = _apply_capturing_build_manifest(monkeypatch)

    result = CliRunner().invoke(main, ["install", "apply"])

    assert result.exit_code == 0, result.output
    # The exported gateway URL rode into the manifest env so the supervised
    # proxy forwards there instead of the public Anthropic endpoint (#2240).
    assert captured["extra_env"]["ANTHROPIC_TARGET_API_URL"] == "https://gateway.internal/v1"


def test_install_apply_explicit_env_overrides_captured(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_TARGET_API_URL", "https://auto.internal/v1")
    captured = _apply_capturing_build_manifest(monkeypatch)

    result = CliRunner().invoke(
        main,
        ["install", "apply", "--env", "ANTHROPIC_TARGET_API_URL=https://explicit.internal/v1"],
    )

    assert result.exit_code == 0, result.output
    # An explicit --env must win over the auto-captured value.
    assert captured["extra_env"]["ANTHROPIC_TARGET_API_URL"] == "https://explicit.internal/v1"


def test_install_status_includes_backend_from_health_probe(monkeypatch) -> None:
    runner = CliRunner()

    class Manifest:
        profile = "default"
        preset = "persistent-service"
        runtime_kind = "python"
        supervisor_kind = "service"
        scope = "user"
        port = 8787
        backend = "anthropic"
        health_url = "http://127.0.0.1:8787/readyz"

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda manifest: "running")
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: True)
    monkeypatch.setattr(
        "headroom.cli.install.probe_json",
        lambda url: {"config": {"backend": "anthropic"}},
    )

    result = runner.invoke(main, ["install", "status"])

    assert result.exit_code == 0, result.output
    assert "Status:     running" in result.output
    assert "Healthy:    yes" in result.output
    assert "Backend:    anthropic" in result.output


def test_install_status_treats_ready_proxy_as_running(monkeypatch) -> None:
    """Health wins when PID/container bookkeeping incorrectly says stopped.

    Regression for #3072 bug 2: status previously printed the contradictory
    pair ``Status: stopped`` and ``Healthy: yes`` while doctor correctly
    reported the same proxy as running.
    """
    manifest = _status_manifest("default")
    monkeypatch.setattr(inst, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(inst, "probe_json", lambda url: None)
    monkeypatch.setattr(inst, "probe_ready", lambda url: True)
    monkeypatch.setattr(inst, "runtime_status", lambda deployment: "stopped")

    result = CliRunner().invoke(main, ["install", "status"])

    assert result.exit_code == 0, result.output
    assert "Status:     running" in result.output
    assert "Healthy:    yes" in result.output
    assert "Status:     stopped" not in result.output


def test_install_status_keeps_runtime_status_when_not_ready(monkeypatch) -> None:
    """An unhealthy live process remains distinguishable from a stopped one."""
    manifest = _status_manifest("default")
    monkeypatch.setattr(inst, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(inst, "probe_json", lambda url: None)
    monkeypatch.setattr(inst, "probe_ready", lambda url: False)
    monkeypatch.setattr(inst, "runtime_status", lambda deployment: "running")

    result = CliRunner().invoke(main, ["install", "status"])

    assert result.exit_code == 0, result.output
    assert "Status:     running" in result.output
    assert "Healthy:    no" in result.output


def test_install_status_survives_non_dict_config(monkeypatch) -> None:
    """A health payload whose `config` is a non-dict (e.g. a different service
    answering on the port returns config: null) must not crash the command."""
    runner = CliRunner()

    class Manifest:
        profile = "default"
        preset = "persistent-service"
        runtime_kind = "python"
        supervisor_kind = "service"
        scope = "user"
        port = 8787
        backend = "anthropic"
        health_url = "http://127.0.0.1:8787/readyz"

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda manifest: "running")
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: True)
    monkeypatch.setattr("headroom.cli.install.probe_json", lambda url: {"config": None})

    result = runner.invoke(main, ["install", "status"])

    # No AttributeError; Backend falls back to the manifest value.
    assert result.exit_code == 0, result.output
    assert "Backend:    anthropic" in result.output


def test_install_restart_uses_internal_helpers(monkeypatch) -> None:
    runner = CliRunner()
    calls: list[str] = []

    class Manifest:
        profile = "default"
        preset = "persistent-service"
        runtime_kind = "python"
        supervisor_kind = "service"
        scope = "user"
        health_url = "http://127.0.0.1:8787/readyz"
        mutations = [object()]

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
    monkeypatch.setattr(
        "headroom.cli.install.revert_mutations", lambda manifest: calls.append("revert")
    )
    monkeypatch.setattr(
        "headroom.cli.install.stop_supervisor", lambda manifest: calls.append("stop_supervisor")
    )
    monkeypatch.setattr(
        "headroom.cli.install.stop_runtime", lambda manifest: calls.append("stop_runtime")
    )
    monkeypatch.setattr(
        "headroom.cli.install.start_supervisor", lambda manifest: calls.append("start_supervisor")
    )
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda manifest, timeout_seconds=45: True
    )
    monkeypatch.setattr(
        "headroom.cli.install.apply_mutations", lambda manifest: calls.append("apply") or []
    )
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda manifest: calls.append("save"))
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: False)
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda manifest: "stopped")

    result = runner.invoke(main, ["install", "restart"])

    assert result.exit_code == 0, result.output
    assert "Restarted deployment 'default'." in result.output
    assert calls == [
        "revert",
        "save",
        "stop_supervisor",
        "stop_runtime",
        "start_supervisor",
        "apply",
        "save",
    ]


def test_install_start_noops_when_already_healthy(monkeypatch) -> None:
    runner = CliRunner()
    calls: list[str] = []

    class Manifest:
        profile = "default"
        preset = "persistent-service"
        runtime_kind = "python"
        supervisor_kind = "service"
        scope = "user"
        health_url = "http://127.0.0.1:8787/readyz"
        mutations = [object()]

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: True)
    monkeypatch.setattr(
        "headroom.cli.install.start_supervisor", lambda manifest: calls.append("start_supervisor")
    )

    result = runner.invoke(main, ["install", "start"])

    assert result.exit_code == 0, result.output
    assert "Started deployment 'default'." in result.output
    assert calls == []


def test_install_start_noops_for_healthy_docker_without_docker_on_path(monkeypatch) -> None:
    runner = CliRunner()

    class Manifest:
        profile = "default"
        preset = "persistent-docker"
        runtime_kind = "docker"
        supervisor_kind = "none"
        scope = "user"
        health_url = "http://127.0.0.1:8787/readyz"
        mutations = [object()]

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: True)
    monkeypatch.setattr("headroom.cli.install.shutil.which", lambda name, *args, **kwargs: None)

    result = runner.invoke(main, ["install", "start"])

    assert result.exit_code == 0, result.output
    assert "Started deployment 'default'." in result.output


def test_install_start_does_not_spawn_when_start_lock_is_contended(monkeypatch) -> None:
    runner = CliRunner()
    calls: list[str] = []

    class Manifest:
        profile = "default"
        preset = "persistent-service"
        runtime_kind = "python"
        supervisor_kind = "service"
        scope = "user"
        health_url = "http://127.0.0.1:8787/readyz"
        mutations = []

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: False)

    import contextlib

    @contextlib.contextmanager
    def fake_lock(profile):
        yield False

    monkeypatch.setattr("headroom.cli.install.acquire_runtime_start_lock", fake_lock)
    monkeypatch.setattr(
        "headroom.cli.install.start_supervisor", lambda manifest: calls.append("start_supervisor")
    )

    result = runner.invoke(main, ["install", "start"])

    assert result.exit_code == 0, result.output
    assert "start is already in progress" in result.output
    assert calls == []


def test_install_start_restarts_wedged_runtime_under_single_lock(monkeypatch) -> None:
    runner = CliRunner()
    calls: list[str] = []

    class Manifest:
        profile = "default"
        preset = "persistent-service"
        runtime_kind = "python"
        supervisor_kind = "service"
        scope = "user"
        health_url = "http://127.0.0.1:8787/readyz"
        mutations = [object()]

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
    probe_calls = {"count": 0}

    def fake_probe_ready(url: str) -> bool:
        probe_calls["count"] += 1
        return probe_calls["count"] > 2

    monkeypatch.setattr("headroom.cli.install.probe_ready", fake_probe_ready)
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda manifest: "running")
    wait_results = iter([False, True])
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda manifest, timeout_seconds: next(wait_results)
    )
    monkeypatch.setattr(
        "headroom.cli.install.revert_mutations", lambda manifest: calls.append("revert")
    )
    monkeypatch.setattr(
        "headroom.cli.install.apply_mutations", lambda manifest: calls.append("apply") or []
    )
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda manifest: calls.append("save"))
    monkeypatch.setattr("headroom.cli.install.stop_runtime", lambda manifest: calls.append("stop"))
    monkeypatch.setattr(
        "headroom.cli.install.start_supervisor", lambda manifest: calls.append("start_supervisor")
    )

    result = runner.invoke(main, ["install", "start"])

    assert result.exit_code == 0, result.output
    assert calls == ["revert", "save", "stop", "start_supervisor", "apply", "save"]


def test_install_apply_rejects_invalid_profile() -> None:
    runner = CliRunner()

    result = runner.invoke(main, ["install", "apply", "--profile", "../bad"])

    assert result.exit_code != 0
    assert "Invalid profile name '../bad'" in result.output


def test_install_apply_rejects_provider_scope_targets_without_support() -> None:
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["install", "apply", "--scope", "provider", "--providers", "manual", "--target", "copilot"],
    )

    assert result.exit_code != 0
    assert "Provider scope supports only claude, codex, openclaw, and opencode" in result.output


def test_install_apply_accepts_opencode_target(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    class Manifest:
        profile = "default"
        preset = "persistent-service"
        runtime_kind = "python"
        supervisor_kind = "service"
        scope = "provider"
        health_url = "http://127.0.0.1:8787/readyz"
        targets = ["opencode"]
        mutations = []
        artifacts = []

    manifest = Manifest()

    def fake_build_manifest(**kwargs):
        captured.update(kwargs)
        return manifest

    monkeypatch.setattr("headroom.cli.install.build_manifest", fake_build_manifest)
    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: None)
    monkeypatch.setattr("headroom.cli.install.apply_mutations", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.start_supervisor", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.start_detached_agent", lambda profile: None)
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45: True
    )

    result = runner.invoke(
        main,
        [
            "install",
            "apply",
            "--scope",
            "provider",
            "--providers",
            "manual",
            "--target",
            "opencode",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["targets"] == ["opencode"]
    assert "Targets: opencode" in result.output


def test_install_apply_restores_previous_deployment_after_failed_update(monkeypatch) -> None:
    runner = CliRunner()
    calls: list[str] = []

    class Manifest:
        def __init__(self, profile: str, targets: list[str]) -> None:
            self.profile = profile
            self.preset = "persistent-service"
            self.runtime_kind = "python"
            self.supervisor_kind = "service"
            self.scope = "user"
            self.health_url = "http://127.0.0.1:8787/readyz"
            self.targets = targets
            self.mutations = []
            self.artifacts = []

    new_manifest = Manifest("default", ["claude"])
    existing_manifest = Manifest("default", ["codex"])
    existing_manifest.mutations = [object()]

    monkeypatch.setattr("headroom.cli.install.build_manifest", lambda **_: new_manifest)
    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: existing_manifest)
    monkeypatch.setattr(
        "headroom.cli.install.apply_mutations",
        lambda deployment: calls.append(f"apply:{','.join(deployment.targets)}") or [],
    )
    monkeypatch.setattr(
        "headroom.cli.install.install_supervisor",
        lambda deployment: calls.append(f"supervisor:{','.join(deployment.targets)}") or [],
    )
    monkeypatch.setattr(
        "headroom.cli.install.save_manifest",
        lambda deployment: calls.append(f"save:{','.join(deployment.targets)}"),
    )
    monkeypatch.setattr(
        "headroom.cli.install.stop_supervisor",
        lambda deployment: calls.append(f"stop-supervisor:{','.join(deployment.targets)}"),
    )
    monkeypatch.setattr(
        "headroom.cli.install.stop_runtime",
        lambda deployment: calls.append(f"stop-runtime:{','.join(deployment.targets)}"),
    )
    monkeypatch.setattr(
        "headroom.cli.install.remove_supervisor",
        lambda deployment: calls.append(f"remove-supervisor:{','.join(deployment.targets)}"),
    )
    monkeypatch.setattr(
        "headroom.cli.install.revert_mutations",
        lambda deployment: calls.append(f"revert:{','.join(deployment.targets)}"),
    )
    monkeypatch.setattr(
        "headroom.cli.install.delete_manifest",
        lambda profile: calls.append(f"delete:{profile}"),
    )

    def _start(deployment) -> None:
        calls.append(f"start:{','.join(deployment.targets)}")
        if deployment is new_manifest:
            raise click.ClickException("boom")

    monkeypatch.setattr("headroom.cli.install._start_deployment", _start)

    result = runner.invoke(main, ["install", "apply"])

    assert result.exit_code != 0
    assert "Restoring previous deployment 'default'" in result.output
    assert calls == [
        "revert:codex",
        "stop-supervisor:codex",
        "stop-runtime:codex",
        "remove-supervisor:codex",
        "delete:default",
        "supervisor:claude",
        "save:claude",
        "start:claude",
        "stop-supervisor:claude",
        "stop-runtime:claude",
        "remove-supervisor:claude",
        "delete:default",
        "supervisor:codex",
        "save:codex",
        "start:codex",
        "apply:codex",
        "save:codex",
    ]


def test_install_start_rejects_task_lifecycle(monkeypatch) -> None:
    runner = CliRunner()

    class Manifest:
        profile = "default"
        preset = "persistent-task"
        runtime_kind = "python"
        supervisor_kind = "task"
        scope = "user"
        health_url = "http://127.0.0.1:8787/readyz"

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())

    result = runner.invoke(main, ["install", "start"])

    assert result.exit_code != 0
    assert "headroom install start" in result.output


def test_install_apply_uses_docker_runtime_for_persistent_docker(monkeypatch) -> None:
    runner = CliRunner()
    calls: list[str] = []

    class Manifest:
        profile = "default"
        preset = "persistent-docker"
        runtime_kind = "docker"
        supervisor_kind = "none"
        scope = "user"
        health_url = "http://127.0.0.1:8787/readyz"
        container_name = "headroom-default"
        targets: list[str] = []
        mutations = []
        artifacts = []

    monkeypatch.setattr("headroom.cli.install.build_manifest", lambda **_: Manifest())
    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: None)
    monkeypatch.setattr("headroom.cli.install.apply_mutations", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: False)
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda deployment: "stopped")

    import contextlib

    @contextlib.contextmanager
    def fake_lock(profile):
        yield True

    monkeypatch.setattr("headroom.cli.install.acquire_runtime_start_lock", fake_lock)
    monkeypatch.setattr(
        "headroom.cli.install.start_persistent_docker",
        lambda deployment: calls.append("start_docker"),
    )
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45: True
    )
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: False)
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda deployment: "stopped")
    # _start_deployment guards the persistent-docker preset with
    # `shutil.which("docker")`. Fake docker as present so the test exercises the
    # runtime-selection path itself rather than the host's docker install —
    # otherwise it passes on dev machines with Docker but fails on CI runners
    # (e.g. macos-latest) that have no docker on PATH.
    monkeypatch.setattr(
        "headroom.cli.install.shutil.which",
        lambda name, *args, **kwargs: "/usr/local/bin/docker" if name == "docker" else None,
    )

    result = runner.invoke(main, ["install", "apply", "--preset", "persistent-docker"])

    assert result.exit_code == 0, result.output
    assert calls == ["start_docker"]


def test_deploy_prefers_docker_when_available(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}
    calls: list[str] = []

    class Manifest:
        profile = "default"
        preset = "persistent-docker"
        runtime_kind = "docker"
        supervisor_kind = "none"
        scope = "user"
        health_url = "http://127.0.0.1:8787/readyz"
        targets = ["claude", "codex"]
        mutations = []
        artifacts = []

    def fake_build(**kwargs):
        captured.update(kwargs)
        return Manifest()

    monkeypatch.setattr(
        "headroom.cli.install._command_available", lambda command: command == "docker"
    )
    monkeypatch.setattr(
        "headroom.cli.install.shutil.which",
        lambda name, *args, **kwargs: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr("headroom.cli.install.build_manifest", fake_build)
    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: None)
    monkeypatch.setattr("headroom.cli.install.apply_mutations", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: False)
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda deployment: "stopped")

    import contextlib

    @contextlib.contextmanager
    def fake_lock(profile):
        yield True

    monkeypatch.setattr("headroom.cli.install.acquire_runtime_start_lock", fake_lock)
    monkeypatch.setattr(
        "headroom.cli.install.start_persistent_docker",
        lambda deployment: calls.append("start_docker"),
    )
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45: True
    )

    result = runner.invoke(main, ["deploy"])

    assert result.exit_code == 0, result.output
    assert "Selected persistent-docker" in result.output
    assert "Deployed turnkey deployment 'default'" in result.output
    assert captured["preset"] == "persistent-docker"
    assert captured["runtime_kind"] == "docker"
    assert calls == ["start_docker"]


def test_deploy_prefers_gpu_docker_when_available(monkeypatch) -> None:
    runner = CliRunner()
    captured: dict[str, object] = {}

    class Manifest:
        profile = "default"
        preset = "persistent-docker"
        runtime_kind = "docker"
        supervisor_kind = "none"
        scope = "user"
        health_url = "http://127.0.0.1:8787/readyz"
        targets: list[str] = []
        base_env: dict[str, str] = {}
        mutations = []
        artifacts = []

    manifest = Manifest()

    def fake_build(**kwargs):
        captured.update(kwargs)
        return manifest

    monkeypatch.setattr("headroom.cli.install._detect_nvidia_gpu_names", lambda: ["RTX 4090"])
    monkeypatch.setattr("headroom.cli.install._docker_supports_nvidia_gpus", lambda: True)
    monkeypatch.setattr(
        "headroom.cli.install.shutil.which",
        lambda name, *args, **kwargs: "/usr/local/bin/docker" if name == "docker" else None,
    )
    monkeypatch.setattr("headroom.cli.install.build_manifest", fake_build)
    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: None)
    monkeypatch.setattr("headroom.cli.install.apply_mutations", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: False)
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda deployment: "stopped")

    import contextlib

    @contextlib.contextmanager
    def fake_lock(profile):
        yield True

    monkeypatch.setattr("headroom.cli.install.acquire_runtime_start_lock", fake_lock)
    monkeypatch.setattr("headroom.cli.install.start_persistent_docker", lambda deployment: None)
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45: True
    )

    result = runner.invoke(main, ["deploy"])

    assert result.exit_code == 0, result.output
    assert "RTX 4090" in result.output
    assert captured["preset"] == "persistent-docker"
    assert captured["runtime_kind"] == "docker"
    assert manifest.base_env["HEADROOM_DOCKER_GPUS"] == "all"


def test_deploy_falls_back_to_detached_python_without_supervisor(monkeypatch) -> None:
    runner = CliRunner()
    calls: list[str] = []

    class Manifest:
        profile = "default"
        preset = "persistent-task"
        runtime_kind = "python"
        supervisor_kind = "task"
        scope = "user"
        health_url = "http://127.0.0.1:8787/readyz"
        targets: list[str] = []
        mutations = []
        artifacts = []

    manifest = Manifest()

    monkeypatch.setattr("headroom.cli.install._command_available", lambda command: False)
    monkeypatch.setattr("headroom.cli.install.build_manifest", lambda **_: manifest)
    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: None)
    monkeypatch.setattr("headroom.cli.install.apply_mutations", lambda deployment: [])
    monkeypatch.setattr(
        "headroom.cli.install.install_supervisor",
        lambda deployment: calls.append(f"supervisor:{deployment.supervisor_kind}") or [],
    )
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: False)
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda deployment: "stopped")

    import contextlib

    @contextlib.contextmanager
    def fake_lock(profile):
        yield True

    monkeypatch.setattr("headroom.cli.install.acquire_runtime_start_lock", fake_lock)
    monkeypatch.setattr(
        "headroom.cli.install.start_detached_agent",
        lambda profile: calls.append(f"agent:{profile}"),
    )
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45: True
    )

    result = runner.invoke(main, ["deploy", "--no-docker"])

    assert result.exit_code == 0, result.output
    assert "No supported supervisor was detected" in result.output
    assert manifest.supervisor_kind == "none"
    assert calls == ["supervisor:none", "agent:default"]


def test_install_remove_continues_when_runtime_teardown_errors(monkeypatch) -> None:
    runner = CliRunner()
    calls: list[str] = []

    class Manifest:
        profile = "default"
        preset = "persistent-service"
        runtime_kind = "python"
        supervisor_kind = "service"
        scope = "user"
        health_url = "http://127.0.0.1:8787/readyz"
        mutations = [object()]

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
    monkeypatch.setattr(
        "headroom.cli.install.revert_mutations", lambda manifest: calls.append("revert")
    )
    monkeypatch.setattr(
        "headroom.cli.install.stop_supervisor",
        lambda manifest: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        "headroom.cli.install.stop_runtime",
        lambda manifest: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        "headroom.cli.install.remove_supervisor", lambda manifest: calls.append("remove_supervisor")
    )
    monkeypatch.setattr(
        "headroom.cli.install.delete_manifest", lambda profile: calls.append("delete")
    )

    result = runner.invoke(main, ["install", "remove"])

    assert result.exit_code == 0, result.output
    assert calls == ["revert", "remove_supervisor", "delete"]


def test_install_agent_ensure_reports_already_healthy(monkeypatch) -> None:
    runner = CliRunner()

    class Manifest:
        profile = "default"
        health_url = "http://127.0.0.1:8787/readyz"

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: True)

    result = runner.invoke(main, ["install", "agent", "ensure"])

    assert result.exit_code == 0, result.output
    assert "already healthy" in result.output


def test_install_agent_run_exits_with_foreground_status(monkeypatch) -> None:
    runner = CliRunner()

    class Manifest:
        profile = "default"
        health_url = "http://127.0.0.1:8787/readyz"

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
    monkeypatch.setattr("headroom.cli.install.run_foreground", lambda manifest: 7)

    result = runner.invoke(main, ["install", "agent", "run"])

    assert result.exit_code == 7


def test_install_agent_ensure_no_spawn_when_lock_not_acquired(monkeypatch) -> None:
    """Ensure does not spawn a runtime when the start lock is contended."""
    runner = CliRunner()
    calls: list[str] = []

    class Manifest:
        profile = "default"
        health_url = "http://127.0.0.1:8787/readyz"

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: False)

    import contextlib

    @contextlib.contextmanager
    def fake_lock(profile):
        yield False

    monkeypatch.setattr("headroom.cli.install.acquire_runtime_start_lock", fake_lock)
    monkeypatch.setattr(
        "headroom.cli.install.start_detached_agent",
        lambda profile: calls.append("start_agent"),
    )
    monkeypatch.setattr(
        "headroom.cli.install.start_persistent_docker",
        lambda manifest: calls.append("start_docker"),
    )

    result = runner.invoke(main, ["install", "agent", "ensure"])
    assert result.exit_code == 0, result.output
    assert "already in progress" in result.output
    assert calls == []


def test_install_agent_ensure_stops_wedged_runtime_before_restart(monkeypatch) -> None:
    """Ensure stops a wedged runtime (running but not ready) before starting fresh."""
    runner = CliRunner()
    calls: list[str] = []

    class Manifest:
        profile = "default"
        health_url = "http://127.0.0.1:8787/readyz"
        preset = "persistent-task"
        supervisor_kind = "none"
        scope = "user"
        mutations = []
        scope = "user"
        mutations = []
        scope = "user"
        mutations = []
        scope = "user"
        mutations = [object()]

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: False)
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda manifest: "running")
    monkeypatch.setattr("headroom.cli.install.wait_ready", lambda manifest, timeout_seconds: False)
    monkeypatch.setattr(
        "headroom.cli.install.revert_mutations", lambda manifest: calls.append("revert")
    )
    monkeypatch.setattr(
        "headroom.cli.install.apply_mutations", lambda manifest: calls.append("apply") or []
    )
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda manifest: calls.append("save"))
    monkeypatch.setattr("headroom.cli.install.stop_runtime", lambda manifest: calls.append("stop"))
    monkeypatch.setattr(
        "headroom.cli.install.start_detached_agent",
        lambda profile: calls.append("start_agent"),
    )
    monkeypatch.setattr(
        "headroom.cli.install.start_persistent_docker",
        lambda manifest: calls.append("start_docker"),
    )

    import contextlib

    @contextlib.contextmanager
    def fake_lock(profile):
        yield True

    monkeypatch.setattr("headroom.cli.install.acquire_runtime_start_lock", fake_lock)
    monkeypatch.setattr(
        "headroom.cli.install._start_deployment",
        lambda manifest, **kwargs: calls.append("start_deployment"),
    )

    result = runner.invoke(main, ["install", "agent", "ensure"])
    assert result.exit_code == 0, result.output
    # stop must come before start_deployment — that's the bug guard.
    assert calls.index("revert") < calls.index("stop")
    assert calls.index("stop") < calls.index("start_deployment")
    assert calls.index("start_deployment") < calls.index("apply")
    assert "start_agent" not in calls
    assert "start_docker" not in calls


def test_install_agent_ensure_starts_when_stopped_and_lock_acquired(monkeypatch) -> None:
    """Ensure starts a runtime when none is running and lock is acquired."""
    runner = CliRunner()
    calls: list[str] = []

    class Manifest:
        profile = "default"
        health_url = "http://127.0.0.1:8787/readyz"
        preset = "persistent-task"
        supervisor_kind = "none"
        scope = "user"
        mutations = []

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: False)
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda manifest: "stopped")
    monkeypatch.setattr(
        "headroom.cli.install.apply_mutations", lambda manifest: calls.append("apply") or []
    )
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda manifest: calls.append("save"))
    monkeypatch.setattr(
        "headroom.cli.install.start_detached_agent",
        lambda profile: calls.append("start_agent"),
    )
    monkeypatch.setattr(
        "headroom.cli.install.start_persistent_docker",
        lambda manifest: calls.append("start_docker"),
    )

    import contextlib

    @contextlib.contextmanager
    def fake_lock(profile):
        yield True

    monkeypatch.setattr("headroom.cli.install.acquire_runtime_start_lock", fake_lock)
    monkeypatch.setattr("headroom.cli.install.wait_ready", lambda manifest, timeout_seconds: True)

    result = runner.invoke(main, ["install", "agent", "ensure"])
    assert result.exit_code == 0, result.output
    assert calls == ["start_agent", "apply", "save"]


def test_install_agent_ensure_no_duplicate_spawn_after_lock_recheck(monkeypatch) -> None:
    """Ensure does not spawn if proxy becomes ready between initial probe and lock."""
    runner = CliRunner()
    calls: list[str] = []

    class Manifest:
        profile = "default"
        health_url = "http://127.0.0.1:8787/readyz"

    # First probe_ready (before lock) returns False, second (after lock) returns True
    probe_results = iter([False, True])
    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: next(probe_results))

    monkeypatch.setattr(
        "headroom.cli.install.start_detached_agent",
        lambda profile: calls.append("start_agent"),
    )

    import contextlib

    @contextlib.contextmanager
    def fake_lock(profile):
        yield True

    monkeypatch.setattr("headroom.cli.install.acquire_runtime_start_lock", fake_lock)

    result = runner.invoke(main, ["install", "agent", "ensure"])
    assert result.exit_code == 0, result.output
    assert "already healthy" in result.output
    assert calls == []


def test_install_agent_ensure_propagates_start_deployment_failure(monkeypatch) -> None:
    """Ensure must exit non-zero and surface the error when _start_deployment fails.

    Regression for review feedback on PR #1301: the previous implementation wrapped
    the guarded block in `except Exception` and returned normally, which made
    a failed ensure indistinguishable from a successful one. Automation callers
    need a non-zero exit code to detect that the deployment did not come up.
    """
    runner = CliRunner()

    class Manifest:
        profile = "default"
        health_url = "http://127.0.0.1:8787/readyz"
        preset = "persistent-task"
        supervisor_kind = "none"
        scope = "user"
        mutations = []

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: False)
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda manifest: "stopped")

    import contextlib

    @contextlib.contextmanager
    def fake_lock(profile):
        yield True

    monkeypatch.setattr("headroom.cli.install.acquire_runtime_start_lock", fake_lock)

    def boom(manifest, **kwargs):
        raise click.ClickException("simulated start failure")

    monkeypatch.setattr("headroom.cli.install._start_deployment", boom)

    result = runner.invoke(main, ["install", "agent", "ensure"])
    assert result.exit_code != 0, f"expected non-zero exit, got {result.exit_code}: {result.output}"
    assert "simulated start failure" in result.output
