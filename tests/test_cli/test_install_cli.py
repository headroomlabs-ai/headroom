from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from headroom.cli import install as inst
from headroom.cli.main import main
from headroom.install.models import DeploymentManifest, ManagedMutation
from headroom.install.state import load_manifest as load_state_manifest


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
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment, **kwargs: [])
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
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45, **kwargs: True
    )
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: False)
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda manifest: "stopped")

    result = runner.invoke(main, ["install", "apply"])

    assert result.exit_code == 0, result.output
    assert "Installed persistent deployment 'default'" in result.output
    assert "Targets: claude, codex" in result.output
    assert calls == ["save", "save", "start_service", "apply", "save"]


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
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment, **kwargs: [])
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.apply_mutations", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: False)
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda manifest: "stopped")
    monkeypatch.setattr(
        "headroom.cli.install.start_detached_agent", lambda profile: calls.append("start_agent")
    )
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45, **kwargs: True
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
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment, **kwargs: [])
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.start_supervisor", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.start_detached_agent", lambda profile: None)
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45, **kwargs: True
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
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment, **kwargs: [])
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.start_supervisor", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.start_detached_agent", lambda profile: None)
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45, **kwargs: True
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
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment, **kwargs: [])
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.start_supervisor", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.start_detached_agent", lambda profile: None)
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45, **kwargs: True
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
    monkeypatch.setattr(
        "headroom.cli.install.runtime_status",
        lambda manifest: "running",
    )
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
        "headroom.cli.install.wait_ready", lambda manifest, timeout_seconds=45, **kwargs: True
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


@pytest.mark.parametrize("supervisor_kind", ["service", "task"])
def test_stop_deployment_stops_external_supervisor_before_docker(
    monkeypatch, supervisor_kind: str
) -> None:
    calls: list[str] = []
    manifest = SimpleNamespace(
        profile="default",
        preset="persistent-service",
        runtime_kind="docker",
        supervisor_kind=supervisor_kind,
    )
    monkeypatch.setattr(inst, "stop_supervisor", lambda current: calls.append("supervisor"))
    monkeypatch.setattr(inst, "stop_runtime", lambda current: calls.append("runtime"))

    inst._stop_deployment(manifest)

    assert calls == ["supervisor", "runtime"]


def test_stop_deployment_stops_docker_even_when_supervisor_stop_fails(monkeypatch) -> None:
    calls: list[str] = []
    manifest = SimpleNamespace(
        profile="default",
        preset="persistent-service",
        runtime_kind="docker",
        supervisor_kind="service",
    )

    def fail_supervisor(current):
        calls.append("supervisor")
        raise RuntimeError("supervisor unavailable")

    monkeypatch.setattr(inst, "stop_supervisor", fail_supervisor)
    monkeypatch.setattr(inst, "stop_runtime", lambda current: calls.append("runtime"))

    with pytest.raises(RuntimeError, match="supervisor unavailable"):
        inst._stop_deployment(manifest)
    assert calls == ["supervisor", "runtime"]


def test_remove_deployment_retains_manifest_when_supervisor_stop_fails(monkeypatch) -> None:
    calls: list[str] = []
    manifest = SimpleNamespace(
        profile="default",
        preset="persistent-service",
        runtime_kind="docker",
        supervisor_kind="service",
        mutations=[],
    )
    monkeypatch.setattr(
        inst,
        "stop_supervisor",
        lambda current: (_ for _ in ()).throw(RuntimeError("supervisor unavailable")),
    )
    monkeypatch.setattr(inst, "stop_runtime", lambda current: calls.append("runtime"))
    monkeypatch.setattr(
        inst, "remove_supervisor", lambda current: calls.append("remove-supervisor")
    )
    monkeypatch.setattr(inst, "delete_manifest", lambda profile: calls.append("delete"))

    with pytest.raises(RuntimeError, match="supervisor unavailable"):
        inst._remove_deployment(manifest)
    assert calls == ["runtime", "remove-supervisor"]


def test_remove_deployment_reports_all_cleanup_failures(monkeypatch) -> None:
    manifest = SimpleNamespace(profile="default", mutations=[object()])
    monkeypatch.setattr(
        inst,
        "_deactivate_deployment_mutations",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("mutation failed")),
    )
    monkeypatch.setattr(
        inst,
        "_stop_deployment",
        lambda current: (_ for _ in ()).throw(RuntimeError("runtime failed")),
    )
    monkeypatch.setattr(
        inst,
        "remove_supervisor",
        lambda current: (_ for _ in ()).throw(RuntimeError("supervisor failed")),
    )

    with pytest.raises(RuntimeError) as exc:
        inst._remove_deployment(manifest)

    message = str(exc.value)
    assert all(
        item in message for item in ("mutation failed", "runtime failed", "supervisor failed")
    )


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
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda manifest: "running")
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: True)
    monkeypatch.setattr(
        "headroom.cli.install.start_supervisor", lambda manifest: calls.append("start_supervisor")
    )

    result = runner.invoke(main, ["install", "start"])

    assert result.exit_code == 0, result.output
    assert "Started deployment 'default'." in result.output
    assert calls == []


def test_start_deployment_requires_identity_for_post_start_readiness(monkeypatch) -> None:
    manifest = SimpleNamespace(
        profile="default",
        preset="persistent-task",
        runtime_kind="python",
        supervisor_kind="none",
        health_url="http://127.0.0.1:8787/readyz",
    )
    waits: list[dict[str, object]] = []
    monkeypatch.setattr(inst, "runtime_status", lambda current: "stopped")
    monkeypatch.setattr(inst, "start_detached_agent", lambda profile: None)
    monkeypatch.setattr(
        inst,
        "wait_ready",
        lambda current, timeout_seconds, **kwargs: waits.append(kwargs) or True,
    )

    inst._start_deployment(manifest, assume_start_lock=True)

    assert waits == [{"require_identity": True}]


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
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda manifest: "running")
    monkeypatch.setattr("headroom.cli.install.probe_ready", lambda url: True)
    monkeypatch.setattr("headroom.cli.install.shutil.which", lambda name, *args, **kwargs: None)

    result = runner.invoke(main, ["install", "start"])

    assert result.exit_code != 0
    assert "docker' was not found on PATH" in result.output


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
        "headroom.cli.install.wait_ready",
        lambda manifest, timeout_seconds, **kwargs: next(wait_results),
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
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment, **kwargs: [])
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.start_supervisor", lambda deployment: None)
    monkeypatch.setattr("headroom.cli.install.start_detached_agent", lambda profile: None)
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45, **kwargs: True
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
        lambda deployment, **kwargs: (
            calls.append(f"supervisor:{','.join(deployment.targets)}") or []
        ),
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
    monkeypatch.setattr(
        "headroom.cli.install.delete_recovery_manifest",
        lambda profile: calls.append(f"delete-recovery:{profile}"),
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
        "save:claude",
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
        "delete-recovery:default",
    ]


def test_install_apply_reports_restore_failure_with_recovery_path(monkeypatch) -> None:
    previous = SimpleNamespace(profile="default")
    new = SimpleNamespace(profile="default", artifacts=[], mutations=[])
    monkeypatch.setattr(inst, "load_manifest", lambda profile: previous)
    monkeypatch.setattr(inst, "_save_recovery_snapshot", lambda *args: None)
    monkeypatch.setattr(inst, "_remove_deployment", lambda manifest: None)
    monkeypatch.setattr(inst, "_save_apply_manifest", lambda manifest: None)
    monkeypatch.setattr(inst, "install_supervisor", lambda manifest, **kwargs: [])
    monkeypatch.setattr(
        inst,
        "_start_deployment",
        lambda manifest: (_ for _ in ()).throw(click.ClickException("startup failed")),
    )
    monkeypatch.setattr(
        inst,
        "_restore_deployment",
        lambda manifest: (_ for _ in ()).throw(RuntimeError("restore failed")),
    )

    with pytest.raises(click.ClickException) as exc:
        inst._apply_manifest(new)

    message = str(exc.value)
    assert "restore failed" in message
    assert "recovery snapshot:" in message
    assert "restore it after resolving the failure" in message


def test_install_apply_old_removal_failure_is_actionable_and_does_not_start_new(
    monkeypatch,
) -> None:
    previous = SimpleNamespace(profile="default")
    new = SimpleNamespace(profile="default", artifacts=[], mutations=[])
    calls: list[str] = []
    monkeypatch.setattr(inst, "load_manifest", lambda profile: previous)
    monkeypatch.setattr(inst, "_save_recovery_snapshot", lambda *args: calls.append("snapshot"))
    monkeypatch.setattr(
        inst,
        "_remove_deployment",
        lambda manifest: (_ for _ in ()).throw(RuntimeError("old removal failed")),
    )
    monkeypatch.setattr(inst, "_start_deployment", lambda manifest: calls.append("start"))

    with pytest.raises(click.ClickException) as exc:
        inst._apply_manifest(new)

    assert "old removal failed" in str(exc.value)
    assert "no new owner was started" in str(exc.value)
    assert calls == ["snapshot"]


def test_install_apply_uses_requested_profile_for_recovery_snapshot(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    previous = DeploymentManifest(
        profile="loaded-profile",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        provider_mode="manual",
        targets=["old"],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
    )
    new = DeploymentManifest(
        profile="requested-profile",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        provider_mode="manual",
        targets=["new"],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
    )
    monkeypatch.setattr(inst, "load_manifest", lambda profile: previous)
    monkeypatch.setattr(inst, "_remove_deployment", lambda manifest: None)
    monkeypatch.setattr(inst, "_save_apply_manifest", lambda manifest: None)
    monkeypatch.setattr(inst, "install_supervisor", lambda manifest, **kwargs: [])
    monkeypatch.setattr(
        inst,
        "_start_deployment",
        lambda manifest: (_ for _ in ()).throw(click.ClickException("startup failed")),
    )

    with pytest.raises(click.ClickException):
        inst._apply_manifest(new)

    assert (tmp_path / ".headroom" / "deploy" / "requested-profile.recovery.json").exists()
    assert not (tmp_path / ".headroom" / "deploy" / "loaded-profile.recovery.json").exists()


def test_restore_deployment_requires_durable_manifest_before_start(monkeypatch) -> None:
    manifest = DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        provider_mode="manual",
        targets=[],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
    )
    calls: list[str] = []
    monkeypatch.setattr(inst, "install_supervisor", lambda current, **kwargs: [])
    monkeypatch.setattr(
        inst,
        "_save_apply_manifest",
        lambda current: (_ for _ in ()).throw(OSError("restore manifest busy")),
    )
    monkeypatch.setattr(inst, "_start_deployment", lambda current: calls.append("start"))

    with pytest.raises(OSError, match="restore manifest busy"):
        inst._restore_deployment(manifest)
    assert calls == []


def test_apply_persists_before_darwin_bootstrap_or_start_on_save_failure(monkeypatch) -> None:
    manifest = DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        provider_mode="manual",
        targets=[],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
    )
    install_calls: list[bool] = []
    bootstrap_calls: list[str] = []
    start_calls: list[str] = []
    save_calls = 0

    def save(current) -> None:
        nonlocal save_calls
        save_calls += 1
        if save_calls == 2:
            raise OSError("active manifest busy")

    def install(current, *, start=True):
        install_calls.append(start)
        if start:
            bootstrap_calls.append(current.profile)
        return []

    monkeypatch.setattr(inst, "load_manifest", lambda profile: None)
    monkeypatch.setattr(inst, "_save_apply_manifest", save)
    monkeypatch.setattr(inst, "install_supervisor", install)
    monkeypatch.setattr(
        inst, "_start_deployment", lambda current: start_calls.append(current.profile)
    )
    monkeypatch.setattr(inst, "_remove_deployment", lambda current: None)

    with pytest.raises(click.ClickException, match="active manifest busy"):
        inst._apply_manifest(manifest)

    assert install_calls == [False]
    assert bootstrap_calls == []
    assert start_calls == []


def test_activate_mutations_reverts_side_effects_when_strict_save_fails(monkeypatch) -> None:
    manifest = DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        provider_mode="manual",
        targets=[],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
    )
    mutation = ManagedMutation(target="env", kind="shell-block", path="settings")
    reverted: list[ManagedMutation] = []
    monkeypatch.setattr(inst, "apply_mutations", lambda current: [mutation])
    monkeypatch.setattr(
        inst,
        "_save_apply_manifest",
        lambda current: (_ for _ in ()).throw(OSError("manifest busy")),
    )
    monkeypatch.setattr(
        inst, "revert_mutations", lambda current: reverted.extend(current.mutations)
    )

    with pytest.raises(OSError, match="manifest busy"):
        inst._activate_deployment_mutations(manifest)
    assert reverted == [mutation]
    assert manifest.mutations == []


def test_recovery_snapshot_delete_failure_is_actionable(monkeypatch) -> None:
    monkeypatch.setattr(
        inst,
        "delete_recovery_manifest",
        lambda profile: (_ for _ in ()).throw(OSError("snapshot busy")),
    )

    with pytest.raises(click.ClickException, match="could not be deleted") as exc:
        inst._delete_recovery_snapshot("default")
    assert "snapshot busy" in str(exc.value)
    assert "retained" in str(exc.value)


def test_install_apply_keeps_new_owner_recovery_fail_closed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    previous = DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        provider_mode="manual",
        targets=["old"],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
    )
    new = DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        provider_mode="manual",
        targets=["new"],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
    )
    calls: list[str] = []

    monkeypatch.setattr(inst, "load_manifest", lambda profile: previous)
    real_save_recovery = inst._save_recovery_snapshot

    def save_recovery(manifest, profile):
        real_save_recovery(manifest, profile)
        calls.append("snapshot")

    monkeypatch.setattr(inst, "_save_recovery_snapshot", save_recovery)

    def remove(manifest):
        calls.append("remove-new" if manifest is new else "remove-old")
        if manifest is new:
            raise RuntimeError("new deployment cleanup failed")

    monkeypatch.setattr(inst, "_remove_deployment", remove)
    real_save_apply = inst._save_apply_manifest

    def save_apply(manifest):
        real_save_apply(manifest)
        calls.append("save-new")

    monkeypatch.setattr(inst, "_save_apply_manifest", save_apply)
    monkeypatch.setattr(
        inst, "install_supervisor", lambda manifest, **kwargs: calls.append("install-new") or []
    )
    monkeypatch.setattr(
        inst,
        "_start_deployment",
        lambda manifest: (_ for _ in ()).throw(click.ClickException("startup failed")),
    )
    monkeypatch.setattr(inst, "_restore_deployment", lambda manifest: calls.append("restore-old"))
    monkeypatch.setattr(
        inst, "delete_recovery_manifest", lambda profile: calls.append("delete-snapshot")
    )

    with pytest.raises(click.ClickException) as exc:
        inst._apply_manifest(new)

    message = str(exc.value)
    assert "startup failed" in message
    assert "new deployment cleanup failed" in message
    assert "recovery snapshot" in message
    assert "remove the new owner before restoring the snapshot" in message
    assert calls == ["snapshot", "remove-old", "save-new", "install-new", "save-new", "remove-new"]
    assert load_state_manifest("default") is not None
    recovery = tmp_path / ".headroom" / "deploy" / "default.recovery.json"
    assert json.loads(recovery.read_text(encoding="utf-8"))["targets"] == ["old"]


def test_install_apply_keeps_snapshot_when_active_persistence_fails(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    previous = DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        provider_mode="manual",
        targets=["old"],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
    )
    new = DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        provider_mode="manual",
        targets=["new"],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
    )
    calls: list[str] = []
    monkeypatch.setattr(inst, "load_manifest", lambda profile: previous)
    monkeypatch.setattr(inst, "_remove_deployment", lambda manifest: calls.append("cleanup"))
    monkeypatch.setattr(inst, "install_supervisor", lambda manifest, **kwargs: [])
    saves = 0

    def fail_active_save(manifest):
        nonlocal saves
        saves += 1
        if saves == 1:
            inst.save_manifest_strict(manifest)
            return
        raise OSError("active manifest busy")

    monkeypatch.setattr(inst, "_save_apply_manifest", fail_active_save)
    monkeypatch.setattr(inst, "_restore_deployment", lambda manifest: calls.append("restore"))
    monkeypatch.setattr(inst, "delete_recovery_manifest", lambda profile: calls.append("delete"))

    with pytest.raises(click.ClickException) as exc:
        inst._apply_manifest(new)

    assert "active manifest persistence failed" in str(exc.value)
    assert "recovery snapshot:" in str(exc.value)
    assert calls == ["cleanup", "cleanup"]
    assert load_state_manifest("default") is not None
    assert (tmp_path / ".headroom" / "deploy" / "default.recovery.json").exists()


def test_install_apply_persists_new_manifest_before_supervisor_install(monkeypatch) -> None:
    new = SimpleNamespace(profile="default", artifacts=[], mutations=[])
    calls: list[str] = []

    monkeypatch.setattr(inst, "load_manifest", lambda profile: None)
    monkeypatch.setattr(inst, "_save_apply_manifest", lambda manifest: calls.append("save"))
    monkeypatch.setattr(
        inst,
        "install_supervisor",
        lambda manifest, **kwargs: (_ for _ in ()).throw(RuntimeError("install failed")),
    )
    monkeypatch.setattr(inst, "_remove_deployment", lambda manifest: calls.append("cleanup"))

    with pytest.raises(click.ClickException, match="install failed"):
        inst._apply_manifest(new)

    assert calls == ["save", "cleanup"]


def test_install_apply_without_previous_has_no_recovery_guidance(monkeypatch) -> None:
    new = SimpleNamespace(profile="default", artifacts=[], mutations=[])
    monkeypatch.setattr(inst, "load_manifest", lambda profile: None)
    monkeypatch.setattr(inst, "_save_apply_manifest", lambda manifest: None)
    monkeypatch.setattr(inst, "install_supervisor", lambda manifest, **kwargs: [])
    monkeypatch.setattr(
        inst,
        "_start_deployment",
        lambda manifest: (_ for _ in ()).throw(click.ClickException("startup failed")),
    )
    monkeypatch.setattr(
        inst,
        "_remove_deployment",
        lambda manifest: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )

    with pytest.raises(click.ClickException) as exc:
        inst._apply_manifest(new)

    message = str(exc.value)
    assert "startup failed" in message and "cleanup failed" in message
    assert "recovery snapshot" not in message


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
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment, **kwargs: [])
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
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45, **kwargs: True
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
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment, **kwargs: [])
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
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45, **kwargs: True
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
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment, **kwargs: [])
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
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45, **kwargs: True
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
        lambda deployment, **kwargs: calls.append(f"supervisor:{deployment.supervisor_kind}") or [],
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
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45, **kwargs: True
    )

    result = runner.invoke(main, ["deploy", "--no-docker"])

    assert result.exit_code == 0, result.output
    assert "No supported supervisor was detected" in result.output
    assert manifest.supervisor_kind == "none"
    assert calls == ["supervisor:none", "agent:default"]


def test_install_remove_retains_manifest_when_runtime_teardown_errors(monkeypatch) -> None:
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

    assert result.exit_code != 0
    assert "Error: Failed to remove deployment 'default': cleanup failed:" in result.output
    assert "Traceback" not in result.output
    assert calls == ["revert", "remove_supervisor"]


def test_install_agent_ensure_reports_already_healthy(monkeypatch) -> None:
    runner = CliRunner()

    class Manifest:
        profile = "default"
        health_url = "http://127.0.0.1:8787/readyz"
        mutations = []

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda manifest: "running")
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
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda manifest: "stopped")
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
    wait_calls: list[dict[str, object]] = []

    def fake_wait_ready(manifest, timeout_seconds, **kwargs):
        wait_calls.append(kwargs)
        return False

    monkeypatch.setattr("headroom.cli.install.wait_ready", fake_wait_ready)
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
    assert wait_calls == [{"require_identity": True}]
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
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda manifest, timeout_seconds, **kwargs: True
    )

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
    monkeypatch.setattr("headroom.cli.install.runtime_status", lambda manifest: "running")
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
