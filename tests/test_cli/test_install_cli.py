from __future__ import annotations

import click
from click.testing import CliRunner

from headroom.cli.main import main


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
        targets = ["claude", "codex"]
        mutations = []
        artifacts = []

    manifest = Manifest()

    monkeypatch.setattr("headroom.cli.install.build_manifest", lambda **_: manifest)
    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: None)
    monkeypatch.setattr("headroom.cli.install.apply_mutations", lambda deployment: [])
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

    result = runner.invoke(main, ["install", "apply"])

    assert result.exit_code == 0, result.output
    assert "Installed persistent deployment 'default'" in result.output
    assert "Targets: claude, codex" in result.output
    assert calls == ["save", "start_service"]


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

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
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

    result = runner.invoke(main, ["install", "restart"])

    assert result.exit_code == 0, result.output
    assert "Restarted deployment 'default'." in result.output
    assert calls == ["stop_supervisor", "stop_runtime", "start_supervisor"]


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
        "stop-supervisor:codex",
        "stop-runtime:codex",
        "remove-supervisor:codex",
        "revert:codex",
        "delete:default",
        "apply:claude",
        "supervisor:claude",
        "save:claude",
        "start:claude",
        "stop-supervisor:claude",
        "stop-runtime:claude",
        "remove-supervisor:claude",
        "revert:claude",
        "delete:default",
        "apply:codex",
        "supervisor:codex",
        "save:codex",
        "start:codex",
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
        targets: list[str] = []
        mutations = []
        artifacts = []

    monkeypatch.setattr("headroom.cli.install.build_manifest", lambda **_: Manifest())
    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: None)
    monkeypatch.setattr("headroom.cli.install.apply_mutations", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.install_supervisor", lambda deployment: [])
    monkeypatch.setattr("headroom.cli.install.save_manifest", lambda deployment: None)
    monkeypatch.setattr(
        "headroom.cli.install.start_persistent_docker",
        lambda deployment: calls.append("start_docker"),
    )
    monkeypatch.setattr(
        "headroom.cli.install.wait_ready", lambda deployment, timeout_seconds=45: True
    )

    result = runner.invoke(main, ["install", "apply", "--preset", "persistent-docker"])

    assert result.exit_code == 0, result.output
    assert calls == ["start_docker"]


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

    monkeypatch.setattr("headroom.cli.install.load_manifest", lambda profile: Manifest())
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
        "headroom.cli.install.revert_mutations", lambda manifest: calls.append("revert")
    )
    monkeypatch.setattr(
        "headroom.cli.install.delete_manifest", lambda profile: calls.append("delete")
    )

    result = runner.invoke(main, ["install", "remove"])

    assert result.exit_code == 0, result.output
    assert calls == ["remove_supervisor", "revert", "delete"]


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
