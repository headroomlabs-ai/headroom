from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.install.planner import build_manifest
from headroom.install.runtime import build_runtime_command
from headroom.install.state import load_manifest, save_manifest


def test_learning_project_storage_round_trips_into_runtime(monkeypatch, tmp_path: Path) -> None:
    """A persistent profile must carry learning/project storage without patches."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    manifest = build_manifest(
        profile="memory-project",
        preset="persistent-service",
        runtime_kind="python",
        scope="user",
        provider_mode="manual",
        targets=["claude"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        proxy_mode=None,
        memory_enabled=True,
        learn_enabled=True,
        memory_storage_mode="project",
        traffic_learning_min_evidence=7,
        memory_project_root=str(tmp_path / "scratch-project"),
        telemetry_enabled=False,
        image="ghcr.io/headroomlabs-ai/headroom:latest",
    )
    save_manifest(manifest)

    loaded = load_manifest("memory-project")

    assert loaded is not None
    assert loaded.learn_enabled is True
    assert loaded.memory_storage_mode == "project"
    assert loaded.traffic_learning_min_evidence == 7
    assert loaded.memory_project_root == str(tmp_path / "scratch-project")
    assert build_runtime_command(loaded)[4:] == [
        "--host",
        "127.0.0.1",
        "--port",
        "8787",
        "--headroom-deployment-profile",
        "memory-project",
        "--headroom-deployment-runtime",
        "python",
        "--backend",
        "anthropic",
        "--no-telemetry",
        "--memory",
        "--memory-db-path",
        str(tmp_path / ".headroom" / "memory.db"),
        "--learn",
        "--memory-storage",
        "project",
        "--min-evidence",
        "7",
        "--memory-project-root",
        str(tmp_path / "scratch-project"),
    ]


def test_install_apply_persists_explicit_memory_options(monkeypatch, tmp_path: Path) -> None:
    captured = []
    monkeypatch.setattr("headroom.cli.install._apply_manifest", captured.append)

    result = CliRunner().invoke(
        main,
        [
            "install",
            "apply",
            "--learn",
            "--memory-storage",
            "user",
            "--min-evidence",
            "7",
            "--memory-project-root",
            str(tmp_path / "scratch-project"),
            "--mode",
            "cache",
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = captured[0]
    assert manifest.memory_enabled is True
    assert manifest.learn_enabled is True
    assert manifest.memory_storage_mode == "user"
    assert manifest.traffic_learning_min_evidence == 7
    assert manifest.memory_project_root == str(tmp_path / "scratch-project")
    assert manifest.proxy_args.count("--mode") == 1
    assert "HEADROOM_MODE" not in manifest.base_env


def test_install_apply_no_learn_is_persistent_override(monkeypatch) -> None:
    captured = []
    monkeypatch.setattr("headroom.cli.install._apply_manifest", captured.append)

    result = CliRunner().invoke(main, ["install", "apply", "--memory", "--no-learn"])

    assert result.exit_code == 0, result.output
    manifest = captured[0]
    assert manifest.memory_enabled is True
    assert manifest.learn_enabled is False
    assert "--no-learn" in manifest.proxy_args


@pytest.mark.parametrize("env_threshold", ["11", ""])
def test_min_evidence_env_precedence(env_threshold: str) -> None:
    manifest = build_manifest(
        profile="memory-env-precedence",
        preset="persistent-service",
        runtime_kind="python",
        scope="user",
        provider_mode="manual",
        targets=["claude"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        proxy_mode=None,
        memory_enabled=True,
        learn_enabled=True,
        memory_storage_mode="project",
        traffic_learning_min_evidence=7,
        memory_project_root="",
        telemetry_enabled=False,
        image="ghcr.io/headroomlabs-ai/headroom:latest",
        extra_env={"HEADROOM_MIN_EVIDENCE": env_threshold},
    )

    assert manifest.base_env["HEADROOM_MIN_EVIDENCE"] == env_threshold
    if env_threshold:
        assert "--min-evidence" not in manifest.proxy_args
    else:
        assert manifest.proxy_args[manifest.proxy_args.index("--min-evidence") + 1] == "7"


def test_docker_learning_options_omit_host_memory_path(tmp_path: Path) -> None:
    manifest = build_manifest(
        profile="memory-docker",
        preset="persistent-docker",
        runtime_kind="docker",
        scope="user",
        provider_mode="manual",
        targets=["claude"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        proxy_mode=None,
        memory_enabled=True,
        learn_enabled=True,
        memory_storage_mode="project",
        traffic_learning_min_evidence=7,
        memory_project_root=str(tmp_path / "scratch-project"),
        telemetry_enabled=False,
        image="ghcr.io/headroomlabs-ai/headroom:latest",
    )

    assert "--memory" in manifest.proxy_args
    assert "--memory-db-path" not in manifest.proxy_args
    assert "--learn" in manifest.proxy_args
    assert manifest.proxy_args[-6:] == [
        "--memory-storage",
        "project",
        "--min-evidence",
        "7",
        "--memory-project-root",
        str(tmp_path / "scratch-project"),
    ]


@pytest.mark.parametrize("runtime_kind", ["python", "docker"])
@pytest.mark.parametrize("project_root", [".", "../other-project", "~/project"])
def test_project_root_is_resolved_before_supervisor_start(
    monkeypatch, tmp_path: Path, runtime_kind: str, project_root: str
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    install_dir = tmp_path / "install"
    install_dir.mkdir()
    monkeypatch.chdir(install_dir)
    expected_root = str(Path(project_root).expanduser().resolve())
    manifest = build_manifest(
        profile="relative-root",
        preset="persistent-service",
        runtime_kind=runtime_kind,
        scope="user",
        provider_mode="manual",
        targets=[],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        proxy_mode=None,
        memory_enabled=True,
        memory_project_root=project_root,
        telemetry_enabled=False,
        image="ghcr.io/headroomlabs-ai/headroom:latest",
    )
    save_manifest(manifest)
    monkeypatch.chdir(tmp_path)

    loaded = load_manifest("relative-root")

    assert loaded is not None
    assert loaded.memory_project_root == expected_root
    command = build_runtime_command(loaded)
    assert command[command.index("--memory-project-root") + 1] == expected_root
