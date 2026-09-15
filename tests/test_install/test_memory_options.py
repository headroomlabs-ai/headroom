from __future__ import annotations

from pathlib import Path

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
        memory_project_root="/tmp/scratch-project",
        telemetry_enabled=False,
        image="ghcr.io/headroomlabs-ai/headroom:latest",
    )
    save_manifest(manifest)

    loaded = load_manifest("memory-project")

    assert loaded is not None
    assert loaded.learn_enabled is True
    assert loaded.memory_storage_mode == "project"
    assert loaded.traffic_learning_min_evidence == 7
    assert loaded.memory_project_root == "/tmp/scratch-project"
    assert build_runtime_command(loaded)[4:] == [
        "--host",
        "127.0.0.1",
        "--port",
        "8787",
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
        "/tmp/scratch-project",
    ]


def test_install_apply_persists_explicit_memory_options(monkeypatch) -> None:
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
            "/tmp/scratch-project",
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
    assert manifest.memory_project_root == "/tmp/scratch-project"
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


def test_explicit_min_evidence_env_wins_over_cli_default() -> None:
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
        extra_env={"HEADROOM_MIN_EVIDENCE": "11"},
    )

    assert manifest.base_env["HEADROOM_MIN_EVIDENCE"] == "11"
    assert "--min-evidence" not in manifest.proxy_args


def test_docker_learning_options_omit_host_memory_path() -> None:
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
        memory_project_root="/tmp/scratch-project",
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
        "/tmp/scratch-project",
    ]
