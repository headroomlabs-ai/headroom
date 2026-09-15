# mypy: disable-error-code="no-untyped-def,func-returns-value"
from __future__ import annotations

import copy
import importlib
import json
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

import click
import pytest
from click.testing import CliRunner

from headroom.install.models import DeploymentManifest


@pytest.fixture(autouse=True)
def _isolate_native_user_state(monkeypatch, tmp_path: Path):
    """Keep every init test away from real manifest, task, and extension paths."""
    from headroom.install import paths as install_paths
    from headroom.install import state as install_state
    from headroom.install import supervisors
    from headroom.providers import pi_extension

    root = tmp_path / "native-user"

    def profile_root(profile):
        return root / "deploy" / profile

    def manifest_path(profile):
        return profile_root(profile) / "manifest.json"

    def run_script(profile):
        return profile_root(profile) / "run-headroom.sh"

    def ensure_script(profile):
        return profile_root(profile) / "ensure-headroom.sh"

    def run_ps1(profile):
        return profile_root(profile) / "run-headroom.ps1"

    def run_cmd(profile):
        return profile_root(profile) / "run-headroom.cmd"

    def ensure_ps1(profile):
        return profile_root(profile) / "ensure-headroom.ps1"

    def ensure_cmd(profile):
        return profile_root(profile) / "ensure-headroom.cmd"

    monkeypatch.setattr(pi_extension, "extension_config_path", lambda: root / "pi-extension.json")
    monkeypatch.setattr(install_paths, "profile_root", profile_root)
    monkeypatch.setattr(install_paths, "manifest_path", manifest_path)
    monkeypatch.setattr(install_paths, "unix_run_script_path", run_script)
    monkeypatch.setattr(install_paths, "unix_ensure_script_path", ensure_script)
    monkeypatch.setattr(install_paths, "windows_run_script_path", run_ps1)
    monkeypatch.setattr(install_paths, "windows_run_cmd_path", run_cmd)
    monkeypatch.setattr(install_paths, "windows_ensure_script_path", ensure_ps1)
    monkeypatch.setattr(install_paths, "windows_ensure_cmd_path", ensure_cmd)
    monkeypatch.setattr(install_state, "profile_root", profile_root)
    monkeypatch.setattr(install_state, "manifest_path", manifest_path)
    monkeypatch.setattr(supervisors, "unix_run_script_path", run_script)
    monkeypatch.setattr(supervisors, "unix_ensure_script_path", ensure_script)
    monkeypatch.setattr(supervisors, "windows_run_script_path", run_ps1)
    monkeypatch.setattr(supervisors, "windows_run_cmd_path", run_cmd)
    monkeypatch.setattr(supervisors, "windows_ensure_script_path", ensure_ps1)
    monkeypatch.setattr(supervisors, "windows_ensure_cmd_path", ensure_cmd)

    sentinel = tmp_path / "real-user-sentinel"
    sentinel.write_bytes(b"untouched")
    yield root
    assert sentinel.read_bytes() == b"untouched"


def _load_init_module(monkeypatch):
    monkeypatch.delitem(sys.modules, "headroom.cli.init", raising=False)
    monkeypatch.delitem(sys.modules, "headroom.cli.main", raising=False)
    fake_main_module: types.ModuleType | SimpleNamespace = types.ModuleType("headroom.cli.main")

    @click.group()
    def fake_main() -> None:
        pass

    fake_main_module.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "headroom.cli.main", fake_main_module)
    importlib.invalidate_caches()
    init_cli = importlib.import_module("headroom.cli.init")
    monkeypatch.delitem(sys.modules, "headroom.cli.init", raising=False)
    return init_cli, fake_main


def test_global_auto_detection_includes_pi_and_omp(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(
        init_cli.shutil,
        "which",
        lambda name: f"/bin/{name}" if name in {"pi", "omp"} else None,
    )

    assert init_cli.detect_init_targets(global_scope=True) == ["pi", "omp"]


def test_local_pi_rejected_before_manifest_or_package_changes(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    touched: list[str] = []
    monkeypatch.setattr(
        init_cli,
        "_ensure_runtime_manifest",
        lambda **kwargs: touched.append("manifest"),
    )
    monkeypatch.setattr(
        init_cli,
        "ensure_host_package",
        lambda *args, **kwargs: touched.append("package"),
        raising=False,
    )

    result = CliRunner().invoke(fake_main, ["init", "pi"])

    assert result.exit_code != 0
    assert "requires -g" in result.output
    assert touched == []


def test_native_init_uses_shared_profile_and_task(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = SimpleNamespace(
        profile="init-user",
        targets=["pi"],
        supervisor_kind=init_cli.SupervisorKind.NONE.value,
        artifacts=[],
    )
    package = init_cli.ArtifactRecord(
        kind="pi-extension-package",
        path="pi",
        metadata={"version": "0.34.0", "owned": True},
    )
    config = init_cli.ArtifactRecord(
        kind="pi-extension-config",
        path="/home/test/.headroom/integrations/pi-extension.json",
        metadata={"managed_sha256": "abc"},
    )
    storage = [manifest]
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: storage[0])
    monkeypatch.setattr(
        init_cli, "save_manifest", lambda value: storage.__setitem__(0, copy.deepcopy(value))
    )
    monkeypatch.setattr(init_cli, "extension_release_version", lambda value: "0.34.0")
    monkeypatch.setattr(init_cli, "ensure_host_package", lambda *args, **kwargs: package)
    monkeypatch.setattr(init_cli, "ensure_extension_config", lambda *args, **kwargs: config)
    monkeypatch.setattr(init_cli, "_snapshot_extension_config", lambda: (False, b"", None))
    installed: list[SimpleNamespace] = []
    monkeypatch.setattr(init_cli, "install_supervisor", lambda value: installed.append(value) or [])
    monkeypatch.setattr(init_cli, "_ensure_profile_running", lambda profile: None)

    init_cli._init_native_host(
        host="pi",
        binary="/bin/pi",
        profile="init-user",
        port=8787,
    )

    assert storage[0].supervisor_kind == init_cli.SupervisorKind.TASK.value
    assert package in storage[0].artifacts
    assert config in storage[0].artifacts
    assert installed[0].profile == "init-user"
    assert installed[0].supervisor_kind == init_cli.SupervisorKind.TASK.value


def test_explicit_omp_init_dispatches_native_host(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(init_cli, "_run_init_targets", lambda **kwargs: calls.append(kwargs))

    result = CliRunner().invoke(fake_main, ["init", "-g", "omp"])

    assert result.exit_code == 0, result.output
    assert calls[0]["targets"] == ["omp"]
    assert calls[0]["global_scope"] is True


def native_manifest(init_cli, targets, artifacts):
    return SimpleNamespace(
        profile=init_cli._GLOBAL_PROFILE,
        targets=list(targets),
        supervisor_kind=init_cli.SupervisorKind.TASK.value,
        artifacts=list(artifacts),
    )


def native_package(init_cli, host, *, owned):
    return init_cli.ArtifactRecord(
        kind="pi-extension-package",
        path=host,
        metadata={"version": "0.34.0", "owned": owned},
    )


def native_config(init_cli):
    return init_cli.ArtifactRecord(
        kind="pi-extension-config",
        path="/home/test/.headroom/integrations/pi-extension.json",
        metadata={"managed_sha256": "abc"},
    )


def test_remove_pi_preserves_shared_omp_runtime(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    pi_package = native_package(init_cli, "pi", owned=True)
    omp_package = native_package(init_cli, "omp", owned=True)
    manifest = native_manifest(
        init_cli,
        ["pi", "omp"],
        [pi_package, omp_package, native_config(init_cli)],
    )
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli, "save_manifest", lambda value: None)
    monkeypatch.setattr(init_cli.shutil, "which", lambda host: f"/bin/{host}")
    monkeypatch.setattr(init_cli, "remove_owned_host_package", lambda *args: "removed")
    stopped = []
    monkeypatch.setattr(init_cli, "remove_supervisor", lambda value: stopped.append(value))

    result = CliRunner().invoke(fake_main, ["init", "-g", "remove", "pi"])

    assert result.exit_code == 0, result.output
    assert manifest.targets == ["omp"]
    assert stopped == []
    assert omp_package in manifest.artifacts
    assert pi_package not in manifest.artifacts


def test_remove_last_native_target_preserves_user_owned_package(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    package = native_package(init_cli, "pi", owned=False)
    config = native_config(init_cli)
    manifest = native_manifest(init_cli, ["pi"], [package, config])
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli.shutil, "which", lambda host: f"/bin/{host}")
    monkeypatch.setattr(init_cli, "remove_owned_host_package", lambda *args: "preserved")
    monkeypatch.setattr(init_cli, "remove_owned_extension_config", lambda item: "restored")
    removed_tasks = []
    stopped = []
    deleted = []
    monkeypatch.setattr(init_cli, "remove_supervisor", lambda value: removed_tasks.append(value))
    monkeypatch.setattr(init_cli, "stop_runtime", lambda value: stopped.append(value))
    monkeypatch.setattr(init_cli, "delete_manifest", lambda profile: deleted.append(profile))

    result = CliRunner().invoke(fake_main, ["init", "-g", "remove", "pi"])

    assert result.exit_code == 0, result.output
    assert removed_tasks == [manifest]
    assert stopped == [manifest]
    assert deleted == [init_cli._GLOBAL_PROFILE]


def test_remove_omp_never_touches_models_yml(monkeypatch, tmp_path) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    models = tmp_path / "models.yml"
    before = b"providers:\n  anthropic:\n    baseUrl: https://example.invalid\n"
    models.write_bytes(before)
    package = native_package(init_cli, "omp", owned=True)
    manifest = native_manifest(init_cli, ["omp"], [package, native_config(init_cli)])
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli.shutil, "which", lambda host: f"/bin/{host}")
    monkeypatch.setattr(init_cli, "remove_owned_host_package", lambda *args: "removed")
    monkeypatch.setattr(init_cli, "remove_owned_extension_config", lambda item: "restored")
    monkeypatch.setattr(init_cli, "remove_supervisor", lambda value: None)
    monkeypatch.setattr(init_cli, "stop_runtime", lambda value: None)
    monkeypatch.setattr(init_cli, "delete_manifest", lambda profile: None)

    result = CliRunner().invoke(fake_main, ["init", "-g", "remove", "omp"])

    assert result.exit_code == 0, result.output
    assert models.read_bytes() == before


def test_local_native_remove_fails_before_mutation(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    touched = []
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: touched.append(profile))
    monkeypatch.setattr(init_cli, "remove_owned_host_package", lambda *args: touched.append("pkg"))

    result = CliRunner().invoke(fake_main, ["init", "remove", "pi"])

    assert result.exit_code != 0
    assert "requires -g" in result.output
    assert touched == []


def test_remove_missing_native_target_is_idempotent(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    manifest = native_manifest(init_cli, ["omp"], [native_package(init_cli, "omp", owned=True)])
    touched = []
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli.shutil, "which", lambda host: touched.append(host))
    monkeypatch.setattr(init_cli, "save_manifest", lambda value: touched.append("manifest"))

    result = CliRunner().invoke(fake_main, ["init", "-g", "remove", "pi"])

    assert result.exit_code == 0, result.output
    assert touched == []
    assert manifest.targets == ["omp"]


def test_remove_preserves_user_owned_or_externally_changed_package(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    package = native_package(init_cli, "pi", owned=True)
    manifest = native_manifest(init_cli, ["pi", "omp"], [package])
    saved = []
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli.shutil, "which", lambda host: "/bin/pi")
    monkeypatch.setattr(init_cli, "remove_owned_host_package", lambda *args: "preserved")
    monkeypatch.setattr(init_cli, "save_manifest", lambda value: saved.append(copy.deepcopy(value)))

    result = CliRunner().invoke(fake_main, ["init", "-g", "remove", "pi"])

    assert result.exit_code == 0, result.output
    assert saved[0].targets == ["omp"]
    assert package not in saved[0].artifacts


def test_removed_package_manifest_save_failure_is_reported(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    package = native_package(init_cli, "pi", owned=True)
    manifest = native_manifest(
        init_cli,
        ["pi", "omp"],
        [package, native_package(init_cli, "omp", owned=True)],
    )
    removed = []
    persisted = copy.deepcopy(manifest)
    loads = iter([manifest, persisted])
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: next(loads))
    monkeypatch.setattr(init_cli, "save_manifest", lambda value: None)
    monkeypatch.setattr(init_cli.shutil, "which", lambda host: "/bin/pi")
    monkeypatch.setattr(
        init_cli,
        "remove_owned_host_package",
        lambda *args: removed.append("package") or "removed",
    )

    result = CliRunner().invoke(fake_main, ["init", "-g", "remove", "pi"])

    assert result.exit_code != 0
    assert "Could not verify persisted ownership state" in result.output
    assert "Removed durable pi integration" not in result.output
    assert removed == ["package"]


def test_remove_user_owned_package_does_not_require_host_binary(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    package = native_package(init_cli, "pi", owned=False)
    manifest = native_manifest(
        init_cli,
        ["pi", "omp"],
        [package, native_package(init_cli, "omp", owned=True)],
    )
    removed = []
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli.shutil, "which", lambda host: None)
    monkeypatch.setattr(
        init_cli,
        "remove_owned_host_package",
        lambda host, binary, artifact: removed.append((host, binary)) or "preserved",
    )
    monkeypatch.setattr(init_cli, "_save_manifest_verified", lambda value: None)

    result = CliRunner().invoke(fake_main, ["init", "-g", "remove", "pi"])

    assert result.exit_code == 0, result.output
    assert removed == [("pi", "")]
    assert manifest.targets == ["omp"]
    assert package not in manifest.artifacts


def test_native_remove_reports_corrupt_manifest(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    monkeypatch.setattr(
        init_cli,
        "load_manifest",
        lambda profile: (_ for _ in ()).throw(init_cli.ManifestError("corrupt state")),
    )

    result = CliRunner().invoke(fake_main, ["init", "-g", "remove", "pi"])

    assert result.exit_code != 0
    assert "Cannot remove durable pi integration" in result.output
    assert "corrupt state" in result.output


def test_last_overall_remove_verifies_profile_deletion(monkeypatch, tmp_path) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    package = native_package(init_cli, "pi", owned=False)
    manifest = native_manifest(init_cli, ["pi"], [package])
    profile_dir = tmp_path / "init-user"
    profile_dir.mkdir()
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli.shutil, "which", lambda host: None)
    monkeypatch.setattr(init_cli, "remove_owned_host_package", lambda *args: "preserved")
    monkeypatch.setattr(init_cli, "remove_supervisor", lambda value: None)
    monkeypatch.setattr(init_cli, "stop_runtime", lambda value: None)
    monkeypatch.setattr(init_cli, "delete_manifest", lambda profile: None)
    monkeypatch.setattr(init_cli, "profile_root", lambda profile: profile_dir)

    result = CliRunner().invoke(fake_main, ["init", "-g", "remove", "pi"])

    assert result.exit_code != 0
    assert "Could not verify deletion" in result.output
    assert "Removed durable pi integration" not in result.output


def test_failed_package_uninstall_keeps_manifest_ownership_for_retry(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    package = native_package(init_cli, "pi", owned=True)
    manifest = native_manifest(init_cli, ["pi"], [package])
    saved = []
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli.shutil, "which", lambda host: "/bin/pi")
    monkeypatch.setattr(
        init_cli,
        "remove_owned_host_package",
        lambda *args: (_ for _ in ()).throw(click.ClickException("uninstall failed")),
    )
    monkeypatch.setattr(init_cli, "save_manifest", lambda value: saved.append(value))

    result = CliRunner().invoke(fake_main, ["init", "-g", "remove", "pi"])

    assert result.exit_code != 0
    assert "uninstall failed" in result.output
    assert manifest.targets == ["pi"]
    assert package in manifest.artifacts
    assert saved == []


def test_last_native_remove_fails_before_mutation_when_start_lock_is_held(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    package = native_package(init_cli, "pi", owned=True)
    manifest = native_manifest(init_cli, ["pi"], [package, native_config(init_cli)])
    touched = []

    @contextmanager
    def held_lock(profile):
        yield False

    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli, "acquire_runtime_start_lock", held_lock)
    monkeypatch.setattr(
        init_cli,
        "remove_owned_host_package",
        lambda *args: touched.append("package"),
    )
    monkeypatch.setattr(init_cli, "remove_supervisor", lambda value: touched.append("task"))
    monkeypatch.setattr(init_cli, "stop_runtime", lambda value: touched.append("runtime"))
    monkeypatch.setattr(init_cli, "delete_manifest", lambda profile: touched.append("manifest"))

    result = CliRunner().invoke(fake_main, ["init", "-g", "remove", "pi"])

    assert result.exit_code != 0
    assert "already being initialized" in result.output
    assert touched == []
    assert manifest.targets == ["pi"]
    assert package in manifest.artifacts


def test_last_native_remove_keeps_manifest_when_supervisor_teardown_fails(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    package = native_package(init_cli, "pi", owned=True)
    manifest = native_manifest(init_cli, ["pi"], [package, native_config(init_cli)])
    saved = []
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli.shutil, "which", lambda host: "/bin/pi")
    monkeypatch.setattr(init_cli, "remove_owned_host_package", lambda *args: "removed")
    monkeypatch.setattr(init_cli, "remove_owned_extension_config", lambda item: "removed")
    monkeypatch.setattr(
        init_cli,
        "remove_supervisor",
        lambda value: (_ for _ in ()).throw(click.ClickException("scheduler failed")),
    )
    monkeypatch.setattr(init_cli, "save_manifest", lambda value: saved.append(copy.deepcopy(value)))

    result = CliRunner().invoke(fake_main, ["init", "-g", "remove", "pi"])

    assert result.exit_code != 0
    assert "scheduler failed" in result.output
    assert saved == []
    assert manifest.targets == ["pi"]
    assert package in manifest.artifacts


def test_last_native_remove_drops_task_but_keeps_non_native_shared_profile(
    monkeypatch, tmp_path
) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    package = native_package(init_cli, "pi", owned=True)
    config = native_config(init_cli)
    task_path = tmp_path / "task"
    task_path.write_text("managed task", encoding="utf-8")
    task = init_cli.ArtifactRecord(kind="cron", path=str(task_path))
    manifest = native_manifest(init_cli, ["claude", "pi"], [package, config, task])
    saved = []
    removed_tasks = []
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli.shutil, "which", lambda host: "/bin/pi")
    monkeypatch.setattr(init_cli, "remove_owned_host_package", lambda *args: "removed")
    monkeypatch.setattr(init_cli, "remove_owned_extension_config", lambda item: "removed")
    monkeypatch.setattr(init_cli, "remove_supervisor", lambda value: removed_tasks.append(value))
    monkeypatch.setattr(init_cli, "save_manifest", lambda value: saved.append(copy.deepcopy(value)))
    monkeypatch.setattr(
        init_cli, "stop_runtime", lambda value: (_ for _ in ()).throw(AssertionError("stop"))
    )

    result = CliRunner().invoke(fake_main, ["init", "-g", "remove", "pi"])

    assert result.exit_code == 0, result.output
    assert removed_tasks == [manifest]
    assert saved[0].targets == ["claude"]
    assert saved[0].supervisor_kind == init_cli.SupervisorKind.NONE.value
    assert saved[0].artifacts == []
    assert not task_path.exists()


def test_last_overall_target_stops_runtime_restores_config_and_deletes_profile(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    package = native_package(init_cli, "omp", owned=True)
    config = native_config(init_cli)
    manifest = native_manifest(init_cli, ["omp"], [package, config])
    calls = []
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli.shutil, "which", lambda host: "/bin/omp")
    monkeypatch.setattr(
        init_cli, "remove_owned_host_package", lambda *args: calls.append("package") or "removed"
    )
    monkeypatch.setattr(
        init_cli,
        "remove_owned_extension_config",
        lambda item: calls.append("config") or "restored",
    )
    monkeypatch.setattr(init_cli, "remove_supervisor", lambda value: calls.append("task"))
    monkeypatch.setattr(init_cli, "stop_runtime", lambda value: calls.append("runtime"))
    monkeypatch.setattr(init_cli, "delete_manifest", lambda profile: calls.append("manifest"))

    result = CliRunner().invoke(fake_main, ["init", "-g", "remove", "omp"])

    assert result.exit_code == 0, result.output
    assert calls == ["package", "config", "task", "runtime", "manifest"]


def test_run_init_targets_includes_native_runtime_targets(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    native: list[tuple[str, str, list[str]]] = []
    monkeypatch.setattr(init_cli, "extension_release_version", lambda version: "1.2.3")
    monkeypatch.setattr(init_cli.shutil, "which", lambda target: f"/bin/{target}")
    monkeypatch.setattr(
        init_cli,
        "_init_native_hosts",
        lambda **kwargs: native.extend(
            (host, kwargs["manifest"].profile, kwargs["manifest"].targets)
            for host, _binary in kwargs["hosts"]
        ),
    )
    monkeypatch.setattr(init_cli, "_install_headroom_mcp_for_targets", lambda **kwargs: None)

    init_cli._run_init_targets(
        targets=["pi", "omp"],
        global_scope=True,
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )

    assert native == [
        ("pi", "init-user", ["omp", "pi"]),
        ("omp", "init-user", ["omp", "pi"]),
    ]


def test_ensure_runtime_manifest_preserves_native_targets_with_real_planner(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    package = init_cli.ArtifactRecord("pi-extension-package", "pi", {"version": "1.2.3"})
    existing = init_cli.build_manifest(
        profile="init-user",
        preset=init_cli.InstallPreset.PERSISTENT_TASK.value,
        runtime_kind=init_cli.RuntimeKind.PYTHON.value,
        scope=init_cli.ConfigScope.USER.value,
        provider_mode="manual",
        targets=["claude"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        proxy_mode="token",
        memory_enabled=False,
        telemetry_enabled=True,
        image="image",
    )
    existing.targets = ["claude", "pi"]
    existing.artifacts = [package]
    saved: list[DeploymentManifest] = []
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: existing)
    monkeypatch.setattr(init_cli, "save_manifest", lambda manifest: saved.append(manifest))

    init_cli._ensure_runtime_manifest(
        global_scope=True,
        targets=["omp"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )

    manifest = saved[0]
    assert manifest.targets == ["claude", "omp", "pi"]
    assert manifest.artifacts == [package]
    assert manifest.supervisor_kind == init_cli.SupervisorKind.TASK.value


def test_native_missing_binary_aborts_before_manifest_or_runtime_mutation(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    touched: list[str] = []
    monkeypatch.setattr(init_cli.shutil, "which", lambda target: None)
    monkeypatch.setattr(init_cli, "save_manifest", lambda manifest: touched.append("manifest"))
    monkeypatch.setattr(init_cli, "stop_runtime", lambda manifest: touched.append("runtime"))
    monkeypatch.setattr(init_cli, "extension_release_version", lambda version: "1.2.3")

    with pytest.raises(click.ClickException, match="not found in PATH"):
        init_cli._run_init_targets(
            targets=["pi"],
            global_scope=True,
            port=8787,
            backend="anthropic",
            anyllm_provider=None,
            region=None,
            memory=False,
        )

    assert touched == []


def test_native_corrupt_manifest_is_untouched(monkeypatch, _isolate_native_user_state) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest_path = _isolate_native_user_state / "deploy" / "init-user" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    corrupt = b'{"targets": [invalid]\n'
    manifest_path.write_bytes(corrupt)
    monkeypatch.setattr(init_cli.shutil, "which", lambda target: f"/bin/{target}")
    monkeypatch.setattr(init_cli, "extension_release_version", lambda version: "1.2.3")

    with pytest.raises(click.ClickException, match="corrupt"):
        init_cli._run_init_targets(
            targets=["pi"],
            global_scope=True,
            port=8787,
            backend="anthropic",
            anyllm_provider=None,
            region=None,
            memory=False,
        )

    assert manifest_path.read_bytes() == corrupt


def test_native_config_publish_then_failure_restores_raw_snapshot(
    monkeypatch, _isolate_native_user_state
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    config_path = _isolate_native_user_state / "pi-extension.json"
    original = b'{"keep": true}\n'
    managed = b'{\n  "keep": true,\n  "baseUrl": "http://127.0.0.1:8787"\n}\n'
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(original)
    manifest = SimpleNamespace(
        profile="init-user",
        targets=["pi"],
        supervisor_kind=init_cli.SupervisorKind.NONE.value,
        artifacts=[],
    )
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli, "save_manifest", lambda value: None)
    monkeypatch.setattr(init_cli, "extension_release_version", lambda version: "1.2.3")
    monkeypatch.setattr(init_cli, "inspect_host_package", lambda host, binary: None)
    monkeypatch.setattr(init_cli, "install_supervisor", lambda value: [])
    monkeypatch.setattr(init_cli, "remove_supervisor", lambda value: None)
    monkeypatch.setattr(init_cli, "_ensure_profile_running", lambda profile: None)

    def publish_then_fail(port, existing):
        config_path.write_bytes(managed)
        raise RuntimeError("config failed after publish")

    monkeypatch.setattr(init_cli, "ensure_extension_config", publish_then_fail)

    with pytest.raises(click.ClickException, match="config failed after publish"):
        init_cli._init_native_host(host="pi", binary="/bin/pi", profile="init-user", port=8787)

    assert config_path.read_bytes() == original


def test_native_config_rollback_preserves_concurrent_change_and_combines_error(
    monkeypatch, _isolate_native_user_state
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    config_path = _isolate_native_user_state / "pi-extension.json"
    original = b'{"keep": true}\n'
    concurrent = b'{"user": "changed"}\n'
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(original)
    manifest = SimpleNamespace(
        profile="init-user",
        targets=["pi"],
        supervisor_kind=init_cli.SupervisorKind.NONE.value,
        artifacts=[],
    )
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli, "save_manifest", lambda value: None)
    monkeypatch.setattr(init_cli, "extension_release_version", lambda version: "1.2.3")
    monkeypatch.setattr(init_cli, "inspect_host_package", lambda host, binary: None)
    monkeypatch.setattr(init_cli, "install_supervisor", lambda value: [])
    monkeypatch.setattr(init_cli, "remove_supervisor", lambda value: None)
    monkeypatch.setattr(init_cli, "_ensure_profile_running", lambda profile: None)

    def concurrent_then_fail(port, existing):
        config_path.write_bytes(concurrent)
        raise RuntimeError("initiating config failure")

    monkeypatch.setattr(init_cli, "ensure_extension_config", concurrent_then_fail)

    with pytest.raises(click.ClickException) as exc_info:
        init_cli._init_native_host(host="pi", binary="/bin/pi", profile="init-user", port=8787)

    assert config_path.read_bytes() == concurrent
    assert "initiating config failure" in str(exc_info.value)
    assert "Rollback also failed" in str(exc_info.value)


@pytest.mark.parametrize("failure", ["package", "config", "task"])
def test_native_init_failure_rolls_back_package_config_task_and_manifest(
    monkeypatch, failure: str
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    previous = SimpleNamespace(
        profile="init-user",
        targets=["claude", "pi"],
        supervisor_kind=init_cli.SupervisorKind.NONE.value,
        artifacts=[],
    )
    manifest = SimpleNamespace(**previous.__dict__)
    saved: list[tuple[str, list[str], list[tuple[str, str]]]] = []
    storage = [manifest]
    removed_tasks: list[str] = []
    restored_configs: list[str] = []
    package = init_cli.ArtifactRecord(
        "pi-extension-package", "pi", {"version": "1.2.3", "owned": True}
    )
    config = init_cli.ArtifactRecord("pi-extension-config", "/config", {"managed_sha256": "x"})

    monkeypatch.setattr(init_cli, "extension_release_version", lambda version: "1.2.3")
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: storage[0])

    def save(value):
        saved.append(
            (
                value.supervisor_kind,
                list(value.targets),
                [(record.kind, record.path) for record in value.artifacts],
            )
        )
        storage[0] = copy.deepcopy(value)

    monkeypatch.setattr(init_cli, "save_manifest", save)
    monkeypatch.setattr(
        init_cli,
        "_snapshot_manifest",
        lambda profile: (True, b"previous", storage[0]),
    )
    monkeypatch.setattr(
        init_cli,
        "_restore_manifest_snapshot",
        lambda profile, existed, content: storage.__setitem__(0, copy.deepcopy(previous)),
    )
    monkeypatch.setattr(init_cli, "inspect_host_package", lambda host, binary: None)
    monkeypatch.setattr(init_cli, "_snapshot_extension_config", lambda: (False, b"", None))
    monkeypatch.setattr(
        init_cli, "remove_supervisor", lambda value: removed_tasks.append(value.profile)
    )
    monkeypatch.setattr(init_cli, "_ensure_profile_running", lambda profile: None)
    monkeypatch.setattr(init_cli, "remove_owned_host_package", lambda *args: "removed")
    monkeypatch.setattr(
        init_cli,
        "install_supervisor",
        lambda value: (
            (_ for _ in ()).throw(RuntimeError("task failed"))
            if failure == "task"
            else [init_cli.ArtifactRecord("crontab", "user:init-user")]
        ),
    )
    monkeypatch.setattr(
        init_cli,
        "ensure_extension_config",
        lambda port, existing: (
            (_ for _ in ()).throw(RuntimeError("config failed")) if failure == "config" else config
        ),
    )
    monkeypatch.setattr(
        init_cli,
        "ensure_host_package",
        lambda host, binary, version, existing: (
            (_ for _ in ()).throw(RuntimeError("package failed"))
            if failure == "package"
            else package
        ),
    )

    with pytest.raises(click.ClickException, match=f"{failure} failed") as exc_info:
        init_cli._init_native_host(host="pi", binary="/bin/pi", profile="init-user", port=8787)

    assert storage[0].supervisor_kind == init_cli.SupervisorKind.NONE.value
    assert storage[0].targets == ["claude", "pi"]
    assert storage[0].artifacts == []
    assert removed_tasks == ["init-user"]
    assert restored_configs == []
    assert "Rollback also failed" not in str(exc_info.value)


def test_native_init_combines_initiating_and_rollback_failures(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = SimpleNamespace(
        profile="init-user",
        targets=["pi"],
        supervisor_kind=init_cli.SupervisorKind.NONE.value,
        artifacts=[],
    )
    monkeypatch.setattr(init_cli, "extension_release_version", lambda version: "1.2.3")
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli, "wait_ready", lambda value, timeout_seconds: False)
    monkeypatch.setattr(init_cli, "save_manifest", lambda value: None)
    monkeypatch.setattr(
        init_cli, "_snapshot_manifest", lambda profile: (True, b"previous", manifest)
    )
    monkeypatch.setattr(init_cli, "_restore_manifest_snapshot", lambda *args: None)
    monkeypatch.setattr(init_cli, "inspect_host_package", lambda host, binary: None)
    monkeypatch.setattr(init_cli, "_snapshot_extension_config", lambda: (False, b"", None))
    monkeypatch.setattr(
        init_cli,
        "ensure_extension_config",
        lambda port, existing: init_cli.ArtifactRecord(
            "pi-extension-config", str(init_cli.extension_config_path()), {}
        ),
    )
    monkeypatch.setattr(
        init_cli,
        "ensure_host_package",
        lambda host, binary, version, existing: init_cli.ArtifactRecord(
            "pi-extension-package", host, {"version": version, "owned": True}
        ),
    )
    monkeypatch.setattr(
        init_cli,
        "install_supervisor",
        lambda value: (_ for _ in ()).throw(RuntimeError("initiating failure")),
    )
    monkeypatch.setattr(
        init_cli,
        "remove_supervisor",
        lambda value: (_ for _ in ()).throw(RuntimeError("rollback failure")),
    )

    with pytest.raises(click.ClickException) as exc_info:
        init_cli._init_native_host(host="pi", binary="/bin/pi", profile="init-user", port=8787)

    assert "initiating failure" in str(exc_info.value)
    assert "Rollback also failed" in str(exc_info.value)
    assert "rollback failure" in str(exc_info.value)


def test_repeated_native_init_deduplicates_artifacts_and_task_records(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    package = init_cli.ArtifactRecord("pi-extension-package", "pi", {"version": "1.2.3"})
    config = init_cli.ArtifactRecord("pi-extension-config", "/config", {"managed_sha256": "x"})
    task = init_cli.ArtifactRecord("crontab", "user:init-user")
    manifest = SimpleNamespace(
        profile="init-user",
        targets=["pi"],
        supervisor_kind=init_cli.SupervisorKind.TASK.value,
        artifacts=[package, config, task],
    )
    storage = [manifest]
    monkeypatch.setattr(init_cli, "extension_release_version", lambda version: "1.2.3")
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: storage[0])
    monkeypatch.setattr(
        init_cli, "save_manifest", lambda value: storage.__setitem__(0, copy.deepcopy(value))
    )
    monkeypatch.setattr(init_cli, "inspect_host_package", lambda host, binary: object())
    monkeypatch.setattr(init_cli, "_snapshot_extension_config", lambda: (True, b"{}", 0o600))
    monkeypatch.setattr(init_cli, "install_supervisor", lambda value: [task])
    monkeypatch.setattr(init_cli, "_ensure_profile_running", lambda profile: None)
    monkeypatch.setattr(init_cli, "ensure_extension_config", lambda port, existing: config)
    monkeypatch.setattr(
        init_cli,
        "ensure_host_package",
        lambda host, binary, version, existing: package,
    )

    init_cli._init_native_host(host="pi", binary="/bin/pi", profile="init-user", port=8787)

    assert storage[0].artifacts.count(package) == 1
    assert storage[0].artifacts.count(config) == 1
    assert storage[0].artifacts.count(task) == 1


def test_native_batch_pi_success_omp_failure_rolls_back_both_hosts(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = init_cli._build_runtime_manifest(
        profile="init-user",
        existing=None,
        targets=["pi", "omp"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )
    packages = {"pi": None, "omp": None}
    installed: list[str] = []
    removed: list[str] = []
    supervisors: list[str] = []
    monkeypatch.setattr(init_cli, "inspect_host_package", lambda host, binary: packages[host])

    def ensure_package(host, binary, version, existing):
        if host == "omp":
            raise RuntimeError("omp failed")
        installed.append(host)
        packages[host] = version
        return init_cli.ArtifactRecord(
            "pi-extension-package", host, {"version": version, "owned": True}
        )

    monkeypatch.setattr(init_cli, "ensure_host_package", ensure_package)
    monkeypatch.setattr(
        init_cli,
        "remove_owned_host_package",
        lambda host, binary, artifact: (
            removed.append(host) or packages.__setitem__(host, None) or "removed"
        ),
    )
    monkeypatch.setattr(
        init_cli,
        "ensure_extension_config",
        lambda port, existing: init_cli.ArtifactRecord(
            "pi-extension-config", str(init_cli.extension_config_path()), {}
        ),
    )
    monkeypatch.setattr(
        init_cli, "install_supervisor", lambda value: supervisors.append(value.profile) or []
    )
    monkeypatch.setattr(init_cli, "remove_supervisor", lambda value: None)
    monkeypatch.setattr(init_cli, "_ensure_profile_running", lambda profile: None)
    monkeypatch.setattr(init_cli, "_save_manifest_verified", lambda value: None)
    monkeypatch.setattr(init_cli, "_restore_manifest_snapshot", lambda *args: None)

    with pytest.raises(click.ClickException, match="omp failed"):
        init_cli._init_native_hosts(
            hosts=[("pi", "/bin/pi"), ("omp", "/bin/omp")],
            release="1.2.3",
            manifest=manifest,
            manifest_snapshot=(False, b"", None),
            port=8787,
        )

    assert installed == ["pi"]
    assert removed == ["pi"]
    assert packages == {"pi": None, "omp": None}
    assert supervisors == []


def test_native_package_helper_publish_then_fail_is_verified_restored(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = init_cli._build_runtime_manifest(
        profile="init-user",
        existing=None,
        targets=["pi"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )
    state = {"pi": None}
    monkeypatch.setattr(init_cli, "inspect_host_package", lambda host, binary: state[host])

    def publish_then_fail(host, binary, version, existing):
        state[host] = version
        state[host] = None  # Task 2 helper's internal rollback.
        raise RuntimeError("package publish failed")

    monkeypatch.setattr(init_cli, "ensure_host_package", publish_then_fail)
    monkeypatch.setattr(
        init_cli,
        "ensure_extension_config",
        lambda port, existing: init_cli.ArtifactRecord(
            "pi-extension-config", str(init_cli.extension_config_path()), {}
        ),
    )
    monkeypatch.setattr(init_cli, "remove_supervisor", lambda value: None)
    monkeypatch.setattr(init_cli, "_restore_manifest_snapshot", lambda *args: None)

    with pytest.raises(click.ClickException, match="package publish failed") as exc_info:
        init_cli._init_native_hosts(
            hosts=[("pi", "/bin/pi")],
            release="1.2.3",
            manifest=manifest,
            manifest_snapshot=(False, b"", None),
            port=8787,
        )

    assert state == {"pi": None}
    assert "package helper did not restore" not in str(exc_info.value)


def test_native_batch_installs_shared_supervisor_once(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = init_cli._build_runtime_manifest(
        profile="init-user",
        existing=None,
        targets=["pi", "omp"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )
    calls: list[str] = []
    monkeypatch.setattr(init_cli, "inspect_host_package", lambda host, binary: None)
    monkeypatch.setattr(
        init_cli,
        "ensure_host_package",
        lambda host, binary, version, existing: init_cli.ArtifactRecord(
            "pi-extension-package", host, {"version": version, "owned": True}
        ),
    )
    monkeypatch.setattr(
        init_cli,
        "ensure_extension_config",
        lambda port, existing: init_cli.ArtifactRecord(
            "pi-extension-config", str(init_cli.extension_config_path()), {}
        ),
    )
    monkeypatch.setattr(
        init_cli, "install_supervisor", lambda value: calls.append("supervisor") or []
    )
    monkeypatch.setattr(
        init_cli, "_start_profile_strict_locked", lambda value: calls.append("runtime")
    )
    monkeypatch.setattr(init_cli, "_save_manifest_verified", lambda value: None)

    init_cli._init_native_hosts(
        hosts=[("pi", "/bin/pi"), ("omp", "/bin/omp")],
        release="1.2.3",
        manifest=manifest,
        manifest_snapshot=(False, b"", None),
        port=8787,
    )

    assert calls == ["supervisor", "runtime"]


def test_native_init_saves_provisional_before_supervisor_and_holds_start_lock(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = init_cli._build_runtime_manifest(
        profile="init-user",
        existing=None,
        targets=["pi"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )
    calls: list[str] = []

    @contextmanager
    def start_lock(profile):
        calls.append("lock-enter")
        yield True
        calls.append("lock-exit")

    monkeypatch.setattr(init_cli, "acquire_runtime_start_lock", start_lock)
    monkeypatch.setattr(init_cli, "inspect_host_package", lambda host, binary: None)
    monkeypatch.setattr(
        init_cli,
        "ensure_host_package",
        lambda host, binary, version, existing: init_cli.ArtifactRecord(
            "pi-extension-package", host, {"version": version, "owned": True}
        ),
    )
    monkeypatch.setattr(
        init_cli,
        "ensure_extension_config",
        lambda port, existing: init_cli.ArtifactRecord(
            "pi-extension-config", str(init_cli.extension_config_path()), {}
        ),
    )
    monkeypatch.setattr(
        init_cli,
        "_save_manifest_verified",
        lambda value: calls.append("final-save" if value.targets else "provisional-save"),
    )
    monkeypatch.setattr(
        init_cli, "install_supervisor", lambda value: calls.append("supervisor") or []
    )
    monkeypatch.setattr(
        init_cli,
        "_start_profile_strict_locked",
        lambda value: calls.append("start:locked"),
    )

    init_cli._init_native_hosts(
        hosts=[("pi", "/bin/pi")],
        release="1.2.3",
        manifest=manifest,
        manifest_snapshot=(False, b"", None),
        port=8787,
    )

    assert calls == [
        "lock-enter",
        "provisional-save",
        "supervisor",
        "start:locked",
        "final-save",
        "lock-exit",
    ]


def test_native_init_start_lock_denial_avoids_duplicate_setup(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = init_cli._build_runtime_manifest(
        profile="init-user",
        existing=None,
        targets=["pi"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )
    touched: list[str] = []

    @contextmanager
    def denied_lock(profile):
        yield False

    monkeypatch.setattr(init_cli, "acquire_runtime_start_lock", denied_lock)
    monkeypatch.setattr(init_cli, "inspect_host_package", lambda host, binary: None)
    monkeypatch.setattr(
        init_cli,
        "ensure_host_package",
        lambda *args: touched.append("package"),
    )
    monkeypatch.setattr(init_cli, "ensure_extension_config", lambda *args: touched.append("config"))
    monkeypatch.setattr(init_cli, "install_supervisor", lambda value: touched.append("supervisor"))

    with pytest.raises(click.ClickException, match="already being initialized"):
        init_cli._init_native_hosts(
            hosts=[("pi", "/bin/pi")],
            release="1.2.3",
            manifest=manifest,
            manifest_snapshot=(False, b"", None),
            port=8787,
        )

    assert touched == []


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_native_platform_scheduler_snapshot_restore(
    monkeypatch, tmp_path: Path, platform: str
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    calls: list[tuple[list[str], dict]] = []

    class Result:
        def __init__(self, returncode=0, stdout: str | bytes = ""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if platform == "darwin" and command[:2] == ["launchctl", "print"]:
            return Result(0)
        if platform == "win32" and command[:2] == ["schtasks", "/Query"]:
            return Result(0, b"<Task>original</Task>")
        return Result(0)

    monkeypatch.setattr(init_cli, "run", fake_run)
    monkeypatch.setattr(init_cli.sys, "platform", platform)
    plist: Path | None = None
    restored: list[tuple[str, bytes]] = []
    if platform == "darwin":
        monkeypatch.setattr(init_cli.Path, "home", lambda: tmp_path)
        plist = tmp_path / "Library" / "LaunchAgents" / "com.headroom.init-user.plist"
        plist.parent.mkdir(parents=True)
        plist.write_bytes(b"original plist")
    else:
        monkeypatch.setattr(
            init_cli, "_restore_windows_task", lambda name, xml: restored.append((name, xml))
        )

    snapshot = init_cli._snapshot_scheduler(
        SimpleNamespace(profile="init-user", service_name="headroom-init-user")
    )
    init_cli._restore_scheduler(snapshot)

    if platform == "darwin":
        assert plist is not None
        assert plist.read_bytes() == b"original plist"
        assert any(command[:2] == ["launchctl", "bootstrap"] for command, _ in calls)
    else:
        assert restored == [
            ("headroom-init-user-startup", b"<Task>original</Task>"),
            ("headroom-init-user-health", b"<Task>original</Task>"),
        ]


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_native_scheduler_snapshot_transient_failure_aborts_before_mutation(
    monkeypatch, platform: str
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    touched: list[str] = []

    class Result:
        returncode = 5
        stdout = b""
        stderr = b"transient failure"

    monkeypatch.setattr(init_cli.sys, "platform", platform)
    monkeypatch.setattr(init_cli, "run", lambda command, **kwargs: Result())
    monkeypatch.setattr(init_cli, "ensure_extension_config", lambda *args: touched.append("config"))

    with pytest.raises(click.ClickException, match="snapshot"):
        init_cli._snapshot_scheduler(
            SimpleNamespace(profile="init-user", service_name="headroom-init-user")
        )

    assert touched == []


def test_native_crontab_snapshot_restores_exact_whitespace(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    original = b"# user cron\n\xff\n  0 1 * * * command  \n"
    writes: list[bytes] = []

    class Result:
        returncode = 0
        stdout = original
        stderr = b""

    def fake_run(command, **kwargs):
        if command == ["crontab", "-"]:
            writes.append(kwargs["input"])
        return Result()

    monkeypatch.setattr(init_cli, "run", fake_run)
    monkeypatch.setattr(init_cli.sys, "platform", "linux")

    snapshot = init_cli._snapshot_scheduler(SimpleNamespace(profile="init-user", service_name="x"))
    init_cli._restore_scheduler(snapshot)

    assert writes == [original]


@pytest.mark.parametrize(
    ("returncode", "stderr"),
    [(113, b"permission denied"), (5, b"Could not find service")],
)
def test_native_macos_scheduler_absence_requires_code_and_message(
    monkeypatch, returncode: int, stderr: bytes
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)

    class Result:
        stdout = b""

        def __init__(self) -> None:
            self.returncode = returncode
            self.stderr = stderr

    monkeypatch.setattr(init_cli.sys, "platform", "darwin")
    monkeypatch.setattr(init_cli, "run", lambda command, **kwargs: Result())

    with pytest.raises(click.ClickException, match="snapshot"):
        init_cli._snapshot_scheduler(
            SimpleNamespace(profile="init-user", service_name="headroom-init-user")
        )


@pytest.mark.parametrize("platform", ["linux", "darwin", "windows"])
def test_native_scheduler_absence_rollback_failure_is_reported(
    monkeypatch, tmp_path: Path, platform: str
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)

    class Result:
        returncode = 7
        stdout = b""
        stderr = b"rollback denied"

    monkeypatch.setattr(init_cli, "run", lambda command, **kwargs: Result())
    snapshots = {
        "linux": ("linux", (False, b"")),
        "darwin": ("darwin", (tmp_path / "task.plist", None, False, "gui/1")),
        "windows": ("windows", {"headroom-init-user-startup": None}),
    }

    with pytest.raises(click.ClickException, match="rollback denied"):
        init_cli._restore_scheduler(snapshots[platform])


@pytest.mark.parametrize(
    ("platform", "returncode", "stderr"),
    [
        ("linux", 1, b"no crontab for test"),
        ("darwin", 113, b"Could not find service"),
        ("win32", 1, b"ERROR: The system cannot find the file specified."),
    ],
)
def test_native_scheduler_authoritative_absence_is_supported(
    monkeypatch, platform: str, returncode: int, stderr: bytes
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)

    class Result:
        stdout = b""

        def __init__(self) -> None:
            self.returncode = returncode
            self.stderr = stderr

    monkeypatch.setattr(init_cli.sys, "platform", platform)
    monkeypatch.setattr(init_cli, "run", lambda command, **kwargs: Result())

    snapshot = init_cli._snapshot_scheduler(
        SimpleNamespace(profile="init-user", service_name="headroom-init-user")
    )

    assert snapshot[0] == ("windows" if platform == "win32" else platform)


def test_native_running_but_unready_runtime_is_restored_and_verified(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    previous = SimpleNamespace(
        profile="init-user",
        supervisor_kind="none",
        artifacts=[],
        health_url="http://ready",
        preset="persistent-task",
    )
    manifest = SimpleNamespace(
        profile="init-user",
        targets=["pi"],
        supervisor_kind="none",
        artifacts=[],
        preset="persistent-task",
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory_enabled=False,
    )
    calls: list[str] = []
    statuses = iter(["running", "stopped", "stopped"])
    monkeypatch.setattr(init_cli, "runtime_status", lambda value: next(statuses))
    monkeypatch.setattr(init_cli, "wait_ready", lambda value, timeout_seconds: False)
    monkeypatch.setattr(init_cli, "inspect_host_package", lambda host, binary: None)
    monkeypatch.setattr(
        init_cli,
        "ensure_extension_config",
        lambda port, existing: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(init_cli, "remove_supervisor", lambda value: None)
    monkeypatch.setattr(init_cli, "_restore_manifest_snapshot", lambda *args: None)
    monkeypatch.setattr(init_cli, "stop_runtime", lambda value: calls.append("stop"))
    monkeypatch.setattr(init_cli, "start_detached_agent", lambda profile: calls.append("restart"))

    with pytest.raises(click.ClickException, match="boom"):
        init_cli._init_native_hosts(
            hosts=[("pi", "/bin/pi")],
            release="1.2.3",
            manifest=manifest,
            manifest_snapshot=(True, b"prior", previous),
            port=8787,
        )

    assert calls == ["stop", "restart"]


def test_native_runtime_restoration_polls_delayed_transitions(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = SimpleNamespace(profile="init-user", preset="persistent-task")
    statuses = iter(["running", "stopped", "stopped", "running"])
    calls: list[str] = []
    monkeypatch.setattr(init_cli, "runtime_status", lambda value: next(statuses))
    monkeypatch.setattr(init_cli.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(init_cli, "stop_runtime", lambda value: calls.append("stop"))
    monkeypatch.setattr(init_cli, "start_detached_agent", lambda profile: calls.append("start"))

    init_cli._restore_runtime_state(manifest, "running", False)

    assert calls == ["stop", "start"]


def test_native_runtime_restoration_reports_stop_timeout(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = SimpleNamespace(profile="init-user", preset="persistent-task")
    monkeypatch.setattr(init_cli, "runtime_status", lambda value: "running")
    monkeypatch.setattr(init_cli.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(init_cli, "stop_runtime", lambda value: None)

    with pytest.raises(click.ClickException, match="stop"):
        init_cli._restore_runtime_state(manifest, "stopped", False)


def test_native_runtime_restart_failure_is_combined(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    previous = SimpleNamespace(
        profile="init-user",
        supervisor_kind="none",
        artifacts=[],
        health_url="http://ready",
        preset="persistent-task",
    )
    manifest = SimpleNamespace(
        **previous.__dict__,
        targets=["pi"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory_enabled=False,
    )
    statuses = iter(["running", "running", "stopped", "stopped"])
    monkeypatch.setattr(init_cli, "runtime_status", lambda value: next(statuses))
    monkeypatch.setattr(init_cli, "wait_ready", lambda value, timeout_seconds: False)
    monkeypatch.setattr(init_cli, "inspect_host_package", lambda host, binary: None)
    monkeypatch.setattr(
        init_cli,
        "ensure_extension_config",
        lambda port, existing: (_ for _ in ()).throw(RuntimeError("initiating")),
    )
    monkeypatch.setattr(init_cli, "remove_supervisor", lambda value: None)
    monkeypatch.setattr(init_cli, "stop_runtime", lambda value: None)
    monkeypatch.setattr(init_cli, "_restore_manifest_snapshot", lambda *args: None)
    monkeypatch.setattr(
        init_cli,
        "start_detached_agent",
        lambda profile: (_ for _ in ()).throw(RuntimeError("restart failed")),
    )

    with pytest.raises(click.ClickException) as exc_info:
        init_cli._init_native_hosts(
            hosts=[("pi", "/bin/pi")],
            release="1.2.3",
            manifest=manifest,
            manifest_snapshot=(True, b"prior", previous),
            port=8787,
        )

    assert "initiating" in str(exc_info.value)
    assert "restart failed" in str(exc_info.value)


@pytest.mark.parametrize("was_running", [True, False])
def test_native_runtime_state_restored_on_rollback(monkeypatch, was_running: bool) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    previous = SimpleNamespace(
        profile="init-user",
        supervisor_kind="none",
        artifacts=[],
        health_url="http://ready",
        preset="persistent-task",
    )
    manifest = SimpleNamespace(
        profile="init-user",
        targets=["pi"],
        supervisor_kind="none",
        artifacts=[],
        preset="persistent-task",
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory_enabled=False,
    )
    calls: list[str] = []
    if was_running:
        statuses = iter(["running", "running", "stopped", "stopped"])
        monkeypatch.setattr(init_cli, "runtime_status", lambda value: next(statuses))
    else:
        monkeypatch.setattr(init_cli, "runtime_status", lambda value: "stopped")
    monkeypatch.setattr(init_cli, "wait_ready", lambda value, timeout_seconds: was_running)
    monkeypatch.setattr(init_cli, "inspect_host_package", lambda host, binary: None)
    monkeypatch.setattr(
        init_cli,
        "ensure_extension_config",
        lambda port, existing: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(init_cli, "install_supervisor", lambda value: [])
    monkeypatch.setattr(init_cli, "remove_supervisor", lambda value: None)
    monkeypatch.setattr(init_cli, "_save_manifest_verified", lambda value: None)
    monkeypatch.setattr(init_cli, "_restore_manifest_snapshot", lambda *args: None)
    monkeypatch.setattr(init_cli, "stop_runtime", lambda value: calls.append("stop"))
    monkeypatch.setattr(
        init_cli, "_start_profile_strict_locked", lambda value: calls.append("restart")
    )

    with pytest.raises(click.ClickException, match="boom"):
        init_cli._init_native_hosts(
            hosts=[("pi", "/bin/pi")],
            release="1.2.3",
            manifest=manifest,
            manifest_snapshot=(True, b"prior", previous),
            port=8787,
        )

    assert calls == (["stop", "restart"] if was_running else ["stop"])


def test_native_ready_runtime_rollback_reuses_held_start_lock(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    previous = init_cli._build_runtime_manifest(
        profile="init-user",
        existing=None,
        targets=["claude"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )
    manifest = init_cli._build_runtime_manifest(
        profile="init-user",
        existing=previous,
        targets=["pi"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )
    lock_held = False

    @contextmanager
    def non_reentrant_lock(profile):
        nonlocal lock_held
        if lock_held:
            yield False
            return
        lock_held = True
        try:
            yield True
        finally:
            lock_held = False

    statuses = iter(["running", "running", "stopped", "stopped", "stopped"])
    monkeypatch.setattr(init_cli, "acquire_runtime_start_lock", non_reentrant_lock)
    monkeypatch.setattr(init_cli, "runtime_status", lambda value: next(statuses))
    monkeypatch.setattr(init_cli, "wait_ready", lambda value, timeout_seconds: True)
    monkeypatch.setattr(init_cli, "inspect_host_package", lambda host, binary: None)
    monkeypatch.setattr(
        init_cli,
        "ensure_extension_config",
        lambda port, existing: (_ for _ in ()).throw(RuntimeError("initiating")),
    )
    monkeypatch.setattr(init_cli, "remove_supervisor", lambda value: None)
    monkeypatch.setattr(init_cli, "_restore_manifest_snapshot", lambda *args: None)
    monkeypatch.setattr(init_cli, "stop_runtime", lambda value: None)
    monkeypatch.setattr(init_cli, "start_detached_agent", lambda profile: None)

    with pytest.raises(click.ClickException, match="initiating") as exc_info:
        init_cli._init_native_hosts(
            hosts=[("pi", "/bin/pi")],
            release="1.2.3",
            manifest=manifest,
            manifest_snapshot=(True, b"prior", previous),
            port=8787,
        )

    assert "already being started" not in str(exc_info.value)


def test_native_strict_startup_failure_leaves_no_native_claims(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    previous = init_cli._build_runtime_manifest(
        profile="init-user",
        existing=None,
        targets=["claude"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )
    manifest = init_cli._build_runtime_manifest(
        profile="init-user",
        existing=previous,
        targets=["pi"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )
    saved: list[list[str]] = []
    removed: list[str] = []
    monkeypatch.setattr(init_cli, "runtime_status", lambda value: "stopped")
    monkeypatch.setattr(init_cli, "wait_ready", lambda value, timeout_seconds: False)
    monkeypatch.setattr(init_cli, "inspect_host_package", lambda host, binary: None)
    monkeypatch.setattr(
        init_cli,
        "ensure_host_package",
        lambda host, binary, version, existing: init_cli.ArtifactRecord(
            "pi-extension-package", host, {"version": version, "owned": True}
        ),
    )
    monkeypatch.setattr(
        init_cli,
        "remove_owned_host_package",
        lambda host, binary, artifact: removed.append(host) or "removed",
    )
    monkeypatch.setattr(
        init_cli,
        "ensure_extension_config",
        lambda port, existing: init_cli.ArtifactRecord(
            "pi-extension-config", str(init_cli.extension_config_path()), {}
        ),
    )
    monkeypatch.setattr(init_cli, "install_supervisor", lambda value: [])
    monkeypatch.setattr(init_cli, "remove_supervisor", lambda value: None)
    monkeypatch.setattr(
        init_cli, "_save_manifest_verified", lambda value: saved.append(list(value.targets))
    )
    monkeypatch.setattr(
        init_cli,
        "_start_profile_strict_locked",
        lambda value: (_ for _ in ()).throw(RuntimeError("startup failed")),
    )
    monkeypatch.setattr(init_cli, "_restore_manifest_snapshot", lambda *args: None)
    monkeypatch.setattr(init_cli, "stop_runtime", lambda value: None)

    with pytest.raises(click.ClickException, match="startup failed"):
        init_cli._init_native_hosts(
            hosts=[("pi", "/bin/pi")],
            release="1.2.3",
            manifest=manifest,
            manifest_snapshot=(True, b"prior", previous),
            port=8787,
        )

    assert saved == [["claude"]]
    assert removed == ["pi"]


def test_pi_and_omp_share_init_user_profile_and_config_artifact(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(init_cli, "extension_release_version", lambda version: "1.2.3")
    monkeypatch.setattr(init_cli.shutil, "which", lambda target: f"/bin/{target}")
    monkeypatch.setattr(
        init_cli,
        "_init_native_hosts",
        lambda **kwargs: calls.extend(
            (host, kwargs["manifest"].profile) for host, _binary in kwargs["hosts"]
        ),
    )
    monkeypatch.setattr(init_cli, "_install_headroom_mcp_for_targets", lambda **kwargs: None)

    init_cli._run_init_targets(
        targets=["pi", "omp"],
        global_scope=True,
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )

    assert calls == [("pi", "init-user"), ("omp", "init-user")]


def test_omp_init_does_not_import_wrapper_runtime_or_touch_models_yml(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.delitem(sys.modules, "headroom.providers.omp.runtime", raising=False)
    init_cli, fake_main = _load_init_module(monkeypatch)
    models = tmp_path / "models.yml"
    monkeypatch.setattr(init_cli, "_run_init_targets", lambda **kwargs: None)

    result = CliRunner().invoke(fake_main, ["init", "-g", "omp"])

    assert result.exit_code == 0, result.output
    assert "headroom.providers.omp.runtime" not in sys.modules
    assert not models.exists()


def test_init_auto_detects_targets(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()
    captured: dict[str, object] = {}

    monkeypatch.setattr(init_cli, "detect_init_targets", lambda global_scope: ["claude", "codex"])
    monkeypatch.setattr(init_cli, "_run_init_targets", lambda **kwargs: captured.update(kwargs))

    result = runner.invoke(fake_main, ["init", "-g"])

    assert result.exit_code == 0, result.output
    assert captured["targets"] == ["claude", "codex"]
    assert captured["global_scope"] is True


def test_init_fails_when_auto_detection_empty(monkeypatch) -> None:
    """Bare ``headroom init`` with no agents on PATH prints a guided error.

    Regression guard for issue #245: the error must list every target that
    was probed, confirm that -g / --global is a valid flag, and show the
    explicit per-target invocation so the user knows how to proceed.
    """

    init_cli, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()
    monkeypatch.setattr(init_cli.shutil, "which", lambda name: None)

    result = runner.invoke(fake_main, ["init", "-g"])

    assert result.exit_code != 0
    assert "No supported user-scope agents were found on PATH" in result.output
    assert "probed the following agents" in result.output
    # Every in-scope target is listed with its lookup status.
    for target in ("claude", "codex", "copilot", "openclaw"):
        assert target in result.output
    # The user is told that -g is still valid and given a concrete next step.
    assert "-g" in result.output
    assert "headroom init -g claude" in result.output


def test_format_empty_detection_error_local_scope(monkeypatch) -> None:
    """Local-scope variant of the guided error only lists local-scope agents."""

    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli.shutil, "which", lambda name: None)

    message = init_cli._format_empty_detection_error(global_scope=False)

    assert "local-scope agents" in message
    assert "claude" in message and "codex" in message
    # Copilot / openclaw are global-only; must not be suggested for local.
    assert "headroom init copilot" not in message
    assert "headroom init openclaw" not in message
    assert "headroom init claude" in message
    assert "headroom init codex" in message


def test_format_empty_detection_error_reports_found_paths(monkeypatch, tmp_path) -> None:
    """When a binary IS present, the error still surfaces its path for debugging."""

    init_cli, _ = _load_init_module(monkeypatch)
    fake_claude = tmp_path / "claude"
    fake_claude.write_text("")
    monkeypatch.setattr(
        init_cli.shutil,
        "which",
        lambda name: str(fake_claude) if name == "claude" else None,
    )

    message = init_cli._format_empty_detection_error(global_scope=True)

    assert f"claude: found at {fake_claude}" in message
    assert "codex: not found" in message


def test_init_verbose_enables_debug_logging_on_stderr(monkeypatch) -> None:
    """``headroom init -v`` should emit diagnostic lines to stderr.

    Different Click 8.x versions expose stderr on ``CliRunner`` results
    differently (``mix_stderr`` was removed in 8.2, and ``result.stderr``
    appeared around the same time). To stay compatible with any Click 8.x
    the repo targets, the test reads ``result.stderr`` when the attribute
    exists AND contains data, otherwise falls back to ``result.output``
    (which is the combined stream when stderr isn't captured separately).
    """

    init_cli, fake_main = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli.shutil, "which", lambda name: None)
    runner = CliRunner()

    result = runner.invoke(fake_main, ["init", "-v", "-g"])

    # Newer Click: stderr captured separately.
    stderr = getattr(result, "stderr", None) or ""
    if not stderr:
        # Older Click: everything in result.output.
        stderr = result.output

    assert result.exit_code != 0, f"output: {result.output!r}"
    assert "[headroom init]" in stderr
    assert "detect_init_targets" in stderr
    assert "global_scope=True" in stderr
    for target in ("claude", "codex", "copilot", "openclaw"):
        assert target in stderr


def test_init_verbose_is_idempotent(monkeypatch) -> None:
    """Calling _enable_verbose_logging repeatedly keeps one handler attached."""

    init_cli, _ = _load_init_module(monkeypatch)
    # Clear any prior handler state on the dedicated init logger.
    init_cli.logger.handlers.clear()
    if hasattr(init_cli.logger, init_cli._VERBOSE_HANDLER_ATTR):
        delattr(init_cli.logger, init_cli._VERBOSE_HANDLER_ATTR)

    init_cli._enable_verbose_logging()
    init_cli._enable_verbose_logging()
    init_cli._enable_verbose_logging()

    assert len(init_cli.logger.handlers) == 1


def test_init_copilot_requires_global(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()
    monkeypatch.setattr(init_cli, "_ensure_runtime_manifest", lambda **kwargs: "init-local-test")

    result = runner.invoke(fake_main, ["init", "copilot"])

    assert result.exit_code != 0
    assert "requires -g" in result.output


def test_init_claude_local_writes_settings_and_installs_marketplace(
    monkeypatch, tmp_path: Path
) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    marketplace_calls: list[str] = []
    monkeypatch.setattr(init_cli, "_ensure_runtime_manifest", lambda **kwargs: "init-local-demo")
    monkeypatch.setattr(
        init_cli,
        "_install_claude_marketplace",
        lambda scope: marketplace_calls.append(scope),
    )

    result = runner.invoke(fake_main, ["init", "claude"])

    assert result.exit_code == 0, result.output
    settings_path = tmp_path / ".claude" / "settings.local.json"
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert marketplace_calls == ["local"]
    assert any(
        "--profile init-local-demo" in hook["command"] and "init hook ensure" in hook["command"]
        for entry in payload["hooks"]["SessionStart"]
        for hook in entry["hooks"]
    )


def test_init_codex_merges_feature_flag_into_existing_table(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("[features]\nshell_tool = true\n", encoding="utf-8")

    init_cli._init_codex(global_scope=False, profile="init-local-demo", port=9000)

    content = config_path.read_text(encoding="utf-8")
    assert 'base_url = "http://127.0.0.1:9000/v1"' in content
    assert content.count("[features]") == 1
    assert "hooks = true" in content
    assert 'env_key = "OPENAI_API_KEY"' not in content
    hooks = json.loads((tmp_path / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    assert "--profile init-local-demo" in hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    assert "init hook ensure" in hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]


def test_init_codex_creates_hooks_feature_flag_on_first_init(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.chdir(tmp_path)

    init_cli._init_codex(global_scope=False, profile="init-local-demo", port=9000)

    content = (tmp_path / ".codex" / "config.toml").read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert parsed["model_provider"] == "headroom"
    assert parsed["features"]["hooks"] is True
    assert "codex_hooks" not in content


def test_init_claude_uses_custom_port(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_cli, "_install_claude_marketplace", lambda scope: None)

    init_cli._init_claude(global_scope=False, profile="init-local-demo", port=9011)

    payload = json.loads((tmp_path / ".claude" / "settings.local.json").read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9011"


def test_init_copilot_global_writes_hooks_and_env(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    captured_env: dict[str, str] = {}
    monkeypatch.setattr(init_cli, "_copilot_config_path", lambda: tmp_path / "copilot-config.json")
    monkeypatch.setattr(init_cli, "_apply_user_env", lambda values: captured_env.update(values))
    monkeypatch.setattr(init_cli, "_install_copilot_marketplace", lambda: None)

    init_cli._init_copilot(global_scope=True, profile="init-user", port=9005, backend="openai")

    payload = json.loads((tmp_path / "copilot-config.json").read_text(encoding="utf-8"))
    assert "SessionStart" in payload["hooks"]
    assert "PreToolUse" in payload["hooks"]
    assert "--profile init-user" in payload["hooks"]["SessionStart"][0]["command"]
    assert captured_env == {
        "COPILOT_PROVIDER_TYPE": "openai",
        "COPILOT_PROVIDER_BASE_URL": "http://127.0.0.1:9005/v1",
        "COPILOT_PROVIDER_WIRE_API": "completions",
    }


def test_init_hook_ensure_prefers_local_profile(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    ensured: list[str] = []

    def fake_load(profile: str):
        return object() if profile == "init-repo-12345678" else None

    monkeypatch.setattr(init_cli, "_local_profile", lambda cwd=None: "init-repo-12345678")
    monkeypatch.setattr(init_cli, "load_manifest", fake_load)
    monkeypatch.setattr(
        init_cli, "_ensure_profile_running", lambda profile: ensured.append(profile)
    )

    runner = CliRunner()
    result = runner.invoke(fake_main, ["init", "hook", "ensure"])

    assert result.exit_code == 0, result.output
    assert ensured == ["init-repo-12345678"]


def test_init_openclaw_requires_global(monkeypatch) -> None:
    _, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(fake_main, ["init", "openclaw"])

    assert result.exit_code != 0
    assert "requires -g" in result.output


def test_init_openclaw_delegates_to_wrap(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    calls: list[list[str]] = []

    class _Result:
        returncode = 0

    monkeypatch.setattr(init_cli, "resolve_headroom_command", lambda: ["headroom"])
    monkeypatch.setattr(
        init_cli.subprocess,
        "run",
        lambda cmd: calls.append(cmd) or _Result(),
    )

    init_cli._init_openclaw(global_scope=True, port=9999)

    assert calls == [["headroom", "wrap", "openclaw", "--proxy-port", "9999"]]


def test_detect_init_targets_respects_scope(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(
        init_cli.shutil,
        "which",
        lambda name: name if name in {"claude", "copilot", "codex", "openclaw"} else None,
    )

    assert init_cli.detect_init_targets(False) == ["claude", "codex"]
    assert init_cli.detect_init_targets(True) == ["claude", "copilot", "codex", "openclaw"]


def test_marketplace_source_prefers_env_override(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setenv("HEADROOM_MARKETPLACE_SOURCE", "custom/source")

    assert init_cli._marketplace_source() == "custom/source"


def test_run_checked_treats_existing_install_as_success(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)

    class _Result:
        returncode = 1
        stderr = "plugin already exists"
        stdout = ""

    monkeypatch.setattr(init_cli.subprocess, "run", lambda *args, **kwargs: _Result())

    init_cli._run_checked(["claude", "plugin", "install"], action="claude plugin install")


def test_command_string_and_matcher_on_windows(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(init_cli.subprocess, "list2cmdline", lambda parts: "joined-command")

    assert init_cli._command_string(["headroom", "init"]) == "joined-command"
    assert init_cli._powershell_matcher() == "Bash|PowerShell"


def test_command_string_normalizes_backslashes_on_windows(monkeypatch) -> None:
    """Backslash paths must become forward slashes so Git Bash hooks work (#724)."""
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli, "os", SimpleNamespace(name="nt"))

    result = init_cli._command_string(
        ["C:\\Users\\user\\.local\\bin\\headroom.exe", "init", "hook", "ensure"]
    )
    assert "\\" not in result
    assert "C:/Users/user/.local/bin/headroom.exe" in result


def test_command_string_quotes_spaces_after_normalization(monkeypatch) -> None:
    """Paths with spaces must stay properly quoted after backslash normalization (#724)."""
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli, "os", SimpleNamespace(name="nt"))

    result = init_cli._command_string(
        ["C:\\Program Files\\headroom\\headroom.exe", "init", "hook", "ensure"]
    )
    assert "\\" not in result
    assert '"C:/Program Files/headroom/headroom.exe"' in result


def test_json_file_handles_missing_empty_and_non_mapping(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    missing = tmp_path / "missing.json"
    empty = tmp_path / "empty.json"
    array_payload = tmp_path / "payload.json"
    empty.write_text("   \n", encoding="utf-8")
    array_payload.write_text('["value"]\n', encoding="utf-8")

    assert init_cli._json_file(missing) == {}
    assert init_cli._json_file(empty) == {}
    assert init_cli._json_file(array_payload) == {}


def test_json_file_rejects_malformed_json(monkeypatch, tmp_path: Path) -> None:
    # A user-owned settings file with a hand-edit typo (trailing comma) must
    # fail with an actionable error rather than crashing init with a raw
    # JSONDecodeError or silently overwriting the file (which the caller's
    # subsequent _write_json would do if this returned {}).
    init_cli, _ = _load_init_module(monkeypatch)
    malformed = tmp_path / "settings.json"
    malformed.write_text('{"env": {"A": "B",}}\n', encoding="utf-8")

    with pytest.raises(click.ClickException, match="invalid JSON"):
        init_cli._json_file(malformed)

    # The file is left untouched (not overwritten).
    assert malformed.read_text(encoding="utf-8") == '{"env": {"A": "B",}}\n'


def test_ensure_claude_hooks_rewrites_existing_entries(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "env": {"KEEP": "1"},
                "hooks": {
                    "SessionStart": [
                        "not-a-dict",
                        {"hooks": "not-a-list"},
                        {
                            "matcher": "startup|resume",
                            "hooks": [{"type": "command", "command": "echo keep-me"}],
                        },
                        {
                            "matcher": "startup|resume",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "headroom init hook ensure --marker headroom-init-claude",
                                }
                            ],
                        },
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(init_cli, "_hook_command", lambda *parts: "headroom init hook ensure")

    init_cli._ensure_claude_hooks(settings_path, "init-local-demo", 9001)

    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["env"] == {
        "KEEP": "1",
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:9001",
        "ENABLE_TOOL_SEARCH": "true",
    }
    session_entries = payload["hooks"]["SessionStart"]
    assert session_entries[0] == "not-a-dict"
    assert session_entries[1] == {"hooks": "not-a-list"}
    assert session_entries[2]["hooks"][0]["command"] == "echo keep-me"
    assert session_entries[-1]["hooks"][0]["command"].endswith("--marker headroom-init-claude")


def test_ensure_copilot_hooks_replaces_existing_marker(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    config_path = tmp_path / "copilot.json"
    config_path.write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {"type": "command", "command": "echo keep"},
                        {
                            "type": "command",
                            "command": "headroom init hook ensure --marker headroom-init-copilot",
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(init_cli, "_hook_command", lambda *parts: "headroom init hook ensure")

    init_cli._ensure_copilot_hooks(config_path, "init-user")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    commands = [entry["command"] for entry in payload["hooks"]["SessionStart"]]
    assert commands == ["echo keep", "headroom init hook ensure --marker headroom-init-copilot"]


def test_ensure_codex_hooks_preserves_user_hooks(monkeypatch, tmp_path: Path) -> None:
    """init codex must merge into hooks.json, not overwrite it — a user's own
    hooks (and unrelated top-level keys) must survive."""
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "hooks.json"
    path.write_text(
        json.dumps(
            {
                "notify": True,  # unrelated top-level key
                "hooks": {
                    "SessionStart": [
                        {
                            "matcher": "startup",
                            "hooks": [{"type": "command", "command": "echo keep"}],
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(init_cli, "_hook_command", lambda *parts: "headroom init hook ensure")

    init_cli._ensure_codex_hooks(path, "init-user")

    payload = json.loads(path.read_text(encoding="utf-8"))
    # The unrelated top-level key survives.
    assert payload["notify"] is True
    # The user's own hook is retained and Headroom's is appended exactly once.
    ss_commands = [
        item["command"] for entry in payload["hooks"]["SessionStart"] for item in entry["hooks"]
    ]
    assert "echo keep" in ss_commands
    assert sum(1 for c in ss_commands if "headroom-init-codex" in c) == 1


def test_replace_marker_block_replaces_existing_block(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    content = "before\n# start\nold\n# end\nafter\n"

    replaced = init_cli._replace_marker_block(content, "# start", "# end", "# start\nnew\n# end")

    assert replaced == "before\n\nafter\n\n# start\nnew\n# end\n"


def test_ensure_codex_provider_replaces_existing_marker(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text(
        f"prefix\n{init_cli._CODEX_PROVIDER_MARKER_START}\nold = true\n{init_cli._CODEX_PROVIDER_MARKER_END}\n",
        encoding="utf-8",
    )

    init_cli._ensure_codex_provider(path, 9100)

    content = path.read_text(encoding="utf-8")
    assert content.count(init_cli._CODEX_PROVIDER_MARKER_START) == 1
    assert 'base_url = "http://127.0.0.1:9100/v1"' in content
    assert "old = true" not in content
    assert 'env_key = "OPENAI_API_KEY"' not in content


def test_ensure_codex_provider_keeps_root_keys_above_existing_table(
    monkeypatch, tmp_path: Path
) -> None:
    """#260: a config ending in a table must not capture the provider root keys.

    Appending the block after a trailing [features] table scoped model_provider
    under it, so Codex refused to start with
    'invalid type: string "headroom", expected a boolean in features'.
    """
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text("[features]\nhooks = true\n", encoding="utf-8")

    init_cli._ensure_codex_provider(path, 8787)

    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    # model_provider belongs at the document root, not under [features].
    assert parsed["model_provider"] == "headroom"
    assert "model_provider" not in parsed["features"]
    assert "openai_base_url" not in parsed["features"]
    # The user's existing table is preserved.
    assert parsed["features"]["hooks"] is True
    assert parsed["model_providers"]["headroom"]["base_url"] == "http://127.0.0.1:8787/v1"


def test_ensure_codex_provider_replaces_existing_model_provider(
    monkeypatch, tmp_path: Path
) -> None:
    """A pre-existing root model_provider is replaced, never duplicated (#260).

    A second top-level model_provider key would be invalid TOML; init owns it.
    """
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text('model_provider = "openai"\n[features]\nhooks = true\n', encoding="utf-8")

    init_cli._ensure_codex_provider(path, 8787)

    parsed = tomllib.loads(path.read_text(encoding="utf-8"))  # raises on a duplicate key
    assert parsed["model_provider"] == "headroom"
    assert parsed["features"]["hooks"] is True


def test_ensure_codex_provider_preserves_profile_overrides(monkeypatch, tmp_path: Path) -> None:
    """Per-profile model_provider/openai_base_url overrides must survive init.

    init owns the ROOT-level keys, but the same keys inside [profiles.*] are the
    user's per-profile routing; a broad strip silently reroutes those profiles
    to the injected "headroom" default (config corruption).
    """
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text(
        'model_provider = "openai"\n\n'
        "[profiles.work]\n"
        'model_provider = "azure"\n'
        'openai_base_url = "https://azure.example/v1"\n',
        encoding="utf-8",
    )

    init_cli._ensure_codex_provider(path, 8787)

    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    # Root is replaced by headroom (no duplicate top-level key).
    assert parsed["model_provider"] == "headroom"
    # The user's per-profile overrides are untouched.
    assert parsed["profiles"]["work"]["model_provider"] == "azure"
    assert parsed["profiles"]["work"]["openai_base_url"] == "https://azure.example/v1"


def test_ensure_codex_provider_emits_requires_openai_auth_for_chatgpt(
    monkeypatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    (tmp_path / "auth.json").write_text('{"auth_mode": "chatgpt"}', encoding="utf-8")

    init_cli._ensure_codex_provider(path, 8787)

    assert "requires_openai_auth = true" in path.read_text(encoding="utf-8")


def test_ensure_codex_provider_omits_requires_openai_auth_for_api_key(
    monkeypatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    (tmp_path / "auth.json").write_text('{"auth_mode": "apikey"}', encoding="utf-8")

    init_cli._ensure_codex_provider(path, 8787)

    assert "requires_openai_auth" not in path.read_text(encoding="utf-8")


def test_ensure_codex_feature_flag_replaces_existing_marker(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text(
        f"[features]\n{init_cli._CODEX_FEATURE_MARKER_START}\ncodex_hooks = false\n{init_cli._CODEX_FEATURE_MARKER_END}\n",
        encoding="utf-8",
    )

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    assert content.count(init_cli._CODEX_FEATURE_MARKER_START) == 1
    assert "hooks = true" in content
    assert "codex_hooks" not in content


def test_ensure_codex_feature_flag_replaces_marker_inside_features_scope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text(
        "[features]\n"
        f"{init_cli._CODEX_FEATURE_MARKER_START}\n"
        "hooks = true\n"
        f"{init_cli._CODEX_FEATURE_MARKER_END}\n"
        "\n[tools]\nhooks = false\n",
        encoding="utf-8",
    )

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert parsed["features"]["hooks"] is True
    assert parsed["tools"]["hooks"] is False
    assert content.count(init_cli._CODEX_FEATURE_MARKER_START) == 1


def test_ensure_codex_feature_flag_migrates_legacy_codex_hooks_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text("[features]\ncodex_hooks = true\nshell_tool = true\n", encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert parsed["features"]["hooks"] is True
    assert parsed["features"]["shell_tool"] is True
    assert "codex_hooks" not in parsed["features"]
    assert content.count("hooks = true") == 1
    assert content.count(init_cli._CODEX_FEATURE_MARKER_START) == 1


def test_ensure_codex_feature_flag_migrates_dotted_legacy_codex_hooks_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text("features.codex_hooks = true\nfeatures.shell_tool = true\n", encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert parsed["features"]["hooks"] is True
    assert parsed["features"]["shell_tool"] is True
    assert "codex_hooks" not in parsed["features"]
    assert "features.hooks = true" in content


def test_ensure_codex_feature_flag_migrates_when_both_keys_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    # A config that carried both the legacy and the correct key must not produce
    # a duplicate `hooks` key (which Codex would reject as invalid TOML).
    path.write_text("[features]\ncodex_hooks = true\nhooks = false\n", encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert "codex_hooks" not in parsed["features"]
    # The user's explicit `hooks` value is respected; only the legacy key is removed.
    assert parsed["features"]["hooks"] is False


def test_ensure_codex_feature_flag_migrates_when_keys_reversed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text("[features]\nhooks = false\ncodex_hooks = true\n", encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert "codex_hooks" not in parsed["features"]
    assert parsed["features"]["hooks"] is False


def test_ensure_codex_feature_flag_ignores_hooks_outside_features(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text(
        "[features]\nshell_tool = true\n\n[some_other_table]\nhooks = true\n",
        encoding="utf-8",
    )

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert parsed["features"]["hooks"] is True
    assert parsed["features"]["shell_tool"] is True
    assert parsed["some_other_table"]["hooks"] is True
    assert content.count(init_cli._CODEX_FEATURE_MARKER_START) == 1


def test_ensure_codex_feature_flag_ignores_hooks_after_commented_table_header(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text(
        "[features]\nshell_tool = true\n\n[some_other_table] # comment\nhooks = true\n",
        encoding="utf-8",
    )

    init_cli._ensure_codex_feature_flag(path)

    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert parsed["features"]["hooks"] is True
    assert parsed["features"]["shell_tool"] is True
    assert parsed["some_other_table"]["hooks"] is True


def test_ensure_codex_feature_flag_respects_commented_features_header_and_quoted_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text('[features] # comment\n"hooks" = false\n', encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert parsed["features"]["hooks"] is False
    assert init_cli._CODEX_FEATURE_MARKER_START not in content


def test_ensure_codex_feature_flag_migrates_quoted_legacy_codex_hooks_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text('[features]\n"codex_hooks" = true\n', encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert parsed["features"]["hooks"] is True
    assert "codex_hooks" not in parsed["features"]


def test_ensure_codex_feature_flag_respects_root_dotted_feature_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text("features.hooks = false\n", encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert parsed["features"]["hooks"] is False
    assert init_cli._CODEX_FEATURE_MARKER_START not in content


def test_ensure_codex_feature_flag_preserves_legacy_key_outside_features(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text("[some_other_table]\ncodex_hooks = true\n", encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    parsed = tomllib.loads(path.read_text(encoding="utf-8"))
    assert parsed["features"]["hooks"] is True
    assert parsed["some_other_table"]["codex_hooks"] is True


def test_ensure_codex_feature_flag_drops_legacy_key_outside_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text(
        "[features]\n"
        "codex_hooks = true\n"
        f"{init_cli._CODEX_FEATURE_MARKER_START}\n"
        "hooks = true\n"
        f"{init_cli._CODEX_FEATURE_MARKER_END}\n",
        encoding="utf-8",
    )

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    parsed = tomllib.loads(content)
    assert "codex_hooks" not in parsed["features"]
    assert parsed["features"]["hooks"] is True
    assert content.count("hooks = true") == 1


def test_ensure_codex_feature_flag_is_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text("[features]\ncodex_hooks = true\n", encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)
    first = path.read_text(encoding="utf-8")
    init_cli._ensure_codex_feature_flag(path)
    second = path.read_text(encoding="utf-8")

    assert first == second
    parsed = tomllib.loads(second)
    assert parsed["features"]["hooks"] is True
    assert second.count("hooks = true") == 1


def test_ensure_codex_feature_flag_creates_features_section_when_missing(
    monkeypatch, tmp_path: Path
) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / "config.toml"
    path.write_text('model = "gpt-5"\n', encoding="utf-8")

    init_cli._ensure_codex_feature_flag(path)

    content = path.read_text(encoding="utf-8")
    assert "[features]" in content
    assert "hooks = true" in content


def test_manifest_changed_detects_differences(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    existing = SimpleNamespace(
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory_enabled=False,
    )

    assert not init_cli._manifest_changed(
        existing,
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )
    assert init_cli._manifest_changed(
        existing,
        port=9000,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )


def test_ensure_runtime_manifest_merges_targets_and_stops_changed_runtime(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    existing = SimpleNamespace(
        targets=["claude"],
        mutations=["mutation"],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory_enabled=False,
    )
    saved: list[object] = []
    stopped: list[object] = []
    built = SimpleNamespace(supervisor_kind="", artifacts=[], mutations=[], targets=[])

    monkeypatch.setattr(init_cli, "_runtime_profile", lambda global_scope, cwd=None: "init-user")
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: existing)
    monkeypatch.setattr(
        init_cli,
        "build_manifest",
        lambda **kwargs: built.__dict__.update(kwargs) or built,
    )
    monkeypatch.setattr(init_cli, "save_manifest", lambda manifest: saved.append(manifest))
    monkeypatch.setattr(init_cli, "stop_runtime", lambda manifest: stopped.append(manifest))

    profile = init_cli._ensure_runtime_manifest(
        global_scope=True,
        targets=["codex"],
        port=9001,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )

    assert profile == "init-user"
    assert stopped == [existing]
    assert saved == [built]
    assert built.targets == ["claude", "codex"]
    assert built.mutations == ["mutation"]
    assert built.supervisor_kind == init_cli.SupervisorKind.NONE.value
    assert built.artifacts == []


def test_ensure_runtime_manifest_ignores_stop_runtime_errors(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    existing = SimpleNamespace(
        targets=[],
        mutations=[],
        port=8787,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory_enabled=False,
    )
    saved: list[object] = []
    built = SimpleNamespace(supervisor_kind="", artifacts=[], mutations=[], targets=[])

    monkeypatch.setattr(init_cli, "_runtime_profile", lambda global_scope, cwd=None: "init-user")
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: existing)
    monkeypatch.setattr(
        init_cli,
        "build_manifest",
        lambda **kwargs: built.__dict__.update(kwargs) or built,
    )
    monkeypatch.setattr(init_cli, "save_manifest", lambda manifest: saved.append(manifest))
    monkeypatch.setattr(
        init_cli, "stop_runtime", lambda manifest: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    init_cli._ensure_runtime_manifest(
        global_scope=True,
        targets=["claude"],
        port=9001,
        backend="anthropic",
        anyllm_provider=None,
        region=None,
        memory=False,
    )

    assert saved == [built]


def test_apply_user_env_routes_by_platform(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = SimpleNamespace(base_env={"OLD": "1"}, tool_envs={})
    windows_calls: list[object] = []
    unix_calls: list[object] = []
    monkeypatch.setattr(init_cli, "_env_manifest", lambda values: manifest)
    monkeypatch.setattr(
        init_cli, "_apply_windows_env_scope", lambda value: windows_calls.append(value)
    )
    monkeypatch.setattr(init_cli, "_apply_unix_env_scope", lambda value: unix_calls.append(value))

    monkeypatch.setattr(init_cli, "os", SimpleNamespace(name="nt"))
    init_cli._apply_user_env({"COPILOT_PROVIDER_TYPE": "openai"})
    monkeypatch.setattr(init_cli, "os", SimpleNamespace(name="posix"))
    init_cli._apply_user_env({"COPILOT_PROVIDER_TYPE": "anthropic"})

    assert manifest.base_env == {}
    assert manifest.tool_envs == {"copilot": {"COPILOT_PROVIDER_TYPE": "anthropic"}}
    assert windows_calls == [manifest]
    assert unix_calls == [manifest]


def test_resolve_copilot_env_supports_anthropic(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)

    assert init_cli._resolve_copilot_env(9010, "anthropic") == {
        "COPILOT_PROVIDER_TYPE": "anthropic",
        "COPILOT_PROVIDER_BASE_URL": "http://127.0.0.1:9010",
    }


def test_marketplace_source_prefers_repo_checkout(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.delenv("HEADROOM_MARKETPLACE_SOURCE", raising=False)

    assert init_cli.__file__ is not None
    assert init_cli._marketplace_source() == str(Path(init_cli.__file__).resolve().parents[2])


def test_run_checked_raises_on_failure(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)

    class _Result:
        returncode = 2
        stderr = "bad stderr"
        stdout = "bad stdout"

    monkeypatch.setattr(init_cli.subprocess, "run", lambda *args, **kwargs: _Result())

    with pytest.raises(
        click.ClickException, match="claude plugin install failed: bad stderr\nbad stdout"
    ):
        init_cli._run_checked(["claude", "plugin", "install"], action="claude plugin install")


def test_install_claude_marketplace_errors_without_binary(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli.shutil, "which", lambda name: None)

    with pytest.raises(click.ClickException, match="'claude' not found"):
        init_cli._install_claude_marketplace("local")


def test_install_claude_marketplace_runs_expected_commands(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    calls: list[tuple[list[str], str]] = []
    monkeypatch.setattr(init_cli.shutil, "which", lambda name: "claude")
    monkeypatch.setattr(init_cli, "_marketplace_source", lambda: "repo/source")
    monkeypatch.setattr(
        init_cli, "_run_checked", lambda command, action: calls.append((command, action))
    )

    init_cli._install_claude_marketplace("user")

    assert calls == [
        (["claude", "plugin", "marketplace", "add", "repo/source"], "claude marketplace add"),
        (
            ["claude", "plugin", "install", "headroom@headroom-marketplace", "--scope", "user"],
            "claude plugin install",
        ),
    ]


def test_install_copilot_marketplace_handles_missing_binary(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(init_cli.shutil, "which", lambda name: None)

    with pytest.raises(click.ClickException, match="'copilot' not found"):
        init_cli._install_copilot_marketplace()


def test_install_copilot_marketplace_runs_expected_commands(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    calls: list[tuple[list[str], str]] = []
    monkeypatch.setattr(init_cli.shutil, "which", lambda name: "copilot")
    monkeypatch.setattr(init_cli, "_marketplace_source", lambda: "repo/source")
    monkeypatch.setattr(
        init_cli, "_run_checked", lambda command, action: calls.append((command, action))
    )

    init_cli._install_copilot_marketplace()

    assert calls == [
        (["copilot", "plugin", "marketplace", "add", "repo/source"], "copilot marketplace add"),
        (
            ["copilot", "plugin", "install", "headroom@headroom-marketplace"],
            "copilot plugin install",
        ),
    ]


def test_ensure_profile_running_covers_runtime_modes(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    docker_manifest = SimpleNamespace(
        preset=init_cli.InstallPreset.PERSISTENT_DOCKER.value,
        supervisor_kind=init_cli.SupervisorKind.NONE.value,
        profile="docker-profile",
    )
    service_manifest = SimpleNamespace(
        preset=init_cli.InstallPreset.PERSISTENT_TASK.value,
        supervisor_kind=init_cli.SupervisorKind.SERVICE.value,
        profile="service-profile",
    )
    task_manifest = SimpleNamespace(
        preset=init_cli.InstallPreset.PERSISTENT_TASK.value,
        supervisor_kind=init_cli.SupervisorKind.NONE.value,
        profile="task-profile",
    )
    manifests = {
        "docker-profile": docker_manifest,
        "service-profile": service_manifest,
        "task-profile": task_manifest,
    }
    docker_calls: list[object] = []
    service_calls: list[object] = []
    detached_calls: list[str] = []
    wait_calls: list[tuple[str, int]] = []

    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifests.get(profile))
    monkeypatch.setattr(init_cli, "runtime_status", lambda manifest: "stopped")

    @contextmanager
    def fake_start_lock(profile: str):
        yield True

    monkeypatch.setattr(init_cli, "acquire_runtime_start_lock", fake_start_lock)

    def fake_wait_ready(manifest, timeout_seconds: int) -> bool:
        wait_calls.append((manifest.profile, timeout_seconds))
        return False

    monkeypatch.setattr(init_cli, "wait_ready", fake_wait_ready)
    monkeypatch.setattr(
        init_cli, "start_persistent_docker", lambda manifest: docker_calls.append(manifest)
    )
    monkeypatch.setattr(
        init_cli, "start_supervisor", lambda manifest: service_calls.append(manifest)
    )
    monkeypatch.setattr(
        init_cli,
        "start_detached_agent",
        lambda profile: detached_calls.append(profile),
    )

    init_cli._ensure_profile_running("missing")
    init_cli._ensure_profile_running("docker-profile")
    init_cli._ensure_profile_running("service-profile")
    init_cli._ensure_profile_running("task-profile")

    assert docker_calls == [docker_manifest]
    assert service_calls == [service_manifest]
    assert detached_calls == ["task-profile"]
    assert ("docker-profile", 1) in wait_calls
    assert ("docker-profile", 45) in wait_calls


def test_ensure_profile_running_suppresses_hook_recovery_output(monkeypatch, capfd) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = SimpleNamespace(
        preset=init_cli.InstallPreset.PERSISTENT_TASK.value,
        supervisor_kind=init_cli.SupervisorKind.SERVICE.value,
        profile="service-profile",
    )

    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli, "wait_ready", lambda manifest, timeout_seconds: False)

    def noisy_start_supervisor(manifest) -> None:
        print("python stdout")
        print("python stderr", file=sys.stderr)
        os.write(1, b"fd stdout\n")
        os.write(2, b"fd stderr\n")
        raise RuntimeError("not permitted")

    monkeypatch.setattr(init_cli, "start_supervisor", noisy_start_supervisor)

    init_cli._ensure_profile_running("service-profile")

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_ensure_profile_running_returns_when_ready_or_on_exception(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = SimpleNamespace(
        preset=init_cli.InstallPreset.PERSISTENT_TASK.value,
        supervisor_kind=init_cli.SupervisorKind.NONE.value,
        profile="task-profile",
    )
    detached_calls: list[str] = []
    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli, "wait_ready", lambda manifest, timeout_seconds: True)
    monkeypatch.setattr(
        init_cli,
        "start_detached_agent",
        lambda profile: detached_calls.append(profile),
    )

    init_cli._ensure_profile_running("task-profile")
    assert detached_calls == []

    @contextmanager
    def fake_start_lock(profile: str):
        yield True

    monkeypatch.setattr(init_cli, "acquire_runtime_start_lock", fake_start_lock)
    monkeypatch.setattr(init_cli, "runtime_status", lambda manifest: "stopped")
    monkeypatch.setattr(init_cli, "wait_ready", lambda manifest, timeout_seconds: False)
    monkeypatch.setattr(
        init_cli,
        "start_detached_agent",
        lambda profile: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    init_cli._ensure_profile_running("task-profile")


def test_ensure_profile_running_skips_spawn_when_start_lock_is_held(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = SimpleNamespace(
        preset=init_cli.InstallPreset.PERSISTENT_TASK.value,
        supervisor_kind=init_cli.SupervisorKind.NONE.value,
        profile="task-profile",
    )
    detached_calls: list[str] = []

    @contextmanager
    def fake_start_lock(profile: str):
        yield False

    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli, "wait_ready", lambda manifest, timeout_seconds: False)
    monkeypatch.setattr(init_cli, "acquire_runtime_start_lock", fake_start_lock)
    monkeypatch.setattr(
        init_cli,
        "start_detached_agent",
        lambda profile: detached_calls.append(profile),
    )

    init_cli._ensure_profile_running("task-profile")

    assert detached_calls == []


def test_ensure_profile_running_does_not_spawn_again_during_slow_startup(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    manifest = SimpleNamespace(
        preset=init_cli.InstallPreset.PERSISTENT_TASK.value,
        supervisor_kind=init_cli.SupervisorKind.NONE.value,
        profile="task-profile",
    )
    detached_calls: list[str] = []
    wait_calls: list[int] = []
    stop_calls: list[object] = []

    @contextmanager
    def fake_start_lock(profile: str):
        yield True

    def fake_wait_ready(manifest, timeout_seconds: int) -> bool:
        wait_calls.append(timeout_seconds)
        return bool(detached_calls and timeout_seconds == init_cli._STARTUP_READY_TIMEOUT_SECONDS)

    monkeypatch.setattr(init_cli, "load_manifest", lambda profile: manifest)
    monkeypatch.setattr(init_cli, "wait_ready", fake_wait_ready)
    monkeypatch.setattr(init_cli, "acquire_runtime_start_lock", fake_start_lock)
    monkeypatch.setattr(
        init_cli,
        "runtime_status",
        lambda manifest: "running" if detached_calls else "stopped",
    )
    monkeypatch.setattr(
        init_cli,
        "start_detached_agent",
        lambda profile: detached_calls.append(profile),
    )
    monkeypatch.setattr(init_cli, "stop_runtime", lambda manifest: stop_calls.append(manifest))

    init_cli._ensure_profile_running("task-profile")
    init_cli._ensure_profile_running("task-profile")

    assert detached_calls == ["task-profile"]
    assert init_cli._STARTUP_READY_TIMEOUT_SECONDS in wait_calls
    assert stop_calls == []


def test_init_codex_windows_warns_about_upstream_hook_limitation(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    messages: list[str] = []
    monkeypatch.setattr(init_cli, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(init_cli, "_codex_scope_path", lambda global_scope: Path("config.toml"))
    monkeypatch.setattr(init_cli, "_codex_hooks_path", lambda global_scope: Path("hooks.json"))
    monkeypatch.setattr(init_cli, "_ensure_codex_provider", lambda path, port: None)
    monkeypatch.setattr(init_cli, "_ensure_codex_feature_flag", lambda path: None)
    monkeypatch.setattr(init_cli, "_ensure_codex_hooks", lambda path, profile: None)
    monkeypatch.setattr(init_cli.click, "echo", lambda message: messages.append(message))

    init_cli._init_codex(global_scope=True, profile="init-user", port=9000)

    assert any("disabled upstream on Windows" in message for message in messages)


def test_init_openclaw_propagates_nonzero_exit(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)

    class _Result:
        returncode = 9

    monkeypatch.setattr(init_cli, "resolve_headroom_command", lambda: ["headroom"])
    monkeypatch.setattr(init_cli.subprocess, "run", lambda command: _Result())

    with pytest.raises(SystemExit) as exc:
        init_cli._init_openclaw(global_scope=True, port=9999)

    assert exc.value.code == 9


def test_run_init_targets_dispatches_supported_targets(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    calls: list[tuple[str, tuple[object, ...]]] = []
    monkeypatch.setattr(init_cli, "_ensure_runtime_manifest", lambda **kwargs: "init-profile")
    monkeypatch.setattr(
        init_cli,
        "_init_claude",
        lambda **kwargs: calls.append(
            ("claude", (kwargs["global_scope"], kwargs["profile"], kwargs["port"]))
        ),
    )
    monkeypatch.setattr(
        init_cli,
        "_init_copilot",
        lambda **kwargs: calls.append(
            ("copilot", (kwargs["global_scope"], kwargs["profile"], kwargs["port"]))
        ),
    )
    monkeypatch.setattr(
        init_cli,
        "_init_codex",
        lambda **kwargs: calls.append(
            ("codex", (kwargs["global_scope"], kwargs["profile"], kwargs["port"]))
        ),
    )
    monkeypatch.setattr(
        init_cli,
        "_init_openclaw",
        lambda **kwargs: calls.append(("openclaw", (kwargs["global_scope"], kwargs["port"]))),
    )

    init_cli._run_init_targets(
        targets=["claude", "copilot", "codex", "openclaw"],
        global_scope=True,
        port=9000,
        backend="openai",
        anyllm_provider="provider",
        region="us-east-1",
        memory=True,
    )

    assert calls == [
        ("claude", (True, "init-profile", 9000)),
        ("copilot", (True, "init-profile", 9000)),
        ("codex", (True, "init-profile", 9000)),
        ("openclaw", (True, 9000)),
    ]


def test_init_subcommand_uses_group_options(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()
    captured: dict[str, object] = {}
    monkeypatch.setattr(init_cli, "_run_init_targets", lambda **kwargs: captured.update(kwargs))

    result = runner.invoke(
        fake_main,
        ["init", "-g", "--port", "9007", "--backend", "openai", "--memory", "claude"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "targets": ["claude"],
        "global_scope": True,
        "port": 9007,
        "backend": "openai",
        "anyllm_provider": None,
        "region": None,
        "memory": True,
    }


def test_init_hook_ensure_prefers_global_when_local_missing(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    ensured: list[str] = []
    monkeypatch.setattr(init_cli, "_local_profile", lambda cwd=None: "init-repo-12345678")
    monkeypatch.setattr(
        init_cli,
        "load_manifest",
        lambda profile: object() if profile == init_cli._GLOBAL_PROFILE else None,
    )
    monkeypatch.setattr(
        init_cli, "_ensure_profile_running", lambda profile: ensured.append(profile)
    )

    runner = CliRunner()
    result = runner.invoke(fake_main, ["init", "hook", "ensure"])

    assert result.exit_code == 0, result.output
    assert ensured == [init_cli._GLOBAL_PROFILE]


def test_init_hook_ensure_uses_explicit_profile(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    ensured: list[str] = []
    monkeypatch.setattr(
        init_cli, "_ensure_profile_running", lambda profile: ensured.append(profile)
    )

    runner = CliRunner()
    result = runner.invoke(fake_main, ["init", "hook", "ensure", "--profile", "init-explicit"])

    assert result.exit_code == 0, result.output
    assert ensured == ["init-explicit"]


# ---------------------------------------------------------------------------
# Bug 3 (#406): _ensure_codex_provider must inject openai_base_url
# ---------------------------------------------------------------------------


def test_init_codex_writes_openai_base_url(monkeypatch, tmp_path: Path) -> None:
    """_ensure_codex_provider must write openai_base_url at the top level so that
    subscription (ChatGPT plan) users are routed through headroom even when the
    init entry point is used instead of wrap."""
    init_cli, _ = _load_init_module(monkeypatch)
    path = tmp_path / ".codex" / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)

    init_cli._ensure_codex_provider(path, 8787)

    content = path.read_text(encoding="utf-8")
    assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in content, (
        f"openai_base_url missing from init codex config:\n{content}"
    )
    # Must NOT appear inside a [section] block.
    lines = content.splitlines()
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("["):
            in_section = True
        if in_section and stripped.startswith("openai_base_url"):
            raise AssertionError(
                f"openai_base_url appeared inside a section block in init output:\n{content}"
            )
    # Bug 3 regression guard.
    assert "requires_openai_auth" not in content, (
        f"requires_openai_auth must not appear in init codex config:\n{content}"
    )


def test_init_codex_provider_retags_existing_threads(monkeypatch, tmp_path: Path) -> None:
    """`headroom init` injects `model_provider = "headroom"` for Codex, which
    Codex Desktop filters its history menu by. Without retagging, existing native
    `openai` threads vanish from the sidebar/search (#961). `_ensure_codex_provider`
    must retag existing threads openai->headroom so the history stays visible —
    the same reconciliation the install and wrap paths already perform."""
    import sqlite3

    init_cli, _ = _load_init_module(monkeypatch)

    codex_home = tmp_path / ".codex"
    config_path = codex_home / "config.toml"
    # Codex Desktop reads <codex_home>/sqlite/state_5.sqlite.
    db = codex_home / "sqlite" / "state_5.sqlite"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, model_provider TEXT NOT NULL)")
        conn.executemany(
            "INSERT INTO threads (id, model_provider) VALUES (?, ?)",
            [("t1", "openai"), ("t2", "openai"), ("t3", "anthropic")],
        )
        conn.commit()
    finally:
        conn.close()

    init_cli._ensure_codex_provider(config_path, 8787)

    conn = sqlite3.connect(str(db))
    try:
        counts = dict(
            conn.execute("SELECT model_provider, COUNT(*) FROM threads GROUP BY model_provider")
        )
    finally:
        conn.close()
    # Native threads now live under the active headroom provider (stay visible);
    # third-party providers are left untouched.
    assert counts.get("headroom") == 2, f"existing openai threads not retagged: {counts}"
    assert counts.get("openai", 0) == 0, f"openai threads still hidden: {counts}"
    assert counts.get("anthropic") == 1, f"third-party provider must be left alone: {counts}"


def test_init_codex_strip_removes_openai_base_url(monkeypatch, tmp_path: Path) -> None:
    """_strip_codex_init_block must remove both the managed block and any orphaned
    openai_base_url lines left by a crashed or partial init."""
    init_cli, _ = _load_init_module(monkeypatch)

    # Normal install-then-strip cycle.
    path = tmp_path / "config.toml"
    path.write_text('model = "gpt-4o"\n', encoding="utf-8")
    init_cli._ensure_codex_provider(path, 8787)
    assert 'openai_base_url = "http://127.0.0.1:8787/v1"' in path.read_text(encoding="utf-8")

    stripped = init_cli._strip_codex_init_block(path.read_text(encoding="utf-8"))
    assert "openai_base_url" not in stripped, (
        f"_strip_codex_init_block must remove openai_base_url after install:\n{stripped}"
    )
    assert "requires_openai_auth" not in stripped
    assert 'model = "gpt-4o"' in stripped

    # Orphan-cleanup path: openai_base_url left outside marker block.
    orphan_content = (
        'model = "gpt-4o"\n'
        'openai_base_url = "http://127.0.0.1:8787/v1"\n'
        'model_provider = "headroom"\n'
    )
    orphan_stripped = init_cli._strip_codex_init_block(orphan_content)
    assert "openai_base_url" not in orphan_stripped, (
        f"_strip_codex_init_block must remove orphaned openai_base_url:\n{orphan_stripped}"
    )
    assert 'model = "gpt-4o"' in orphan_stripped
