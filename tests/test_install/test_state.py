from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.install.models import ArtifactRecord, DeploymentManifest, ManagedMutation
from headroom.install.state import (
    ManifestError,
    delete_manifest,
    list_manifests,
    load_manifest,
    save_manifest,
)


def _manifest() -> DeploymentManifest:
    return DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        provider_mode="manual",
        targets=["claude"],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        mutations=[ManagedMutation(target="env", kind="shell-block", path="x")],
        artifacts=[ArtifactRecord(kind="script", path="run-headroom.sh")],
    )


def test_save_and_load_manifest_round_trip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = _manifest()

    save_manifest(manifest)
    loaded = load_manifest("default")

    assert loaded is not None
    assert loaded.profile == "default"
    assert loaded.mutations[0].kind == "shell-block"
    assert loaded.artifacts[0].kind == "script"


def test_load_manifest_raises_manifest_error_on_corrupt_payload(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Simulate a crash mid-write: a truncated/garbage manifest left on disk.
    profile_dir = tmp_path / ".headroom" / "deploy" / "default"
    profile_dir.mkdir(parents=True)
    (profile_dir / "manifest.json").write_text("{not json", encoding="utf-8")

    # A present-but-corrupt manifest surfaces a typed ManifestError so callers can
    # report cleanly or degrade, rather than a raw JSONDecodeError traceback.
    with pytest.raises(ManifestError):
        load_manifest("default")


def test_save_manifest_writes_atomically(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    save_manifest(_manifest())

    # No leftover temp file from the atomic write; only the manifest itself.
    profile_dir = tmp_path / ".headroom" / "deploy" / "default"
    assert sorted(p.name for p in profile_dir.iterdir()) == ["manifest.json"]
    # And the persisted manifest still round-trips.
    assert load_manifest("default") is not None


def test_list_manifests_ignores_invalid_payloads(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    valid = _manifest()
    save_manifest(valid)

    broken_dir = tmp_path / ".headroom" / "deploy" / "broken"
    broken_dir.mkdir(parents=True)
    (broken_dir / "manifest.json").write_text("{not json", encoding="utf-8")

    manifests = list_manifests()

    assert [manifest.profile for manifest in manifests] == ["default"]


def _write_manifest_with_image(profile_dir: Path, image: str) -> None:
    profile_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": profile_dir.name,
        "preset": "persistent-docker",
        "runtime_kind": "docker",
        "supervisor_kind": "none",
        "scope": "user",
        "provider_mode": "manual",
        "targets": ["claude"],
        "port": 8787,
        "host": "127.0.0.1",
        "backend": "anthropic",
        "image": image,
    }
    (profile_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def test_load_manifest_migrates_retired_image_repo(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile_dir = tmp_path / ".headroom" / "deploy" / "default"
    # A manifest written before the org move still pins the retired personal
    # repo, which is frozen at 0.27.0. Loading it must rewrite the repo while
    # preserving the tag, so the deployment tracks the current image (#2426).
    _write_manifest_with_image(profile_dir, "ghcr.io/chopratejas/headroom:latest")

    loaded = load_manifest("default")

    assert loaded is not None
    assert loaded.image == "ghcr.io/headroomlabs-ai/headroom:latest"


def test_load_manifest_leaves_unrelated_image_untouched(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile_dir = tmp_path / ".headroom" / "deploy" / "default"
    _write_manifest_with_image(profile_dir, "ghcr.io/headroomlabs-ai/headroom:0.31.0")

    loaded = load_manifest("default")

    assert loaded is not None
    # An already-current image, and any third-party image, must pass through
    # unchanged so the migration only ever rewrites the one retired repo.
    assert loaded.image == "ghcr.io/headroomlabs-ai/headroom:0.31.0"


def test_list_manifests_migrates_retired_image_repo(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _write_manifest_with_image(
        tmp_path / ".headroom" / "deploy" / "default",
        "ghcr.io/chopratejas/headroom:0.27.0",
    )

    manifests = list_manifests()

    assert [m.image for m in manifests] == ["ghcr.io/headroomlabs-ai/headroom:0.27.0"]


def test_delete_manifest_removes_profile_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = _manifest()
    save_manifest(manifest)
    extra_file = tmp_path / ".headroom" / "deploy" / "default" / "runner.log"
    extra_file.write_text("log", encoding="utf-8")

    delete_manifest("default")

    assert load_manifest("default") is None
    assert not extra_file.parent.exists()
