from __future__ import annotations

from pathlib import Path

from headroom.install.models import ArtifactRecord, DeploymentManifest, ManagedMutation
from headroom.install.state import delete_manifest, list_manifests, load_manifest, save_manifest


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


def test_list_manifests_ignores_invalid_payloads(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    valid = _manifest()
    save_manifest(valid)

    broken_dir = tmp_path / ".headroom" / "deploy" / "broken"
    broken_dir.mkdir(parents=True)
    (broken_dir / "manifest.json").write_text("{not json", encoding="utf-8")

    manifests = list_manifests()

    assert [manifest.profile for manifest in manifests] == ["default"]


def test_delete_manifest_removes_profile_root(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = _manifest()
    save_manifest(manifest)
    extra_file = tmp_path / ".headroom" / "deploy" / "default" / "runner.log"
    extra_file.write_text("log", encoding="utf-8")

    delete_manifest("default")

    assert load_manifest("default") is None
    assert not extra_file.parent.exists()
