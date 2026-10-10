from __future__ import annotations

import json
import logging
import os
import stat
from pathlib import Path

import pytest

from headroom.install.models import ArtifactRecord, DeploymentManifest, ManagedMutation
from headroom.install.state import (
    ManifestError,
    delete_manifest,
    delete_recovery_manifest,
    list_manifests,
    load_manifest,
    save_manifest,
    save_recovery_manifest,
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


def test_recovery_manifest_preserves_mutations_and_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    manifest = _manifest()

    save_recovery_manifest(manifest)

    recovery = tmp_path / ".headroom" / "deploy" / "default.recovery.json"
    assert recovery.exists()
    payload = json.loads(recovery.read_text(encoding="utf-8"))
    assert payload["mutations"][0]["kind"] == "shell-block"
    assert payload["artifacts"][0]["kind"] == "script"
    assert load_manifest("default") is None
    delete_recovery_manifest("default")
    assert not recovery.exists()


def test_list_manifests_ignores_invalid_payloads(monkeypatch, tmp_path: Path, caplog) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    valid = _manifest()
    save_manifest(valid)

    broken_dir = tmp_path / ".headroom" / "deploy" / "broken"
    broken_dir.mkdir(parents=True)
    (broken_dir / "manifest.json").write_text("{not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="headroom.install.state"):
        manifests = list_manifests()

    assert [manifest.profile for manifest in manifests] == ["default"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(str(broken_dir / "manifest.json") in message for message in warnings)
    assert not any(str(tmp_path / ".headroom" / "deploy" / "default") in m for m in warnings)


def _broken_profile(tmp_path: Path, name: str) -> Path:
    path = tmp_path / ".headroom" / "deploy" / name / "manifest.json"
    path.parent.mkdir(parents=True)
    return path


def test_unreadable_and_malformed_manifests_get_distinct_messages(
    monkeypatch, tmp_path: Path, caplog
) -> None:
    """A read failure must not be reported as corruption: the fix differs."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    malformed = _broken_profile(tmp_path, "malformed")
    malformed.write_text("{not json", encoding="utf-8")
    unreadable = _broken_profile(tmp_path, "unreadable")
    unreadable.write_text(json.dumps({"profile": "unreadable"}), encoding="utf-8")

    real_read_text = Path.read_text

    def read_text(self: Path, *args, **kwargs) -> str:
        if self == unreadable:
            raise PermissionError(13, "Permission denied", str(self))
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    with caplog.at_level(logging.WARNING, logger="headroom.install.state"):
        assert list_manifests() == []
    messages = [r.getMessage() for r in caplog.records if r.name == "headroom.install.state"]
    assert len(messages) == 2
    corrupt = next(m for m in messages if str(malformed) in m)
    unread = next(m for m in messages if str(unreadable) in m)
    assert corrupt.startswith("Skipping corrupt deployment manifest")
    assert unread.startswith("Skipping unreadable deployment manifest")
    assert "Permission denied" in unread and "permissions" in unread

    with pytest.raises(ManifestError, match="is corrupt"):
        load_manifest("malformed")
    with pytest.raises(ManifestError, match="could not be read .*Permission denied"):
        load_manifest("unreadable")


@pytest.mark.skipif(os.name != "posix", reason="mode bits decide access only on POSIX")
def test_save_manifest_keeps_manifest_owner_only(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    save_manifest(_manifest())
    path = tmp_path / ".headroom" / "deploy" / "default" / "manifest.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    # A manifest left group/world-readable is tightened on the next save.
    path.chmod(0o644)
    save_manifest(_manifest())
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


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


def test_delete_manifest_propagates_removal_failure(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    root = tmp_path / ".headroom" / "deploy" / "default"
    root.mkdir(parents=True)
    monkeypatch.setattr(
        "headroom.install.state.shutil.rmtree", lambda path: (_ for _ in ()).throw(OSError("busy"))
    )

    with pytest.raises(OSError, match="busy"):
        delete_manifest("default")


@pytest.mark.windows_newline
def test_manifest_writes_pin_lf(monkeypatch, tmp_path: Path) -> None:
    import headroom.install.state as state

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    calls: list[dict] = []
    real_fdopen = state.os.fdopen

    def fdopen(fd, *args, **kwargs):
        calls.append(kwargs)
        return real_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(state.os, "fdopen", fdopen)
    save_manifest(_manifest())
    save_recovery_manifest(_manifest())

    assert [call.get("newline") for call in calls] == ["\n", "\n"]
