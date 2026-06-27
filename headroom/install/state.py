"""Persistence helpers for deployment manifests."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from .models import ArtifactRecord, DeploymentManifest, ManagedMutation, iso_utc_now
from .paths import deploy_root, manifest_path, profile_root

logger = logging.getLogger(__name__)


def _atomic_write_text(path: Path, data: str) -> None:
    """Write ``data`` to ``path`` atomically.

    The payload is written to a temporary file in the same directory, flushed and
    fsynced, then moved into place with :func:`os.replace` (an atomic rename on
    both POSIX and Windows). A crash between truncate and full write therefore
    leaves either the previous file or the complete new one on disk, never a
    truncated manifest.
    """
    directory = path.parent
    fd, tmp_name = tempfile.mkstemp(dir=directory, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def save_manifest(manifest: DeploymentManifest) -> None:
    """Persist a deployment manifest to disk.

    The write is atomic so an interrupted save (SIGKILL, system restart, OOM)
    cannot leave a truncated ``manifest.json`` behind. Gracefully handles
    read-only filesystems by logging a warning instead of crashing.
    """
    try:
        root = profile_root(manifest.profile)
        root.mkdir(parents=True, exist_ok=True)
        manifest.updated_at = iso_utc_now()
        path = manifest_path(manifest.profile)
        _atomic_write_text(path, json.dumps(asdict(manifest), indent=2) + "\n")
    except OSError as e:
        logger.warning("Cannot save deployment manifest: %s — continuing without persistence", e)


def load_manifest(profile: str = "default") -> DeploymentManifest | None:
    """Load a deployment manifest when present.

    Returns ``None`` when the manifest is missing or unreadable — e.g. a
    truncated file left by a crash mid-write — mirroring the resilient
    behaviour of :func:`list_manifests` rather than raising on a corrupt
    payload.
    """

    path = manifest_path(profile)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["mutations"] = [ManagedMutation(**item) for item in payload.get("mutations", [])]
        payload["artifacts"] = [ArtifactRecord(**item) for item in payload.get("artifacts", [])]
        return DeploymentManifest(**payload)
    except (OSError, ValueError, TypeError) as e:
        logger.warning("Ignoring unreadable deployment manifest %s: %s", path, e)
        return None


def list_manifests() -> list[DeploymentManifest]:
    """Load all deployment manifests under the deployment root."""

    root = deploy_root()
    if not root.exists():
        return []

    manifests: list[DeploymentManifest] = []
    for candidate in sorted(root.glob("*/manifest.json")):
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            payload["mutations"] = [
                ManagedMutation(**item) for item in payload.get("mutations", [])
            ]
            payload["artifacts"] = [ArtifactRecord(**item) for item in payload.get("artifacts", [])]
            manifests.append(DeploymentManifest(**payload))
        except (OSError, ValueError, TypeError):
            continue
    return manifests


def delete_manifest(profile: str = "default") -> None:
    """Delete the full deployment profile state if present."""

    root = profile_root(profile)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
