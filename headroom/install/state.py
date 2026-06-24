"""Persistence helpers for deployment manifests."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict

from .models import ArtifactRecord, DeploymentManifest, ManagedMutation, iso_utc_now
from .paths import deploy_root, manifest_path, profile_root

logger = logging.getLogger(__name__)


def save_manifest(manifest: DeploymentManifest) -> None:
    """Persist a deployment manifest to disk.

    Gracefully handles read-only filesystems by logging a warning
    instead of crashing.
    """
    try:
        root = profile_root(manifest.profile)
        root.mkdir(parents=True, exist_ok=True)
        manifest.updated_at = iso_utc_now()
        path = manifest_path(manifest.profile)
        path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        logger.warning("Cannot save deployment manifest: %s — continuing without persistence", e)


def load_manifest(profile: str = "default") -> DeploymentManifest | None:
    """Load a deployment manifest when present."""

    path = manifest_path(profile)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["mutations"] = [ManagedMutation(**item) for item in payload.get("mutations", [])]
    payload["artifacts"] = [ArtifactRecord(**item) for item in payload.get("artifacts", [])]
    return DeploymentManifest(**payload)


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
