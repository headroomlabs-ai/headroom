"""Persistence helpers for deployment manifests."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import ArtifactRecord, DeploymentManifest, ManagedMutation, iso_utc_now
from .paths import deploy_root, manifest_path, profile_root

logger = logging.getLogger(__name__)


class ManifestError(Exception):
    """A deployment manifest exists on disk but could not be parsed."""


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


# The Docker image org moved from a personal repo to the project org. The old
# ``ghcr.io/chopratejas/headroom`` repo is frozen at 0.27.0, so a manifest that
# still pins it silently runs ~5 minor versions behind the CLI with no drift
# signal (#2426). Rewrite it to the org repo on load, preserving the tag.
_DEPRECATED_IMAGE_REPO = "ghcr.io/chopratejas/headroom"
_CURRENT_IMAGE_REPO = "ghcr.io/headroomlabs-ai/headroom"


def _migrate_deprecated_image(image: Any) -> Any:
    """Rewrite the retired ``chopratejas`` Docker repo to the org repo (#2426)."""
    if isinstance(image, str) and image.startswith(_DEPRECATED_IMAGE_REPO):
        migrated = _CURRENT_IMAGE_REPO + image[len(_DEPRECATED_IMAGE_REPO) :]
        logger.info("Migrating deployment image from retired repo %s to %s", image, migrated)
        return migrated
    return image


def load_manifest(profile: str = "default") -> DeploymentManifest | None:
    """Load a deployment manifest when present."""

    path = manifest_path(profile)
    if not path.exists():
        return None
    # A present-but-corrupt manifest (partial write, hand-edit, schema drift)
    # must not crash callers with a raw traceback — every install lifecycle
    # command and the auto-run `init hook ensure` route through here. Raise a
    # typed error so callers can report cleanly or degrade gracefully.
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["mutations"] = [ManagedMutation(**item) for item in payload.get("mutations", [])]
        payload["artifacts"] = [ArtifactRecord(**item) for item in payload.get("artifacts", [])]
        if "image" in payload:
            payload["image"] = _migrate_deprecated_image(payload["image"])
        return DeploymentManifest(**payload)
    except (json.JSONDecodeError, ValueError, TypeError, OSError) as e:
        raise ManifestError(f"deployment profile '{profile}' is corrupt ({path}): {e}") from e


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
            if "image" in payload:
                payload["image"] = _migrate_deprecated_image(payload["image"])
            manifests.append(DeploymentManifest(**payload))
        except (OSError, ValueError, TypeError):
            continue
    return manifests


def delete_manifest(profile: str = "default") -> None:
    """Delete the full deployment profile state if present."""

    root = profile_root(profile)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
