"""OpenCode install-time helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from headroom import fsutil
from headroom.install.models import ConfigScope, DeploymentManifest, ManagedMutation, ToolTarget
from headroom.install.paths import opencode_config_path

from .config import (
    _inject_key_into_json,
    _parse_json_loose,
    snapshot_opencode_config_if_unwrapped,
    strip_opencode_headroom_blocks,
)
from .runtime import proxy_base_url


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for OpenCode."""
    del backend
    del port
    return {}


def apply_provider_scope(manifest: DeploymentManifest) -> ManagedMutation | None:
    """Apply OpenCode provider-scope configuration when requested."""
    if manifest.scope != ConfigScope.PROVIDER.value:
        return None

    config_file = opencode_config_path()
    config_file.parent.mkdir(parents=True, exist_ok=True)

    snapshot_opencode_config_if_unwrapped(
        config_file, config_file.with_suffix(".json.headroom-backup")
    )

    if config_file.exists():
        content = fsutil.read_text(config_file)
        data = _parse_json_loose(content)
    else:
        data = {}

    provider = {
        "headroom": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "Headroom Proxy",
            "options": {"baseURL": proxy_base_url(manifest.port)},
        }
    }
    data = _inject_key_into_json(data, "provider", provider)

    config_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return ManagedMutation(
        target=ToolTarget.OPENCODE.value,
        kind="json-block",
        path=str(config_file),
    )


def revert_provider_scope(mutation: ManagedMutation, manifest: DeploymentManifest) -> None:
    """Revert OpenCode provider-scope configuration.

    Restores from pre-wrap backup when available, otherwise strips the
    headroom provider from the config file.
    """
    del manifest
    if not mutation.path:
        return
    path = Path(mutation.path)
    backup_file = path.with_suffix(".json.headroom-backup")
    if backup_file.exists():
        try:
            shutil.copy2(backup_file, path)
            backup_file.unlink()
            return
        except OSError:
            pass
    if not path.exists():
        return
    content = fsutil.read_text(path)
    cleaned = strip_opencode_headroom_blocks(content)
    if cleaned:
        path.write_text(cleaned + "\n", encoding="utf-8")
    else:
        path.unlink(missing_ok=True)
