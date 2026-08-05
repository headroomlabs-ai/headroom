"""Claude install-time helpers."""

from __future__ import annotations

import json
from pathlib import Path

from headroom.install.models import ConfigScope, DeploymentManifest, ManagedMutation, ToolTarget
from headroom.install.paths import claude_settings_path

from .runtime import TOOL_SEARCH_DEFAULT, TOOL_SEARCH_ENV, proxy_base_url


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent install environment for Claude."""
    del backend
    # TOOL_SEARCH_ENV keeps Claude Code deferring MCP/system tool schemas behind
    # the server-side Tool Search Tool when pointed at the proxy's custom
    # ANTHROPIC_BASE_URL; without it Claude Code materializes every schema into
    # its context window (GH #746) — breaking sub-agents and forcing compaction.
    # The install env is headroom-managed and reverted on uninstall, so it is
    # authoritative — unlike `init`, it always writes the default rather than
    # deferring to a pre-existing user value.
    return {
        "ANTHROPIC_BASE_URL": proxy_base_url(port),
        TOOL_SEARCH_ENV: TOOL_SEARCH_DEFAULT,
    }


def apply_provider_scope(manifest: DeploymentManifest) -> ManagedMutation | None:
    """Apply Claude provider-scope configuration when requested."""
    if manifest.scope != ConfigScope.PROVIDER.value:
        return None

    path = claude_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    env = payload.get("env")
    env_map = dict(env) if isinstance(env, dict) else {}
    values = manifest.tool_envs.get(ToolTarget.CLAUDE.value, {})
    previous = {name: env_map.get(name) for name in values}
    env_map.update(values)
    payload["env"] = env_map
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return ManagedMutation(
        target=ToolTarget.CLAUDE.value,
        kind="json-env",
        path=str(path),
        data={"previous": previous},
    )


def revert_provider_scope(mutation: ManagedMutation, manifest: DeploymentManifest) -> None:
    """Revert Claude provider-scope configuration."""
    if not mutation.path:
        return
    path = Path(mutation.path)
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    env = payload.get("env")
    env_map = dict(env) if isinstance(env, dict) else {}
    previous: dict[str, object] = mutation.data.get("previous", {})
    values = manifest.tool_envs.get(ToolTarget.CLAUDE.value, {})
    for name in values:
        if previous.get(name) is None:
            env_map.pop(name, None)
        else:
            env_map[name] = previous[name]
    payload["env"] = env_map
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
