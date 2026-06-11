"""OpenCode install-time helpers.

OpenCode reads provider base URLs from ``opencode.json`` under
``provider.<id>.options.baseURL``.  The primary provider for most users
is ``github-copilot``; Anthropic and OpenAI are also common.  We patch all
three so headroom intercepts every model the user might select.

headroom's proxy must be running with a valid ``GITHUB_TOKEN`` (or
``GITHUB_COPILOT_GITHUB_TOKEN``) so it can exchange tokens with GitHub
Copilot on behalf of OpenCode.  The wrap command exposes a note about this.
"""

from __future__ import annotations

import json
from pathlib import Path

from headroom.install.models import ConfigScope, DeploymentManifest, ManagedMutation
from headroom.install.paths import opencode_config_path

from .runtime import proxy_base_url, proxy_openai_url

# Providers whose baseURL we patch in opencode.json.
_PROVIDERS: dict[str, str] = {
    "github-copilot": "",  # Anthropic-protocol (no /v1 suffix)
    "anthropic": "",  # Anthropic-protocol
    "openai": "/v1",  # OpenAI-protocol
}


def build_install_env(*, port: int, backend: str) -> dict[str, str]:
    """Build the persistent env-var block for OpenCode.

    OpenCode's github-copilot provider does not read env vars for its base
    URL; that is controlled via opencode.json.  However, Anthropic and OpenAI
    providers respect the standard env vars, so we set them here for coverage.
    """
    del backend
    return {
        "ANTHROPIC_BASE_URL": proxy_base_url(port),
        "OPENAI_BASE_URL": proxy_openai_url(port),
    }


def apply_provider_scope(manifest: DeploymentManifest) -> ManagedMutation | None:
    """Patch ``opencode.json`` to route all providers through the proxy."""
    if manifest.scope != ConfigScope.PROVIDER.value:
        return None

    port = manifest.port
    path = opencode_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {}
    if path.exists():
        payload = json.loads(path.read_text())

    provider_section = payload.setdefault("provider", {})
    previous: dict[str, str | None] = {}

    for prov_id, suffix in _PROVIDERS.items():
        base = proxy_base_url(port) + suffix
        opts = provider_section.setdefault(prov_id, {}).setdefault("options", {})
        previous[prov_id] = opts.get("baseURL")
        opts["baseURL"] = base

    path.write_text(json.dumps(payload, indent=2) + "\n")

    return ManagedMutation(
        target="opencode",
        kind="json-provider-baseurl",
        path=str(path),
        data={"previous": previous},
    )


def revert_provider_scope(mutation: ManagedMutation, manifest: DeploymentManifest) -> None:
    """Restore the previous baseURL values (or remove them if there were none)."""
    del manifest
    if not mutation.path:
        return
    path = Path(mutation.path)
    if not path.exists():
        return

    payload = json.loads(path.read_text())
    provider_section = payload.get("provider", {})
    previous: dict[str, str | None] = mutation.data.get("previous", {})

    for prov_id in _PROVIDERS:
        if prov_id not in provider_section:
            continue
        opts = provider_section[prov_id].get("options", {})
        prev_url = previous.get(prov_id)
        if prev_url is None:
            opts.pop("baseURL", None)
        else:
            opts["baseURL"] = prev_url
        # Clean up empty dicts we may have created.
        if not opts:
            provider_section[prov_id].pop("options", None)
        if not provider_section[prov_id]:
            del provider_section[prov_id]

    path.write_text(json.dumps(payload, indent=2) + "\n")


def _patch_copilot_base_url(port: int) -> None:
    """Ensure the github-copilot provider in opencode.json points to the proxy.

    Called by ``headroom wrap opencode`` so the github-copilot backend routes
    through headroom even without a full ``headroom mcp install``.
    """
    path = opencode_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            # Don't overwrite a broken config silently.
            return

    base = proxy_base_url(port)
    opts = (
        payload.setdefault("provider", {})
        .setdefault("github-copilot", {})
        .setdefault("options", {})
    )
    if opts.get("baseURL") == base:
        return  # already patched
    opts["baseURL"] = base
    path.write_text(json.dumps(payload, indent=2) + "\n")
