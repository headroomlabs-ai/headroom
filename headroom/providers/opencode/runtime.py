"""Runtime helpers for OpenCode integrations."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from .config import HEADROOM_OPENCODE_PLUGIN


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by OpenCode integrations."""
    return f"http://127.0.0.1:{port}/v1"


def proxy_server_url(port: int) -> str:
    """Return the local Headroom proxy origin used by the OpenCode plugin."""
    return f"http://127.0.0.1:{port}"


_opencode_plugin_spec_override: str | None = None


def _resolve_opencode_plugin_spec() -> str:
    """Resolve a plugin spec that OpenCode can load.
    """
    if _opencode_plugin_spec_override is not None:
        return _opencode_plugin_spec_override
    candidates = (
        Path(__file__).resolve().parents[2] / "plugins" / "opencode",
        Path(__file__).resolve().parents[3] / "plugins" / "opencode",
    )
    for candidate in candidates:
        manifest = candidate / "package.json"
        dist_entry = candidate / "dist" / "index.js"
        if manifest.is_file() and dist_entry.is_file():
            return candidate.as_uri()
    return HEADROOM_OPENCODE_PLUGIN


def build_opencode_config_content(
    *,
    port: int,
    include_mcp: bool = True,
    include_plugin: bool = True,
) -> dict[str, object]:
    """Build JSON payload for ``OPENCODE_CONFIG_CONTENT``.

    Runtime wrap keeps OpenCode's provider/model selection intact and injects
    only the Headroom plugin. The plugin sits under the user's existing
    provider configuration and transparently routes outbound traffic through the
    local proxy.

    ``include_mcp`` is currently a no-op here. Runtime MCP wiring is handled by
    the OpenCode MCP registrar so we do not inject an invalid remote ``/mcp``
    entry into ``OPENCODE_CONFIG_CONTENT``.
    """
    del include_mcp
    config: dict[str, object] = {}
    if include_plugin:
        plugin_spec = _resolve_opencode_plugin_spec()
        config["plugin"] = [[
            plugin_spec,
            {
                "mode": "native-fetch",
                "proxyUrl": proxy_server_url(port),
            },
        ]]
    return config


def build_launch_env(
    port: int,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
    *,
    include_mcp: bool = True,
    include_plugin: bool = True,
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables for launching OpenCode through Headroom.

    ``OPENCODE_CONFIG_CONTENT`` carries only the Headroom plugin bootstrap.
    Existing provider/base URL environment variables are preserved.
    """
    env = dict(environ or os.environ)

    config_content = build_opencode_config_content(
        port=port,
        include_mcp=include_mcp,
        include_plugin=include_plugin,
    )
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config_content, separators=(",", ":"))

    display = ["OPENCODE_CONFIG_CONTENT={plugin: headroom-opencode}"]
    if include_plugin:
        display.append(f"plugin={HEADROOM_OPENCODE_PLUGIN}")

    if project and "HEADROOM_PROJECT" not in env:
        env["HEADROOM_PROJECT"] = project

    return env, display
