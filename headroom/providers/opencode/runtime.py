"""Runtime helpers for OpenCode integrations."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

from headroom.mcp_registry.install import DEFAULT_PROXY_URL

from .config import HEADROOM_OPENCODE_PLUGIN


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by OpenCode integrations."""
    return f"http://127.0.0.1:{port}/v1"


def build_opencode_config_content(
    *,
    port: int,
    include_mcp: bool = True,
    include_plugin: bool = True,
) -> dict[str, object]:
    """Build JSON payload for ``OPENCODE_CONFIG_CONTENT``.

    Runtime wrap injects the Headroom provider as a stable explicit fallback,
    plus the Headroom plugin which transparently routes provider fetch traffic
    through the local proxy without rewriting user provider config URLs.
    """
    base_url = proxy_base_url(port)
    config: dict[str, object] = {
        "provider": {
            "headroom": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Headroom Proxy",
                "options": {"baseURL": base_url},
            }
        }
    }
    if include_mcp:
        proxy_url = f"http://127.0.0.1:{port}"
        mcp_entry: dict[str, object] = {
            "type": "local",
            "command": ["headroom", "mcp", "serve"],
            "enabled": True,
        }
        if proxy_url != DEFAULT_PROXY_URL:
            mcp_entry["environment"] = {"HEADROOM_PROXY_URL": proxy_url}
        config["mcp"] = {
            "headroom": mcp_entry,
        }
    if include_plugin:
        config["plugin"] = [[HEADROOM_OPENCODE_PLUGIN, {"proxyUrl": base_url}]]
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

    ``OPENCODE_CONFIG_CONTENT`` carries Headroom provider/MCP/plugin config.
    Existing provider/base URL environment variables are preserved.
    """
    env = dict(environ or os.environ)

    config_content = build_opencode_config_content(
        port=port,
        include_mcp=include_mcp,
        include_plugin=include_plugin,
    )
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config_content, separators=(",", ":"))

    display = ["OPENCODE_CONFIG_CONTENT={provider: headroom}"]
    if include_plugin:
        display.append(f"plugin={HEADROOM_OPENCODE_PLUGIN}")

    if project and "HEADROOM_PROJECT" not in env:
        env["HEADROOM_PROJECT"] = project

    return env, display
