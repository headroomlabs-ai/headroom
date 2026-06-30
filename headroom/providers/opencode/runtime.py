"""Runtime helpers for OpenCode integrations."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from headroom.mcp_registry.install import DEFAULT_PROXY_URL

from .config import HEADROOM_OPENCODE_PLUGIN


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by OpenCode integrations."""
    return f"http://127.0.0.1:{port}/v1"


def headroom_opencode_plugin_path() -> str | None:
    """Return the absolute path to the built OpenCode transport plugin, or None.

    OpenCode loads a plugin from an absolute file path (verified against
    opencode 1.17). The plugin's loader entry exports ONLY the plugin function
    (``plugins/opencode/dist/entry.opencode.js``) — the library barrel cannot
    be loaded directly ("Plugin export is not a function"). Returns ``None``
    when the plugin has not been built (e.g. a pip-only install that does not
    ship ``plugins/``), in which case wrap falls back to the native-provider
    baseURL override, which already covers Anthropic/OpenAI.

    ``HEADROOM_OPENCODE_PLUGIN_PATH`` overrides the resolved path.
    """
    override = os.environ.get("HEADROOM_OPENCODE_PLUGIN_PATH", "").strip()
    if override:
        return override if Path(override).is_file() else None
    # runtime.py → opencode → providers → headroom → <repo root>
    candidate = (
        Path(__file__).resolve().parents[3] / "plugins" / "opencode" / "dist" / "entry.opencode.js"
    )
    return str(candidate) if candidate.is_file() else None


def build_opencode_config_content(
    *,
    port: int,
    include_mcp: bool = True,
    include_plugin: bool = True,
) -> dict[str, object]:
    """Build JSON payload for ``OPENCODE_CONFIG_CONTENT``.

    Two complementary routing layers (both verified against opencode 1.17):

    1. **Native-provider baseURL override** — points OpenCode's built-in
       ``anthropic`` / ``openai`` providers at the proxy. Keeps native provider
       identity (model metadata, output-token limits) and reuses the user's own
       API keys (env / ``opencode auth``); the proxy forwards upstream by path
       (``/v1/messages`` → Anthropic, ``/v1/chat/completions`` → OpenAI). This
       is the reliable always-on layer and the only one shipped pip-only
       installs need.

    2. **Transparent transport plugin** — when the local plugin is built, it is
       loaded by absolute path and patches ``fetch``/``http`` to reroute *every*
       provider's traffic through the proxy, tagging the real upstream via
       ``x-headroom-base-url``. This covers providers we don't name (Gemini,
       Copilot, custom gateways) and providers added mid-session. The plugin
       self-configures from ``HEADROOM_PROXY_URL`` (set in :func:`build_launch_env`).
       Loopback URLs are not double-routed, so it coexists with layer 1.

    ponytail: config-level ``options.baseURL`` is reliable where the env-var
    override (``ANTHROPIC_BASE_URL``) is not — verified against opencode 1.17.
    """
    base_url = proxy_base_url(port)
    config: dict[str, object] = {
        "provider": {
            "anthropic": {"options": {"baseURL": base_url}},
            "openai": {"options": {"baseURL": base_url}},
            "headroom": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Headroom Proxy",
                "options": {"baseURL": base_url},
            },
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
        plugin_path = headroom_opencode_plugin_path()
        if plugin_path:
            # Plain absolute-path string; the plugin reads HEADROOM_PROXY_URL
            # from the launch env (build_launch_env sets it).
            config["plugin"] = [plugin_path]
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
    Existing provider/base URL environment variables are preserved. When the
    transport plugin is loaded, ``HEADROOM_PROXY_URL`` tells it which proxy to
    route to.
    """
    env = dict(environ or os.environ)

    config_content = build_opencode_config_content(
        port=port,
        include_mcp=include_mcp,
        include_plugin=include_plugin,
    )
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config_content, separators=(",", ":"))

    display = ["OPENCODE_CONFIG_CONTENT={provider: headroom}"]
    if "plugin" in config_content:
        env["HEADROOM_PROXY_URL"] = f"http://127.0.0.1:{port}"
        display.append(f"plugin={HEADROOM_OPENCODE_PLUGIN}")

    if project and "HEADROOM_PROJECT" not in env:
        env["HEADROOM_PROJECT"] = project

    return env, display
