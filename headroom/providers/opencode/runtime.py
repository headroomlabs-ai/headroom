"""Runtime helpers for OpenCode integrations."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL used by OpenCode-compatible integrations."""
    return f"http://127.0.0.1:{port}/v1"


def build_opencode_config_content(
    *,
    port: int,
    include_mcp: bool = True,
) -> dict[str, object]:
    """Build the JSON payload for ``OPENCODE_CONFIG_CONTENT``.

    Registers a transparent headroom provider via ``@ai-sdk/openai-compatible``
    without restricting which models are available.  The user's existing
    model selection is left untouched.
    """
    config: dict[str, object] = {
        "provider": {
            "headroom": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "Headroom Proxy",
                "options": {"baseURL": proxy_base_url(port)},
            }
        }
    }
    if include_mcp:
        config["mcp"] = {
            "headroom": {
                "type": "remote",
                "url": f"http://127.0.0.1:{port}/mcp",
                "enabled": True,
            }
        }
    return config


def build_launch_env(
    port: int,
    environ: Mapping[str, str] | None = None,
    project: str | None = None,
    *,
    include_mcp: bool = True,
) -> tuple[dict[str, str], list[str]]:
    """Build environment variables for OpenCode through the local proxy.

    Sets ``OPENCODE_CONFIG_CONTENT`` with the headroom provider definition.
    Also sets ``OPENAI_BASE_URL`` and ``ANTHROPIC_BASE_URL`` as fallbacks.
    """
    env = dict(environ or os.environ)
    base_url = proxy_base_url(port)

    config_content = build_opencode_config_content(
        port=port,
        include_mcp=include_mcp,
    )
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(config_content, separators=(",", ":"))

    # Fallback env vars for OpenCode versions that respect them.
    env["OPENAI_BASE_URL"] = base_url
    env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{port}"

    env_vars_display = [
        "OPENCODE_CONFIG_CONTENT={provider: headroom}",
        f"OPENAI_BASE_URL={base_url}",
        f"ANTHROPIC_BASE_URL=http://127.0.0.1:{port}",
    ]

    # Per-project savings attribution (same pattern as codex).
    if project and "HEADROOM_PROJECT" not in env:
        env["HEADROOM_PROJECT"] = project

    return env, env_vars_display
