"""Runtime helpers for ZCode (zcode.z.ai desktop app) integrations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from headroom.install.paths import zcode_config_dir
from headroom.providers.claude import proxy_base_url as _claude_proxy_base_url

_log = logging.getLogger(__name__)

ZAI_ANTHROPIC_DEFAULT = "https://api.z.ai/api/anthropic"


@dataclass(frozen=True)
class ZCodeProxyTargets:
    """Resolved local proxy targets shown in ZCode setup instructions."""

    openai_base_url: str
    anthropic_base_url: str


@dataclass(frozen=True)
class ZCodeUpstream:
    """Detected upstream provider from ZCode config."""

    provider_name: str
    base_url: str
    kind: str  # "anthropic" or "openai-compatible"


def build_proxy_targets(port: int) -> ZCodeProxyTargets:
    """Build the local proxy URLs shown to ZCode users."""
    return ZCodeProxyTargets(
        openai_base_url=f"http://127.0.0.1:{port}/v1",
        anthropic_base_url=_claude_proxy_base_url(port),
    )


def detect_upstream(config_path: Path | None = None) -> ZCodeUpstream:
    """Read ZCode config.json and find the enabled provider's upstream URL.

    Resolution order:
    1. First provider with ``enabled=True`` and a non-empty ``baseURL``.
    2. Fallback: Z.ai Anthropic endpoint.

    Parameters
    ----------
    config_path:
        Override for the config file path.  When *None* the default
        ``~/.zcode/v2/config.json`` is used.
    """
    if config_path is None:
        config_path = zcode_config_dir() / "v2" / "config.json"

    try:
        raw = config_path.read_text(encoding="utf-8")
        cfg = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        _log.debug("Cannot read ZCode config %s: %s", config_path, exc)
        return ZCodeUpstream(
            provider_name="Z.ai (default)",
            base_url=ZAI_ANTHROPIC_DEFAULT,
            kind="anthropic",
        )

    providers = cfg.get("provider")
    if not isinstance(providers, dict):
        return ZCodeUpstream(
            provider_name="Z.ai (default)",
            base_url=ZAI_ANTHROPIC_DEFAULT,
            kind="anthropic",
        )

    for _key, provider in providers.items():
        if not isinstance(provider, dict):
            continue
        if not provider.get("enabled"):
            continue
        opts = provider.get("options", {})
        if not isinstance(opts, dict):
            continue
        base_url = opts.get("baseURL", "").strip()
        if not base_url:
            continue
        kind = provider.get("kind", "anthropic")
        name = provider.get("name", "ZCode provider")
        return ZCodeUpstream(provider_name=name, base_url=base_url, kind=kind)

    return ZCodeUpstream(
        provider_name="Z.ai (default)",
        base_url=ZAI_ANTHROPIC_DEFAULT,
        kind="anthropic",
    )


def upstream_to_proxy_urls(upstream: ZCodeUpstream) -> tuple[str | None, str | None]:
    """Map :class:`ZCodeUpstream` to headroom proxy URL params.

    Returns ``(anthropic_api_url, openai_api_url)`` — exactly one is set.
    """
    if upstream.kind == "anthropic":
        return upstream.base_url, None
    return None, upstream.base_url


def render_setup_lines(port: int) -> list[str]:
    """Render the ZCode setup instructions for the local proxy."""
    targets = build_proxy_targets(port)
    return [
        "  Headroom proxy is running. Configure ZCode:",
        "",
        "  Open ZCode > Settings > Model Settings:",
        "",
        f"    OpenAI Base URL:      {targets.openai_base_url}",
        f"    Anthropic Base URL:   {targets.anthropic_base_url}",
        "",
        "  Select a model through the new provider in ZCode's model selector.",
        "",
        "  To add the Headroom MCP server (optional):",
        "    Settings > MCP Servers > New MCP Server > Full configuration",
        '    Paste: {"headroom": {"type": "stdio", "command": "headroom",',
        '             "args": ["mcp", "serve"], "enabled": true}}',
    ]
