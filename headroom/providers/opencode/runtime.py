"""Runtime helpers for OpenCode (sst/opencode) integrations.

OpenCode is configured through a JSON config file (``opencode.json``) rather
than environment variables — it does **not** read ``OPENAI_BASE_URL`` /
``OPENAI_API_BASE`` for built-in providers.  To route traffic through the
local Headroom proxy we override the ``baseURL`` of the built-in ``anthropic``
and ``openai`` providers, which keeps the user's chosen model IDs
(``anthropic/claude-…``, ``openai/gpt-…``) working while sending every request
through Headroom first.

The helpers here are pure functions over plain ``dict`` config objects so they
can be unit-tested without touching the filesystem.  The file I/O, pre-wrap
snapshot, and CLI wiring live in :mod:`headroom.cli.wrap`.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from headroom.proxy.project_context import with_project_prefix

# OpenCode's built-in providers whose baseURL Headroom overrides.  Both the
# AI-SDK ``anthropic`` and ``openai`` providers append their own path suffix
# (``/messages`` and ``/chat/completions`` respectively) to ``baseURL``, so the
# same ``…/v1`` proxy base works for both.
MANAGED_PROVIDERS: tuple[str, ...] = ("anthropic", "openai")

CONFIG_SCHEMA_URL = "https://opencode.ai/config.json"

# Recognises a baseURL that points at a local Headroom proxy (optionally with a
# ``/p/<project>`` per-project savings prefix).  Used to identify Headroom-owned
# overrides when no pre-wrap backup is available (crash-recovery path), so we
# never strip a baseURL the user configured themselves.
_HEADROOM_BASE_URL_RE = re.compile(r"^http://127\.0\.0\.1:\d+(?:/p/[^/]+)?/v1$")


def proxy_base_url(port: int) -> str:
    """Return the local proxy base URL OpenCode providers should target."""
    return f"http://127.0.0.1:{port}/v1"


def build_provider_overrides(port: int, project: str | None = None) -> dict[str, Any]:
    """Build the ``provider`` overrides that route OpenCode through the proxy.

    ``project`` (the wrap launch directory) is encoded as a ``/p/<name>``
    base-URL prefix because OpenCode reads a static config file rather than
    sending custom headers; the proxy strips the prefix and attributes savings
    per project.
    """
    base_url = with_project_prefix(proxy_base_url(port), project)
    return {name: {"options": {"baseURL": base_url}} for name in MANAGED_PROVIDERS}


def is_headroom_base_url(value: Any) -> bool:
    """Return True when *value* is a local Headroom proxy baseURL."""
    return isinstance(value, str) and bool(_HEADROOM_BASE_URL_RE.match(value))


def config_has_headroom_overrides(config: dict[str, Any]) -> bool:
    """Return True when the config already routes a provider through Headroom."""
    providers = config.get("provider")
    if not isinstance(providers, dict):
        return False
    for name in MANAGED_PROVIDERS:
        provider = providers.get(name)
        if not isinstance(provider, dict):
            continue
        options = provider.get("options")
        if isinstance(options, dict) and is_headroom_base_url(options.get("baseURL")):
            return True
    return False


def apply_provider_overrides(
    config: dict[str, Any], port: int, project: str | None = None
) -> dict[str, Any]:
    """Return a copy of *config* with the Headroom provider overrides applied.

    Existing provider settings (``apiKey``, ``models``, …) are preserved — only
    each managed provider's ``options.baseURL`` is set to the proxy URL.  Any
    prior Headroom override is replaced first via :func:`strip_provider_overrides`
    so re-wrapping with a different port stays idempotent.
    """
    result = strip_provider_overrides(config)
    overrides = build_provider_overrides(port, project)
    providers = result.setdefault("provider", {})
    if not isinstance(providers, dict):  # defensive: user set provider to a scalar
        providers = {}
        result["provider"] = providers
    for name, override in overrides.items():
        provider = providers.setdefault(name, {})
        if not isinstance(provider, dict):
            provider = {}
            providers[name] = provider
        options = provider.setdefault("options", {})
        if not isinstance(options, dict):
            options = {}
            provider["options"] = options
        options["baseURL"] = override["options"]["baseURL"]
    return result


def strip_provider_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *config* with Headroom provider baseURL overrides removed.

    Only baseURLs recognised as a local Headroom proxy (see
    :func:`is_headroom_base_url`) are removed, so a baseURL the user set
    themselves is left untouched.  Empty ``options`` / provider / ``provider``
    containers left behind by the removal are pruned so the config returns to
    its pre-override shape.
    """
    result = copy.deepcopy(config)
    providers = result.get("provider")
    if not isinstance(providers, dict):
        return result
    for name in MANAGED_PROVIDERS:
        provider = providers.get(name)
        if not isinstance(provider, dict):
            continue
        options = provider.get("options")
        if isinstance(options, dict) and is_headroom_base_url(options.get("baseURL")):
            options.pop("baseURL", None)
            if not options:
                provider.pop("options", None)
            if not provider:
                providers.pop(name, None)
    if not providers:
        result.pop("provider", None)
    return result


def _is_headroom_retrieve_entry(entry: Any) -> bool:
    """Return True for the ``mcp.headroom`` retrieve server Headroom registers."""
    if not isinstance(entry, dict):
        return False
    command = entry.get("command")
    if isinstance(command, list) and command:
        return str(command[0]) == "headroom"
    return False


def strip_managed_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *config* with all Headroom-managed entries removed.

    Strips the provider ``baseURL`` overrides (see
    :func:`strip_provider_overrides`) and the ``mcp.headroom`` retrieve server
    Headroom registers. The Serena MCP entry is tracked in the install ledger
    and removed by the CLI unwrap path, not here.
    """
    result = strip_provider_overrides(config)
    mcp = result.get("mcp")
    if isinstance(mcp, dict):
        if _is_headroom_retrieve_entry(mcp.get("headroom")):
            mcp.pop("headroom", None)
        if not mcp:
            result.pop("mcp", None)
    return result
