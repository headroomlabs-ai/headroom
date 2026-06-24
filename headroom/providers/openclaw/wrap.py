"""OpenClaw wrapper provider helpers."""

from __future__ import annotations

import json
from typing import Any

DEFAULT_GATEWAY_PROVIDER_IDS = ["openai-codex"]


def normalize_gateway_provider_ids(provider_ids: tuple[str, ...] | None) -> list[str]:
    """Normalize configured OpenClaw provider ids."""
    values = provider_ids or ()
    seen: set[str] = set()
    normalized: list[str] = []

    for entry in values:
        provider_id = entry.strip()
        if not provider_id or provider_id in seen:
            continue
        seen.add(provider_id)
        normalized.append(provider_id)

    return normalized or DEFAULT_GATEWAY_PROVIDER_IDS.copy()


def decode_entry_json(raw_value: str | None) -> Any | None:
    """Decode a JSON payload captured from `openclaw config get` when available."""
    if not raw_value:
        return None

    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


# Keys we know newer openclaw plugin schemas reject when echoed back.
# We strip them defensively from `existing_entry` so a stale entry left
# over from an older Headroom or older OpenClaw install doesn't cause
# `openclaw config set` to fail with "Unrecognized key". The list is
# narrow on purpose — anything else is assumed user-managed and
# preserved verbatim.
_LEGACY_REJECTED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"mcpServers"})


def build_plugin_entry(
    *,
    existing_entry: Any,
    proxy_port: int,
    startup_timeout_ms: int,
    python_path: str | None,
    no_auto_start: bool,
    gateway_provider_ids: tuple[str, ...] | None,
    enabled: bool,
) -> dict[str, object]:
    """Merge managed Headroom plugin settings with any existing entry payload."""
    raw_base = existing_entry if isinstance(existing_entry, dict) else {}
    base_entry = {k: v for k, v in raw_base.items() if k not in _LEGACY_REJECTED_TOP_LEVEL_KEYS}
    existing_config = base_entry.get("config")
    next_config = dict(existing_config) if isinstance(existing_config, dict) else {}

    next_config["proxyPort"] = proxy_port
    next_config["autoStart"] = not no_auto_start
    next_config["startupTimeoutMs"] = startup_timeout_ms
    next_config["gatewayProviderIds"] = normalize_gateway_provider_ids(gateway_provider_ids)

    if python_path:
        next_config["pythonPath"] = python_path
    else:
        next_config.pop("pythonPath", None)

    return {
        **base_entry,
        "enabled": enabled,
        "config": next_config,
    }


def build_unwrap_entry(existing_entry: Any) -> dict[str, object]:
    """Disable the managed plugin while preserving unrelated user config."""
    base_entry = existing_entry if isinstance(existing_entry, dict) else {}
    existing_config: dict[str, object] = {}
    if isinstance(existing_entry, dict) and isinstance(existing_entry.get("config"), dict):
        existing_config = {
            key: value
            for key, value in existing_entry["config"].items()
            if key
            not in {
                "gatewayProviderIds",
                "proxyUrl",
                "proxyPort",
                "autoStart",
                "startupTimeoutMs",
                "pythonPath",
            }
        }

    return {**base_entry, "enabled": False, "config": existing_config}
