from __future__ import annotations

from headroom.providers.openclaw.wrap import (
    DEFAULT_GATEWAY_PROVIDER_IDS,
    build_plugin_entry,
    build_unwrap_entry,
    decode_entry_json,
    normalize_gateway_provider_ids,
)


def test_normalize_gateway_provider_ids_trims_deduplicates_and_defaults() -> None:
    # Arrange / Act / Assert
    assert normalize_gateway_provider_ids((" openai-codex ", "anthropic", "anthropic", "")) == [
        "openai-codex",
        "anthropic",
    ]
    assert normalize_gateway_provider_ids(None) == DEFAULT_GATEWAY_PROVIDER_IDS


def test_decode_entry_json_handles_empty_valid_and_invalid_payloads() -> None:
    # Arrange / Act / Assert
    assert decode_entry_json(None) is None
    assert decode_entry_json("") is None
    assert decode_entry_json('{"enabled": true}') == {"enabled": True}
    assert decode_entry_json("{not-json}") == "{not-json}"


def test_build_plugin_entry_preserves_unmanaged_values_and_removes_empty_python_path() -> None:
    # Arrange
    existing_entry = {
        "enabled": False,
        "name": "headroom",
        "config": {
            "keep": "value",
            "proxyUrl": "https://user.example",
            "pythonPath": "/old/python",
        },
    }

    # Act
    entry = build_plugin_entry(
        existing_entry=existing_entry,
        proxy_port=8787,
        startup_timeout_ms=1500,
        python_path=None,
        no_auto_start=True,
        gateway_provider_ids=(" openai-codex ", "anthropic", "anthropic"),
        enabled=True,
    )

    # Assert
    assert entry["enabled"] is True
    assert entry["name"] == "headroom"
    assert entry["config"] == {
        "keep": "value",
        "proxyUrl": "https://user.example",
        "proxyPort": 8787,
        "autoStart": False,
        "startupTimeoutMs": 1500,
        "gatewayProviderIds": ["openai-codex", "anthropic"],
    }


def test_build_plugin_entry_creates_managed_defaults_for_non_mapping_input() -> None:
    # Arrange / Act
    entry = build_plugin_entry(
        existing_entry="not-a-dict",
        proxy_port=9000,
        startup_timeout_ms=2500,
        python_path="/usr/bin/python",
        no_auto_start=False,
        gateway_provider_ids=None,
        enabled=False,
    )

    # Assert
    assert entry == {
        "enabled": False,
        "config": {
            "proxyPort": 9000,
            "autoStart": True,
            "startupTimeoutMs": 2500,
            "gatewayProviderIds": ["openai-codex"],
            "pythonPath": "/usr/bin/python",
        },
    }


def test_build_unwrap_entry_disables_plugin_and_removes_managed_keys_only() -> None:
    # Arrange
    existing_entry = {
        "enabled": True,
        "name": "headroom",
        "config": {
            "keep": "value",
            "gatewayProviderIds": ["openai-codex"],
            "proxyUrl": "https://managed.example",
            "proxyPort": 8787,
            "autoStart": True,
            "startupTimeoutMs": 1000,
            "pythonPath": "/usr/bin/python",
        },
    }

    # Act
    entry = build_unwrap_entry(existing_entry)

    # Assert
    assert entry == {
        "enabled": False,
        "name": "headroom",
        "config": {"keep": "value"},
    }


def test_build_unwrap_entry_handles_non_mapping_input() -> None:
    # Arrange / Act / Assert
    assert build_unwrap_entry("not-a-dict") == {"enabled": False, "config": {}}


def test_build_plugin_entry_strips_mcpServers_from_existing_entry() -> None:
    """Newer OpenClaw schemas reject `mcpServers` at the plugin-entry root.

    `headroom init -g` was failing with `Config invalid: Unrecognized
    key: "mcpServers"` because the prior plugin entry in the user's
    config still had that legacy field, and we were spreading it back in
    via `**existing_entry`. Pin the strip so we don't regress.
    """
    existing_entry = {
        "enabled": True,
        "name": "headroom",
        "mcpServers": {"some": "stale-block"},  # legacy, must be removed
        "config": {"keep": "value"},
    }

    entry = build_plugin_entry(
        existing_entry=existing_entry,
        proxy_port=8787,
        startup_timeout_ms=1500,
        python_path=None,
        no_auto_start=False,
        gateway_provider_ids=None,
        enabled=True,
    )

    assert "mcpServers" not in entry
    assert entry["enabled"] is True
    assert entry["name"] == "headroom"
    assert entry["config"]["keep"] == "value"
