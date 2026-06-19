from __future__ import annotations

import json

from headroom.providers.opencode import build_launch_env


def test_opencode_build_launch_env_sets_config_content_without_mutating_input() -> None:
    # Arrange
    source_env = {"EXISTING": "value"}

    # Act
    env, lines = build_launch_env(8787, environ=source_env)

    # Assert
    assert source_env == {"EXISTING": "value"}
    assert env["EXISTING"] == "value"
    assert "OPENCODE_CONFIG_CONTENT" in env
    assert lines == [f"OPENCODE_CONFIG_CONTENT={env['OPENCODE_CONFIG_CONTENT']}"]


def test_opencode_base_urls_end_with_v1_for_both_providers() -> None:
    # Act
    env, _lines = build_launch_env(8787, environ={})
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])

    # Assert
    openai_base = config["provider"]["openai"]["options"]["baseURL"]
    anthropic_base = config["provider"]["anthropic"]["options"]["baseURL"]
    assert openai_base.endswith("/v1")
    assert anthropic_base.endswith("/v1")
    assert openai_base == "http://127.0.0.1:8787/v1"
    assert anthropic_base == "http://127.0.0.1:8787/v1"


def test_opencode_config_pins_autoupdate_false() -> None:
    # Act
    env, _lines = build_launch_env(8787, environ={})
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])

    # Assert
    assert config["autoupdate"] is False


def test_opencode_project_sets_sanitized_header_under_both_providers() -> None:
    # Act
    env, _lines = build_launch_env(8787, environ={}, project="My Proj")
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])

    # Assert
    for provider_id in ("openai", "anthropic"):
        options = config["provider"][provider_id]["options"]
        assert options["headers"]["X-Headroom-Project"] == "My Proj"


def test_opencode_no_project_sets_no_headers_key() -> None:
    # Act
    env, _lines = build_launch_env(8787, environ={}, project=None)
    config = json.loads(env["OPENCODE_CONFIG_CONTENT"])

    # Assert
    for provider_id in ("openai", "anthropic"):
        options = config["provider"][provider_id]["options"]
        assert "headers" not in options
