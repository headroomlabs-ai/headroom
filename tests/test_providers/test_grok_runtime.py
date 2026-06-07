"""Tests for Grok provider runtime helpers."""

from headroom.providers.grok.runtime import (
    DEFAULT_API_URL,
    build_launch_env,
    proxy_base_url,
)


def test_proxy_base_url() -> None:
    assert proxy_base_url(8787) == "http://127.0.0.1:8787/v1"


def test_default_api_url() -> None:
    assert DEFAULT_API_URL == "https://cli-chat-proxy.grok.com/v1"


def test_build_launch_env_sets_grok_proxy_override() -> None:
    env, display = build_launch_env(9999, {"HOME": "/tmp"})
    assert env["GROK_CLI_CHAT_PROXY_BASE_URL"] == "http://127.0.0.1:9999/v1"
    assert env["HOME"] == "/tmp"
    assert display == ["GROK_CLI_CHAT_PROXY_BASE_URL=http://127.0.0.1:9999/v1"]


def test_build_launch_env_respects_empty_environ() -> None:
    env, display = build_launch_env(8787, {})

    assert env == {"GROK_CLI_CHAT_PROXY_BASE_URL": "http://127.0.0.1:8787/v1"}
    assert display == ["GROK_CLI_CHAT_PROXY_BASE_URL=http://127.0.0.1:8787/v1"]
