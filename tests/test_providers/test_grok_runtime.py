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


def test_build_launch_env_overrides_existing_grok_proxy_url() -> None:
    """Wrap must always point Grok at the local Headroom proxy, not a stale override."""
    env, display = build_launch_env(
        4242,
        {
            "GROK_CLI_CHAT_PROXY_BASE_URL": "https://cli-chat-proxy.grok.com/v1",
            "HOME": "/home/dev",
        },
    )

    assert env["GROK_CLI_CHAT_PROXY_BASE_URL"] == "http://127.0.0.1:4242/v1"
    assert env["HOME"] == "/home/dev"
    assert display == ["GROK_CLI_CHAT_PROXY_BASE_URL=http://127.0.0.1:4242/v1"]


def test_build_launch_env_preserves_unrelated_env_vars() -> None:
    env, _ = build_launch_env(
        8787,
        {
            "PATH": "/usr/bin",
            "GROK_COMPACTION_MODE": "segments",
            "GROK_COMPACTION_DETAIL": "verbose",
        },
    )

    assert env["PATH"] == "/usr/bin"
    assert env["GROK_COMPACTION_MODE"] == "segments"
    assert env["GROK_COMPACTION_DETAIL"] == "verbose"
    assert env["GROK_CLI_CHAT_PROXY_BASE_URL"] == "http://127.0.0.1:8787/v1"
