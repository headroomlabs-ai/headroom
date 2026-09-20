from __future__ import annotations

from headroom.providers.grok import (
    PROXY_ENV_KEY,
    build_launch_env,
    is_grok_cli_request,
    proxy_base_url,
)
from headroom.providers.grok.install import build_install_env


def test_grok_proxy_base_url_uses_local_headroom_proxy() -> None:
    assert proxy_base_url(8787) == "http://127.0.0.1:8787/v1"


def test_is_grok_cli_request_matches_xai_token_auth() -> None:
    # Official Grok CLI stamps x-xai-token-auth: xai-grok-cli on inference.
    assert is_grok_cli_request({"x-xai-token-auth": "xai-grok-cli"}) is True
    assert is_grok_cli_request({"X-Xai-Token-Auth": "xai-grok-cli"}) is True
    assert is_grok_cli_request({"x-xai-token-auth": "other"}) is False


def test_is_grok_cli_request_matches_known_user_agents() -> None:
    assert (
        is_grok_cli_request(
            {"user-agent": "grok-pager/0.2.117 grok-shell/0.2.117 (macos; aarch64)"}
        )
        is True
    )
    assert is_grok_cli_request({"user-agent": "grok-shell/0.2.112"}) is True
    # Non-Grok OpenAI-compatible clients must not match, including wrappers
    # whose name merely contains "grok" — they carry OpenAI credentials.
    assert is_grok_cli_request({"user-agent": "codex-tui/0.146.0"}) is False
    assert is_grok_cli_request({"user-agent": "litellm-grok/1.0"}) is False
    assert is_grok_cli_request({"user-agent": "my-grok-shell/1.0"}) is False
    assert is_grok_cli_request({"user-agent": "grok/0.1.0"}) is False
    assert is_grok_cli_request({}) is False


def test_grok_build_launch_env_sets_models_base_url() -> None:
    env, display = build_launch_env(9999, environ={})

    assert env[PROXY_ENV_KEY] == "http://127.0.0.1:9999/v1"
    assert "GROK_CLI_CHAT_PROXY_BASE_URL" not in env
    assert display == [f"{PROXY_ENV_KEY}=http://127.0.0.1:9999/v1"]


def test_grok_build_launch_env_applies_project_prefix() -> None:
    env, _display = build_launch_env(8787, environ={}, project="frontend")

    assert env[PROXY_ENV_KEY] == "http://127.0.0.1:8787/p/frontend/v1"
    assert "GROK_CLI_CHAT_PROXY_BASE_URL" not in env


def test_grok_build_install_env_returns_proxy_url() -> None:
    assert build_install_env(port=7654, backend="ignored") == {
        PROXY_ENV_KEY: "http://127.0.0.1:7654/v1",
    }
