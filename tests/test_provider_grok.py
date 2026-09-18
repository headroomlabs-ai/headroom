from __future__ import annotations

from headroom.providers.grok import (
    PROXY_ENV_KEY,
    SESSION_API_URL,
    build_launch_env,
    proxy_base_url,
    session_upstream,
)
from headroom.providers.grok.install import build_install_env


def test_grok_proxy_base_url_uses_local_headroom_proxy() -> None:
    assert proxy_base_url(8787) == "http://127.0.0.1:8787/v1"


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


_SESSION_JWT = "Bearer eyJ0eXAiOiJhdCtqd3QiLCJhbGciOiJFUzI1NiJ9.x.y"


def test_session_upstream_routes_grok_login_token_to_session_host() -> None:
    # `grok login` mode: auth.x.ai session token, marker header, grok-shell UA.
    headers = {
        "Authorization": _SESSION_JWT,
        "X-Xai-Token-Auth": "xai-grok-cli",
        "User-Agent": "grok-shell/0.2.112 (macos; aarch64)",
    }
    assert session_upstream(headers) == SESSION_API_URL
    # Either signal alone is enough; header lookup is case-insensitive.
    assert session_upstream({"authorization": _SESSION_JWT, "x-xai-token-auth": "xai-grok-cli"})
    assert session_upstream({"authorization": _SESSION_JWT, "user-agent": "grok/0.1.0"})


def test_session_upstream_keeps_api_keys_on_configured_target() -> None:
    headers = {"authorization": "Bearer xai-abc123", "user-agent": "grok-shell/0.2.112"}
    assert session_upstream(headers) is None


def test_session_upstream_ignores_other_clients() -> None:
    assert session_upstream({"authorization": _SESSION_JWT, "user-agent": "codex-cli/1.0"}) is None
    assert session_upstream({"authorization": _SESSION_JWT}) is None
    # Wrapper UAs that merely contain "grok" never match.
    assert (
        session_upstream({"authorization": _SESSION_JWT, "user-agent": "litellm-grok/1.0"}) is None
    )
