from __future__ import annotations

import pytest

from headroom.providers.grok import (
    PROXY_ENV_KEY,
    SESSION_API_URL,
    SESSION_ROUTING_ENV_KEY,
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
# What `headroom wrap grok` and the install planner configure as the OpenAI target.
_XAI_TARGET = "https://api.x.ai"


def test_session_upstream_routes_grok_login_token_to_session_host() -> None:
    # `grok login` mode: auth.x.ai session token, marker header, grok-shell UA.
    headers = {
        "Authorization": _SESSION_JWT,
        "X-Xai-Token-Auth": "xai-grok-cli",
        "User-Agent": "grok-shell/0.2.112 (macos; aarch64)",
    }
    assert session_upstream(headers, _XAI_TARGET) == SESSION_API_URL
    # Either strong signal alone is enough; header lookup is case-insensitive.
    assert session_upstream(
        {"authorization": _SESSION_JWT, "x-xai-token-auth": "xai-grok-cli"}, _XAI_TARGET
    )
    assert session_upstream(
        {"authorization": _SESSION_JWT, "user-agent": "grok-shell/0.2.112"}, _XAI_TARGET
    )


# Re-pointing a request sends somebody's Authorization header to a public
# third-party host, so every signal that decides it is client-controlled and
# must be treated as hostile. These are the shapes an install could see when
# Headroom fronts an internal gateway (Kong, LiteLLM) rather than xAI.
@pytest.mark.parametrize(
    ("headers", "why"),
    [
        (
            {"authorization": "Bearer sk-proj-REALKEY", "user-agent": "grok/1.0"},
            "bare grok/ UA is claimed elsewhere in auth_policy and is not a session signal",
        ),
        (
            {"authorization": "Bearer sk-litellm-abc", "x-xai-token-auth": "xai-grok-cli"},
            "marker header cannot launder a gateway key",
        ),
        ({"user-agent": "grok/1.0"}, "no credential at all"),
        (
            {"authorization": "Bearer ghu_abcdef", "x-xai-token-auth": "xai-grok-cli"},
            "GitHub token is not JWT-shaped",
        ),
        (
            {"authorization": "Bearer eyJabc", "user-agent": "grok-shell/0.2.112"},
            "eyJ prefix without three segments is not a JWT",
        ),
        (
            {"authorization": "Bearer eyJa..y", "user-agent": "grok-shell/0.2.112"},
            "empty JWT segment",
        ),
    ],
)
def test_session_upstream_never_redirects_a_foreign_credential(headers, why) -> None:
    assert session_upstream(headers, _XAI_TARGET) is None, why


def test_session_routing_kill_switch(monkeypatch) -> None:
    """An operator must be able to refuse this entirely, whatever a client claims."""
    headers = {"authorization": _SESSION_JWT, "user-agent": "grok-shell/0.2.112"}
    assert session_upstream(headers, _XAI_TARGET) == SESSION_API_URL
    monkeypatch.setenv(SESSION_ROUTING_ENV_KEY, "0")
    assert session_upstream(headers, _XAI_TARGET) is None
    monkeypatch.setenv(SESSION_ROUTING_ENV_KEY, "1")
    assert session_upstream(headers, _XAI_TARGET) == SESSION_API_URL


def test_session_upstream_keeps_api_keys_on_configured_target() -> None:
    headers = {"authorization": "Bearer xai-abc123", "user-agent": "grok-shell/0.2.112"}
    assert session_upstream(headers, _XAI_TARGET) is None


def test_session_upstream_ignores_other_clients() -> None:
    assert (
        session_upstream(
            {"authorization": _SESSION_JWT, "user-agent": "codex-cli/1.0"}, _XAI_TARGET
        )
        is None
    )
    assert session_upstream({"authorization": _SESSION_JWT}, _XAI_TARGET) is None
    # Wrapper UAs that merely contain "grok" never match.
    assert (
        session_upstream(
            {"authorization": _SESSION_JWT, "user-agent": "litellm-grok/1.0"}, _XAI_TARGET
        )
        is None
    )


_SPOOFED_GROK_SESSION = {
    "authorization": "Bearer eyJhbGciOiJSUzI1NiJ9.eyJpc3MiOiJodHRwczovL3Nzby5jb3JwIn0.sig",
    "x-xai-token-auth": "xai-grok-cli",
    "user-agent": "grok-shell/0.2.112 (macos; aarch64)",
}


@pytest.mark.parametrize(
    "configured_target",
    [
        "https://gateway.internal",
        "https://api.openai.com",
        None,
        "http://api.x.ai",
        "https://api.x.ai:8443",
        "https://api.x.ai.evil.example",
    ],
)
def test_session_upstream_never_bypasses_a_non_xai_target(configured_target) -> None:
    """A JWT-shaped bearer plus spoofed Grok headers proves nothing about who
    issued the token (an internal gateway's SSO JWT looks the same), so a
    target the operator did not point at xAI must keep the request."""
    assert session_upstream(_SPOOFED_GROK_SESSION, configured_target) is None


@pytest.mark.parametrize(
    "configured_target", ["https://api.x.ai", "https://api.x.ai/", "https://API.X.AI:443/v1"]
)
def test_session_upstream_accepts_any_spelling_of_the_xai_target(configured_target) -> None:
    assert session_upstream(_SPOOFED_GROK_SESSION, configured_target) == SESSION_API_URL
