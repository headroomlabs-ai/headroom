from __future__ import annotations

from headroom.providers.grok import PROXY_ENV_KEY, build_launch_env, proxy_base_url
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
