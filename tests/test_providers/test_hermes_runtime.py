"""Tests for Hermes provider runtime helpers."""

from headroom.providers.hermes.runtime import (
    DEFAULT_HERMES_API_URL,
    build_launch_env,
    proxy_base_url,
)


def test_proxy_base_url() -> None:
    assert proxy_base_url(8787) == "http://127.0.0.1:8787/v1"


def test_default_hermes_api_url_has_v1_suffix() -> None:
    assert DEFAULT_HERMES_API_URL.endswith("/v1")


def test_build_launch_env_sets_openai_base_url() -> None:
    env, display = build_launch_env(9999, {"HOME": "/tmp"})
    assert env["OPENAI_BASE_URL"] == "http://127.0.0.1:9999/v1"
    assert env["HOME"] == "/tmp"
    assert display == ["OPENAI_BASE_URL=http://127.0.0.1:9999/v1"]