"""Tests for the DeepSeek Harness (dsh) provider runtime."""

from __future__ import annotations

import pytest

from headroom.providers.dsh.install import build_install_env
from headroom.providers.dsh.runtime import (
    DEFAULT_API_URL,
    build_launch_env,
    proxy_base_url,
    resolve_dsh_command,
)
from headroom.providers.install_registry import build_install_target_envs


def test_proxy_base_url_includes_v1() -> None:
    assert proxy_base_url(8787) == "http://127.0.0.1:8787/v1"


def test_default_api_url_is_deepseek_public() -> None:
    assert DEFAULT_API_URL == "https://api.deepseek.com"


def test_build_launch_env_sets_deepseek_base_url_and_passes_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    env, display = build_launch_env(9000)
    assert env["DEEPSEEK_BASE_URL"] == "http://127.0.0.1:9000/v1"
    assert env["DEEPSEEK_API_KEY"] == "sk-test"
    assert display == ["DEEPSEEK_BASE_URL=http://127.0.0.1:9000/v1"]


def test_resolve_dsh_command_web_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("headroom.providers.dsh.runtime.shutil.which", lambda _name: "/usr/bin/dsh")
    assert resolve_dsh_command() == ["/usr/bin/dsh", "web"]


def test_resolve_dsh_command_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("headroom.providers.dsh.runtime.shutil.which", lambda _name: "/usr/bin/dsh")
    assert resolve_dsh_command(profile="headless", task_args=("explain foo",)) == [
        "/usr/bin/dsh",
        "--profile",
        "headless",
        "explain foo",
    ]


def test_resolve_dsh_command_explicit_command_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("headroom.providers.dsh.runtime.shutil.which", lambda _name: "/usr/bin/dsh")
    assert resolve_dsh_command(command="pnpm dsh") == ["pnpm", "dsh", "web"]


def test_resolve_dsh_command_missing_binary_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("headroom.providers.dsh.runtime.shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="not found in PATH"):
        resolve_dsh_command()


def test_dsh_build_install_env_sets_deepseek_base_url() -> None:
    assert build_install_env(port=9000, backend="anthropic") == {
        "DEEPSEEK_BASE_URL": "http://127.0.0.1:9000/v1"
    }


def test_install_registry_includes_dsh() -> None:
    envs = build_install_target_envs(port=9000, backend="anthropic", targets=["dsh"])
    assert envs["dsh"] == {"DEEPSEEK_BASE_URL": "http://127.0.0.1:9000/v1"}
