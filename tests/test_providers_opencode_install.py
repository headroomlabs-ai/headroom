"""Tests for OpenCode install-time helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from headroom.install.models import ConfigScope, DeploymentManifest
from headroom.providers.opencode.install import (
    apply_provider_scope,
    build_install_env,
    revert_provider_scope,
)


def _manifest(port: int = 8787) -> DeploymentManifest:
    return DeploymentManifest(
        profile="test",
        preset="persistent-task",
        runtime_kind="python",
        supervisor_kind="none",
        scope=ConfigScope.PROVIDER.value,
        provider_mode="auto",
        targets=[],
        port=port,
        host="127.0.0.1",
        backend="anthropic",
        proxy_args=[],
        base_env={},
        tool_envs={},
    )


def test_build_install_env() -> None:
    """build_install_env leaves OpenCode provider env vars untouched."""
    env = build_install_env(port=8787, backend="anthropic")
    assert env == {}


def test_apply_provider_scope_creates_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """apply_provider_scope creates the opencode config with headroom provider."""
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)

    manifest = _manifest(port=8787)
    mutation = apply_provider_scope(manifest)
    assert mutation is not None
    assert mutation.target == "opencode"
    assert mutation.kind == "json-block"

    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    assert config_file.exists()
    import json

    config = json.loads(config_file.read_text())
    assert config["provider"]["headroom"]["options"]["baseURL"] == "http://127.0.0.1:8787/v1"


def test_apply_provider_scope_skips_when_scope_is_not_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """apply_provider_scope returns None when scope is not PROVIDER."""
    manifest = _manifest()
    manifest.scope = ConfigScope.USER.value
    result = apply_provider_scope(manifest)
    assert result is None


def test_revert_provider_scope_restores_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """revert_provider_scope strips the Headroom block from the config."""
    home = str(tmp_path)
    monkeypatch.setenv("HOME", home)
    monkeypatch.setenv("USERPROFILE", home)
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)

    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text('{"model": "openai/gpt-4o"}')

    from headroom.install.models import ManagedMutation

    mutation = ManagedMutation(
        target="opencode",
        kind="json-block",
        path=str(config_file),
    )
    manifest = _manifest()
    revert_provider_scope(mutation, manifest)
    assert config_file.exists()
    assert config_file.read_text().strip() == '{"model": "openai/gpt-4o"}'


def test_revert_provider_scope_noop_when_file_missing(
    tmp_path: Path,
) -> None:
    """revert_provider_scope is a safe no-op when the config file is gone."""
    from headroom.install.models import ManagedMutation

    mutation = ManagedMutation(
        target="opencode",
        kind="json-block",
        path=str(tmp_path / "nonexistent.json"),
    )
    manifest = _manifest()
    revert_provider_scope(mutation, manifest)
    # Should not raise
