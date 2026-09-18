"""Tests for CodeBuddy install-time helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from headroom.install.models import DeploymentManifest, ToolTarget
from headroom.providers.codebuddy.install import (
    apply_provider_scope,
    build_install_env,
    revert_provider_scope,
)


def _manifest(tmp_path: Path) -> DeploymentManifest:
    return DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="provider",
        provider_mode="manual",
        targets=["codebuddy"],
        port=8787,
        host="127.0.0.1",
        backend="codebuddy",
        memory_db_path=str(tmp_path / "memory.db"),
        tool_envs={
            "codebuddy": {"CODEBUDDY_BASE_URL": "http://127.0.0.1:8787/v2"},
        },
    )


def test_build_install_env():
    env = build_install_env(port=8787, backend="codebuddy")
    assert env["CODEBUDDY_BASE_URL"] == "http://127.0.0.1:8787/v2"


def test_apply_and_revert_provider_scope(monkeypatch, tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps({"env": {"CODEBUDDY_BASE_URL": "https://old.upstream"}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "headroom.providers.codebuddy.install.codebuddy_settings_path",
        lambda: settings_path,
    )
    manifest = _manifest(tmp_path)

    mutation = apply_provider_scope(manifest)
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["env"]["CODEBUDDY_BASE_URL"] == "http://127.0.0.1:8787/v2"

    assert mutation is not None
    revert_provider_scope(mutation, manifest)
    reverted = json.loads(settings_path.read_text(encoding="utf-8"))
    assert reverted["env"]["CODEBUDDY_BASE_URL"] == "https://old.upstream"


def test_apply_provider_scope_creates_settings_if_missing(monkeypatch, tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    monkeypatch.setattr(
        "headroom.providers.codebuddy.install.codebuddy_settings_path",
        lambda: settings_path,
    )
    manifest = _manifest(tmp_path)

    mutation = apply_provider_scope(manifest)
    assert settings_path.exists()
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["env"]["CODEBUDDY_BASE_URL"] == "http://127.0.0.1:8787/v2"
    assert mutation is not None


def test_apply_provider_scope_returns_none_for_non_provider_scope(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = DeploymentManifest(
        profile="default",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="user",
        provider_mode="manual",
        targets=["codebuddy"],
        port=8787,
        host="127.0.0.1",
        backend="codebuddy",
        memory_db_path=str(tmp_path / "memory.db"),
        tool_envs={},
    )
    result = apply_provider_scope(manifest)
    assert result is None


def test_apply_provider_scope_handles_invalid_json(monkeypatch, tmp_path: Path) -> None:
    import json as json_mod

    settings_path = tmp_path / "settings.json"
    settings_path.write_text("not valid json", encoding="utf-8")
    monkeypatch.setattr(
        "headroom.providers.codebuddy.install.codebuddy_settings_path",
        lambda: settings_path,
    )
    manifest = _manifest(tmp_path)

    with pytest.raises(json_mod.JSONDecodeError):
        apply_provider_scope(manifest)


def test_revert_provider_scope_no_path():
    from headroom.install.models import ManagedMutation

    mutation = ManagedMutation(
        target=ToolTarget.CODEBUDDY.value,
        kind="json-env",
        path="",
        data={},
    )
    manifest = _manifest(Path("/tmp"))
    revert_provider_scope(mutation, manifest)


def test_revert_provider_scope_missing_file(monkeypatch, tmp_path: Path) -> None:
    from headroom.install.models import ManagedMutation

    settings_path = tmp_path / "nonexistent.json"
    mutation = ManagedMutation(
        target=ToolTarget.CODEBUDDY.value,
        kind="json-env",
        path=str(settings_path),
        data={"previous": {}},
    )
    manifest = _manifest(tmp_path)
    revert_provider_scope(mutation, manifest)
