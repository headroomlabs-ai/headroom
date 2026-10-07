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
    assert "mcp" not in config


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


@pytest.mark.parametrize("suffix", [".json", ".jsonc"])
def test_temporary_deactivation_preserves_edits_and_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    """Reactivation removes only Headroom's provider and keeps the unwrap snapshot."""
    import json

    from headroom.cli import install as cli_install
    from headroom.install.providers import revert_mutations

    config_file = tmp_path / f"opencode{suffix}"
    monkeypatch.setenv("OPENCODE_CONFIG", str(config_file))
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    monkeypatch.setattr(cli_install, "_save_apply_manifest", lambda manifest: None)
    monkeypatch.setattr(cli_install, "save_manifest", lambda manifest: None)
    original_json = (
        b'{"theme":"original","provider":{"anthropic":{"name":"Claude"}},"mcp":{"local":{}}}\n'
    )
    original = (
        b"// user comment in OpenCode JSONC\n" + original_json
        if suffix == ".jsonc"
        else original_json
    )
    config_file.write_bytes(original)

    manifest = _manifest()
    manifest.targets = ["opencode"]
    cli_install._activate_deployment_mutations(manifest)
    backup_file = config_file.with_name(config_file.name + ".headroom-backup")
    assert backup_file.read_bytes() == original

    for theme in ("edited after install", "edited again"):
        data = json.loads(config_file.read_text())
        data["theme"] = theme
        data["provider"]["other-user-provider"] = {"name": "Other"}
        data["mcp"]["remote"] = {"url": "https://example.test"}
        config_file.write_text(json.dumps(data))

        cli_install._deactivate_deployment_mutations(manifest)
        assert manifest.mutations == []
        assert backup_file.read_bytes() == original
        inactive = json.loads(config_file.read_text())
        assert inactive["theme"] == theme
        assert inactive["provider"] == {
            "anthropic": {"name": "Claude"},
            "other-user-provider": {"name": "Other"},
        }
        assert inactive["mcp"]["remote"] == {"url": "https://example.test"}

        cli_install._activate_deployment_mutations(manifest)
        active = json.loads(config_file.read_text())
        assert active["theme"] == theme
        assert active["provider"]["headroom"]["options"]["baseURL"] == ("http://127.0.0.1:8787/v1")
        assert active["provider"]["other-user-provider"] == {"name": "Other"}
        assert active["mcp"]["remote"] == {"url": "https://example.test"}

    revert_mutations(manifest, restore_backup=True)
    assert config_file.read_bytes() == original
    assert not backup_file.exists()


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


@pytest.mark.windows_newline
def test_provider_config_writes_pin_lf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider edits and stale managed-block cleanup pin LF and preserve user config."""
    config_file = tmp_path / "opencode.json"
    monkeypatch.setenv("OPENCODE_CONFIG", str(config_file))
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    config_file.write_bytes(b'{"theme":"before"}\n')

    writes: list[tuple[Path, str | None]] = []
    original_write_text = Path.write_text

    def write_text_spy(self, data, encoding=None, errors=None, newline=None):
        writes.append((self, newline))
        return original_write_text(self, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "write_text", write_text_spy)

    manifest = _manifest()
    mutation = apply_provider_scope(manifest)
    assert mutation is not None
    revert_provider_scope(mutation, manifest, restore_backup=False)

    marker_config = (
        '{\n  "theme": "kept",\n'
        "// --- Headroom proxy provider ---\n"
        "  stale generated provider block\n"
        "// --- end Headroom proxy provider ---\n"
        '  "mcp": {"remote": {}}\n}\n'
    )
    config_file.write_bytes(marker_config.encode())
    revert_provider_scope(mutation, manifest, restore_backup=False)

    assert len(writes) == 3
    assert all(path == config_file and newline == "\n" for path, newline in writes)
    import json

    cleaned = json.loads(config_file.read_text())
    assert cleaned == {"theme": "kept", "mcp": {"remote": {}}}
