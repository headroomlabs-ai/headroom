"""Tests for OpenCode install-time helpers."""

from __future__ import annotations

import json
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


@pytest.mark.parametrize("suffix", [".json", ".jsonc"])
def test_install_stop_remove_restores_original_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    from headroom.cli import install as cli_install

    config_file = tmp_path / f"opencode{suffix}"
    monkeypatch.setenv("OPENCODE_CONFIG", str(config_file))
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    original = (
        b'{"theme":"before","provider":{"anthropic":{}},}\n'
        if suffix == ".jsonc"
        else b'{"theme":"before","provider":{"anthropic":{}}}\n'
    )
    config_file.write_bytes(original)
    manifest = _manifest()
    manifest.targets = ["opencode"]
    monkeypatch.setattr(cli_install, "_save_apply_manifest", lambda current: None)
    monkeypatch.setattr(cli_install, "save_manifest", lambda current: None)
    monkeypatch.setattr(cli_install, "stop_runtime", lambda current: None)
    monkeypatch.setattr(cli_install, "wait_stopped", lambda current: True)
    monkeypatch.setattr(cli_install, "remove_supervisor", lambda current: None)
    monkeypatch.setattr(cli_install, "delete_manifest", lambda profile: None)

    cli_install._activate_deployment_mutations(manifest)
    cli_install._deactivate_deployment_mutations(manifest)
    assert manifest.mutations == []
    cli_install._remove_deployment(manifest, restore_backup=True)

    assert config_file.read_bytes() == original
    assert not config_file.with_name(config_file.name + ".headroom-backup").exists()


def test_failed_activation_retains_backup_for_final_remove_after_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from headroom.cli import install as cli_install

    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "headroom-state"))
    config_file = tmp_path / "opencode.json"
    monkeypatch.setenv("OPENCODE_CONFIG", str(config_file))
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    original = b'{"theme":"original"}\n'
    config_file.write_bytes(original)
    manifest = _manifest()
    manifest.targets = ["opencode"]
    real_save = cli_install._save_apply_manifest
    real_save(manifest)

    def fail_activation_save(current):
        if current.mutations:
            raise OSError("manifest save failed")
        real_save(current)

    monkeypatch.setattr(cli_install, "_save_apply_manifest", fail_activation_save)
    with pytest.raises(OSError, match="manifest save failed"):
        cli_install._activate_deployment_mutations(manifest)

    backup_file = config_file.with_name(config_file.name + ".headroom-backup")
    assert manifest.mutations == []
    assert backup_file.read_bytes() == original
    from headroom.install.state import load_manifest as load_state_manifest

    persisted = load_state_manifest(manifest.profile)
    assert persisted is not None

    monkeypatch.setattr(cli_install, "_save_apply_manifest", lambda current: None)
    monkeypatch.setattr(cli_install, "stop_runtime", lambda current: None)
    monkeypatch.setattr(cli_install, "wait_stopped", lambda current: True)
    monkeypatch.setattr(cli_install, "remove_supervisor", lambda current: None)
    monkeypatch.setattr(cli_install, "delete_manifest", lambda profile: None)
    cli_install._remove_deployment(persisted, restore_backup=True)
    assert config_file.read_bytes() == original
    assert not backup_file.exists()


@pytest.mark.parametrize("config_name", ["opencode.jsonc", "custom-settings.jsonc"])
def test_legacy_jsonc_backup_migrates_before_restart_and_final_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_name: str
) -> None:
    from headroom.cli import install as cli_install

    config_file = tmp_path / config_name
    monkeypatch.setenv("OPENCODE_CONFIG", str(config_file))
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    original = b'// user\'s original JSONC\n{"theme":"original",}\n'
    config_file.write_bytes(b'{"theme":"edited","provider":{"headroom":{}}}\n')
    legacy_backup = config_file.with_suffix(".json.headroom-backup")
    legacy_backup.write_bytes(original)
    canonical_backup = config_file.with_name(config_file.name + ".headroom-backup")
    manifest = _manifest()
    manifest.targets = ["opencode"]
    monkeypatch.setattr(cli_install, "_save_apply_manifest", lambda current: None)
    monkeypatch.setattr(cli_install, "save_manifest", lambda current: None)
    monkeypatch.setattr(cli_install, "stop_runtime", lambda current: None)
    monkeypatch.setattr(cli_install, "wait_stopped", lambda current: True)
    monkeypatch.setattr(cli_install, "remove_supervisor", lambda current: None)
    monkeypatch.setattr(cli_install, "delete_manifest", lambda profile: None)

    cli_install._activate_deployment_mutations(manifest)
    assert canonical_backup.read_bytes() == original
    assert not legacy_backup.exists()
    cli_install._deactivate_deployment_mutations(manifest)
    cli_install._activate_deployment_mutations(manifest)
    cli_install._remove_deployment(
        manifest,
        restore_backup=True,
    )
    assert config_file.read_bytes() == original
    assert not canonical_backup.exists()


def test_jsonc_trailing_commas_preserve_consumer_edits_across_reactivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from headroom.cli import install as cli_install
    from headroom.install.providers import revert_mutations
    from headroom.providers.opencode.config import _parse_json_loose

    config_file = tmp_path / "opencode.jsonc"
    monkeypatch.setenv("OPENCODE_CONFIG", str(config_file))
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    original = (
        b'// original comment\n{"theme":"original","provider":{"anthropic":{"name":"Claude"},},'
        b'"mcp":{"local":{},},}\n'
    )
    config_file.write_bytes(original)
    manifest = _manifest()
    manifest.targets = ["opencode"]

    monkeypatch.setattr(cli_install, "_save_apply_manifest", lambda current: None)
    cli_install._activate_deployment_mutations(manifest)
    config_file.write_text(
        '// consumer edit\n{"theme":"edited","provider":{"anthropic":{"name":"Claude",},'
        '"headroom":{"options":{"baseURL":"http://127.0.0.1:8787/v1",},},},'
        '"mcp":{"remote":{"url":"https://example.test",},},}\n'
    )
    cli_install._deactivate_deployment_mutations(manifest)
    inactive = _parse_json_loose(config_file.read_text())
    assert inactive["theme"] == "edited"
    assert inactive["provider"] == {"anthropic": {"name": "Claude"}}
    assert inactive["mcp"]["remote"]["url"] == "https://example.test"
    cli_install._activate_deployment_mutations(manifest)
    active = _parse_json_loose(config_file.read_text())
    assert active["theme"] == "edited"
    assert active["provider"]["headroom"]["options"]["baseURL"].endswith("/v1")
    assert active["mcp"]["remote"]["url"] == "https://example.test"
    revert_mutations(manifest, restore_backup=True)
    assert config_file.read_bytes() == original


def test_successful_replacement_dropping_opencode_restores_original_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from headroom.cli import install as cli_install
    from headroom.install.state import save_manifest_strict

    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "headroom-state"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OPENCODE_CONFIG", str(tmp_path / "opencode.json"))
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    config_file = tmp_path / "opencode.json"
    original = b'{"theme":"original"}\n'
    config_file.write_bytes(original)
    previous = _manifest()
    previous.targets = ["opencode"]
    cli_install._activate_deployment_mutations(previous)
    save_manifest_strict(previous)
    replacement = _manifest()
    replacement.targets = []

    monkeypatch.setattr(cli_install, "load_manifest", lambda profile: previous)
    monkeypatch.setattr(cli_install, "install_supervisor", lambda current, **kwargs: [])
    monkeypatch.setattr(cli_install, "_start_deployment", lambda current: None)
    monkeypatch.setattr(cli_install, "stop_runtime", lambda current: None)
    monkeypatch.setattr(cli_install, "wait_stopped", lambda current: True)
    monkeypatch.setattr(cli_install, "remove_supervisor", lambda current: None)

    cli_install._apply_manifest(replacement)

    assert config_file.read_bytes() == original
    assert not config_file.with_name("opencode.json.headroom-backup").exists()


def test_failed_replacement_retains_opencode_snapshot_and_reinstates_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import click

    from headroom.cli import install as cli_install
    from headroom.install.state import save_manifest_strict

    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "headroom-state"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OPENCODE_CONFIG", str(tmp_path / "opencode.json"))
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    config_file = tmp_path / "opencode.json"
    original = b'{"theme":"original"}\n'
    config_file.write_bytes(original)
    previous = _manifest()
    previous.targets = ["opencode"]
    cli_install._activate_deployment_mutations(previous)
    save_manifest_strict(previous)
    replacement = _manifest()
    replacement.targets = []
    backup_file = config_file.with_name("opencode.json.headroom-backup")

    monkeypatch.setattr(cli_install, "load_manifest", lambda profile: previous)
    monkeypatch.setattr(cli_install, "install_supervisor", lambda current, **kwargs: [])

    def start(current):
        if not current.targets:
            raise click.ClickException("replacement startup failed")

    monkeypatch.setattr(cli_install, "_start_deployment", start)
    monkeypatch.setattr(cli_install, "stop_runtime", lambda current: None)
    monkeypatch.setattr(cli_install, "wait_stopped", lambda current: True)
    monkeypatch.setattr(cli_install, "remove_supervisor", lambda current: None)

    with pytest.raises(click.ClickException, match="replacement startup failed"):
        cli_install._apply_manifest(replacement)

    assert backup_file.read_bytes() == original
    from headroom.install.state import load_manifest as load_state_manifest

    restored = load_state_manifest(previous.profile)
    assert restored is not None and restored.mutations
    active = json.loads(config_file.read_text())
    assert active["provider"]["headroom"]["options"]["baseURL"].endswith("/v1")


def test_jsonc_migration_does_not_claim_json_installation_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from headroom.cli import install as cli_install

    config_file = tmp_path / "opencode.jsonc"
    json_config = tmp_path / "opencode.json"
    monkeypatch.setenv("OPENCODE_CONFIG", str(config_file))
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    json_config.write_bytes(b'{"theme":"json install"}\n')
    config_file.write_bytes(b'{"theme":"jsonc install"}\n')
    legacy_backup = tmp_path / "opencode.json.headroom-backup"
    legacy_original = b'{"theme":"json original"}\n'
    legacy_backup.write_bytes(legacy_original)
    manifest = _manifest()
    manifest.targets = ["opencode"]
    monkeypatch.setattr(cli_install, "_save_apply_manifest", lambda current: None)

    cli_install._activate_deployment_mutations(manifest)

    canonical_backup = tmp_path / "opencode.jsonc.headroom-backup"
    assert canonical_backup.read_bytes() == b'{"theme":"jsonc install"}\n'
    assert legacy_backup.read_bytes() == legacy_original


@pytest.fixture
def opencode_transaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from headroom.cli import install as cli_install

    monkeypatch.setenv("HEADROOM_WORKSPACE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OPENCODE_CONFIG", str(tmp_path / "opencode.json"))
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    monkeypatch.setattr(cli_install, "install_supervisor", lambda current, **kwargs: [])
    monkeypatch.setattr(cli_install, "_start_deployment", lambda current: None)
    monkeypatch.setattr(cli_install, "stop_runtime", lambda current: None)
    monkeypatch.setattr(cli_install, "wait_stopped", lambda current: True)
    monkeypatch.setattr(cli_install, "remove_supervisor", lambda current: None)
    return cli_install


def test_failed_replacement_consumes_new_unowned_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opencode_transaction
) -> None:
    import click

    cli_install = opencode_transaction
    config = tmp_path / "opencode.json"
    original = b'{"theme":"dark"}\n'
    config.write_bytes(original)
    previous = _manifest()
    cli_install.save_manifest_strict(previous)
    replacement = _manifest()
    replacement.targets = ["opencode"]
    real_save = cli_install._save_apply_manifest

    def fail_active_save(current):
        if current.mutations:
            raise OSError("activation persistence failed")
        real_save(current)

    with monkeypatch.context() as fault:
        fault.setattr(cli_install, "_save_apply_manifest", fail_active_save)
        with pytest.raises(click.ClickException):
            cli_install._apply_manifest(replacement)

    assert config.read_bytes() == original
    assert not config.with_name(config.name + ".headroom-backup").exists()
    later = b'{"theme":"light"}\n'
    config.write_bytes(later)
    retry = _manifest()
    retry.targets = ["opencode"]
    cli_install._apply_manifest(retry)
    cli_install._remove_deployment(retry, restore_backup=True)
    assert config.read_bytes() == later


@pytest.mark.parametrize("legacy_manifest", [False, True])
@pytest.mark.parametrize("retain_opencode", [False, True])
def test_changed_config_path_restores_previous_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opencode_transaction,
    legacy_manifest: bool,
    retain_opencode: bool,
) -> None:
    cli_install = opencode_transaction
    original_config = tmp_path / "opencode.json"
    original = b'{"theme":"old config original"}\n'
    original_config.write_bytes(original)
    previous = _manifest()
    previous.targets = ["opencode"]
    cli_install._activate_deployment_mutations(previous)
    if legacy_manifest:
        previous.artifacts = []
    cli_install.save_manifest_strict(previous)
    new_config = tmp_path / "different.json"
    new_original = b'{"theme":"new config original"}\n'
    new_config.write_bytes(new_original)
    monkeypatch.setenv("OPENCODE_CONFIG", str(new_config))
    replacement = _manifest()
    replacement.targets = ["opencode"] if retain_opencode else []

    cli_install._apply_manifest(replacement)

    assert original_config.read_bytes() == original
    assert not original_config.with_name(original_config.name + ".headroom-backup").exists()
    if retain_opencode:
        assert "headroom" in json.loads(new_config.read_text())["provider"]
    cli_install._remove_deployment(replacement, restore_backup=True)
    assert new_config.read_bytes() == new_original


def test_canonical_snapshot_wins_over_ambiguous_legacy_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opencode_transaction
) -> None:
    cli_install = opencode_transaction
    config = tmp_path / "opencode.jsonc"
    monkeypatch.setenv("OPENCODE_CONFIG", str(config))
    config.write_bytes(b'{"theme":"live"}\n')
    canonical = config.with_name("opencode.jsonc.headroom-backup")
    original = b'// current original\n{"theme":"light",}\n'
    canonical.write_bytes(original)
    legacy = config.with_suffix(".json.headroom-backup")
    legacy_original = b'{"theme":"dark"}\n'
    legacy.write_bytes(legacy_original)
    manifest = _manifest()
    manifest.targets = ["opencode"]

    cli_install._activate_deployment_mutations(manifest)
    cli_install._deactivate_deployment_mutations(manifest)
    cli_install._activate_deployment_mutations(manifest)
    cli_install._remove_deployment(manifest, restore_backup=True)

    assert config.read_bytes() == original
    assert not canonical.exists()
    assert legacy.read_bytes() == legacy_original


def test_replacement_restore_failure_can_finish_after_manifest_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, opencode_transaction
) -> None:
    import click

    cli_install = opencode_transaction
    config = tmp_path / "opencode.json"
    original = b'{"theme":"original"}\n'
    config.write_bytes(original)
    previous = _manifest()
    previous.targets = ["opencode"]
    cli_install._activate_deployment_mutations(previous)
    replacement = _manifest()
    restore = cli_install.restore_opencode_backup

    def fail_restore(*args):
        raise OSError("backup temporarily unreadable")

    with monkeypatch.context() as fault:
        fault.setattr(cli_install, "restore_opencode_backup", fail_restore)
        with pytest.raises((OSError, click.ClickException)):
            cli_install._apply_manifest(replacement)

    persisted = cli_install.load_manifest(replacement.profile)
    assert persisted is not None and persisted.targets == []
    monkeypatch.setattr(cli_install, "restore_opencode_backup", restore)
    retry = _manifest()
    cli_install._apply_manifest(retry)

    assert config.read_bytes() == original
    assert not config.with_name(config.name + ".headroom-backup").exists()
