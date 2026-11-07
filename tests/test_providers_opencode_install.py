"""Tests for OpenCode install-time helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import pytest

from headroom.install.models import ConfigScope, DeploymentManifest
from headroom.providers.opencode.config import HEADROOM_OPENCODE_MODELS
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


def _opencode_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)
    return tmp_path / ".config" / "opencode" / "opencode.json"


def _headroom_models(config_file: Path) -> dict:
    return json.loads(config_file.read_text())["provider"]["headroom"]["models"]


def test_apply_provider_scope_lists_default_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OpenCode only resolves headroom/<id> for listed ids, so install lists them.

    Extra models come from the manifest only: a scheduled restart does not run
    in the shell that ran ``install apply``.
    """
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    monkeypatch.setenv("HEADROOM_OPENCODE_EXTRA_MODELS", "from-shell")

    apply_provider_scope(_manifest())

    assert _headroom_models(config_file) == HEADROOM_OPENCODE_MODELS


def test_apply_provider_scope_ignores_the_shell_when_the_manifest_has_no_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest without base_env must not fall back to the process env.

    ``extra_models_from_env`` reads ``os.environ`` when it is handed no
    mapping, so a manifest whose ``base_env`` is null (hand-edited or corrupt)
    would otherwise write the shell's specs into the config -- the opposite of
    the manifest-env guarantee.
    """
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    monkeypatch.setenv("HEADROOM_OPENCODE_EXTRA_MODELS", "from-shell")
    manifest = _manifest()
    manifest.base_env = None  # type: ignore[assignment]

    apply_provider_scope(manifest)

    assert _headroom_models(config_file) == HEADROOM_OPENCODE_MODELS


def test_apply_provider_scope_adds_manifest_extra_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HEADROOM_OPENCODE_EXTRA_MODELS from ``install apply --env`` is written."""
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    manifest = _manifest()
    manifest.base_env = {
        "HEADROOM_OPENCODE_EXTRA_MODELS": "deepseek-chat=DeepSeek Chat=65536,qwen2.5-coder:7b"
    }

    apply_provider_scope(manifest)

    models = _headroom_models(config_file)
    assert models["deepseek-chat"] == {
        "name": "DeepSeek Chat",
        "limit": {"context": 65536, "output": 16384},
    }
    assert models["qwen2.5-coder:7b"]["name"] == "qwen2.5-coder:7b"
    assert "gpt-4o" in models


@pytest.mark.parametrize("prior_config", [True, False], ids=["with-backup", "no-prior-config"])
def test_reapply_keeps_manifest_extra_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prior_config: bool
) -> None:
    """Re-running install apply reverts the old deployment, then applies (#3970).

    The revert restores the pre-install backup when there is one, and the apply
    replaces the whole ``headroom`` provider, so the models must come back from
    the manifest.
    """
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    if prior_config:
        config_file.parent.mkdir(parents=True)
        config_file.write_text('{"theme": "system"}')
    manifest = _manifest()
    manifest.base_env = {"HEADROOM_OPENCODE_EXTRA_MODELS": "deepseek-chat"}

    mutation = apply_provider_scope(manifest)
    assert mutation is not None
    revert_provider_scope(mutation, manifest)
    apply_provider_scope(manifest)

    models = _headroom_models(config_file)
    assert "deepseek-chat" in models
    assert "gpt-4o" in models


def _write_credential_config(config_file: Path, api_key: str) -> None:
    """Write an OpenCode config whose headroom provider carries a credential."""
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        json.dumps(
            {
                "provider": {
                    "headroom": {
                        "npm": "@ai-sdk/openai-compatible",
                        "name": "Headroom Proxy",
                        "options": {
                            "baseURL": "http://127.0.0.1:8787/v1",
                            "apiKey": api_key,
                        },
                        "models": {"deepseek-chat": {"name": "DeepSeek Chat"}},
                    }
                }
            },
            indent=2,
        )
        + "\n"
    )


def _headroom_options(config_file: Path) -> dict:
    return json.loads(config_file.read_text())["provider"]["headroom"]["options"]


def test_revert_without_a_credential_stores_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A provider entry without a credential must not land in the state."""
    _opencode_config_file(tmp_path, monkeypatch)
    manifest = _upstream_manifest()

    mutation = apply_provider_scope(manifest)
    assert mutation is not None
    revert_provider_scope(mutation, manifest, restore_backup=False)

    assert manifest.provider_options == {}


def test_apply_keeps_user_credential_for_openai_api_url_arg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--openai-api-url`` in the proxy args names the upstream too."""
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    _write_credential_config(config_file, "{env:DEEPSEEK_API_KEY}")
    manifest = _manifest()
    manifest.proxy_args = [
        "--host",
        "127.0.0.1",
        "--openai-api-url",
        "https://api.deepseek.com/v1",
    ]

    apply_provider_scope(manifest)

    assert _headroom_options(config_file)["apiKey"] == "{env:DEEPSEEK_API_KEY}"


@pytest.mark.parametrize(
    "config",
    [
        {"provider": {"headroom": "not-a-provider"}},
        {"provider": "not-a-mapping"},
        {"provider": {"headroom": {"options": "not-a-mapping"}}},
    ],
    ids=["provider-not-a-mapping", "headroom-not-a-mapping", "options-not-a-mapping"],
)
def test_apply_survives_a_malformed_provider_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: dict
) -> None:
    """A hand-mangled config must not crash the install."""
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(json.dumps(config, indent=2) + "\n")

    apply_provider_scope(_upstream_manifest())

    assert _headroom_options(config_file)["baseURL"] == "http://127.0.0.1:8787/v1"


@pytest.mark.parametrize(
    "store",
    [None, {"opencode": "not-a-mapping"}],
    ids=["store-not-a-mapping", "options-not-a-mapping"],
)
def test_apply_survives_a_malformed_option_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, store: dict | None
) -> None:
    """A hand-edited manifest must not crash the install."""
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    _write_credential_config(config_file, "{env:DEEPSEEK_API_KEY}")
    manifest = _upstream_manifest()
    manifest.provider_options = store  # type: ignore[assignment]

    apply_provider_scope(manifest)

    assert _headroom_options(config_file)["apiKey"] == "{env:DEEPSEEK_API_KEY}"


def _upstream_manifest() -> DeploymentManifest:
    manifest = _manifest()
    manifest.base_env = {"OPENAI_TARGET_API_URL": "https://api.deepseek.com/v1"}
    return manifest


def test_apply_keeps_user_credential_for_named_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The provider apiKey survives the install when an upstream is named (#4015).

    OpenCode sends that key to the proxy, which forwards it to the upstream, so
    dropping it leaves the listed models unusable.
    """
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    _write_credential_config(config_file, "{env:DEEPSEEK_API_KEY}")

    apply_provider_scope(_upstream_manifest())

    options = _headroom_options(config_file)
    assert options["apiKey"] == "{env:DEEPSEEK_API_KEY}"
    # Headroom still owns the proxy URL.
    assert options["baseURL"] == "http://127.0.0.1:8787/v1"


def test_apply_drops_user_credential_without_named_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a named upstream the proxy forwards to OpenAI: drop the key."""
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    _write_credential_config(config_file, "{env:DEEPSEEK_API_KEY}")

    apply_provider_scope(_manifest())

    options = _headroom_options(config_file)
    assert "apiKey" not in options
    assert options["baseURL"] == "http://127.0.0.1:8787/v1"


def test_reapply_keeps_user_credential(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-running install apply keeps the credential (#4015).

    The deactivate rewrites the config without the provider entry, so the
    credential has to come back from the deployment state.
    """
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    _write_credential_config(config_file, "{env:DEEPSEEK_API_KEY}")
    manifest = _upstream_manifest()

    mutation = apply_provider_scope(manifest)
    assert mutation is not None
    revert_provider_scope(mutation, manifest, restore_backup=False)
    assert "headroom" not in json.loads(config_file.read_text()).get("provider", {})
    apply_provider_scope(manifest)

    assert _headroom_options(config_file)["apiKey"] == "{env:DEEPSEEK_API_KEY}"


def test_credential_added_after_install_survives_reactivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A key written into the config after the install survives the cycle."""
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    manifest = _upstream_manifest()

    mutation = apply_provider_scope(manifest)
    assert mutation is not None
    data = json.loads(config_file.read_text())
    data["provider"]["headroom"]["options"]["apiKey"] = "sk-user-key"
    config_file.write_text(json.dumps(data, indent=2) + "\n")

    revert_provider_scope(mutation, manifest, restore_backup=False)
    # provider_options now includes _upstream for upstream tracking
    assert manifest.provider_options["opencode"] == {
        "apiKey": "sk-user-key",
        "_upstream": "https://api.deepseek.com/v1",
        "_upstream_v": 2,
    }
    apply_provider_scope(manifest)

    assert _headroom_options(config_file)["apiKey"] == "sk-user-key"


def test_install_apply_carries_credential_into_the_new_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-apply builds a fresh manifest: the credential must carry over (#4015)."""
    from headroom.cli import install as cli_install

    config_file = _opencode_config_file(tmp_path, monkeypatch)
    _write_credential_config(config_file, "{env:DEEPSEEK_API_KEY}")
    existing = _upstream_manifest()
    existing.targets = ["opencode"]
    mutation = apply_provider_scope(existing)
    assert mutation is not None
    existing.mutations = [mutation]

    new = _upstream_manifest()
    new.targets = ["opencode"]

    monkeypatch.setattr(cli_install, "load_manifest", lambda profile: existing)
    monkeypatch.setattr(cli_install, "_save_apply_manifest", lambda manifest: None)
    monkeypatch.setattr(cli_install, "save_manifest", lambda manifest: None)
    monkeypatch.setattr(cli_install, "install_supervisor", lambda manifest, **kwargs: [])
    monkeypatch.setattr(cli_install, "_start_deployment", lambda manifest: None)
    monkeypatch.setattr(cli_install, "_stop_deployment", lambda manifest: None)
    monkeypatch.setattr(cli_install, "remove_supervisor", lambda manifest: None)
    monkeypatch.setattr(cli_install, "delete_manifest", lambda profile: None)

    cli_install._apply_manifest(new)

    # provider_options now includes _upstream for upstream tracking
    assert new.provider_options["opencode"] == {
        "apiKey": "{env:DEEPSEEK_API_KEY}",
        "_upstream": "https://api.deepseek.com/v1",
        "_upstream_v": 2,
    }
    assert _headroom_options(config_file)["apiKey"] == "{env:DEEPSEEK_API_KEY}"


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

    later = b'// edited after removal\n{"theme":"new choice",}\n'
    config.write_bytes(later)
    second = _manifest()
    second.targets = ["opencode"]
    cli_install._activate_deployment_mutations(second)
    assert canonical.read_bytes() == later
    cli_install._remove_deployment(second, restore_backup=True)
    assert config.read_bytes() == later
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


@pytest.mark.parametrize(
    ("config_name", "location"),
    [
        ("opencode.jsonc", "override"),
        ("custom-settings.jsonc", "override"),
        ("opencode.jsonc", "opencode_home"),
        ("opencode.jsonc", "default_home"),
    ],
)
@pytest.mark.parametrize("ambiguity", ["canonical", "sibling"])
def test_ambiguous_legacy_backup_stays_unowned_across_two_install_remove_cycles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_name: str,
    location: str,
    ambiguity: str,
) -> None:
    """Removing the conflicting file must not grant ownership of legacy bytes."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    config_dir = tmp_path / ".config" / "opencode" if location == "default_home" else tmp_path
    config_dir.mkdir(parents=True, exist_ok=True)
    config = config_dir / config_name
    if location == "override":
        monkeypatch.setenv("OPENCODE_CONFIG", str(config))
    elif location == "opencode_home":
        monkeypatch.setenv("OPENCODE_HOME", str(config_dir))
    live = b'{"theme":"live"}\n'
    original = b'// current original\r\n{"theme":"light",}\r\n'
    unrelated = b'// unrelated original\r\n{"theme":"dark",}\r\n'
    config.write_bytes(live if ambiguity == "canonical" else original)
    canonical = config.with_name(config.name + ".headroom-backup")
    legacy = config.with_suffix(".json.headroom-backup")
    legacy.write_bytes(unrelated)
    sibling = config.with_suffix(".json")
    if ambiguity == "canonical":
        canonical.write_bytes(original)
    else:
        sibling.write_bytes(b'{"theme":"separate JSON installation"}\n')

    first = _manifest()
    mutation = apply_provider_scope(first)
    assert mutation is not None
    revert_provider_scope(mutation, first)
    assert config.read_bytes() == original
    assert not canonical.exists()
    assert legacy.read_bytes() == unrelated
    if ambiguity == "sibling":
        sibling.unlink()

    # A new deployment after the first snapshot has been consumed must take a
    # fresh snapshot, even though the formerly ambiguous legacy file remains.
    later = b'// edited between installations\r\n{"theme":"new choice",}\r\n'
    config.write_bytes(later)
    second = _manifest()
    mutation = apply_provider_scope(second)
    assert mutation is not None
    assert canonical.read_bytes() == later
    assert legacy.read_bytes() == unrelated
    revert_provider_scope(mutation, second, restore_backup=False)
    mutation = apply_provider_scope(second)
    assert mutation is not None
    revert_provider_scope(mutation, second)
    assert config.read_bytes() == later
    assert not canonical.exists()
    assert legacy.read_bytes() == unrelated


@pytest.mark.parametrize("operation", ["apply", "restore"])
def test_ambiguous_legacy_backup_marker_failure_preserves_all_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    from headroom.providers.opencode.install import restore_opencode_backup

    config = tmp_path / "opencode.jsonc"
    monkeypatch.setenv("OPENCODE_CONFIG", str(config))
    live = b'{"theme":"live"}\n'
    original = b'{"theme":"original"}\n'
    unrelated = b'{"theme":"unrelated"}\n'
    config.write_bytes(live)
    canonical = config.with_name(config.name + ".headroom-backup")
    canonical.write_bytes(original)
    legacy = config.with_suffix(".json.headroom-backup")
    legacy.write_bytes(unrelated)
    touch = Path.touch

    def fail_marker(path: Path, *args, **kwargs) -> None:
        if path.name.endswith(".jsonc-migration-blocked"):
            raise PermissionError("migration marker is not writable")
        touch(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "touch", fail_marker)
        with pytest.raises(PermissionError, match="migration marker"):
            if operation == "apply":
                apply_provider_scope(_manifest())
            else:
                restore_opencode_backup(str(config), str(canonical))

    assert config.read_bytes() == live
    assert canonical.read_bytes() == original
    assert legacy.read_bytes() == unrelated
    restore_opencode_backup(str(config), str(canonical))
    assert config.read_bytes() == original
    assert not canonical.exists()
    assert legacy.read_bytes() == unrelated


@pytest.mark.parametrize("config_name", ["opencode.jsonc", "custom-settings.jsonc"])
def test_unambiguous_legacy_backup_still_migrates_and_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_name: str
) -> None:
    config = tmp_path / config_name
    monkeypatch.setenv("OPENCODE_CONFIG", str(config))
    config.write_bytes(b'{"theme":"live"}\n')
    legacy = config.with_suffix(".json.headroom-backup")
    original = b'// original\r\n{"theme":"original",}\r\n'
    legacy.write_bytes(original)
    canonical = config.with_name(config.name + ".headroom-backup")
    manifest = _manifest()

    mutation = apply_provider_scope(manifest)
    assert mutation is not None
    assert canonical.read_bytes() == original
    assert not legacy.exists()
    revert_provider_scope(mutation, manifest)
    assert config.read_bytes() == original
    assert not canonical.exists()


def test_install_apply_carries_credential_same_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same upstream: credential carries over on install apply."""
    from headroom.cli import install as cli_install

    config_file = _opencode_config_file(tmp_path, monkeypatch)
    _write_credential_config(config_file, "{env:DEEPSEEK_API_KEY}")
    existing = _upstream_manifest()
    existing.targets = ["opencode"]
    mutation = apply_provider_scope(existing)
    assert mutation is not None
    existing.mutations = [mutation]

    new = _upstream_manifest()
    new.targets = ["opencode"]

    monkeypatch.setattr(cli_install, "load_manifest", lambda profile: existing)
    monkeypatch.setattr(cli_install, "_save_apply_manifest", lambda manifest: None)
    monkeypatch.setattr(cli_install, "save_manifest", lambda manifest: None)
    monkeypatch.setattr(cli_install, "install_supervisor", lambda manifest, **kwargs: [])
    monkeypatch.setattr(cli_install, "_start_deployment", lambda manifest: None)
    monkeypatch.setattr(cli_install, "_stop_deployment", lambda manifest: None)
    monkeypatch.setattr(cli_install, "remove_supervisor", lambda manifest: None)
    monkeypatch.setattr(cli_install, "delete_manifest", lambda profile: None)

    cli_install._apply_manifest(new)

    # provider_options now includes _upstream for upstream tracking
    assert new.provider_options["opencode"] == {
        "apiKey": "{env:DEEPSEEK_API_KEY}",
        "_upstream": "https://api.deepseek.com/v1",
        "_upstream_v": 2,
    }
    assert _headroom_options(config_file)["apiKey"] == "{env:DEEPSEEK_API_KEY}"


def test_install_apply_drops_credential_changed_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changed upstream: credential is dropped on install apply."""
    from headroom.cli import install as cli_install

    config_file = _opencode_config_file(tmp_path, monkeypatch)
    _write_credential_config(config_file, "{env:DEEPSEEK_API_KEY}")
    existing = _upstream_manifest()
    existing.targets = ["opencode"]
    mutation = apply_provider_scope(existing)
    assert mutation is not None
    existing.mutations = [mutation]

    # New manifest with different upstream
    new = _upstream_manifest()
    new.targets = ["opencode"]
    new.base_env = {"OPENAI_TARGET_API_URL": "https://different-upstream.example/v1"}

    monkeypatch.setattr(cli_install, "load_manifest", lambda profile: existing)
    monkeypatch.setattr(cli_install, "_save_apply_manifest", lambda manifest: None)
    monkeypatch.setattr(cli_install, "save_manifest", lambda manifest: None)
    monkeypatch.setattr(cli_install, "install_supervisor", lambda manifest, **kwargs: [])
    monkeypatch.setattr(cli_install, "_start_deployment", lambda manifest: None)
    monkeypatch.setattr(cli_install, "_stop_deployment", lambda manifest: None)
    monkeypatch.setattr(cli_install, "remove_supervisor", lambda manifest: None)
    monkeypatch.setattr(cli_install, "delete_manifest", lambda profile: None)

    cli_install._apply_manifest(new)

    # Credential should be dropped - no opencode entry in provider_options
    assert "opencode" not in new.provider_options
    # Config should not have apiKey
    options = _headroom_options(config_file)
    assert "apiKey" not in options


def test_apply_provider_scope_restart_keeps_same_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restart with same upstream keeps stored credential."""
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    _write_credential_config(config_file, "{env:DEEPSEEK_API_KEY}")
    manifest = _upstream_manifest()

    # First apply - stores credential with upstream
    mutation = apply_provider_scope(manifest)
    assert mutation is not None

    # Revert without backup - captures credential
    revert_provider_scope(mutation, manifest, restore_backup=False)
    assert manifest.provider_options["opencode"] == {
        "apiKey": "{env:DEEPSEEK_API_KEY}",
        "_upstream": "https://api.deepseek.com/v1",
        "_upstream_v": 2,
    }

    # Re-apply with same upstream - credential should be restored
    apply_provider_scope(manifest)

    assert _headroom_options(config_file)["apiKey"] == "{env:DEEPSEEK_API_KEY}"


def test_apply_provider_scope_restart_drops_changed_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restart with changed upstream drops stored credential."""
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    _write_credential_config(config_file, "{env:DEEPSEEK_API_KEY}")
    manifest = _upstream_manifest()

    # First apply - stores credential with upstream
    mutation = apply_provider_scope(manifest)
    assert mutation is not None

    # Revert without backup - captures credential
    revert_provider_scope(mutation, manifest, restore_backup=False)
    assert manifest.provider_options["opencode"] == {
        "apiKey": "{env:DEEPSEEK_API_KEY}",
        "_upstream": "https://api.deepseek.com/v1",
        "_upstream_v": 2,
    }

    # Change upstream
    manifest.base_env = {"OPENAI_TARGET_API_URL": "https://different-upstream.example/v1"}

    # Re-apply with changed upstream - credential should be dropped
    apply_provider_scope(manifest)

    options = _headroom_options(config_file)
    assert "apiKey" not in options


def test_host_case_difference_matches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Host case difference still matches on install apply and restart."""
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    _write_credential_config(config_file, "{env:DEEPSEEK_API_KEY}")
    manifest = _upstream_manifest()
    manifest.base_env = {"OPENAI_TARGET_API_URL": "https://api.DEEPSEEK.com/v1"}

    mutation = apply_provider_scope(manifest)
    assert mutation is not None
    revert_provider_scope(mutation, manifest, restore_backup=False)
    assert manifest.provider_options["opencode"]["_upstream"] == "https://api.deepseek.com/v1"
    assert manifest.provider_options["opencode"]["_upstream_v"] == 2

    apply_provider_scope(manifest)
    assert _headroom_options(config_file)["apiKey"] == "{env:DEEPSEEK_API_KEY}"


def test_path_case_difference_drops_on_install_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/V1 vs /v1 is rejected on install apply."""
    from headroom.cli import install as cli_install

    config_file = _opencode_config_file(tmp_path, monkeypatch)
    _write_credential_config(config_file, "{env:DEEPSEEK_API_KEY}")
    existing = _upstream_manifest()
    existing.targets = ["opencode"]
    mutation = apply_provider_scope(existing)
    assert mutation is not None
    existing.mutations = [mutation]

    new = _upstream_manifest()
    new.targets = ["opencode"]
    new.base_env = {"OPENAI_TARGET_API_URL": "https://api.deepseek.com/V1"}

    monkeypatch.setattr(cli_install, "load_manifest", lambda profile: existing)
    monkeypatch.setattr(cli_install, "_save_apply_manifest", lambda manifest: None)
    monkeypatch.setattr(cli_install, "save_manifest", lambda manifest: None)
    monkeypatch.setattr(cli_install, "install_supervisor", lambda manifest, **kwargs: [])
    monkeypatch.setattr(cli_install, "_start_deployment", lambda manifest: None)
    monkeypatch.setattr(cli_install, "_stop_deployment", lambda manifest: None)
    monkeypatch.setattr(cli_install, "remove_supervisor", lambda manifest: None)
    monkeypatch.setattr(cli_install, "delete_manifest", lambda profile: None)

    cli_install._apply_manifest(new)

    assert "opencode" not in new.provider_options
    assert "apiKey" not in _headroom_options(config_file)


def test_path_case_difference_drops_on_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/V1 vs /v1 is rejected on restart."""
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    _write_credential_config(config_file, "{env:DEEPSEEK_API_KEY}")
    manifest = _upstream_manifest()

    mutation = apply_provider_scope(manifest)
    assert mutation is not None
    revert_provider_scope(mutation, manifest, restore_backup=False)

    manifest.base_env = {"OPENAI_TARGET_API_URL": "https://api.deepseek.com/V1"}

    apply_provider_scope(manifest)
    assert "apiKey" not in _headroom_options(config_file)


def test_query_difference_drops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Query difference is rejected."""
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    _write_credential_config(config_file, "{env:DEEPSEEK_API_KEY}")
    manifest = _upstream_manifest()

    mutation = apply_provider_scope(manifest)
    assert mutation is not None
    revert_provider_scope(mutation, manifest, restore_backup=False)

    manifest.base_env = {"OPENAI_TARGET_API_URL": "https://api.deepseek.com/v1?foo=bar"}

    apply_provider_scope(manifest)
    assert "apiKey" not in _headroom_options(config_file)


def test_v1_lowercase_binding_migrates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A v1 binding migrates to v2 when nothing was lost."""
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    _write_credential_config(config_file, "{env:DEEPSEEK_API_KEY}")
    manifest = _upstream_manifest()

    mutation = apply_provider_scope(manifest)
    assert mutation is not None
    revert_provider_scope(mutation, manifest, restore_backup=False)

    # Simulate a v1 binding (no _upstream_v)
    stored = manifest.provider_options["opencode"]
    del stored["_upstream_v"]
    stored["_upstream"] = "https://api.deepseek.com/v1"

    apply_provider_scope(manifest)
    assert manifest.provider_options["opencode"]["_upstream_v"] == 2
    assert _headroom_options(config_file)["apiKey"] == "{env:DEEPSEEK_API_KEY}"


def test_v1_binding_vs_mixed_case_path_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A v1 binding against a mixed-case path is dropped."""
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    _write_credential_config(config_file, "{env:DEEPSEEK_API_KEY}")
    manifest = _upstream_manifest()
    manifest.base_env = {"OPENAI_TARGET_API_URL": "https://api.deepseek.com/V1"}

    mutation = apply_provider_scope(manifest)
    assert mutation is not None
    revert_provider_scope(mutation, manifest, restore_backup=False)

    # Simulate a v1 binding (no _upstream_v)
    stored = manifest.provider_options["opencode"]
    del stored["_upstream_v"]
    stored["_upstream"] = "https://api.deepseek.com/v1"

    apply_provider_scope(manifest)
    assert "apiKey" not in _headroom_options(config_file)


def test_legacy_untagged_state_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy state without _upstream is treated as bound to current upstream."""
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    _write_credential_config(config_file, "{env:DEEPSEEK_API_KEY}")
    manifest = _upstream_manifest()

    # First apply - stores credential with upstream
    mutation = apply_provider_scope(manifest)
    assert mutation is not None

    # Revert without backup - captures credential
    revert_provider_scope(mutation, manifest, restore_backup=False)

    # Manually remove _upstream to simulate legacy state
    legacy_options = manifest.provider_options["opencode"].copy()
    del legacy_options["_upstream"]
    manifest.provider_options["opencode"] = legacy_options

    # Re-apply with same upstream - legacy state should be used
    apply_provider_scope(manifest)

    assert _headroom_options(config_file)["apiKey"] == "{env:DEEPSEEK_API_KEY}"


def test_install_apply_carries_untagged_credential_for_mixed_case_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Untagged legacy state stays bound to its own upstream, path case included."""
    from headroom.cli import install as cli_install

    _opencode_config_file(tmp_path, monkeypatch)
    url = "https://gateway.example/Tenant/V1"
    existing = _manifest()
    existing.base_env = {"OPENAI_TARGET_API_URL": url}
    existing.targets = ["opencode"]
    existing.provider_options = {"opencode": {"apiKey": "{env:GATEWAY_KEY}"}}

    new = _manifest()
    new.base_env = {"OPENAI_TARGET_API_URL": url}
    new.targets = ["opencode"]

    monkeypatch.setattr(cli_install, "load_manifest", lambda profile: existing)
    monkeypatch.setattr(cli_install, "_save_apply_manifest", lambda manifest: None)
    monkeypatch.setattr(cli_install, "save_manifest", lambda manifest: None)
    monkeypatch.setattr(cli_install, "install_supervisor", lambda manifest, **kwargs: [])
    monkeypatch.setattr(cli_install, "_start_deployment", lambda manifest: None)
    monkeypatch.setattr(cli_install, "_stop_deployment", lambda manifest: None)
    monkeypatch.setattr(cli_install, "remove_supervisor", lambda manifest: None)
    monkeypatch.setattr(cli_install, "delete_manifest", lambda profile: None)

    cli_install._apply_manifest(new)

    assert new.provider_options["opencode"]["apiKey"] == "{env:GATEWAY_KEY}"
    assert new.provider_options["opencode"]["_upstream"] == url


def _apply_stored_only(
    monkeypatch: pytest.MonkeyPatch, old_url: str, new_url: str, stored: dict[str, Any]
) -> DeploymentManifest:
    """Run install apply where the credential lives only in the deployment state."""
    from headroom.cli import install as cli_install

    existing = _manifest()
    existing.base_env = {"OPENAI_TARGET_API_URL": old_url}
    existing.targets = ["opencode"]
    existing.provider_options = {"opencode": dict(stored)}

    new = _manifest()
    new.base_env = {"OPENAI_TARGET_API_URL": new_url}
    new.targets = ["opencode"]

    monkeypatch.setattr(cli_install, "load_manifest", lambda profile: existing)
    monkeypatch.setattr(cli_install, "_save_apply_manifest", lambda manifest: None)
    monkeypatch.setattr(cli_install, "save_manifest", lambda manifest: None)
    monkeypatch.setattr(cli_install, "install_supervisor", lambda manifest, **kwargs: [])
    monkeypatch.setattr(cli_install, "_start_deployment", lambda manifest: None)
    monkeypatch.setattr(cli_install, "_stop_deployment", lambda manifest: None)
    monkeypatch.setattr(cli_install, "remove_supervisor", lambda manifest: None)
    monkeypatch.setattr(cli_install, "delete_manifest", lambda profile: None)

    cli_install._apply_manifest(new)
    return new


def test_install_apply_drops_stored_v1_credential_from_mixed_case_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A v1 binding of a /V1 deployment cannot be trusted for a new /v1 one."""
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    new = _apply_stored_only(
        monkeypatch,
        "https://api.deepseek.com/V1",
        "https://api.deepseek.com/v1",
        {"apiKey": "{env:DEEPSEEK_API_KEY}", "_upstream": "https://api.deepseek.com/v1"},
    )

    assert "opencode" not in new.provider_options
    assert "apiKey" not in _headroom_options(config_file)


def test_install_apply_migrates_stored_v1_credential_for_same_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A v1 binding whose deployment URL lost nothing to lowercasing carries as v2."""
    config_file = _opencode_config_file(tmp_path, monkeypatch)
    new = _apply_stored_only(
        monkeypatch,
        "https://api.deepseek.com/v1",
        "https://api.deepseek.com/v1",
        {"apiKey": "{env:DEEPSEEK_API_KEY}", "_upstream": "https://api.deepseek.com/v1"},
    )

    assert new.provider_options["opencode"] == {
        "apiKey": "{env:DEEPSEEK_API_KEY}",
        "_upstream": "https://api.deepseek.com/v1",
        "_upstream_v": 2,
    }
    assert _headroom_options(config_file)["apiKey"] == "{env:DEEPSEEK_API_KEY}"


def test_apply_provider_scope_leaves_unparseable_config_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """apply_provider_scope reports an unparseable config instead of replacing it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("OPENCODE_HOME", raising=False)
    monkeypatch.delenv("OPENCODE_CONFIG", raising=False)
    config_file = tmp_path / ".config" / "opencode" / "opencode.json"
    config_file.parent.mkdir(parents=True)
    original = '{"model": "gpt-4o" "oops": 1}'
    config_file.write_text(original, encoding="utf-8")

    with pytest.raises(click.ClickException, match="could not parse"):
        apply_provider_scope(_manifest())

    assert config_file.read_text(encoding="utf-8") == original
