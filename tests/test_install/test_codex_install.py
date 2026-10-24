from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

from headroom.install.models import DeploymentManifest
from headroom.providers.codex.install import (
    CodexAuthConfigError,
    _codex_login_status,
    apply_provider_scope,
    build_codex_auth_config,
    build_provider_section,
    codex_auth_helper_path,
    codex_uses_chatgpt_auth,
    revert_provider_scope,
)


def _manifest(tmp_path: Path) -> DeploymentManifest:
    return DeploymentManifest(
        profile="test",
        preset="persistent-service",
        runtime_kind="python",
        supervisor_kind="service",
        scope="provider",
        provider_mode="manual",
        targets=["codex"],
        port=8787,
        host="127.0.0.1",
        backend="anthropic",
        memory_db_path=str(tmp_path / "memory.db"),
        tool_envs={},
    )


def _login_status(
    stdout: str = "",
    *,
    stderr: str = "",
    returncode: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_keyring_chatgpt_auth_emits_provider_flag(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('cli_auth_credentials_store = "keyring"\n', encoding="utf-8")
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config)
    monkeypatch.setattr(
        "headroom.providers.codex.install.run",
        lambda *args, **kwargs: _login_status(stderr="Logged in using ChatGPT\n"),
    )

    apply_provider_scope(_manifest(tmp_path))

    assert "requires_openai_auth = true" in config.read_text(encoding="utf-8")


def test_keyring_non_chatgpt_auth_keeps_provider_flag_off(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    config.write_text('cli_auth_credentials_store = "keyring"\n', encoding="utf-8")
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config)
    monkeypatch.setattr(
        "headroom.providers.codex.install.run",
        lambda *args, **kwargs: _login_status(stderr="Logged in using API key\n"),
    )

    apply_provider_scope(_manifest(tmp_path))

    assert "requires_openai_auth" not in config.read_text(encoding="utf-8")


def test_auto_store_chatgpt_auth_is_detected(monkeypatch, tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    (tmp_path / "config.toml").write_text('cli_auth_credentials_store = "auto"\n', encoding="utf-8")
    monkeypatch.setattr(
        "headroom.providers.codex.install.run",
        lambda *args, **kwargs: _login_status(stderr="Logged in using ChatGPT\n"),
    )

    assert codex_uses_chatgpt_auth(auth) is True


def test_file_backed_auth_preserves_existing_modes(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text('{"auth_mode": "CHATGPT"}', encoding="utf-8")
    assert codex_uses_chatgpt_auth(auth) is True
    auth.write_text('{"auth_mode": "apikey", "tokens": {"account_id": "acct"}}', encoding="utf-8")
    assert codex_uses_chatgpt_auth(auth) is False


def test_api_key_auth_is_persisted_as_a_codex_command(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.toml"
    auth = tmp_path / "auth.json"
    config.write_text('model = "gpt-5"\n', encoding="utf-8")
    auth.write_text('{"OPENAI_API_KEY": "sk-test-only"}', encoding="utf-8")
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config)

    mutation = apply_provider_scope(_manifest(tmp_path))
    assert mutation is not None

    content = config.read_text(encoding="utf-8")
    helper = codex_auth_helper_path(auth, config_path=config)
    assert "auth = { command =" in content
    assert tomllib.loads(content)["model_providers"]["headroom"]["auth"]["args"] == [
        str(helper.resolve())
    ]
    assert "sk-test-only" not in content
    assert helper.exists()

    revert_provider_scope(mutation, _manifest(tmp_path))

    assert not helper.exists()


def test_persistent_install_helper_collision_preserves_config(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.toml"
    auth = tmp_path / "auth.json"
    auth.write_text('{"OPENAI_API_KEY": "sk-test-only"}', encoding="utf-8")
    original = 'model_provider = "openai"\n'
    config.write_text(original, encoding="utf-8")
    helper = codex_auth_helper_path(auth, config_path=config)
    helper.write_text("user content", encoding="utf-8")
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config)

    with pytest.raises(CodexAuthConfigError, match="Codex provider configuration was not updated"):
        apply_provider_scope(_manifest(tmp_path))

    assert config.read_text(encoding="utf-8") == original
    assert helper.read_text(encoding="utf-8") == "user content"


def test_persistent_uninstall_preserves_project_authentication(tmp_path: Path, monkeypatch) -> None:
    from headroom.cli.init import _ensure_codex_provider

    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    config = tmp_path / "config.toml"
    auth = tmp_path / "auth.json"
    auth.write_text('{"OPENAI_API_KEY": "sk-test-only"}', encoding="utf-8")
    monkeypatch.setattr("headroom.providers.codex.install.codex_config_path", lambda: config)
    project_config = tmp_path / "project" / ".codex" / "config.toml"
    _ensure_codex_provider(project_config, 8787)
    project_helper = codex_auth_helper_path(auth, config_path=project_config)
    mutation = apply_provider_scope(_manifest(tmp_path))
    assert mutation is not None

    revert_provider_scope(mutation, _manifest(tmp_path))

    assert project_helper.exists()
    assert tomllib.loads(project_config.read_text(encoding="utf-8"))["model_providers"]["headroom"][
        "auth"
    ]["args"] == [str(project_helper.resolve())]
    assert not codex_auth_helper_path(auth, config_path=config).exists()


def test_uninstall_retains_legacy_shared_helper(tmp_path: Path) -> None:
    from headroom.install.models import ManagedMutation

    auth = tmp_path / "auth.json"
    auth.write_text('{"OPENAI_API_KEY": "sk-test-only"}', encoding="utf-8")
    build_codex_auth_config(auth)
    legacy_helper = tmp_path / ".headroom-codex-auth.py"
    legacy_helper.write_text(codex_auth_helper_path(auth).read_text(encoding="utf-8"))
    config = tmp_path / "config.toml"
    config.write_text(
        "# --- Headroom persistent provider ---\n"
        "[model_providers.headroom]\n"
        f'auth = {{ command = "python", args = [{json.dumps(str(legacy_helper), ensure_ascii=False)}] }}\n'
        "# --- end Headroom persistent provider ---\n",
        encoding="utf-8",
    )

    revert_provider_scope(
        ManagedMutation(target="codex", kind="toml-block", path=str(config)), _manifest(tmp_path)
    )

    assert legacy_helper.exists()


def test_legacy_file_backed_account_id_stays_supported(tmp_path: Path) -> None:
    auth = tmp_path / "auth.json"
    auth.write_text('{"tokens": {"account_id": "acct"}}', encoding="utf-8")
    assert codex_uses_chatgpt_auth(auth) is True


def test_missing_or_failed_login_status_fails_closed(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text(
        'cli_auth_credentials_store = "keyring"\n', encoding="utf-8"
    )
    monkeypatch.setattr(
        "headroom.providers.codex.install.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError()),
    )
    assert codex_uses_chatgpt_auth(tmp_path / "auth.json") is False


def test_login_status_probe_uses_codex_contract(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict]] = []

    def probe(command: list[str], **kwargs):
        calls.append((command, kwargs))
        return _login_status(stderr="Logged in using ChatGPT\n")

    monkeypatch.setattr("headroom.providers.codex.install.run", probe)

    assert _codex_login_status(tmp_path) is True
    assert calls[0][0] == ["codex", "login", "status"]
    assert calls[0][1]["timeout"] == 3
    assert calls[0][1]["env"]["CODEX_HOME"] == str(tmp_path)
    assert calls[0][1]["capture_output"] is True


def test_login_status_probe_accepts_stdout_or_stderr(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "headroom.providers.codex.install.run",
        lambda *args, **kwargs: _login_status(stdout="Logged in using ChatGPT\n"),
    )

    assert _codex_login_status(tmp_path) is True


def test_provider_section_still_emits_flag_when_requested() -> None:
    assert "requires_openai_auth = true" in build_provider_section(
        port=8787, name="Headroom", requires_openai_auth=True
    )
