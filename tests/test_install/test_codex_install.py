from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from headroom.install.models import DeploymentManifest
from headroom.providers.codex.install import (
    _codex_login_status,
    apply_provider_scope,
    build_provider_section,
    codex_uses_chatgpt_auth,
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
