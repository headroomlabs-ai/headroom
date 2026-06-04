from __future__ import annotations

from types import SimpleNamespace

import pytest

from headroom import copilot_macos_keychain


def test_read_copilot_oauth_token_uses_security(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        assert kwargs["capture_output"] is True
        assert kwargs["timeout"] == 5
        return SimpleNamespace(returncode=0, stdout="gho-keychain\n")

    monkeypatch.setattr(copilot_macos_keychain.sys, "platform", "darwin")
    monkeypatch.setenv("GITHUB_COPILOT_KEYCHAIN_SERVICE", "GitHub Copilot")
    monkeypatch.setenv("GITHUB_COPILOT_KEYCHAIN_ACCOUNT", "chopratejas")
    monkeypatch.setattr(copilot_macos_keychain.subprocess, "run", fake_run)

    assert copilot_macos_keychain.read_copilot_oauth_token(host="github.com") == "gho-keychain"
    assert calls[0] == ["security", "find-generic-password", "-s", "GitHub Copilot", "-w"]


def test_read_copilot_oauth_token_returns_none_off_macos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(copilot_macos_keychain.sys, "platform", "linux")

    assert copilot_macos_keychain.read_copilot_oauth_token() is None


def test_candidate_security_commands_include_account_specific_lookup() -> None:
    commands = copilot_macos_keychain._candidate_security_commands(
        "github.com",
        ["GitHub Copilot"],
        ["chopratejas"],
    )

    assert ["security", "find-generic-password", "-s", "GitHub Copilot", "-w"] in commands
    assert [
        "security",
        "find-generic-password",
        "-s",
        "GitHub Copilot",
        "-a",
        "chopratejas",
        "-w",
    ] in commands


def test_read_copilot_oauth_token_tries_copilot_cli_host_login_account(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls: list[list[str]] = []
    copilot_home = tmp_path / ".copilot"
    copilot_home.mkdir()
    (copilot_home / "config.json").write_text(
        '{"lastLoggedInUser":{"host":"https://github.com","login":"chopratejas"}}',
        encoding="utf-8",
    )

    def fake_run(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        stdout = "gho-keychain\n" if command == expected else ""
        return type("CompletedProcess", (), {"returncode": 0 if stdout else 44, "stdout": stdout})()

    expected = [
        "security",
        "find-generic-password",
        "-s",
        "copilot-cli",
        "-a",
        "https://github.com:chopratejas",
        "-w",
    ]
    monkeypatch.setattr(copilot_macos_keychain.sys, "platform", "darwin")
    monkeypatch.setenv("COPILOT_HOME", str(copilot_home))
    monkeypatch.delenv("GITHUB_COPILOT_KEYCHAIN_SERVICE", raising=False)
    monkeypatch.delenv("GITHUB_COPILOT_KEYCHAIN_ACCOUNT", raising=False)
    monkeypatch.setattr(copilot_macos_keychain.subprocess, "run", fake_run)

    assert copilot_macos_keychain.read_copilot_oauth_token(host="github.com") == "gho-keychain"
    assert expected in calls
