from __future__ import annotations

from types import SimpleNamespace

from headroom import copilot_linux_secret


def test_read_copilot_oauth_token_uses_secret_tool(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN003
        calls.append(command)
        assert kwargs["capture_output"] is True
        return SimpleNamespace(returncode=0, stdout="gho-secret\n")

    monkeypatch.setattr(copilot_linux_secret.sys, "platform", "linux")
    monkeypatch.setattr(copilot_linux_secret, "_read_copilot_config_login", lambda: "octo")
    monkeypatch.setattr(copilot_linux_secret.subprocess, "run", fake_run)

    assert copilot_linux_secret.read_copilot_oauth_token(host="github.com") == "gho-secret"
    assert calls[0] == ["secret-tool", "lookup", "service", "copilot-cli"]


def test_read_copilot_oauth_token_returns_none_off_linux(monkeypatch) -> None:
    monkeypatch.setattr(copilot_linux_secret.sys, "platform", "darwin")

    assert copilot_linux_secret.read_copilot_oauth_token() is None


def test_candidate_secret_tool_commands_include_login_specific_lookup() -> None:
    commands = copilot_linux_secret._candidate_secret_tool_commands(
        "secret-tool",
        "github.com",
        "octo",
    )

    assert [
        "secret-tool",
        "lookup",
        "service",
        "copilot-cli",
        "account",
        "https://github.com:octo",
    ] in commands


def test_read_copilot_oauth_token_tries_until_match(monkeypatch) -> None:
    expected = [
        "secret-tool",
        "lookup",
        "service",
        "copilot-cli",
        "account",
        "https://github.com:octo",
    ]
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ANN001, ANN003
        calls.append(command)
        stdout = "gho-secret\n" if command == expected else ""
        return SimpleNamespace(returncode=0, stdout=stdout)

    monkeypatch.setattr(copilot_linux_secret.sys, "platform", "linux")
    monkeypatch.setattr(copilot_linux_secret, "_read_copilot_config_login", lambda: "octo")
    monkeypatch.setattr(copilot_linux_secret.subprocess, "run", fake_run)

    assert copilot_linux_secret.read_copilot_oauth_token(host="github.com") == "gho-secret"
    assert expected in calls
