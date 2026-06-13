from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from headroom.cli import main


def test_copilot_auth_login_saves_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    auth_file = tmp_path / "copilot_auth.json"
    monkeypatch.setenv("HEADROOM_COPILOT_AUTH_FILE", str(auth_file))
    monkeypatch.setattr(
        "headroom.cli.copilot_auth.start_copilot_device_authorization",
        lambda domain: {
            "verification_uri": "https://github.com/login/device",
            "user_code": "ABCD-1234",
            "device_code": "device-code",
            "interval": 1,
            "expires_in": 900,
        },
    )
    monkeypatch.setattr(
        "headroom.cli.copilot_auth.poll_copilot_device_authorization",
        lambda device_code, *, domain, interval, expires_in: "gho-headroom",
    )

    result = CliRunner().invoke(main, ["copilot-auth", "login"])

    assert result.exit_code == 0, result.output
    assert "https://github.com/login/device" in result.output
    assert "ABCD-1234" in result.output
    assert "gho-headroom" not in result.output
    assert auth_file.exists()


def test_copilot_auth_status_reports_missing_login(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HEADROOM_COPILOT_AUTH_FILE", str(tmp_path / "missing.json"))

    result = CliRunner().invoke(main, ["copilot-auth", "status"])

    assert result.exit_code == 0, result.output
    assert "Status: not logged in" in result.output
