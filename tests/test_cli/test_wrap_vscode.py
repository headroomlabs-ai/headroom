"""CLI coverage for transparent VS Code Copilot setup and undo."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.copilot_auth import CopilotSubscriptionTokenResolution


def _resolution() -> CopilotSubscriptionTokenResolution:
    return CopilotSubscriptionTokenResolution(
        token="copilot-token",
        source="test",
        confidence="test",
        api_url="https://api.githubcopilot.com",
        token_fingerprint="sha256:test",
    )


def test_wrap_vscode_configures_actual_port_and_seeds_subscription(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    captured = {}

    def fake_watcher(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        kwargs["print_setup_lines"](9999)

    with (
        patch(
            "headroom.cli.wrap._require_copilot_subscription_resolution", return_value=_resolution()
        ),
        patch("headroom.cli.wrap._run_proxy_only_watcher", side_effect=fake_watcher),
    ):
        result = CliRunner().invoke(main, ["wrap", "vscode", "--settings-file", str(path)])

    assert result.exit_code == 0, result.output
    settings = path.read_text(encoding="utf-8")
    assert "http://127.0.0.1:9999/" in settings
    assert "model" not in settings.lower()
    assert "normal model picker" in result.output
    assert captured["openai_api_url"] == "https://api.githubcopilot.com"
    assert captured["copilot_api_token"] == "copilot-token"


def _run_wrap_vscode(*args: str) -> str:
    def fake_watcher(**kwargs):  # noqa: ANN003, ANN202
        kwargs["print_setup_lines"](8787)

    with (
        patch(
            "headroom.cli.wrap._require_copilot_subscription_resolution", return_value=_resolution()
        ),
        patch("headroom.cli.wrap._run_proxy_only_watcher", side_effect=fake_watcher),
    ):
        result = CliRunner().invoke(main, ["wrap", "vscode", *args])
    assert result.exit_code == 0, result.output
    return result.output


def _write_profiles(user_dir: Path, *profiles: dict[str, object]) -> None:
    storage = user_dir / "globalStorage" / "storage.json"
    storage.parent.mkdir(parents=True)
    storage.write_text(json.dumps({"userDataProfiles": list(profiles)}), encoding="utf-8")


def _default_user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for var in ("APPDATA", "HOME", "USERPROFILE", "XDG_CONFIG_HOME"):
        monkeypatch.setenv(var, str(tmp_path))
    from headroom.providers.copilot.vscode import vscode_user_dir

    return vscode_user_dir()


def test_wrap_vscode_names_profiles_that_ignore_default_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_dir = _default_user_dir(tmp_path, monkeypatch)
    _write_profiles(
        user_dir,
        {"location": "6c87cdb4", "name": "Work"},
        {"location": "1a2b", "name": "Shared", "useDefaultFlags": {"settings": True}},
    )
    work_settings = user_dir / "profiles" / "6c87cdb4" / "settings.json"

    implicit = _run_wrap_vscode()
    assert "overrideCapiUrl" in (user_dir / "settings.json").read_text(encoding="utf-8")
    # A window in the "Work" profile reads only its own settings.json, so the
    # block above never reaches it; say so instead of reporting plain success.
    assert "VS Code profile 'Work'" in implicit
    assert str(work_settings) in implicit
    # A profile that shares the Default profile's settings already sees the block.
    assert "Shared" not in implicit
    # Naming the Default file explicitly configures the same file, so it warns too.
    explicit = _run_wrap_vscode("--settings-file", str(user_dir / "settings.json"))
    assert "VS Code profile 'Work'" in explicit
    # Following the advice routes the profile, and nothing is left to warn about.
    assert "Warning" not in _run_wrap_vscode("--settings-file", str(work_settings))
    assert "Warning" not in _run_wrap_vscode()


def test_wrap_vscode_checks_the_profiles_of_the_user_dir_it_configures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_profiles(
        _default_user_dir(tmp_path, monkeypatch), {"location": "6c87cdb4", "name": "Work"}
    )
    # e.g. VS Code Insiders, VSCodium, or a portable install with its own profiles.
    other_user_dir = tmp_path / "Code - Insiders" / "User"
    _write_profiles(other_user_dir, {"location": "-2b3c4d", "name": "Beta"})

    output = _run_wrap_vscode("--settings-file", str(other_user_dir / "settings.json"))

    assert "VS Code profile 'Beta'" in output
    assert "Work" not in output


def test_wrap_vscode_no_configure_prints_transparent_settings(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"

    def fake_watcher(**kwargs):  # noqa: ANN003, ANN202
        kwargs["print_setup_lines"](8787)

    with (
        patch(
            "headroom.cli.wrap._require_copilot_subscription_resolution", return_value=_resolution()
        ),
        patch("headroom.cli.wrap._run_proxy_only_watcher", side_effect=fake_watcher),
    ):
        result = CliRunner().invoke(
            main,
            ["wrap", "vscode", "--no-configure", "--settings-file", str(path)],
        )

    assert result.exit_code == 0, result.output
    assert not path.exists()
    assert "overrideProxyUrl" in result.output
    assert "overrideCapiUrl" in result.output
    # No `overrideAuthType`: the setting does not exist in the modern Copilot
    # Chat extension, so printing it told users to add a key VS Code flags as
    # unknown and which does nothing (#3076).
    assert "overrideAuthType" not in result.output


def test_unwrap_vscode_removes_only_managed_settings(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    original = '{\n  "editor.fontSize": 14\n}\n'
    path.write_text(original, encoding="utf-8")
    from headroom.providers.copilot.vscode import configure_vscode_proxy_settings

    configure_vscode_proxy_settings(path, "http://127.0.0.1:8787")
    result = CliRunner().invoke(main, ["unwrap", "vscode", "--settings-file", str(path)])
    assert result.exit_code == 0, result.output
    assert path.read_text(encoding="utf-8") == original
