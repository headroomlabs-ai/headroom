"""CLI coverage for transparent VS Code Copilot setup and undo."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

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
