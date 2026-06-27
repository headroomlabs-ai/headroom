"""Tests for ``headroom wrap grok`` and ``headroom unwrap grok``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main
from headroom.providers.grok import (
    DEFAULT_API_URL,
    DEFAULT_MODEL,
    HEADROOM_MODEL_ALIAS,
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _set_test_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))


def test_wrap_help_lists_grok(runner: CliRunner) -> None:
    result = runner.invoke(main, ["wrap", "--help"])

    assert result.exit_code == 0, result.output
    assert "headroom wrap grok" in result.output


def test_unwrap_help_lists_grok(runner: CliRunner) -> None:
    result = runner.invoke(main, ["unwrap", "--help"])

    assert result.exit_code == 0, result.output
    assert "grok" in result.output


def test_wrap_grok_injects_config_backed_model_alias(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs: object) -> None:
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="grok"),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            ["wrap", "grok", "--no-rtk", "--", "--model", "grok-beta"],
        )

    assert result.exit_code == 0, result.output
    config_file = tmp_path / ".grok" / "config.toml"
    assert config_file.exists()
    content = config_file.read_text(encoding="utf-8")
    assert "[model.headroom-grok-proxy]" in content
    assert 'model = "grok-beta"' in content
    assert 'base_url = "http://127.0.0.1:8787/v1"' in content
    assert 'name = "Headroom proxy"' in content
    assert "env_key" not in content

    env = captured["env"]
    assert isinstance(env, dict)
    assert "GROK_PROXY_URL" not in env
    assert captured["tool_label"] == "GROK"
    assert captured["agent_type"] == "grok"
    assert captured["args"] == ("--model", HEADROOM_MODEL_ALIAS)
    assert captured["env_vars_display"] == [
        f"{config_file}: model.{HEADROOM_MODEL_ALIAS} -> http://127.0.0.1:8787/v1",
        "upstream model = grok-beta",
    ]
    assert captured["openai_api_url"] == DEFAULT_API_URL


def test_wrap_grok_resolves_existing_model_alias(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    grok_dir = tmp_path / ".grok"
    grok_dir.mkdir(parents=True)
    (grok_dir / "config.toml").write_text(
        '[model.my-model]\nmodel = "grok-beta"\nbase_url = "https://api.x.ai/v1"\n',
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs: object) -> None:
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="grok"),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            ["wrap", "grok", "--no-rtk", "--", "--model", "my-model"],
        )

    assert result.exit_code == 0, result.output
    content = (grok_dir / "config.toml").read_text(encoding="utf-8")
    assert 'model = "grok-beta"' in content
    assert captured["args"] == ("--model", HEADROOM_MODEL_ALIAS)
    assert captured["env_vars_display"][-1] == "upstream model = grok-beta"


def test_wrap_grok_uses_default_model_when_none_requested(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs: object) -> None:
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="grok"),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(main, ["wrap", "grok", "--no-rtk"])

    assert result.exit_code == 0, result.output
    content = (tmp_path / ".grok" / "config.toml").read_text(encoding="utf-8")
    assert f'model = "{DEFAULT_MODEL}"' in content
    assert captured["args"] == ("--model", HEADROOM_MODEL_ALIAS)
    assert captured["env_vars_display"][-1] == f"upstream model = {DEFAULT_MODEL}"


def test_wrap_grok_prepare_only_skips_binary_lookup(runner: CliRunner) -> None:
    with (
        patch("headroom.cli.wrap.shutil.which") as which_mock,
        patch("headroom.cli.wrap._launch_tool") as launch_mock,
    ):
        result = runner.invoke(main, ["wrap", "grok", "--prepare-only", "--no-rtk"])

    assert result.exit_code == 0, result.output
    which_mock.assert_not_called()
    launch_mock.assert_not_called()


def test_wrap_grok_prepare_only_sets_up_rtk_instructions(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with (
        patch.object(wrap_mod, "_selected_context_tool", return_value=wrap_mod._CONTEXT_TOOL_RTK),
        patch.object(wrap_mod, "_ensure_rtk_binary", return_value="rtk"),
        patch.object(wrap_mod, "_inject_rtk_instructions") as inject_mock,
        patch.object(wrap_mod.shutil, "which") as which_mock,
    ):
        result = runner.invoke(main, ["wrap", "grok", "--prepare-only"])

    assert result.exit_code == 0, result.output
    inject_mock.assert_called_once_with(tmp_path / "CONVENTIONS.md", verbose=False)
    which_mock.assert_not_called()


def test_wrap_grok_prepare_only_sets_up_lean_ctx(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with (
        patch.object(
            wrap_mod, "_selected_context_tool", return_value=wrap_mod._CONTEXT_TOOL_LEAN_CTX
        ),
        patch.object(wrap_mod, "_setup_lean_ctx_agent") as setup_mock,
        patch.object(wrap_mod.shutil, "which") as which_mock,
    ):
        result = runner.invoke(main, ["wrap", "grok", "--prepare-only"])

    assert result.exit_code == 0, result.output
    setup_mock.assert_called_once_with("grok", verbose=False)
    which_mock.assert_not_called()


def test_wrap_grok_fails_when_binary_missing(runner: CliRunner) -> None:
    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        result = runner.invoke(main, ["wrap", "grok", "--no-rtk"])

    assert result.exit_code == 1
    assert "Error: 'grok' not found in PATH." in result.output
    assert "Install Grok Build CLI: https://docs.x.ai/build/overview" in result.output


def test_grok_config_restore_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_test_home(monkeypatch, tmp_path)
    grok_dir = tmp_path / ".grok"
    grok_dir.mkdir(parents=True)
    config_file = grok_dir / "config.toml"
    original = '[model.custom]\nmodel = "grok-custom"\n'
    config_file.write_text(original, encoding="utf-8")

    wrap_mod._inject_grok_provider_config(8787, "grok-beta")

    status, restored_file = wrap_mod._restore_grok_provider_config()

    assert status == "restored"
    assert restored_file == config_file
    assert config_file.read_text(encoding="utf-8") == original
    assert not (grok_dir / "config.toml.headroom-backup").exists()


def test_grok_config_restore_removes_generated_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_test_home(monkeypatch, tmp_path)

    wrap_mod._inject_grok_provider_config(8787, "grok-beta")
    status, config_file = wrap_mod._restore_grok_provider_config()

    assert status == "removed"
    assert config_file == tmp_path / ".grok" / "config.toml"
    assert not config_file.exists()


def test_unwrap_grok_restores_prior_config_and_stops_proxy(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_test_home(monkeypatch, tmp_path)
    grok_dir = tmp_path / ".grok"
    grok_dir.mkdir(parents=True)
    config_file = grok_dir / "config.toml"
    original = '[model.custom]\nmodel = "grok-custom"\n'
    config_file.write_text(original, encoding="utf-8")
    wrap_mod._inject_grok_provider_config(8787, "grok-beta")

    with patch(
        "headroom.cli.wrap._stop_local_proxy_for_unwrap", return_value="stopped"
    ) as stop_proxy:
        result = runner.invoke(main, ["unwrap", "grok", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert f"Restored prior {config_file} from pre-wrap backup." in result.output
    assert config_file.read_text(encoding="utf-8") == original
    assert "Stopped local Headroom proxy on port 9999" in result.output
    stop_proxy.assert_called_once_with(9999)


def test_unwrap_grok_is_noop_when_never_wrapped(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_test_home(monkeypatch, tmp_path)

    with patch("headroom.cli.wrap._stop_local_proxy_for_unwrap") as stop_proxy:
        result = runner.invoke(main, ["unwrap", "grok", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert "Nothing to undo:" in result.output
    stop_proxy.assert_not_called()
