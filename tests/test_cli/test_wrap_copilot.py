"""Tests for `headroom wrap copilot` command."""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import patch

import click
import pytest
from click.testing import CliRunner

from headroom.copilot_auth import DEFAULT_API_URL


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def wrap_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[types.ModuleType, click.Group]:
    headroom_pkg = sys.modules.get("headroom")
    saved_headroom_cli_attr = (
        headroom_pkg.cli if headroom_pkg is not None and hasattr(headroom_pkg, "cli") else None
    )
    saved_modules = {
        name: sys.modules.get(name)
        for name in ("headroom.cli", "headroom.cli.main", "headroom.cli.wrap")
    }

    fake_main_module = types.ModuleType("headroom.cli.main")
    fake_main_module.main = click.Group()
    sys.modules["headroom.cli.main"] = fake_main_module
    sys.modules.pop("headroom.cli", None)
    sys.modules.pop("headroom.cli.wrap", None)

    wrap_cli = importlib.import_module("headroom.cli.wrap")
    monkeypatch.setattr(wrap_cli, "_check_proxy", lambda _port: False)

    try:
        yield wrap_cli, fake_main_module.main
    finally:
        for name in ("headroom.cli.wrap", "headroom.cli.main", "headroom.cli"):
            sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
        if saved_modules["headroom.cli"] is not None:
            cli_pkg = saved_modules["headroom.cli"]
            if saved_modules["headroom.cli.main"] is not None:
                cli_pkg.main = saved_modules["headroom.cli.main"]
            if saved_modules["headroom.cli.wrap"] is not None:
                cli_pkg.wrap = saved_modules["headroom.cli.wrap"]
        if headroom_pkg is not None:
            if saved_headroom_cli_attr is None:
                if hasattr(headroom_pkg, "cli"):
                    delattr(headroom_pkg, "cli")
            else:
                headroom_pkg.cli = saved_headroom_cli_attr


def test_wrap_copilot_auto_anthropic_injects_instructions(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrap_cli, main = wrap_modules
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-dummy")
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="copilot"),
        patch("headroom.cli.wrap.has_oauth_auth", return_value=False),
        patch("headroom.cli.wrap._ensure_rtk_binary", return_value=Path("/tmp/rtk")),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            ["wrap", "copilot", "--", "--model", "claude-sonnet-4-20250514"],
        )

    assert result.exit_code == 0, result.output
    instructions = tmp_path / ".github" / "copilot-instructions.md"
    assert instructions.exists()
    content = instructions.read_text()
    assert wrap_cli._RTK_MARKER in content
    assert "RTK (Rust Token Killer)" in content

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["COPILOT_PROVIDER_TYPE"] == "anthropic"
    assert env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787"
    assert "COPILOT_PROVIDER_WIRE_API" not in env
    assert captured["agent_type"] == "copilot"
    assert captured["tool_label"] == "COPILOT"
    assert captured["args"] == ("--model", "claude-sonnet-4-20250514")


def test_wrap_copilot_openai_backend_sets_completions_env(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wrap_cli, main = wrap_modules
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="copilot"),
        patch("headroom.cli.wrap.has_oauth_auth", return_value=False),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            [
                "wrap",
                "copilot",
                "--no-rtk",
                "--backend",
                "anyllm",
                "--anyllm-provider",
                "groq",
                "--region",
                "us-central1",
                "--",
                "--model",
                "gpt-4o",
            ],
        )

    assert result.exit_code == 0, result.output

    env = captured["env"]
    assert isinstance(env, dict)
    assert env["COPILOT_PROVIDER_TYPE"] == "openai"
    assert env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert env["COPILOT_PROVIDER_WIRE_API"] == "completions"
    assert captured["backend"] == "anyllm"
    assert captured["anyllm_provider"] == "groq"
    assert captured["region"] == "us-central1"
    assert captured["args"] == ("--model", "gpt-4o")


def test_wrap_copilot_auto_detects_running_proxy_backend(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wrap_cli, main = wrap_modules
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="copilot"),
        patch("headroom.cli.wrap.has_oauth_auth", return_value=False),
        patch("headroom.cli.wrap._check_proxy", return_value=True),
        patch("headroom.cli.wrap._detect_running_proxy_backend", return_value="anyllm"),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            ["wrap", "copilot", "--no-rtk", "--", "--model", "gpt-4o"],
        )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["COPILOT_PROVIDER_TYPE"] == "openai"
    assert env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert env["COPILOT_PROVIDER_WIRE_API"] == "completions"


def test_wrap_copilot_prefers_existing_oauth_session(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wrap_cli, main = wrap_modules
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-dummy")
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="copilot"):
        with patch("headroom.cli.wrap.resolve_client_bearer_token", return_value="gho-existing"):
            with patch("headroom.cli.wrap.has_oauth_auth", return_value=True):
                with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
                    result = runner.invoke(
                        main,
                        ["wrap", "copilot", "--no-rtk", "--", "--model", "claude-sonnet-4.6"],
                    )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["COPILOT_PROVIDER_TYPE"] == "openai"
    assert env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert env["COPILOT_PROVIDER_WIRE_API"] == "completions"
    assert env["COPILOT_PROVIDER_BEARER_TOKEN"] == "gho-existing"
    assert "COPILOT_PROVIDER_API_KEY" not in env
    assert captured["openai_api_url"] == DEFAULT_API_URL


def test_wrap_copilot_translated_backend_still_requires_byok(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wrap_cli, main = wrap_modules
    # The point of the test is that BYOK is required even with `--backend
    # anyllm`, but the BYOK check only fires when no provider key is in
    # the environment. The test runs against the real `os.environ`, so
    # explicitly clear every key the CLI checks first.
    for var in (
        "COPILOT_PROVIDER_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "TOGETHER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    with patch("headroom.cli.wrap.shutil.which", return_value="copilot"):
        with patch("headroom.cli.wrap.has_oauth_auth", return_value=True):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "copilot",
                    "--no-rtk",
                    "--backend",
                    "anyllm",
                    "--",
                    "--model",
                    "gpt-4o",
                ],
            )

    assert result.exit_code == 1
    assert "Copilot BYOK mode requires a provider API key" in result.output


def test_wrap_copilot_rejects_wire_api_for_anthropic_provider(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    _wrap_cli, main = wrap_modules
    with patch("headroom.cli.wrap.shutil.which", return_value="copilot"):
        result = runner.invoke(
            main,
            [
                "wrap",
                "copilot",
                "--wire-api",
                "responses",
                "--",
                "--model",
                "claude-sonnet-4-20250514",
            ],
        )

    assert result.exit_code != 0
    assert "--wire-api is only valid" in result.output


def test_wrap_copilot_rejects_responses_for_translated_backends(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    _wrap_cli, main = wrap_modules
    with patch("headroom.cli.wrap.shutil.which", return_value="copilot"):
        result = runner.invoke(
            main,
            [
                "wrap",
                "copilot",
                "--backend",
                "anyllm",
                "--wire-api",
                "responses",
                "--",
                "--model",
                "gpt-4o",
            ],
        )

    assert result.exit_code != 0
    assert "not supported with translated backends" in result.output


def test_wrap_copilot_clears_stale_wire_api_in_anthropic_mode(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wrap_cli, main = wrap_modules
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-dummy")
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="copilot"),
        patch("headroom.cli.wrap.has_oauth_auth", return_value=False),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            ["wrap", "copilot", "--no-rtk", "--", "--model", "claude-sonnet-4-20250514"],
            env={
                "COPILOT_PROVIDER_WIRE_API": "responses",
                "ANTHROPIC_API_KEY": "sk-test-dummy",
            },
        )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["COPILOT_PROVIDER_TYPE"] == "anthropic"
    assert "COPILOT_PROVIDER_WIRE_API" not in env


def test_wrap_copilot_fails_when_binary_missing(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    _wrap_cli, main = wrap_modules
    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        result = runner.invoke(main, ["wrap", "copilot", "--", "--model", "gpt-4o"])

    assert result.exit_code == 1
    assert "'copilot' not found in PATH" in result.output
    assert "Install GitHub Copilot CLI" in result.output
