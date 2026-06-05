"""Tests for `headroom wrap copilot` command."""

from __future__ import annotations

import importlib
import os
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
    assert env["GITHUB_COPILOT_API_URL"] == DEFAULT_API_URL
    assert env["OPENAI_TARGET_API_URL"] == DEFAULT_API_URL
    assert "COPILOT_PROVIDER_API_KEY" not in env
    assert captured["openai_api_url"] == DEFAULT_API_URL
    assert f"COPILOT_PROVIDER_API_URL={DEFAULT_API_URL}" in captured["env_vars_display"]


def test_wrap_copilot_subscription_uses_github_auth_without_provider_key(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wrap_cli, main = wrap_modules
    for var in ("COPILOT_PROVIDER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="copilot"),
        patch("headroom.cli.wrap.resolve_subscription_bearer_token", return_value="gho-existing"),
        patch("headroom.cli.wrap.has_oauth_auth", return_value=False),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            ["wrap", "copilot", "--subscription", "--no-rtk"],
        )

    assert result.exit_code == 0, result.output
    assert "Copilot BYOK requires a model" not in result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["COPILOT_PROVIDER_TYPE"] == "openai"
    assert env["COPILOT_PROVIDER_BASE_URL"] == "http://127.0.0.1:8787/v1"
    assert env["COPILOT_PROVIDER_WIRE_API"] == "completions"
    assert env["COPILOT_PROVIDER_BEARER_TOKEN"] == "gho-existing"
    assert "COPILOT_PROVIDER_API_KEY" not in env
    assert captured["openai_api_url"] == DEFAULT_API_URL


def test_wrap_copilot_subscription_defaults_to_responses_for_reasoning_model(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wrap_cli, main = wrap_modules
    _clear_copilot_env(monkeypatch)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="copilot"),
        patch("headroom.cli.wrap.resolve_subscription_bearer_token", return_value="gho-existing"),
        patch("headroom.cli.wrap.has_oauth_auth", return_value=False),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            ["wrap", "copilot", "--subscription", "--no-rtk", "--", "--model", "gpt-5.4"],
        )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["COPILOT_PROVIDER_TYPE"] == "openai"
    assert env["COPILOT_PROVIDER_WIRE_API"] == "responses"
    assert "COPILOT_PROVIDER_WIRE_API=responses" in captured["env_vars_display"]


def test_wrap_copilot_subscription_keeps_gpt4_on_completions(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subscription routing must not blanket-promote every model to the responses
    API: a non-reasoning model such as gpt-4.1 still defaults to ``completions``.
    The provider-helper unit tests cover the wire-API decision in isolation; this
    exercises the full CLI path (args -> subscription resolution -> launch env) so
    the default can't silently regress to ``responses`` for GPT-4 traffic.
    """
    _wrap_cli, main = wrap_modules
    _clear_copilot_env(monkeypatch)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="copilot"),
        patch("headroom.cli.wrap.resolve_subscription_bearer_token", return_value="gho-existing"),
        patch("headroom.cli.wrap.has_oauth_auth", return_value=False),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            ["wrap", "copilot", "--subscription", "--no-rtk", "--", "--model", "gpt-4.1"],
        )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["COPILOT_PROVIDER_TYPE"] == "openai"
    assert env["COPILOT_PROVIDER_WIRE_API"] == "completions"


def test_wrap_copilot_subscription_allows_explicit_responses_wire_api(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _wrap_cli, main = wrap_modules
    _clear_copilot_env(monkeypatch)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="copilot"),
        patch("headroom.cli.wrap.resolve_subscription_bearer_token", return_value="gho-existing"),
        patch("headroom.cli.wrap.has_oauth_auth", return_value=False),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            [
                "wrap",
                "copilot",
                "--subscription",
                "--wire-api",
                "responses",
                "--no-rtk",
                "--",
                "--model",
                "gpt-5.4",
            ],
        )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["COPILOT_PROVIDER_TYPE"] == "openai"
    assert env["COPILOT_PROVIDER_WIRE_API"] == "responses"


def test_wrap_copilot_subscription_pins_validated_token_for_proxy(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--subscription` must hand the *validated* token to the proxy.

    The proxy honours ``GITHUB_COPILOT_API_TOKEN``; the wrapper passes the
    resolved token as the ``copilot_api_token`` launch argument so the proxy
    pins exactly it (rather than re-discovering a possibly different,
    unvalidated token). The token rides the launch arg, never the child env or
    the parent's global ``os.environ``. This guards the deterministic handoff.
    """
    _wrap_cli, main = wrap_modules
    for var in ("COPILOT_PROVIDER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    business_api = "https://api.business.githubcopilot.com"
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs: object) -> None:
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="copilot"),
        patch(
            "headroom.cli.wrap.resolve_subscription_bearer_token",
            return_value="gho-validated",
        ),
        patch("headroom.cli.wrap.resolve_copilot_api_url", return_value=business_api),
        patch("headroom.cli.wrap.has_oauth_auth", return_value=False),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(main, ["wrap", "copilot", "--subscription", "--no-rtk"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    # The validated token is handed to the proxy as an explicit launch
    # argument — not via the child env, not via the parent's os.environ.
    assert captured["copilot_api_token"] == "gho-validated"
    assert "GITHUB_COPILOT_API_TOKEN" not in env
    assert os.environ.get("GITHUB_COPILOT_API_TOKEN") is None
    assert env["COPILOT_PROVIDER_TYPE"] == "openai"
    assert env["COPILOT_PROVIDER_BEARER_TOKEN"] == "gho-validated"
    assert env["GITHUB_COPILOT_USE_TOKEN_EXCHANGE"] == "false"
    assert env["OPENAI_TARGET_API_URL"] == business_api
    assert captured["openai_api_url"] == business_api
    assert "COPILOT_PROVIDER_API_KEY" not in env
    # The secret must never be echoed to the terminal.
    assert "gho-validated" not in result.output


def test_wrap_copilot_subscription_requires_reusable_auth(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    _wrap_cli, main = wrap_modules
    with (
        patch("headroom.cli.wrap.shutil.which", return_value="copilot"),
        patch("headroom.cli.wrap.resolve_subscription_bearer_token", return_value=None),
    ):
        result = runner.invoke(main, ["wrap", "copilot", "--subscription", "--no-rtk"])

    assert result.exit_code != 0
    assert "subscription mode requires a reusable GitHub/Copilot bearer token" in result.output


def test_wrap_copilot_subscription_rejects_translated_backend(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    _wrap_cli, main = wrap_modules
    with patch("headroom.cli.wrap.shutil.which", return_value="copilot"):
        result = runner.invoke(
            main,
            ["wrap", "copilot", "--subscription", "--backend", "anyllm", "--no-rtk"],
        )

    assert result.exit_code != 0
    assert "cannot be combined with translated backends" in result.output


def test_wrap_copilot_subscription_rejects_anthropic_provider_type(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
) -> None:
    _wrap_cli, main = wrap_modules
    with patch("headroom.cli.wrap.shutil.which", return_value="copilot"):
        result = runner.invoke(
            main,
            ["wrap", "copilot", "--subscription", "--provider-type", "anthropic", "--no-rtk"],
        )

    assert result.exit_code != 0
    assert "do not combine it with --provider-type anthropic" in result.output


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


# ---------------------------------------------------------------------------
# Regression suite for #610 — GitHub Copilot endpoint routing per auth mode.
#
# 0.23.0 (commit f4dff9b) re-pointed the *shared* OAuth branch away from the
# generic https://api.githubcopilot.com to the account-specific endpoints.api
# host returned by /copilot_internal/user, and made resolve_copilot_api_url()
# ignore the GITHUB_COPILOT_API_URL override whenever a token resolves. For
# individual-plan users that broke newer models (gpt-5.4) on the responses API
# that had worked on 0.22.4. The pre-existing oauth test passed only because it
# left _fetch_copilot_user_info unmocked — the network call fails in CI, so
# resolve_copilot_api_url() fell back to the generic host and the real-world
# success path was never exercised. These tests mock a *successful* user-info
# response (the real world) so the routing for every auth mode is locked.
# ---------------------------------------------------------------------------

_ACCOUNT_USER_INFO = {"endpoints": {"api": "https://api.individual.githubcopilot.com"}}


def _clear_copilot_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "COPILOT_PROVIDER_API_KEY",
        "COPILOT_PROVIDER_BEARER_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GITHUB_COPILOT_API_URL",
        "GITHUB_COPILOT_TOKEN",
        "GITHUB_COPILOT_GITHUB_TOKEN",
        "COPILOT_MODEL",
        "COPILOT_PROVIDER_MODEL_ID",
        "COPILOT_PROVIDER_WIRE_API",
    ):
        monkeypatch.delenv(var, raising=False)


def test_wrap_copilot_oauth_keeps_generic_endpoint_when_account_advertised(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#610: non-subscription OAuth must route to the generic Copilot endpoint
    even when /copilot_internal/user advertises an account-specific host. The
    account host (api.individual.githubcopilot.com) does not serve newer models
    such as gpt-5.4 on the responses API — exactly what regressed after 0.22.4.
    """
    _wrap_cli, main = wrap_modules
    _clear_copilot_env(monkeypatch)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="copilot"),
        patch("headroom.cli.wrap.resolve_client_bearer_token", return_value="gho-oauth"),
        patch("headroom.cli.wrap.has_oauth_auth", return_value=True),
        patch("headroom.copilot_auth._fetch_copilot_user_info", return_value=_ACCOUNT_USER_INFO),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(main, ["wrap", "copilot", "--no-rtk", "--", "--model", "gpt-5.4"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["COPILOT_PROVIDER_BEARER_TOKEN"] == "gho-oauth"
    assert captured["openai_api_url"] == DEFAULT_API_URL
    assert env["OPENAI_TARGET_API_URL"] == DEFAULT_API_URL
    assert env["GITHUB_COPILOT_API_URL"] == DEFAULT_API_URL


def test_wrap_copilot_oauth_honors_api_url_override(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The GITHUB_COPILOT_API_URL escape hatch must be honored even when a token
    resolves and user-info advertises a different host (it was silently lost)."""
    _wrap_cli, main = wrap_modules
    _clear_copilot_env(monkeypatch)
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://proxy.internal.example.com")
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="copilot"),
        patch("headroom.cli.wrap.resolve_client_bearer_token", return_value="gho-oauth"),
        patch("headroom.cli.wrap.has_oauth_auth", return_value=True),
        patch("headroom.copilot_auth._fetch_copilot_user_info", return_value=_ACCOUNT_USER_INFO),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(main, ["wrap", "copilot", "--no-rtk", "--", "--model", "gpt-5.4"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert captured["openai_api_url"] == "https://proxy.internal.example.com"
    assert env["OPENAI_TARGET_API_URL"] == "https://proxy.internal.example.com"


def test_wrap_copilot_byok_never_resolves_copilot_endpoint(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BYOK (provider key, no OAuth) routes to the model provider through the
    proxy and must never resolve the Copilot hosted endpoint. It was unaffected
    by #610 — this pins that independence so a future change can't entangle it.
    """
    _wrap_cli, main = wrap_modules
    _clear_copilot_env(monkeypatch)
    monkeypatch.setenv("COPILOT_PROVIDER_API_KEY", "sk-test-dummy")
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    def tripwire(*_args, **_kwargs):  # noqa: ANN002,ANN003
        raise AssertionError("BYOK must not resolve the Copilot hosted endpoint")

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="copilot"),
        patch("headroom.cli.wrap.has_oauth_auth", return_value=False),
        patch("headroom.cli.wrap.resolve_copilot_api_url", side_effect=tripwire),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            ["wrap", "copilot", "--no-rtk", "--provider-type", "openai", "--", "--model", "gpt-4o"],
        )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert captured["openai_api_url"] is None
    assert env["COPILOT_PROVIDER_TYPE"] == "openai"


def test_wrap_copilot_subscription_uses_generic_endpoint_not_account(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#610 (subscription has the same latent bug): --subscription must route to
    the generic host too, even when /copilot_internal/user advertises an
    account-specific host. The segmented host does not serve newer models on the
    responses API, and it is not the host the official Copilot client uses."""
    _wrap_cli, main = wrap_modules
    _clear_copilot_env(monkeypatch)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="copilot"),
        patch("headroom.cli.wrap.resolve_subscription_bearer_token", return_value="gho-sub"),
        patch("headroom.cli.wrap.has_oauth_auth", return_value=True),
        patch("headroom.copilot_auth._fetch_copilot_user_info", return_value=_ACCOUNT_USER_INFO),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            ["wrap", "copilot", "--subscription", "--no-rtk", "--", "--model", "gpt-5.4"],
        )

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert captured["openai_api_url"] == DEFAULT_API_URL
    assert env["OPENAI_TARGET_API_URL"] == DEFAULT_API_URL
    assert env["COPILOT_PROVIDER_BEARER_TOKEN"] == "gho-sub"


def test_wrap_copilot_subscription_honors_api_url_override(
    runner: CliRunner,
    wrap_modules: tuple[types.ModuleType, click.Group],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enterprise / data-residency accounts that require a dedicated host pin it
    via GITHUB_COPILOT_API_URL — the override must flow through --subscription."""
    _wrap_cli, main = wrap_modules
    _clear_copilot_env(monkeypatch)
    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://api.enterprise.example.com")
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with (
        patch("headroom.cli.wrap.shutil.which", return_value="copilot"),
        patch("headroom.cli.wrap.resolve_subscription_bearer_token", return_value="gho-sub"),
        patch("headroom.cli.wrap.has_oauth_auth", return_value=True),
        patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool),
    ):
        result = runner.invoke(
            main,
            ["wrap", "copilot", "--subscription", "--no-rtk", "--", "--model", "gpt-5.4"],
        )

    assert result.exit_code == 0, result.output
    assert captured["openai_api_url"] == "https://api.enterprise.example.com"


def test_resolve_copilot_api_url_ignores_user_info_and_never_calls_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unit lock for #610: routing is override -> generic and must NOT depend on a
    user-info lookup. Even with a token in hand and user-info advertising an
    account host, the generic host is returned and no network call is made."""
    from headroom import copilot_auth

    monkeypatch.delenv("GITHUB_COPILOT_API_URL", raising=False)
    with patch.object(copilot_auth, "_fetch_copilot_user_info") as fetch:
        assert copilot_auth.resolve_copilot_api_url("gho-real") == copilot_auth.DEFAULT_API_URL
    fetch.assert_not_called()

    monkeypatch.setenv("GITHUB_COPILOT_API_URL", "https://pin.example.com")
    with patch.object(copilot_auth, "_fetch_copilot_user_info") as fetch:
        assert copilot_auth.resolve_copilot_api_url("gho-real") == "https://pin.example.com"
    fetch.assert_not_called()
