"""Tests for `headroom wrap grok` command.

Grok Build routes cli-chat-proxy traffic via ``GROK_CLI_CHAT_PROXY_BASE_URL``.
These tests mirror real ``grok`` CLI invocations (from ``grok --help``) and
verify Headroom only injects the proxy env — Grok's own flags pass through
unchanged.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.providers.grok.runtime import DEFAULT_API_URL, GROK_PROXY_ENV


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_grok_env_stub(tmp_path: Path) -> Path:
    """Stub `grok` that prints the proxy env var Grok Build reads at launch."""
    stub = tmp_path / "grok"
    stub.write_text(
        "#!/bin/sh\n"
        'printf "proxy=%s\\n" "${GROK_CLI_CHAT_PROXY_BASE_URL:-<unset>}"\n'
        "exit 0\n"
    )
    stub.chmod(0o755)
    return stub


# Real Grok CLI shapes lifted from `grok --help` (2026-06).
REAL_GROK_INVOCATIONS: list[tuple[list[str], tuple[str, ...]]] = [
    # TUI — no args, proxy env only.
    ([], ()),
    # Headless single-turn (`-p` / `--single`).
    (
        ["-p", "fix the failing test in test_wrap_grok.py"],
        ("-p", "fix the failing test in test_wrap_grok.py"),
    ),
    (
        ["--single", "summarize the diff and suggest a commit message"],
        ("--single", "summarize the diff and suggest a commit message"),
    ),
    # Continue the cwd's most recent session.
    (["--continue"], ("--continue",)),
    (["-c"], ("-c",)),
    # Scoped working directory + agent override.
    (
        ["--cwd", "/tmp/myproject", "--agent", "reviewer"],
        ("--cwd", "/tmp/myproject", "--agent", "reviewer"),
    ),
    # Headless parallel best-of-n with auto-approve (common CI-style invocation).
    (
        ["-p", "add input validation", "--best-of-n", "3", "--always-approve"],
        ("-p", "add input validation", "--best-of-n", "3", "--always-approve"),
    ),
    # Prompt from file + structured output (headless).
    (
        [
            "--prompt-file",
            "tasks/refactor.md",
            "--output-format",
            "json",
            "--permission-mode",
            "bypassPermissions",
        ],
        (
            "--prompt-file",
            "tasks/refactor.md",
            "--output-format",
            "json",
            "--permission-mode",
            "bypassPermissions",
        ),
    ),
    # Force passthrough after `--` (escape hatch for future flag collisions).
    (
        ["--", "--port", "4242"],
        ("--port", "4242"),
    ),
]


@pytest.mark.parametrize("cli_args,expected_grok_args", REAL_GROK_INVOCATIONS)
def test_wrap_grok_forwards_real_cli_shapes(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cli_args: list[str],
    expected_grok_args: tuple[str, ...],
) -> None:
    """Each tuple mirrors a documented `grok` invocation; Headroom must not eat Grok flags."""
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="grok"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "grok", *cli_args])

    assert result.exit_code == 0, result.output
    assert captured["args"] == expected_grok_args
    env = captured["env"]
    assert isinstance(env, dict)
    assert env[GROK_PROXY_ENV] == "http://127.0.0.1:8787/v1"


def test_wrap_grok_sets_proxy_env(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="grok"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "grok", "-p", "fix the bug"])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env[GROK_PROXY_ENV] == "http://127.0.0.1:8787/v1"
    assert captured["tool_label"] == "GROK"
    assert captured["agent_type"] == "grok"
    assert captured["args"] == ("-p", "fix the bug")


def test_wrap_grok_keeps_headroom_port_long_option(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grok owns `-p`; Headroom uses `--port` only so both can coexist on one command line."""
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="grok"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(
                main,
                ["wrap", "grok", "--port", "9999", "-p", "ship the feature"],
            )

    assert result.exit_code == 0, result.output
    assert captured["port"] == 9999
    assert captured["args"] == ("-p", "ship the feature")
    env = captured["env"]
    assert isinstance(env, dict)
    assert env[GROK_PROXY_ENV] == "http://127.0.0.1:9999/v1"


def test_wrap_grok_sets_grok_upstream_for_proxy(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="grok"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "grok"])

    assert result.exit_code == 0, result.output
    assert captured["openai_api_url"] == DEFAULT_API_URL


def test_wrap_grok_forwards_headroom_backend_options(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="grok"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(
                main,
                [
                    "wrap",
                    "grok",
                    "--backend",
                    "anyllm",
                    "--anyllm-provider",
                    "groq",
                    "--learn",
                    "--memory",
                    "-p",
                    "refactor auth module",
                ],
            )

    assert result.exit_code == 0, result.output
    assert captured["backend"] == "anyllm"
    assert captured["anyllm_provider"] == "groq"
    assert captured["learn"] is True
    assert captured["memory"] is True
    assert captured["args"] == ("-p", "refactor auth module")


def test_wrap_grok_forwards_no_proxy(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap.shutil.which", return_value="grok"):
        with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
            result = runner.invoke(main, ["wrap", "grok", "--no-proxy"])

    assert result.exit_code == 0, result.output
    assert captured["no_proxy"] is True


def test_wrap_grok_missing_binary(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with patch("headroom.cli.wrap.shutil.which", return_value=None):
        result = runner.invoke(main, ["wrap", "grok"])

    assert result.exit_code == 1
    assert "grok" in result.output.lower()


class TestGrokHeadroomVsPlain:
    """Grok without Headroom uses Grok's hosted proxy; wrap routes through localhost."""

    def test_plain_grok_leaves_proxy_env_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub = _write_grok_env_stub(tmp_path)
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
        monkeypatch.delenv(GROK_PROXY_ENV, raising=False)

        result = subprocess.run([str(stub)], capture_output=True, text=True, check=True)

        assert result.stdout.strip() == "proxy=<unset>"

    def test_plain_grok_with_manual_export_still_hits_remote_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub = _write_grok_env_stub(tmp_path)
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
        monkeypatch.setenv(GROK_PROXY_ENV, DEFAULT_API_URL)

        result = subprocess.run([str(stub)], capture_output=True, text=True, check=True)

        assert result.stdout.strip() == f"proxy={DEFAULT_API_URL}"

    def test_headroom_wrap_points_grok_at_local_proxy(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_grok_env_stub(tmp_path)
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
        monkeypatch.setenv(GROK_PROXY_ENV, DEFAULT_API_URL)

        with patch("headroom.cli.wrap._ensure_proxy", return_value=None):
            result = runner.invoke(main, ["wrap", "grok", "--no-proxy", "-p", "ship it"])

        assert result.exit_code == 0, result.output
        assert f"{GROK_PROXY_ENV}=http://127.0.0.1:8787/v1" in result.output
        assert DEFAULT_API_URL not in result.output
        assert "HEADROOM WRAP: GROK" in result.output

    def test_headroom_wrap_custom_port_overrides_remote_default(
        self, runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_grok_env_stub(tmp_path)
        monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")

        with patch("headroom.cli.wrap._ensure_proxy", return_value=None):
            result = runner.invoke(main, ["wrap", "grok", "--no-proxy", "--port", "4242"])

        assert result.exit_code == 0, result.output
        assert f"{GROK_PROXY_ENV}=http://127.0.0.1:4242/v1" in result.output
        assert DEFAULT_API_URL not in result.output
