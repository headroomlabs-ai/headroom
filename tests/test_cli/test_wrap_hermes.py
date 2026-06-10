"""Tests for `headroom wrap hermes` command."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main
from headroom.providers.hermes.runtime import DEFAULT_HERMES_API_URL, OPENAI_BASE_ENV


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_wrap_hermes_sets_openai_base_and_hermes_upstream(
    runner: CliRunner, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_run_proxy_only_watcher(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap._run_proxy_only_watcher", side_effect=fake_run_proxy_only_watcher):
        result = runner.invoke(main, ["wrap", "hermes", "--port", "4242"])

    assert result.exit_code == 0, result.output
    assert captured["port"] == 4242
    assert captured["agent_type"] == "hermes"
    assert captured["openai_api_url"] == DEFAULT_HERMES_API_URL


def test_wrap_hermes_custom_upstream_url(
    runner: CliRunner, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_run_proxy_only_watcher(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap._run_proxy_only_watcher", side_effect=fake_run_proxy_only_watcher):
        result = runner.invoke(
            main,
            ["wrap", "hermes", "--hermes-url", "http://10.0.0.5:38765/v1"],
        )

    assert result.exit_code == 0, result.output
    assert captured["openai_api_url"] == "http://10.0.0.5:38765/v1"


def test_wrap_hermes_runs_command_with_openai_env(
    runner: CliRunner, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = tmp_path / "client"
    stub.write_text('#!/bin/sh\nprintf "openai=%s\\n" "$OPENAI_BASE_URL"\n')
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.chdir(tmp_path)
    captured: dict[str, object] = {}

    def fake_launch_tool(**kwargs):  # noqa: ANN003
        captured.update(kwargs)

    with patch("headroom.cli.wrap._launch_tool", side_effect=fake_launch_tool):
        result = runner.invoke(main, ["wrap", "hermes", "--no-proxy", "--", str(stub)])

    assert result.exit_code == 0, result.output
    env = captured["env"]
    assert isinstance(env, dict)
    assert env[OPENAI_BASE_ENV] == "http://127.0.0.1:8787/v1"
    assert captured["openai_api_url"] == DEFAULT_HERMES_API_URL
    assert captured["tool_label"] == "HERMES"