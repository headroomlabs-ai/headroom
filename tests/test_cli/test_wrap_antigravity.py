"""Tests for the `headroom wrap antigravity` command."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from click.testing import CliRunner

from headroom.cli import wrap as wrap_mod
from headroom.cli.main import main


def test_wrap_help_lists_antigravity() -> None:
    result = CliRunner().invoke(main, ["wrap", "--help"])
    assert result.exit_code == 0, result.output
    assert "antigravity" in result.output


def test_wrap_antigravity_help_mentions_custom_model_provider() -> None:
    result = CliRunner().invoke(main, ["wrap", "antigravity", "--help"])
    assert result.exit_code == 0, result.output
    assert "Antigravity" in result.output
    assert "OpenAI-compatible" in result.output


def test_wrap_antigravity_prepare_only_exits_cleanly() -> None:
    result = CliRunner().invoke(main, ["wrap", "antigravity", "--prepare-only"])
    assert result.exit_code == 0, result.output


def test_wrap_antigravity_uses_proxy_only_watcher_with_antigravity_type() -> None:
    captured: dict[str, Any] = {}

    def fake_watcher(**kwargs: Any) -> None:
        captured.update(kwargs)

    with patch.object(wrap_mod, "_run_proxy_only_watcher", side_effect=fake_watcher):
        result = CliRunner().invoke(main, ["wrap", "antigravity", "--port", "8787"])

    assert result.exit_code == 0, result.output
    assert captured["agent_label"] == "antigravity"
    assert captured["agent_type"] == "antigravity"
    assert captured["port"] == 8787


def test_wrap_antigravity_setup_lines_render_actual_port(monkeypatch) -> None:
    monkeypatch.setattr(wrap_mod, "_project_name_from_cwd", lambda: None)
    captured: dict[str, Any] = {}

    def fake_watcher(**kwargs: Any) -> None:
        captured.update(kwargs)

    with patch.object(wrap_mod, "_run_proxy_only_watcher", side_effect=fake_watcher):
        result = CliRunner().invoke(main, ["wrap", "antigravity"])

    assert result.exit_code == 0, result.output
    print_setup_lines = captured["print_setup_lines"]

    with patch("click.echo") as echo_mock:
        print_setup_lines(9999)

    rendered = "\n".join(call.args[0] for call in echo_mock.call_args_list)
    assert "http://127.0.0.1:9999/v1" in rendered
    assert "Antigravity" in rendered


def test_wrap_antigravity_honors_headroom_port_env(monkeypatch) -> None:
    """HEADROOM_PORT configures the proxy port like every other wrap command."""
    monkeypatch.setenv("HEADROOM_PORT", "9999")
    monkeypatch.delenv("HEADROOM_PORT_DISCOVERY", raising=False)
    captured: dict[str, Any] = {}

    def fake_watcher(**kwargs: Any) -> None:
        captured.update(kwargs)

    with patch.object(wrap_mod, "_run_proxy_only_watcher", side_effect=fake_watcher):
        result = CliRunner().invoke(main, ["wrap", "antigravity"])

    assert result.exit_code == 0, result.output
    assert captured["port"] == 9999


def test_wrap_antigravity_explicit_port_overrides_env(monkeypatch) -> None:
    """An explicit --port always wins over HEADROOM_PORT."""
    monkeypatch.setenv("HEADROOM_PORT", "9999")
    monkeypatch.delenv("HEADROOM_PORT_DISCOVERY", raising=False)
    captured: dict[str, Any] = {}

    def fake_watcher(**kwargs: Any) -> None:
        captured.update(kwargs)

    with patch.object(wrap_mod, "_run_proxy_only_watcher", side_effect=fake_watcher):
        result = CliRunner().invoke(main, ["wrap", "antigravity", "--port", "7777"])

    assert result.exit_code == 0, result.output
    assert captured["port"] == 7777
