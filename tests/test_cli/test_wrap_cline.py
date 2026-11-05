"""CLI coverage for Cline CLI and extension proxy routing."""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from headroom.cli.main import main


def test_wrap_cline_forwards_custom_openai_upstream() -> None:
    captured: dict[str, object] = {}

    def fake_watcher(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)
        kwargs["print_setup_lines"](9123)

    with patch("headroom.cli.wrap._run_proxy_only_watcher", side_effect=fake_watcher):
        result = CliRunner().invoke(
            main,
            [
                "wrap",
                "cline",
                "--openai-api-url",
                "https://api.example.com/v1",
            ],
        )

    assert result.exit_code == 0, result.output
    assert captured["openai_api_url"] == "https://api.example.com/v1"
    assert "cline auth --provider openai-native" in result.output
    assert "--baseurl http://127.0.0.1:9123/v1" in result.output
    assert "OpenAI-compatible upstream: https://api.example.com/v1" in result.output


def test_wrap_cline_keeps_default_openai_routing_explicit() -> None:
    captured: dict[str, object] = {}

    def fake_watcher(**kwargs):  # noqa: ANN003, ANN202
        captured.update(kwargs)

    with patch("headroom.cli.wrap._run_proxy_only_watcher", side_effect=fake_watcher):
        result = CliRunner().invoke(main, ["wrap", "cline"])

    assert result.exit_code == 0, result.output
    assert captured["openai_api_url"] is None


def test_openai_upstream_option_belongs_to_cline_not_codex() -> None:
    runner = CliRunner()

    cline_help = runner.invoke(main, ["wrap", "cline", "--help"])
    codex_help = runner.invoke(main, ["wrap", "codex", "--help"])

    assert cline_help.exit_code == 0, cline_help.output
    assert "--openai-api-url URL" in cline_help.output
    assert codex_help.exit_code == 0, codex_help.output
    assert "--openai-api-url" not in codex_help.output


@pytest.mark.parametrize(
    ("running_url", "accepted"),
    [("https://api.example.com/v1/", True), ("https://api.openai.com/v1", False), (None, False)],
)
def test_cline_reuse_checks_the_custom_route_before_setup(monkeypatch, running_url, accepted):
    wrap = importlib.import_module("headroom.cli.wrap")
    calls = []
    monkeypatch.setattr(wrap, "_check_proxy", lambda port: True)
    monkeypatch.setattr(wrap, "_query_proxy_health", lambda port: None)

    def config(port):
        calls.append("config")
        return {"openai_api_url": running_url}

    monkeypatch.setattr(wrap, "_query_proxy_config", config)
    monkeypatch.setattr(wrap, "_register_proxy_client", lambda port: calls.append("register"))
    monkeypatch.setattr(wrap, "_make_cleanup", lambda *args: lambda *args: None)
    monkeypatch.setattr(wrap, "_push_runtime_env", lambda *args: None)
    monkeypatch.setattr(wrap.signal, "signal", lambda *args: None)

    def stop(_seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(wrap.time, "sleep", stop)
    with patch.object(wrap, "_ensure_proxy", return_value=(None, 8787)) as ensure:
        result = CliRunner().invoke(
            main, ["wrap", "cline", "--no-proxy", "--openai-api-url", "https://api.example.com/v1"]
        )

    if accepted:
        assert result.exit_code == 0, result.output
        assert calls[:2] == ["config", "register"]
        assert ensure.call_args.kwargs["require_openai_api_url"] is True
        assert "cline auth --provider openai-native" in result.output
    else:
        assert result.exit_code != 0
        assert calls == ["config"]
        ensure.assert_not_called()
        assert "cline auth --provider openai-native" not in result.output
