"""CLI coverage for Cline CLI and extension proxy routing."""

from __future__ import annotations

from unittest.mock import patch

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
