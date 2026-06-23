"""Tests for the `headroom dashboard` command (#1277)."""

from __future__ import annotations

import webbrowser

from click.testing import CliRunner

from headroom.cli.main import main


def test_dashboard_no_open_prints_url(monkeypatch):
    """--no-open prints the dashboard URL and never opens a browser."""
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda u, *a, **k: opened.append(u) or True)

    result = CliRunner().invoke(main, ["dashboard", "--no-open", "--port", "9999"])

    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:9999/dashboard" in result.output
    assert opened == []  # browser must not be launched


def test_dashboard_opens_browser_by_default(monkeypatch):
    """Without --no-open the command opens the URL in a browser."""
    opened: list[str] = []
    monkeypatch.setattr(webbrowser, "open", lambda u, *a, **k: opened.append(u) or True)

    result = CliRunner().invoke(main, ["dashboard", "--port", "1234"])

    assert result.exit_code == 0, result.output
    assert opened == ["http://127.0.0.1:1234/dashboard"]


def test_dashboard_browser_failure_is_swallowed(monkeypatch):
    """A headless box where webbrowser.open raises must not crash the command."""

    def _boom(*_a, **_k):
        raise RuntimeError("no display")

    monkeypatch.setattr(webbrowser, "open", _boom)

    result = CliRunner().invoke(main, ["dashboard"])

    assert result.exit_code == 0, result.output
    assert "/dashboard" in result.output
