"""Tests for anonymous telemetry warning feature.

Covers:
- is_telemetry_warn_enabled() feature flag
- format_telemetry_notice() helper
- proxy CLI banner includes telemetry status
- wrap CLI prints telemetry notice
- /stats endpoint exposes anon_telemetry_shipping flag
"""

from unittest.mock import patch

import pytest

click = pytest.importorskip("click")
from click.testing import CliRunner  # noqa: E402

from headroom.telemetry.beacon import (  # noqa: E402
    format_telemetry_notice,
    is_telemetry_warn_enabled,
)

# ---------------------------------------------------------------------------
# is_telemetry_warn_enabled
# ---------------------------------------------------------------------------


class TestIsTelemetryWarnEnabled:
    """Tests for the HEADROOM_TELEMETRY_WARN feature flag."""

    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("HEADROOM_TELEMETRY_WARN", raising=False)
        assert is_telemetry_warn_enabled() is True

    @pytest.mark.parametrize("value", ["off", "OFF", "false", "0", "no", "disable", "disabled"])
    def test_disabled_by_env_var(self, monkeypatch, value):
        monkeypatch.setenv("HEADROOM_TELEMETRY_WARN", value)
        assert is_telemetry_warn_enabled() is False

    @pytest.mark.parametrize("value", ["on", "ON", "1", "yes", "true"])
    def test_enabled_by_truthy_env_var(self, monkeypatch, value):
        monkeypatch.setenv("HEADROOM_TELEMETRY_WARN", value)
        assert is_telemetry_warn_enabled() is True


# ---------------------------------------------------------------------------
# format_telemetry_notice
# ---------------------------------------------------------------------------


class TestFormatTelemetryNotice:
    """Tests for format_telemetry_notice()."""

    def test_returns_notice_when_telemetry_on(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_TELEMETRY", "on")
        monkeypatch.delenv("HEADROOM_TELEMETRY_WARN", raising=False)
        notice = format_telemetry_notice()
        assert notice != ""
        assert "ENABLED" in notice
        assert "HEADROOM_TELEMETRY=off" in notice
        assert "--no-telemetry" in notice

    def test_empty_when_telemetry_off(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_TELEMETRY", "off")
        monkeypatch.delenv("HEADROOM_TELEMETRY_WARN", raising=False)
        assert format_telemetry_notice() == ""

    def test_empty_when_warn_flag_off(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_TELEMETRY", "on")
        monkeypatch.setenv("HEADROOM_TELEMETRY_WARN", "off")
        assert format_telemetry_notice() == ""

    def test_prefix_is_applied(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_TELEMETRY", "on")
        monkeypatch.delenv("HEADROOM_TELEMETRY_WARN", raising=False)
        notice = format_telemetry_notice(prefix="  ")
        assert notice.startswith("  ")

    def test_no_prefix_by_default(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_TELEMETRY", "on")
        monkeypatch.delenv("HEADROOM_TELEMETRY_WARN", raising=False)
        notice = format_telemetry_notice()
        # Default prefix is "" so the string should start with "Telemetry"
        assert notice.startswith("Telemetry:")


# ---------------------------------------------------------------------------
# proxy CLI banner
# ---------------------------------------------------------------------------


class TestProxyCLITelemetryBanner:
    """Proxy CLI startup banner must include telemetry status."""

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_banner_shows_telemetry_enabled(self, runner, monkeypatch):
        # Telemetry is opt-in: it only shows ENABLED once explicitly turned on.
        monkeypatch.setenv("HEADROOM_TELEMETRY", "on")

        from headroom.cli.main import main

        with patch("headroom.proxy.server.run_server", side_effect=SystemExit(0)):
            result = runner.invoke(main, ["proxy"])

        assert "Telemetry:" in result.output
        assert "ENABLED" in result.output

    def test_banner_disabled_by_default(self, runner, monkeypatch):
        # The whole point of opt-in: unset env => telemetry off, banner says so
        # and surfaces how to opt in.
        monkeypatch.delenv("HEADROOM_TELEMETRY", raising=False)

        from headroom.cli.main import main

        with patch("headroom.proxy.server.run_server", side_effect=SystemExit(0)):
            result = runner.invoke(main, ["proxy"])

        assert "Telemetry:" in result.output
        assert "DISABLED" in result.output
        assert "HEADROOM_TELEMETRY=on" in result.output or "--telemetry" in result.output

    def test_telemetry_flag_opts_in(self, runner, monkeypatch):
        monkeypatch.delenv("HEADROOM_TELEMETRY", raising=False)

        from headroom.cli.main import main

        with patch("headroom.proxy.server.run_server", side_effect=SystemExit(0)):
            result = runner.invoke(main, ["proxy", "--telemetry"])

        assert "Telemetry:" in result.output
        assert "ENABLED" in result.output

    def test_banner_shows_telemetry_disabled(self, runner, monkeypatch):
        monkeypatch.setenv("HEADROOM_TELEMETRY", "off")

        from headroom.cli.main import main

        with patch("headroom.proxy.server.run_server", side_effect=SystemExit(0)):
            result = runner.invoke(main, ["proxy"])

        assert "Telemetry:" in result.output
        assert "DISABLED" in result.output

    def test_no_telemetry_flag_disables(self, runner, monkeypatch):
        monkeypatch.delenv("HEADROOM_TELEMETRY", raising=False)

        from headroom.cli.main import main

        with patch("headroom.proxy.server.run_server", side_effect=SystemExit(0)):
            result = runner.invoke(main, ["proxy", "--no-telemetry"])

        assert "Telemetry:" in result.output
        assert "DISABLED" in result.output

    def test_banner_shows_opt_out_instructions_when_enabled(self, runner, monkeypatch):
        monkeypatch.setenv("HEADROOM_TELEMETRY", "on")

        from headroom.cli.main import main

        with patch("headroom.proxy.server.run_server", side_effect=SystemExit(0)):
            result = runner.invoke(main, ["proxy"])

        assert "HEADROOM_TELEMETRY=off" in result.output or "--no-telemetry" in result.output

    def test_banner_shows_context_tool(self, runner, monkeypatch):
        monkeypatch.setenv("HEADROOM_CONTEXT_TOOL", "lean-ctx")

        from headroom.cli.main import main

        with patch("headroom.proxy.server.run_server", side_effect=SystemExit(0)):
            result = runner.invoke(main, ["proxy"])

        assert result.exit_code == 0
        assert "Context Tool: lean-ctx" in result.output


# ---------------------------------------------------------------------------
# wrap CLI telemetry notice
# ---------------------------------------------------------------------------


class TestWrapCLITelemetryNotice:
    """_print_telemetry_notice() is called from wrap commands."""

    def test_print_notice_outputs_when_telemetry_on(self, monkeypatch, capsys):
        monkeypatch.setenv("HEADROOM_TELEMETRY", "on")
        monkeypatch.delenv("HEADROOM_TELEMETRY_WARN", raising=False)

        from headroom.cli.wrap import _print_telemetry_notice

        _print_telemetry_notice()
        captured = capsys.readouterr()
        assert "Telemetry" in captured.out
        assert "HEADROOM_TELEMETRY=off" in captured.out

    def test_print_notice_silent_when_telemetry_off(self, monkeypatch, capsys):
        monkeypatch.setenv("HEADROOM_TELEMETRY", "off")

        from headroom.cli.wrap import _print_telemetry_notice

        _print_telemetry_notice()
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_print_notice_silent_when_warn_flag_off(self, monkeypatch, capsys):
        monkeypatch.setenv("HEADROOM_TELEMETRY", "on")
        monkeypatch.setenv("HEADROOM_TELEMETRY_WARN", "off")

        from headroom.cli.wrap import _print_telemetry_notice

        _print_telemetry_notice()
        captured = capsys.readouterr()
        assert captured.out == ""


# ---------------------------------------------------------------------------
# /stats endpoint – anon_telemetry_shipping flag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStatsEndpointTelemetryFlag:
    """The /stats endpoint must expose anon_telemetry_shipping."""

    pytest.importorskip("fastapi")

    async def test_stats_includes_anon_telemetry_shipping_true(self, monkeypatch):
        # Opt-in: shipping is only true once telemetry is explicitly enabled.
        monkeypatch.setenv("HEADROOM_TELEMETRY", "on")
        from headroom.proxy.server import ProxyConfig, create_app

        app = create_app(
            ProxyConfig(
                cache_enabled=False,
                rate_limit_enabled=False,
                cost_tracking_enabled=False,
            )
        )

        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert "anon_telemetry_shipping" in data
        assert data["anon_telemetry_shipping"] is True

    async def test_stats_includes_anon_telemetry_shipping_false(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_TELEMETRY", "off")
        from headroom.proxy.server import ProxyConfig, create_app

        app = create_app(
            ProxyConfig(
                cache_enabled=False,
                rate_limit_enabled=False,
                cost_tracking_enabled=False,
            )
        )

        from httpx import ASGITransport, AsyncClient

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/stats")

        assert resp.status_code == 200
        data = resp.json()
        assert "anon_telemetry_shipping" in data
        assert data["anon_telemetry_shipping"] is False


# ---------------------------------------------------------------------------
# telemetry __init__ exports
# ---------------------------------------------------------------------------


class TestTelemetryModuleExports:
    """New helpers must be exported from headroom.telemetry."""

    def test_is_telemetry_warn_enabled_exported(self):
        from headroom.telemetry import is_telemetry_warn_enabled as fn

        assert callable(fn)

    def test_is_telemetry_enabled_exported(self):
        from headroom.telemetry import is_telemetry_enabled as fn

        assert callable(fn)

    def test_format_telemetry_notice_exported(self):
        from headroom.telemetry import format_telemetry_notice as fn

        assert callable(fn)
