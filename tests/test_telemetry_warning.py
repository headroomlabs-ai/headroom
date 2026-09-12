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
        monkeypatch.setenv("HEADROOM_BEACON", "off")
        monkeypatch.delenv("HEADROOM_TELEMETRY_WARN", raising=False)
        notice = format_telemetry_notice()
        assert notice != ""
        assert "ENABLED" in notice
        assert "HEADROOM_TELEMETRY=off" in notice
        assert "--no-telemetry" in notice

    def test_empty_when_telemetry_off(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_TELEMETRY", "off")
        monkeypatch.setenv("HEADROOM_BEACON", "off")
        monkeypatch.delenv("HEADROOM_TELEMETRY_WARN", raising=False)
        assert format_telemetry_notice() == ""

    def test_beacon_is_announced_by_default(self, monkeypatch):
        """The beacon is opt-out, so the notice is the only place a user finds
        out it is running. Silence here is how anonymous telemetry becomes a
        trust incident."""
        for var in ("HEADROOM_TELEMETRY", "HEADROOM_BEACON", "DO_NOT_TRACK"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv("HEADROOM_TELEMETRY_WARN", raising=False)
        notice = format_telemetry_notice()
        assert "compression stats" in notice
        assert "HEADROOM_BEACON=off" in notice

    def test_beacon_notice_names_what_is_not_sent(self, monkeypatch):
        """Vague reassurance is worse than none. The notice has to name the
        three things users actually worry about."""
        for var in ("HEADROOM_TELEMETRY", "HEADROOM_BEACON", "DO_NOT_TRACK"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv("HEADROOM_TELEMETRY_WARN", raising=False)
        notice = format_telemetry_notice()
        assert "never prompts" in notice
        assert "code" in notice
        assert "file paths" in notice

    def test_silent_when_beacon_disabled_and_no_local(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_BEACON", "off")
        for var in ("HEADROOM_TELEMETRY", "DO_NOT_TRACK"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv("HEADROOM_TELEMETRY_WARN", raising=False)
        assert format_telemetry_notice() == ""

    def test_do_not_track_silences_the_beacon_notice(self, monkeypatch):
        monkeypatch.setenv("DO_NOT_TRACK", "1")
        for var in ("HEADROOM_TELEMETRY", "HEADROOM_BEACON"):
            monkeypatch.delenv(var, raising=False)
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
    """Proxy CLI startup banner must include telemetry status.

    Every scenario below sets HEADROOM_BEACON=off explicitly, isolating the
    local-telemetry-only wording (mirrors the convention already used by
    TestFormatTelemetryNotice, e.g. test_returns_notice_when_telemetry_on).
    Before the beacon-disclosure fix the banner never looked at the beacon
    at all, so these tests would have passed unchanged whether the beacon
    was on or off -- which is exactly the bug: an operator relying on this
    banner had no way to tell the two apart. The two tests at the bottom of
    this class (*_beacon_default_on_is_surfaced,
    *_beacon_off_local_off_says_fully_off) cover the beacon-on-by-default
    case the old banner never surfaced.
    """

    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_banner_shows_telemetry_enabled(self, runner, monkeypatch):
        # Telemetry is opt-in: it only shows ENABLED once explicitly turned on.
        monkeypatch.setenv("HEADROOM_TELEMETRY", "on")
        monkeypatch.setenv("HEADROOM_BEACON", "off")

        from headroom.cli.main import main

        with patch("headroom.proxy.server.run_server", side_effect=SystemExit(0)):
            result = runner.invoke(main, ["proxy"])

        assert "Telemetry:" in result.output
        assert "ENABLED" in result.output

    def test_banner_disabled_by_default(self, runner, monkeypatch):
        # The whole point of opt-in: unset env => local telemetry off, banner
        # says so and surfaces how to opt in. (Beacon pinned off here so this
        # isolates the local-only wording; see the beacon-specific tests
        # below for the on-by-default beacon case.)
        monkeypatch.delenv("HEADROOM_TELEMETRY", raising=False)
        monkeypatch.setenv("HEADROOM_BEACON", "off")

        from headroom.cli.main import main

        with patch("headroom.proxy.server.run_server", side_effect=SystemExit(0)):
            result = runner.invoke(main, ["proxy"])

        assert "Telemetry:" in result.output
        assert "OFF" in result.output
        assert "HEADROOM_TELEMETRY=on" in result.output

    def test_telemetry_flag_opts_in(self, runner, monkeypatch):
        monkeypatch.delenv("HEADROOM_TELEMETRY", raising=False)
        monkeypatch.setenv("HEADROOM_BEACON", "off")

        from headroom.cli.main import main

        with patch("headroom.proxy.server.run_server", side_effect=SystemExit(0)):
            result = runner.invoke(main, ["proxy", "--telemetry"])

        assert "Telemetry:" in result.output
        assert "ENABLED" in result.output

    def test_banner_shows_telemetry_disabled(self, runner, monkeypatch):
        monkeypatch.setenv("HEADROOM_TELEMETRY", "off")
        monkeypatch.setenv("HEADROOM_BEACON", "off")

        from headroom.cli.main import main

        with patch("headroom.proxy.server.run_server", side_effect=SystemExit(0)):
            result = runner.invoke(main, ["proxy"])

        assert "Telemetry:" in result.output
        assert "OFF" in result.output

    def test_no_telemetry_flag_disables(self, runner, monkeypatch):
        monkeypatch.delenv("HEADROOM_TELEMETRY", raising=False)
        monkeypatch.setenv("HEADROOM_BEACON", "off")

        from headroom.cli.main import main

        with patch("headroom.proxy.server.run_server", side_effect=SystemExit(0)):
            result = runner.invoke(main, ["proxy", "--no-telemetry"])

        assert "Telemetry:" in result.output
        assert "OFF" in result.output

    def test_banner_shows_opt_out_instructions_when_enabled(self, runner, monkeypatch):
        monkeypatch.setenv("HEADROOM_TELEMETRY", "on")
        monkeypatch.setenv("HEADROOM_BEACON", "off")

        from headroom.cli.main import main

        with patch("headroom.proxy.server.run_server", side_effect=SystemExit(0)):
            result = runner.invoke(main, ["proxy"])

        assert "HEADROOM_TELEMETRY=off" in result.output or "--no-telemetry" in result.output

    def test_banner_beacon_default_on_is_surfaced(self, runner, monkeypatch):
        """The actual bug: HEADROOM_TELEMETRY=off / --no-telemetry alone used
        to make the banner print a bare "Telemetry: DISABLED" even though the
        anonymous upload beacon (a separate, on-by-default switch) was still
        active and never mentioned. After the fix, an operator who sets only
        --no-telemetry still sees the beacon disclosed."""
        monkeypatch.setenv("HEADROOM_TELEMETRY", "off")
        monkeypatch.delenv("HEADROOM_BEACON", raising=False)
        monkeypatch.delenv("DO_NOT_TRACK", raising=False)

        from headroom.cli.main import main

        with patch("headroom.proxy.server.run_server", side_effect=SystemExit(0)):
            result = runner.invoke(main, ["proxy", "--no-telemetry"])

        assert "Telemetry:" in result.output
        assert "compression stats" in result.output
        assert "HEADROOM_BEACON=off" in result.output
        # The old banner's bare "DISABLED" claim must not appear on the
        # Telemetry line specifically (the banner has an unrelated
        # "Memory: DISABLED" line that must not make this assertion a
        # false pass) -- that combination is exactly the false assurance
        # this fix removes.
        telemetry_line = next(
            line for line in result.output.splitlines() if line.strip().startswith("Telemetry:")
        )
        assert "DISABLED" not in telemetry_line

    def test_banner_beacon_off_and_local_off_says_fully_off(self, runner, monkeypatch):
        """Only when BOTH switches are off does the banner claim nothing is
        happening -- the one case where a bare OFF claim is actually true."""
        monkeypatch.setenv("HEADROOM_TELEMETRY", "off")
        monkeypatch.setenv("HEADROOM_BEACON", "off")

        from headroom.cli.main import main

        with patch("headroom.proxy.server.run_server", side_effect=SystemExit(0)):
            result = runner.invoke(main, ["proxy"])

        assert "Telemetry:" in result.output
        assert "OFF" in result.output
        assert "compression stats" not in result.output


# ---------------------------------------------------------------------------
# wrap CLI telemetry notice
# ---------------------------------------------------------------------------


class TestWrapCLITelemetryNotice:
    """_print_telemetry_notice() is called from wrap commands."""

    def test_print_notice_outputs_when_telemetry_on(self, monkeypatch, capsys):
        monkeypatch.setenv("HEADROOM_TELEMETRY", "on")
        monkeypatch.setenv("HEADROOM_BEACON", "off")
        monkeypatch.delenv("HEADROOM_TELEMETRY_WARN", raising=False)

        from headroom.cli.wrap import _print_telemetry_notice

        _print_telemetry_notice()
        captured = capsys.readouterr()
        assert "Telemetry" in captured.out
        assert "HEADROOM_TELEMETRY=off" in captured.out

    def test_print_notice_announces_beacon_by_default(self, monkeypatch, capsys):
        for var in ("HEADROOM_TELEMETRY", "HEADROOM_BEACON", "DO_NOT_TRACK"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv("HEADROOM_TELEMETRY_WARN", raising=False)

        from headroom.cli.wrap import _print_telemetry_notice

        _print_telemetry_notice()
        captured = capsys.readouterr()
        assert "compression stats" in captured.out
        assert "HEADROOM_BEACON=off" in captured.out

    def test_print_notice_silent_when_telemetry_off(self, monkeypatch, capsys):
        monkeypatch.setenv("HEADROOM_TELEMETRY", "off")
        monkeypatch.setenv("HEADROOM_BEACON", "off")

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
    """The /stats endpoint's anon_telemetry_shipping field must reflect the
    live HEADROOM_BEACON state, not a hardcoded constant.

    Previously this field was hardcoded to False on the premise that "the
    anonymous telemetry beacon was removed" -- it was not (telemetry/beacon.py
    and telemetry/session.py fully implement and wire it via
    record_outcome() -> SessionAggregator -> a POST to
    headroom-beacon.headroom-beacon.workers.dev, gated on is_beacon_enabled(),
    which is ON BY DEFAULT). An operator polling /stats to confirm nothing
    ships externally got a false assurance regardless of their actual
    HEADROOM_BEACON setting. These two tests previously asserted `False`
    unconditionally, encoding the same incorrect premise as the field they
    were testing.
    """

    pytest.importorskip("fastapi")

    async def test_stats_anon_telemetry_shipping_reflects_beacon_on(self, monkeypatch):
        # HEADROOM_BEACON is on by default (BEACON_DEFAULT_ON = True) and
        # HEADROOM_TELEMETRY does not touch it -- setting local telemetry on
        # must not by itself make this field misreport False.
        monkeypatch.setenv("HEADROOM_TELEMETRY", "on")
        monkeypatch.delenv("HEADROOM_BEACON", raising=False)
        monkeypatch.delenv("DO_NOT_TRACK", raising=False)
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

    async def test_stats_anon_telemetry_shipping_false_when_beacon_off(self, monkeypatch):
        monkeypatch.setenv("HEADROOM_TELEMETRY", "off")
        monkeypatch.setenv("HEADROOM_BEACON", "off")
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

    async def test_stats_anon_telemetry_shipping_false_when_do_not_track(self, monkeypatch):
        # DO_NOT_TRACK silences the beacon regardless of HEADROOM_BEACON.
        monkeypatch.setenv("DO_NOT_TRACK", "1")
        monkeypatch.delenv("HEADROOM_BEACON", raising=False)
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
