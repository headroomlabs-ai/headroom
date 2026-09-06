"""``headroom telemetry`` — show exactly what the beacon would send.

The beacon is opt-OUT and on by default. "The receiver is open source, go read
worker.js" is a good answer to what happens to the data, but it is a poor
answer to what LEAVES YOUR MACHINE — that requires reading two files, trusting
that the copy on GitHub matches the copy installed, and reconstructing a
payload by hand.

This prints the real thing: the current process's resource attributes and a
session payload built by the same `_Session.payload()` the beacon uses, with
representative counters folded in. Nothing here is a mock-up of the schema; if
a field can appear on the wire, it appears here.
"""

from __future__ import annotations

import json
from typing import Any

import click

from .main import main


def _sample_payload() -> dict[str, Any]:
    """One session payload, built through the real aggregator.

    Uses the actual fold path rather than hand-writing a dict, so this cannot
    drift from what is sent. The token counts are invented; the SHAPE — every
    key, and the fact that every value is a counter, a ratio or a bounded slug
    — is exactly what a real session produces.
    """
    from headroom.telemetry import session as telemetry_session

    class _Example:
        provider = "anthropic"
        model = "gpt-4o"
        original_tokens = 120_000
        attempted_input_tokens = 40_000
        optimized_tokens = 90_000
        provider_input_tokens = 90_000
        output_tokens = 1_200
        tokens_saved = 30_000
        cache_read_tokens = 60_000
        cache_write_tokens = 8_000
        cache_write_5m_tokens = 6_000
        cache_write_1h_tokens = 2_000
        uncached_input_tokens = 22_000
        status_code = 200
        total_latency_ms = 4_200.0
        overhead_ms = 48.0
        ttfb_ms = 900.0
        num_messages = 140
        client = "claude_code"
        transforms_applied = ("crush",)
        tags: dict[str, Any] = {}
        waste_signals = {"reread": 900, "reread_compressed": 400, "json_bloat": 50}

    rows: list[dict[str, Any]] = []
    aggregator = telemetry_session.SessionAggregator(rows.append)
    for index in range(3):
        aggregator.record(_Example(), now=1_000.0 + index * 45.0)
    aggregator.flush_all()
    return rows[-1]


@main.command()
@click.option(
    "--show",
    "show",
    is_flag=True,
    help="Print the exact payload the beacon would send, as JSON.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Print JSON only, with no explanatory footer. Implies --show.",
)
def telemetry(show: bool, as_json: bool) -> None:
    """Show what Headroom's telemetry beacon collects, and whether it is on.

    \b
    Examples:
        headroom telemetry           Current state and the two switches
        headroom telemetry --show    The exact payload, field by field
    """
    from headroom.telemetry.beacon import is_beacon_enabled, is_telemetry_enabled
    from headroom.telemetry.session import (
        DEFAULT_ENDPOINT,
        SCHEMA_VERSION,
        _gzip_enabled,
        read_install_id,
        resource_attributes,
    )

    beacon_on = is_beacon_enabled()
    local_on = is_telemetry_enabled()
    known_id = read_install_id()
    # Reported because it changes what goes on the wire, and because while the
    # transport is staged this is the fastest way to answer "was that upload
    # compressed?" without instrumenting anything.
    gzip_on = _gzip_enabled()

    if show or as_json:
        report = {
            "beacon_enabled": beacon_on,
            "schema_version": SCHEMA_VERSION,
            "endpoint": DEFAULT_ENDPOINT,
            "gzip": gzip_on,
            "install_id": known_id,
            # create_install_id=False, and it is load-bearing: the default
            # mints and persists an id, so building this report would be the
            # act that identifies the machine. Filtering the id out of the
            # OUTPUT is not enough -- by then the file exists.
            "resource": resource_attributes(create_install_id=False),
            "session_event": _sample_payload(),
        }
        click.echo(json.dumps(report, indent=2, sort_keys=True))
        if as_json:
            return
        click.echo()
        click.echo(
            "The COUNTS above are made up. The FIELDS are not: this payload is built\n"
            "by the same code the beacon uses, so every key that can ever be sent is\n"
            "shown, and no other key can be.\n"
            "\n"
            "Every value is a counter, a ratio, or a slug from a fixed list in the\n"
            "source. No prompts, code, file paths, tool names, project names or error\n"
            "text can appear — see headroom/telemetry/session.py for what is built,\n"
            "and deploy/beacon/worker.js for the allowlist it is filtered through on\n"
            "arrival. Your install_id is a random UUID, not a machine fingerprint;\n"
            "delete ~/.headroom/install_id to reset it. Your IP is never read."
        )
        return

    click.echo(f"Upload beacon:   {'ON (opt-out)' if beacon_on else 'OFF'}")
    click.echo(f"Local stats:     {'ON' if local_on else 'OFF (opt-in)'}")
    click.echo(f"Endpoint:        {DEFAULT_ENDPOINT}")
    click.echo(
        f"Compression:     {'gzip' if gzip_on else 'off (plain JSON)'}"
        f"{'' if gzip_on else '   — enable: HEADROOM_BEACON_GZIP=1'}"
    )
    click.echo(f"Install ID:      {known_id or '(none yet — created on first upload)'}")
    click.echo()
    click.echo("Turn the beacon off:   HEADROOM_BEACON=off   (or DO_NOT_TRACK=1)")
    click.echo("See the exact payload: headroom telemetry --show")
