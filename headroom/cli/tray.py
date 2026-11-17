"""`headroom tray` — standalone system tray for an already-running proxy."""

from __future__ import annotations

import click

from .main import main


@main.command("tray")
@click.option("--host", default="127.0.0.1", envvar="HEADROOM_HOST", show_default=True)
@click.option("--port", "-p", default=8787, envvar="HEADROOM_PORT", show_default=True)
def tray(host: str, port: int) -> None:
    """Show a system tray icon with savings, dashboard link and update alerts.

    Attach it to a proxy that is already running, or start both at once with
    `headroom proxy --tray`. Requires: pip install 'headroom-ai[tray]'
    """
    from headroom.tray.app import run_tray, tray_available
    from headroom.tray.core import base_url

    ok, reason = tray_available()
    if not ok:
        raise click.ClickException(f"Tray unavailable: {reason}")
    raise SystemExit(run_tray(base_url(host, port)))
