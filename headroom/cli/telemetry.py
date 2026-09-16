"""Telemetry transparency CLI commands."""

from __future__ import annotations

import json

import click
from rich.console import Console
from rich.table import Table

from headroom.cli.main import main
from headroom.telemetry.surfaces import list_telemetry_surface_dicts


@main.group("telemetry")
def telemetry_group() -> None:
    """Inspect telemetry and observability surfaces."""


@telemetry_group.command("list")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def list_surfaces(as_json: bool) -> None:
    """List every known telemetry/observability surface."""

    surfaces = list_telemetry_surface_dicts()
    if as_json:
        click.echo(json.dumps(surfaces, indent=2, sort_keys=True))
        return

    table = Table(title="Headroom Telemetry Surfaces")
    table.add_column("Surface", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Leaves Host", no_wrap=True)
    table.add_column("Prompt Content", no_wrap=True)
    table.add_column("Observe")
    table.add_column("Retention")

    for surface in surfaces:
        table.add_row(
            surface["surface"],
            surface["status"],
            "yes" if surface["leaves_host_by_default"] else "no",
            "yes" if surface["includes_prompt_content"] else "no",
            surface["observe"],
            surface["retention"],
        )

    Console().print(table)
