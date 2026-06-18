"""CLI: show durable compression savings over time.

Reads the append-only savings ledger (``~/.headroom/savings_events.jsonl``,
written by both the MCP tool path and the proxy) and renders a cost-avoided
summary with Today / Last 7 days / All time bars plus per-model and
per-client breakdowns. Durable across restarts; aggregated on read.
"""

from __future__ import annotations

import json
from typing import Any

import click

from headroom import savings_ledger

from .main import main

_BAR_WIDTH = 16


def _bar(percent: float, width: int = _BAR_WIDTH) -> str:
    filled = int(round(percent / 100 * width))
    filled = max(0, min(width, filled))
    return "█" * filled + "░" * (width - filled)


def _money(value: float, places: int = 4) -> str:
    return f"${value:,.{places}f}"


def _tokens(value: int) -> str:
    return f"{value:,}"


def _window_line(label: str, window: dict[str, Any]) -> str:
    pct = float(window.get("savings_percent", 0.0) or 0.0)
    saved = int(window.get("tokens_saved", 0) or 0)
    before = int(window.get("tokens_before", 0) or 0)
    cost = float(window.get("cost_usd", 0.0) or 0.0)
    return (
        f"{label:<11} {_bar(pct)} {pct:5.1f}%  "
        f"saved {_tokens(saved)} / {_tokens(before)} tokens  {_money(cost)}"
    )


@main.command(name="savings")
@click.option("--json", "as_json", is_flag=True, help="Emit the raw report as JSON.")
@click.option(
    "--days",
    type=int,
    default=savings_ledger.DEFAULT_RETENTION_DAYS,
    show_default=True,
    help="Retention/lookback window for the ledger, in days.",
)
@click.option("--reset", is_flag=True, help="Delete the savings ledger and start fresh.")
def savings(as_json: bool, days: int, reset: bool) -> None:
    """Show durable compression savings over time."""

    if reset:
        path = savings_ledger._resolve_path(None)
        if path.exists():
            path.unlink()
            click.echo(f"Ledger reset: {path}")
        else:
            click.echo("Nothing to reset — ledger does not exist.")
        return

    report = savings_ledger.aggregate_savings(retention_days=days)

    if as_json:
        click.echo(json.dumps(report.to_dict(), indent=2))
        return

    lifetime = report.lifetime
    calls = int(lifetime.get("calls", 0) or 0)
    if calls == 0:
        click.echo("No savings recorded yet.")
        click.echo(
            "Compress via the Headroom MCP tool or route traffic through the "
            "proxy, then re-run `headroom savings`."
        )
        click.echo(f"Ledger: {report.path}")
        return

    saved = int(lifetime.get("tokens_saved", 0) or 0)
    cost = float(lifetime.get("cost_usd", 0.0) or 0.0)

    click.echo("")
    click.echo(
        f"  {_money(cost, 2)}  cost avoided   "
        f"{report.top_model} · {calls:,} calls · {_tokens(saved)} tokens saved"
    )
    click.echo("")
    click.echo(_window_line("Today", report.windows["today"]))
    click.echo(_window_line("Last 7 days", report.windows["last_7_days"]))
    click.echo(_window_line("All time", report.windows["all_time"]))

    if report.by_model:
        click.echo("")
        click.echo("Cost avoided per model (all time):")
        for row in report.by_model:
            click.echo(f"  {str(row['model']):<24} {_money(float(row['cost_usd']))}")

    if report.by_client:
        click.echo("")
        click.echo("Savings by client:")
        for row in report.by_client:
            click.echo(
                f"  {str(row['client']):<24} {int(row['calls']):,} calls · "
                f"{_tokens(int(row['tokens_saved']))} tokens saved"
            )


