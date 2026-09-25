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
    # The cache-aware figure is the headline: what the removed tokens would
    # actually have been billed at. `cost_usd` (flat list price) is shown after
    # it as the ceiling, and only when the two differ enough to be worth the
    # ink -- on a cold-cache workload they legitimately coincide.
    cost = float(window.get("cost_effective_usd", window.get("cost_usd", 0.0)) or 0.0)
    ceiling = float(window.get("cost_usd", 0.0) or 0.0)
    money = _money(cost)
    if ceiling > 0 and abs(ceiling - cost) / ceiling >= 0.01:
        money = f"{money} (list {_money(ceiling)})"
    line = (
        f"{label:<12} {_bar(pct)} {pct:5.1f}%  "
        f"saved {_tokens(saved)} / {_tokens(before)} tokens  {money}"
    )
    # Second basis, only when the window has provider cache data: the share of
    # tokens that newly entered context. The bar's ratio counts a session's
    # cached history on every turn, so a long session reads near 0% there
    # while compression is working; this is the figure the dashboard leads with.
    if int(window.get("new_input_tokens", 0) or 0) > 0:
        new_pct = float(window.get("new_input_savings_percent", 0.0) or 0.0)
        line += f"  · of new input {new_pct:.1f}%"
    return line


#: What each pricing basis means for a reader deciding how much to trust the
#: number. Only the weaker ones are worth a line of output -- a catalog-priced
#: total needs no apology.
_BASIS_NOTES = {
    "list": (
        "Some events are priced at flat list price: either they predate "
        "cache-aware pricing or their provider reported no cache breakdown. "
        "The real saving is likely LOWER."
    ),
    "no-mix": (
        "No provider cache breakdown was available, so savings are priced at "
        "list. The real saving is likely LOWER."
    ),
    "unpriced": "Some events could not be priced at all and contribute $0.",
    "provider-ratio": (
        "Some models publish no cache pricing; their savings use a "
        "provider-level ratio rather than a per-model rate."
    ),
}


def _echo_basis_note(basis: Any) -> None:
    """Print a caveat when the headline rests on something weaker than the catalog."""
    note = _BASIS_NOTES.get(str(basis or ""))
    if note:
        click.echo("")
        click.echo(f"  note: {note}")


@main.command(name="savings")
@click.option("--json", "as_json", is_flag=True, help="Emit the raw report as JSON.")
@click.option(
    "--days",
    type=click.IntRange(min=1, max=savings_ledger.MAX_RETENTION_DAYS),
    default=savings_ledger.DEFAULT_RETENTION_DAYS,
    show_default=True,
    help=f"Retention/lookback window for the ledger, in days (max {savings_ledger.MAX_RETENTION_DAYS}).",
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

    click.echo("")
    click.echo(_window_line("Today", report.windows["today"]))
    click.echo(_window_line("Last 7 days", report.windows["last_7_days"]))
    click.echo(_window_line("Last 30 days", report.windows["last_30_days"]))
    if any(int(w.get("new_input_tokens", 0) or 0) > 0 for w in report.windows.values()):
        click.echo(
            "  % is of all forwarded input (cached history recounted every turn); "
            "'of new input' is of tokens that newly entered context."
        )

    if report.by_model:
        click.echo("")
        click.echo("Cost avoided per model:")
        for row in report.by_model:
            effective = float(row.get("cost_effective_usd", row["cost_usd"]))
            click.echo(f"  {str(row['model']):<24} {_money(effective)}")

    _echo_basis_note(report.lifetime.get("basis"))

    if report.by_client:
        click.echo("")
        click.echo("Savings by client:")
        for row in report.by_client:
            click.echo(
                f"  {str(row['client']):<24} {int(row['calls']):,} calls · "
                f"{_tokens(int(row['tokens_saved']))} tokens saved"
            )
