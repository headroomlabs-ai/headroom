"""CLI: show durable compression savings over time.

Reads the append-only savings ledger (``~/.headroom/savings_events.jsonl``,
written by both the MCP tool path and the proxy) and renders a cost-avoided
summary with Today / Last 7 days / All time bars plus per-model and
per-client breakdowns. Durable across restarts; aggregated on read.

When the local proxy runs a model router (a routing-stats provider, see
``headroom.proxy.routing_stats``), its savings are shown beside: what each
moved request cost on the cheaper model against the model it asked for.
"""

from __future__ import annotations

import json
import os
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


def _proxy_routing() -> dict[str, Any] | None:
    """The running proxy's routing rollup (``/stats`` ``routing``), or None.

    None when no proxy answers or it routes nothing: the ledger sections are
    unaffected either way, so a stopped proxy only hides this section.
    """
    from headroom.install.health import probe_json
    from headroom.install.state import list_manifests

    from .doctor import resolve_probe_port

    env_port = os.environ.get("HEADROOM_PORT")
    requested = int(env_port) if env_port and env_port.isdigit() else None
    port = resolve_probe_port(requested, list_manifests())
    stats = probe_json(f"http://127.0.0.1:{port}/stats", timeout=2.0)
    routing = stats.get("routing") if isinstance(stats, dict) else None
    if not isinstance(routing, dict) or not routing.get("decisions"):
        return None
    return routing


#: How the routing section names the provider's ``window`` (routing_stats contract).
_ROUTING_WINDOWS = {"lifetime": "lifetime", "session": "this proxy session"}


def _echo_routing(routing: dict[str, Any]) -> None:
    # Per the routing_stats contract: a pair's `enforced` requests are the ones whose
    # model was rewritten (holdouts and shadow decisions are not), and the saving is
    # the sum of the pairs' signed `savings_usd`.
    pairs = [p for p in routing.get("pairs") or [] if p.get("direction") != "same"]
    moved = sum(int(p.get("enforced") or 0) for p in pairs)
    saved = sum(float(p.get("savings_usd") or 0.0) for p in pairs)
    window = _ROUTING_WINDOWS.get(str(routing.get("window")), "all decisions on record")
    click.echo("")
    click.echo(
        f"Model routing ({window}): {moved:,} of {int(routing.get('decisions') or 0):,} "
        f"requests moved · {_money(saved)} saved"
    )
    for pair in pairs:
        pair_name = f"{pair.get('requested')} → {pair.get('served')}"
        priced = (
            _money(float(pair.get("savings_usd") or 0.0))
            if int(pair.get("measured") or 0)
            else "not priced (no usage reported)"
        )
        click.echo(f"  {pair_name:<44} {int(pair.get('enforced') or 0):,} moved  {priced}")
    click.echo(
        "  Saved = each moved request's usage priced on the model asked for, minus its cost "
        "on the model served."
    )


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
    routing = _proxy_routing()

    if as_json:
        data = report.to_dict()
        if routing is not None:  # no router installed: not even an empty key
            data["routing"] = routing
        click.echo(json.dumps(data, indent=2))
        return

    lifetime = report.lifetime
    calls = int(lifetime.get("calls", 0) or 0)
    if calls == 0 and routing is not None:
        _echo_routing(routing)
        return
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

    if routing is not None:
        _echo_routing(routing)
