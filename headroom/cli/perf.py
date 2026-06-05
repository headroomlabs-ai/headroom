"""Performance analysis CLI command."""

import json

import click

from .main import main


@main.command()
@click.option(
    "--hours",
    type=float,
    default=168.0,
    help="Analyze logs from the last N hours (default: 168 = 7 days)",
)
@click.option("--raw", is_flag=True, help="Show raw PERF records instead of report")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json", "csv"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format. `json` emits aggregated metrics, `csv` emits one row per request.",
)
def perf(hours: float, raw: bool, fmt: str) -> None:
    """Analyze proxy performance from logs.

    \b
    Reads logs from ~/.headroom/logs/proxy.log and shows:
    - Token savings and compression effectiveness
    - Cache hit rates and prefix stability
    - Transform and routing breakdown
    - TOIN learning status
    - Actionable recommendations

    \b
    Examples:
        headroom perf                       Analyze last 7 days
        headroom perf --hours 24            Analyze last 24 hours
        headroom perf --raw                 Show raw parsed records
        headroom perf --format json         Emit aggregated metrics as JSON
        headroom perf --format csv          Emit per-request rows as CSV
    """
    from headroom.perf.analyzer import (
        format_csv,
        format_report,
        parse_log_files,
        summary_dict,
    )

    report = parse_log_files(last_n_hours=hours)

    fmt = fmt.lower()
    if fmt == "json":
        click.echo(json.dumps(summary_dict(report), indent=2))
        return
    if fmt == "csv":
        click.echo(format_csv(report), nl=False)
        return

    if raw:
        for r in report.perf_records:
            click.echo(
                f"{r.timestamp} {r.request_id} model={r.model} msgs={r.num_messages} "
                f"before={r.tokens_before} after={r.tokens_after} saved={r.tokens_saved} "
                f"cache_read={r.cache_read} cache_write={r.cache_write} "
                f"cache_hit={r.cache_hit_pct}% opt={r.optimization_ms:.0f}ms"
            )
        if not report.perf_records:
            click.echo("No PERF records found. Run the proxy first: headroom proxy")
    else:
        click.echo(format_report(report))
