"""Performance analysis CLI command."""

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
    "output_format",
    type=click.Choice(["text", "json", "csv"], case_sensitive=False),
    default="text",
    help="Output format: text (default), json, or csv",
)
def perf(hours: float, raw: bool, output_format: str) -> None:
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
        headroom perf                Analyze last 7 days
        headroom perf --hours 24     Analyze last 24 hours
        headroom perf --raw          Show raw parsed records
        headroom perf --format json  Output as JSON (for CI / agents)
        headroom perf --format csv   Output as CSV (for spreadsheets)
    """
    from headroom.perf.analyzer import (
        format_report,
        format_report_csv,
        format_report_json,
        parse_log_files,
    )

    report = parse_log_files(last_n_hours=hours)

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
    elif output_format == "json":
        click.echo(format_report_json(report))
    elif output_format == "csv":
        click.echo(format_report_csv(report))
    else:
        click.echo(format_report(report))
