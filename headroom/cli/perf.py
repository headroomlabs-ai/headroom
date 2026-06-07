"""Performance analysis CLI command."""

import csv
import io
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
    "output_format",
    type=click.Choice(["text", "json", "csv"]),
    default="text",
    help="Output format (default: text). json/csv emit machine-readable data.",
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
        headroom perf                      Analyze last 7 days
        headroom perf --hours 24           Analyze last 24 hours
        headroom perf --raw                Show raw parsed records
        headroom perf --format json        Aggregated report as JSON
        headroom perf --format csv --hours 24 > last-24h.csv
        headroom perf --format json --raw  Raw records as a JSON array
    """
    from headroom.perf.analyzer import (
        PERF_RECORD_FIELDS,
        build_perf_summary,
        format_report,
        parse_log_files,
        perf_records_as_dicts,
    )

    report = parse_log_files(last_n_hours=hours)

    if output_format == "json":
        payload = perf_records_as_dicts(report) if raw else build_perf_summary(report)
        click.echo(json.dumps(payload, indent=2))
        return

    if output_format == "csv":
        buf = io.StringIO()
        if raw:
            writer = csv.DictWriter(buf, fieldnames=PERF_RECORD_FIELDS)
            writer.writeheader()
            for rec in perf_records_as_dicts(report):
                row = dict(rec)
                # Flatten the transforms list for a single CSV cell.
                row["transforms"] = ",".join(row.get("transforms", []))
                writer.writerow(row)
        else:
            # Non-raw CSV is the per-model breakdown — the most useful tabular
            # aggregate for spreadsheets and longitudinal charts.
            summary = build_perf_summary(report)
            fieldnames = [
                "model",
                "requests",
                "tokens_before",
                "tokens_after",
                "tokens_saved",
                "savings_pct",
                "list_price_per_mtok",
            ]
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary["by_model"]:
                writer.writerow(row)
        click.echo(buf.getvalue(), nl=False)
        return

    # Default: human-readable text.
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
