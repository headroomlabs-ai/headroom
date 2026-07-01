"""Proxy benchmark helpers for comparing Headroom /stats snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import click

from .main import main


def _load_snapshot(source: str) -> dict:
    if source.startswith(("http://", "https://")):
        try:
            with urlopen(source, timeout=10) as response:  # noqa: S310 - user-provided local stats URL
                return json.loads(response.read().decode("utf-8"))
        except (OSError, URLError, json.JSONDecodeError) as exc:
            raise click.ClickException(f"failed to read stats URL {source}: {exc}") from exc

    path = Path(source)
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"failed to read stats file {path}: {exc}") from exc


def _input_tokens(snapshot: dict, label: str) -> int:
    try:
        value = snapshot["tokens"]["input"]
    except KeyError as exc:
        raise click.ClickException(f"{label} snapshot is missing tokens.input") from exc
    try:
        tokens = int(value)
    except (TypeError, ValueError) as exc:
        raise click.ClickException(f"{label} snapshot tokens.input must be an integer") from exc
    if tokens < 0:
        raise click.ClickException(f"{label} snapshot tokens.input must be non-negative")
    return tokens


def compare_snapshots(baseline: dict, optimized: dict) -> dict[str, int | float]:
    """Compare passthrough and optimized proxy stats snapshots."""
    baseline_input = _input_tokens(baseline, "baseline")
    optimized_input = _input_tokens(optimized, "optimized")
    tokens_saved = max(0, baseline_input - optimized_input)
    savings_percent = round(100 * tokens_saved / baseline_input, 2) if baseline_input else 0.0
    return {
        "baseline_input_tokens": baseline_input,
        "optimized_input_tokens": optimized_input,
        "tokens_saved": tokens_saved,
        "savings_percent": savings_percent,
    }


@main.group(name="proxy-benchmark")
def proxy_benchmark_cmd() -> None:
    """Compare Headroom proxy stats from benchmark runs."""


@proxy_benchmark_cmd.command(name="compare")
@click.argument("baseline")
@click.argument("optimized")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format (default: text)",
)
def compare_cmd(baseline: str, optimized: str, output_format: str) -> None:
    """Compare passthrough and optimized /stats snapshots.

    BASELINE and OPTIMIZED can be JSON files or http(s) URLs returning the
    Headroom proxy /stats payload. For local LLM prefill benchmarks, capture
    the baseline after a --no-optimize run and the optimized snapshot after
    rerunning the same task without --no-optimize.
    """
    result = compare_snapshots(_load_snapshot(baseline), _load_snapshot(optimized))
    if output_format == "json":
        click.echo(json.dumps(result, indent=2, sort_keys=True))
        return

    click.echo("Local LLM prefill benchmark")
    click.echo(f"  baseline input tokens:  {result['baseline_input_tokens']:,}")
    click.echo(f"  optimized input tokens: {result['optimized_input_tokens']:,}")
    click.echo(f"  tokens saved:           {result['tokens_saved']:,}")
    click.echo(f"  savings:                {result['savings_percent']:.2f}%")
