"""Network capture and differential report commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from .main import main


@main.group("capture")
def capture_group() -> None:
    """Capture and compare network traffic for Headroom investigations."""


@capture_group.command("network-diff")
@click.option(
    "--direct",
    "direct_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSONL capture from the direct Claude Code lane.",
)
@click.option(
    "--headroom",
    "headroom_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSONL capture from the Headroom-proxied Claude Code lane.",
)
@click.option(
    "--output",
    "markdown_output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write a Markdown report to this path. Defaults to stdout.",
)
@click.option(
    "--json-output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional machine-readable JSON diff output.",
)
@click.option(
    "--pair-by",
    type=click.Choice(["path", "route"]),
    default="path",
    show_default=True,
    help="Pair exchanges by method+path or by method+host+path.",
)
def network_diff(
    direct_path: Path,
    headroom_path: Path,
    markdown_output: Path | None,
    json_output: Path | None,
    pair_by: str,
) -> None:
    """Compare direct and Headroom MITM capture JSONL files."""

    from headroom.capture.network_diff import (
        compare_captures,
        load_capture_file,
        render_markdown_report,
    )

    direct = load_capture_file(direct_path, fallback_lane="direct")
    headroom = load_capture_file(headroom_path, fallback_lane="headroom")
    diff = compare_captures(direct, headroom, pair_by=pair_by)
    markdown = render_markdown_report(diff)

    if markdown_output:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(markdown, encoding="utf-8")
        click.echo(f"Wrote Markdown report: {markdown_output}")
    else:
        click.echo(markdown)

    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        json_output.write_text(json.dumps(diff.to_dict(), indent=2), encoding="utf-8")
        click.echo(f"Wrote JSON report: {json_output}")
