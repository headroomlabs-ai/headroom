"""Traffic audit CLI commands."""

from pathlib import Path

import click

from .main import main


@main.command(name="audit-reads")
@click.option(
    "--path",
    "root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Transcript directory to audit (default: ~/.claude/projects)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"]),
    default="text",
    help="Output format (default: text)",
)
@click.option(
    "--simulate-maturation",
    is_flag=True,
    help="Also simulate Mechanism B (read maturation): re-read rates, "
    "never-touched-again share, quiesce-window coverage, at-risk edits.",
)
@click.option(
    "--codex",
    "codex_mode",
    is_flag=True,
    help="Audit Codex transcripts instead (shell-based reads: cat/sed/rtk read). "
    "Default path becomes ~/.codex/sessions.",
)
def audit_reads_cmd(
    root: Path | None, output_format: str, simulate_maturation: bool, codex_mode: bool
) -> None:
    """Audit Read-tool traffic for compression opportunities.

    Streams local Claude Code transcripts (read-only) and sizes the
    addressable bytes for each Read mechanism: identical repeats, subset
    containment, write-readback, stale reads, line-number scaffolding,
    context residency, and cache-death windows.

    \b
    Run this on a deployment's transcripts BEFORE tuning compression
    defaults — opportunity sizes vary heavily by workload.

    \b
    Examples:
        headroom audit-reads
        headroom audit-reads --path /var/transcripts --format json
        headroom audit-reads --codex
    """
    import json as _json

    from headroom.audit.reads import audit_reads, render_text

    if codex_mode:
        if root is None:
            root = Path.home() / ".codex" / "sessions"
            if not root.exists():
                raise click.ClickException(
                    f"{root} does not exist — pass --path to the Codex sessions directory"
                )
        from headroom.audit.codex import audit_codex, render_codex_text

        codex_report = audit_codex(root)
        if codex_report.sessions == 0:
            raise click.ClickException(f"no *.jsonl transcripts found under {root}")
        if output_format == "json":
            click.echo(_json.dumps(codex_report.to_dict(), indent=2, sort_keys=True))
        else:
            click.echo(render_codex_text(codex_report))
        return

    if root is None:
        root = Path.home() / ".claude" / "projects"
        if not root.exists():
            raise click.ClickException(
                f"{root} does not exist — pass --path to the transcript directory"
            )

    report = audit_reads(root)
    if report.sessions == 0:
        raise click.ClickException(f"no *.jsonl transcripts found under {root}")

    sim = None
    if simulate_maturation:
        from headroom.audit.maturation import render_sim_text
        from headroom.audit.maturation import simulate_maturation as run_sim

        sim = run_sim(root)

    if output_format == "json":
        data = _json.loads(report.to_json())
        if sim is not None:
            data["maturation_simulation"] = sim.to_dict()
        click.echo(_json.dumps(data, indent=2, sort_keys=True))
    else:
        click.echo(render_text(report))
        if sim is not None:
            click.echo()
            click.echo(render_sim_text(sim))
