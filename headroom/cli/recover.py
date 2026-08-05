"""Recovery commands for state left behind by interrupted wrappers."""

from __future__ import annotations

import os
from pathlib import Path

import click

from headroom.cli.main import main
from headroom.providers.codex.recovery import (
    audit_codex_history,
    discover_dangling_homes,
    discover_referenced_temp_homes,
    discover_retained_sources,
    recover_codex_home,
)


@main.group()
def recover() -> None:
    """Recover agent state left in a temporary Headroom home."""


@recover.command("codex")
@click.option(
    "sources",
    "--source",
    type=click.Path(path_type=Path, exists=True, file_okay=False, resolve_path=True),
    multiple=True,
    help="Temporary Codex home to merge. Repeat to merge more than one.",
)
@click.option(
    "--target",
    type=click.Path(path_type=Path, file_okay=False, resolve_path=True),
    help="Active Codex home. Defaults to CODEX_HOME or ~/.codex.",
)
@click.option("--yes", is_flag=True, help="Apply the recovery without prompting.")
def recover_codex(sources: tuple[Path, ...], target: Path | None, yes: bool) -> None:
    """Merge sessions and configuration from dangling Codex homes."""
    target = target or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    selected_sources = list(sources) or [
        *discover_dangling_homes(),
        *discover_retained_sources(target),
    ]
    if not selected_sources:
        deleted_sources = [
            source for source in discover_referenced_temp_homes(target) if not source.exists()
        ]
        if deleted_sources:
            click.echo("Referenced temporary Codex homes were already deleted:")
            for source in deleted_sources:
                click.echo(f"  {source}")
        audit = audit_codex_history(target)
        if audit is not None:
            click.echo(
                f"Durable Codex history: {audit.indexed} indexed chats "
                f"({audit.active} active, {audit.archived} archived)."
            )
            if audit.unindexed_rollouts:
                click.echo(
                    "Surviving rollout files missing from the thread database: "
                    f"{len(audit.unindexed_rollouts)}"
                )
                for session_id in audit.unindexed_rollouts:
                    click.echo(f"  {session_id}")
            if audit.history_without_rollout:
                click.echo(
                    "History-only records without a surviving rollout: "
                    f"{len(audit.history_without_rollout)}"
                )
                for session_id in audit.history_without_rollout:
                    click.echo(f"  {session_id}")
                click.echo("Their full transcripts cannot be restored without a retained rollout.")
            if audit.indexed:
                click.echo("Run `codex resume --all` to show chats from every working directory.")
        click.echo("No recoverable Headroom Codex homes were found.")
        return

    click.echo(f"Target Codex home: {target}")
    click.echo("Sources:")
    for source in selected_sources:
        click.echo(f"  {source}")
    click.echo("Both the current target and each source will be backed up before merging.")
    if not yes and not click.confirm("Recover these Codex homes?", default=True):
        click.echo("Recovery cancelled. No Codex state was changed.")
        return

    for source in selected_sources:
        try:
            report = recover_codex_home(source=source, target=target)
        except (OSError, RuntimeError, ValueError) as exc:
            raise click.ClickException(f"Codex recovery failed: {exc}") from exc
        click.echo(f"Recovery complete. Backup retained at {report.backup_dir}")
