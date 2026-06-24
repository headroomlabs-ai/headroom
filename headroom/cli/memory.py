"""Memory management CLI commands."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import click

from ..memory.adapters.sqlite import SQLiteMemoryStore
from ..memory.models import Memory, ScopeLevel
from ..memory.ports import MemoryFilter
from ._utils.formatting import (
    console,
    format_age,
    format_bytes,
    print_error,
    print_stats,
    print_success,
    print_table,
    print_warning,
    truncate,
)
from ._utils.parsers import parse_duration
from .main import main


def db_path_option(fn: Any) -> Any:
    """Shared --db-path option for memory commands."""
    return click.option(
        "--db-path",
        type=click.Path(),
        default="headroom_memory.db",
        help="Path to the memory database file.",
        show_default=True,
    )(fn)


def get_store(db_path: str) -> SQLiteMemoryStore:
    """Get a SQLiteMemoryStore instance."""
    return SQLiteMemoryStore(db_path)


def get_scope_label(memory: Memory) -> str:
    """Get a human-readable scope label for a memory."""
    if memory.turn_id is not None:
        return "TURN"
    if memory.agent_id is not None:
        return "AGENT"
    if memory.session_id is not None:
        return "SESSION"
    return "USER"


def _get_stats(store: SQLiteMemoryStore) -> dict[str, Any]:
    """Get statistics about the memory store.

    This is a helper function that directly queries the database
    to get stats, since SQLiteMemoryStore doesn't have a get_stats method.
    """
    stats: dict[str, Any] = {}

    with store._get_conn() as conn:
        # Total count
        cursor = conn.execute("SELECT COUNT(*) FROM memories")
        stats["total_count"] = cursor.fetchone()[0]

        # Database file size
        try:
            stats["db_size_bytes"] = os.path.getsize(store.db_path)
        except OSError:
            stats["db_size_bytes"] = 0

        # Count by scope level
        by_scope: dict[str, int] = {}

        cursor = conn.execute(
            "SELECT COUNT(*) FROM memories "
            "WHERE session_id IS NULL AND agent_id IS NULL AND turn_id IS NULL"
        )
        by_scope["USER"] = cursor.fetchone()[0]

        cursor = conn.execute(
            "SELECT COUNT(*) FROM memories "
            "WHERE session_id IS NOT NULL AND agent_id IS NULL AND turn_id IS NULL"
        )
        by_scope["SESSION"] = cursor.fetchone()[0]

        cursor = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE agent_id IS NOT NULL AND turn_id IS NULL"
        )
        by_scope["AGENT"] = cursor.fetchone()[0]

        cursor = conn.execute("SELECT COUNT(*) FROM memories WHERE turn_id IS NOT NULL")
        by_scope["TURN"] = cursor.fetchone()[0]

        stats["by_scope"] = by_scope

        # Count by age buckets
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        by_age: dict[str, int] = {}

        one_day_ago = (now - timedelta(days=1)).isoformat()
        seven_days_ago = (now - timedelta(days=7)).isoformat()
        thirty_days_ago = (now - timedelta(days=30)).isoformat()

        cursor = conn.execute("SELECT COUNT(*) FROM memories WHERE created_at > ?", (one_day_ago,))
        by_age["< 1 day"] = cursor.fetchone()[0]

        cursor = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE created_at <= ? AND created_at > ?",
            (one_day_ago, seven_days_ago),
        )
        by_age["1-7 days"] = cursor.fetchone()[0]

        cursor = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE created_at <= ? AND created_at > ?",
            (seven_days_ago, thirty_days_ago),
        )
        by_age["7-30 days"] = cursor.fetchone()[0]

        cursor = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE created_at <= ?", (thirty_days_ago,)
        )
        by_age["> 30 days"] = cursor.fetchone()[0]

        stats["by_age"] = by_age

        # Oldest memory age
        cursor = conn.execute("SELECT MIN(created_at) FROM memories")
        oldest = cursor.fetchone()[0]
        if oldest:
            oldest_dt = datetime.fromisoformat(oldest)
            delta = now - oldest_dt
            stats["oldest_memory_age_days"] = delta.total_seconds() / 86400.0
        else:
            stats["oldest_memory_age_days"] = None

        # Average importance
        cursor = conn.execute("SELECT AVG(importance) FROM memories")
        avg_imp = cursor.fetchone()[0]
        stats["avg_importance"] = float(avg_imp) if avg_imp is not None else 0.0

    return stats


def _search_content(store: SQLiteMemoryStore, query: str, limit: int = 50) -> list[Memory]:
    """Search memories by content using LIKE.

    This is a helper function that directly queries the database.
    """
    escaped_query = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    search_pattern = f"%{escaped_query}%"

    with store._get_conn() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM memories
            WHERE content LIKE ? ESCAPE '\\'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (search_pattern, limit),
        )
        return [store._row_to_memory(row) for row in cursor]


def _export_all(store: SQLiteMemoryStore) -> list[dict[str, Any]]:
    """Export all memories as a list of dictionaries."""
    with store._get_conn() as conn:
        cursor = conn.execute("SELECT * FROM memories ORDER BY created_at ASC")
        memories = [store._row_to_memory(row) for row in cursor]

    return [m.to_dict() for m in memories]


def _import_memories(store: SQLiteMemoryStore, memories: list[dict[str, Any]]) -> int:
    """Import memories from a list of dictionaries."""
    if not memories:
        return 0

    imported_count = 0
    with store._get_conn() as conn:
        for mem_dict in memories:
            try:
                memory = Memory.from_dict(mem_dict)
                row = store._memory_to_row(memory)
                conn.execute(
                    """
                    INSERT OR REPLACE INTO memories (
                        id, content, user_id, session_id, agent_id, turn_id,
                        created_at, valid_from, valid_until,
                        category, importance,
                        supersedes, superseded_by, promoted_from, promotion_chain,
                        access_count, last_accessed,
                        entity_refs, embedding, metadata
                    ) VALUES (
                        :id, :content, :user_id, :session_id, :agent_id, :turn_id,
                        :created_at, :valid_from, :valid_until,
                        :category, :importance,
                        :supersedes, :superseded_by, :promoted_from, :promotion_chain,
                        :access_count, :last_accessed,
                        :entity_refs, :embedding, :metadata
                    )
                    """,
                    row,
                )
                imported_count += 1
            except (KeyError, ValueError, TypeError):
                # Skip malformed entries
                continue

        conn.commit()

    return imported_count


@main.group()
@click.pass_context
def memory(ctx: click.Context) -> None:
    """Manage memories stored in Headroom.

    \b
    Examples:
        headroom memory list                     List all stored memories
        headroom memory list --limit 10          List the 10 most recent memories
        headroom memory list --scope USER        List only USER-level memories
        headroom memory list --since 7d          List memories from the last 7 days
        headroom memory show <id>                Show full details of a memory
        headroom memory stats                    Show memory statistics
        headroom memory edit <id> --content ...  Edit a memory's content
        headroom memory delete <id>              Delete a memory
        headroom memory prune --older-than 30d   Delete memories older than 30 days
        headroom memory purge --confirm          Delete ALL memories
        headroom memory export --output file.json  Export all memories to JSON
        headroom memory import file.json         Import memories from JSON
    """
    pass


@memory.command("list")
@db_path_option
@click.option("--limit", "-n", type=int, default=50, help="Maximum number of memories to show.")
@click.option("--session", "-s", "session_id", type=str, help="Filter by session ID.")
@click.option(
    "--scope",
    type=click.Choice(["USER", "SESSION", "AGENT", "TURN"], case_sensitive=False),
    help="Filter by scope level.",
)
@click.option(
    "--since",
    "since_duration",
    type=str,
    help="Show memories created within duration (e.g., 7d, 2w, 1m).",
)
@click.option("--search", "-q", "search_query", type=str, help="Search memories by content.")
@click.pass_context
def list_memories(
    ctx: click.Context,
    db_path: str,
    limit: int,
    session_id: str | None,
    scope: str | None,
    since_duration: str | None,
    search_query: str | None,
) -> None:
    """List stored memories with optional filters.

    \b
    Examples:
        headroom memory list                  List most recent memories
        headroom memory list --limit 10       Show only 10 memories
        headroom memory list --scope USER     Show only USER-level memories
        headroom memory list --since 7d       Show memories from the last 7 days
        headroom memory list --search "api"   Search for memories containing "api"
    """
    store = get_store(db_path)

    try:
        if search_query:
            # Use text search
            memories = _search_content(store, search_query, limit=limit)
        else:
            # Build filter
            filter_kwargs: dict[str, Any] = {
                "limit": limit,
                "order_by": "created_at",
                "order_desc": True,
            }

            if session_id:
                filter_kwargs["session_id"] = session_id

            if scope:
                scope_level = ScopeLevel[scope.upper()]
                filter_kwargs["scope_levels"] = [scope_level]

            if since_duration:
                duration = parse_duration(since_duration)
                cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - duration
                filter_kwargs["created_after"] = cutoff

            mem_filter = MemoryFilter(**filter_kwargs)
            memories = asyncio.run(store.query(mem_filter))

        if not memories:
            click.echo("No memories found.")
            return

        # Build table
        headers = ["ID", "SCOPE", "AGE", "IMPORTANCE", "CONTENT"]
        rows: list[list[str]] = []

        for mem in memories:
            rows.append(
                [
                    mem.id[:8],
                    get_scope_label(mem),
                    format_age(mem.created_at),
                    f"{mem.importance:.2f}",
                    truncate(mem.content.replace("\n", " "), 50),
                ]
            )

        title = f"Memories ({len(memories)} shown)"
        if search_query:
            title = f"Search results for '{search_query}' ({len(memories)} found)"

        print_table(headers, rows, title=title)

    except click.BadParameter as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Failed to list memories: {e}")
        sys.exit(1)


@memory.command("show")
@db_path_option
@click.argument("memory_id", type=str)
@click.option("--json", "output_json", is_flag=True, help="Output as raw JSON.")
@click.pass_context
def show_memory(
    ctx: click.Context,
    db_path: str,
    memory_id: str,
    output_json: bool,
) -> None:
    """Show full details of a single memory.

    \b
    Examples:
        headroom memory show abc123            Show memory details
        headroom memory show abc123 --json     Output as raw JSON
    """
    store = get_store(db_path)

    try:
        # Try to find the memory - support partial IDs
        mem = asyncio.run(store.get(memory_id))

        # If not found with exact ID, try partial match
        if mem is None:
            # Query all and filter by prefix
            all_memories = asyncio.run(store.query(MemoryFilter(limit=1000)))
            matches = [m for m in all_memories if m.id.startswith(memory_id)]

            if len(matches) == 0:
                print_error(f"Memory not found: {memory_id}")
                sys.exit(1)
            elif len(matches) > 1:
                print_error(f"Ambiguous ID '{memory_id}'. Matches: {[m.id[:8] for m in matches]}")
                sys.exit(1)
            else:
                mem = matches[0]

        if output_json:
            click.echo(json.dumps(mem.to_dict(), indent=2, default=str))
            return

        # Display formatted output
        console.print(f"\n[bold]Memory ID:[/bold] {mem.id}")
        console.print(f"[bold]Scope:[/bold] {get_scope_label(mem)}")
        console.print(f"[bold]User ID:[/bold] {mem.user_id}")

        if mem.session_id:
            console.print(f"[bold]Session ID:[/bold] {mem.session_id}")
        if mem.agent_id:
            console.print(f"[bold]Agent ID:[/bold] {mem.agent_id}")
        if mem.turn_id:
            console.print(f"[bold]Turn ID:[/bold] {mem.turn_id}")

        console.print(f"[bold]Created:[/bold] {mem.created_at.isoformat()}")
        console.print(f"[bold]Valid From:[/bold] {mem.valid_from.isoformat()}")
        if mem.valid_until:
            console.print(f"[bold]Valid Until:[/bold] {mem.valid_until.isoformat()}")

        console.print(f"[bold]Importance:[/bold] {mem.importance:.2f}")
        console.print(f"[bold]Access Count:[/bold] {mem.access_count}")

        if mem.last_accessed:
            console.print(f"[bold]Last Accessed:[/bold] {mem.last_accessed.isoformat()}")

        if mem.supersedes:
            console.print(f"[bold]Supersedes:[/bold] {mem.supersedes}")
        if mem.superseded_by:
            console.print(f"[bold]Superseded By:[/bold] {mem.superseded_by}")
        if mem.promoted_from:
            console.print(f"[bold]Promoted From:[/bold] {mem.promoted_from}")

        if mem.entity_refs:
            console.print(f"[bold]Entity Refs:[/bold] {', '.join(mem.entity_refs)}")

        if mem.metadata:
            console.print(f"[bold]Metadata:[/bold] {json.dumps(mem.metadata, indent=2)}")

        console.print(f"\n[bold]Content:[/bold]\n{mem.content}")

    except Exception as e:
        print_error(f"Failed to show memory: {e}")
        sys.exit(1)


@memory.command("stats")
@db_path_option
@click.pass_context
def show_stats(ctx: click.Context, db_path: str) -> None:
    """Show memory store statistics.

    Displays counts by scope level, age distribution, and storage information.
    """
    store = get_store(db_path)

    try:
        stats = _get_stats(store)

        # Format stats for display
        formatted: dict[str, Any] = {}

        formatted["Total Memories"] = stats["total_count"]
        formatted["Database Size"] = format_bytes(stats["db_size_bytes"])

        # By scope
        by_scope = stats.get("by_scope", {})
        formatted["USER scope"] = by_scope.get("USER", 0)
        formatted["SESSION scope"] = by_scope.get("SESSION", 0)
        formatted["AGENT scope"] = by_scope.get("AGENT", 0)
        formatted["TURN scope"] = by_scope.get("TURN", 0)

        # By age
        by_age = stats.get("by_age", {})
        formatted["< 1 day old"] = by_age.get("< 1 day", 0)
        formatted["1-7 days old"] = by_age.get("1-7 days", 0)
        formatted["7-30 days old"] = by_age.get("7-30 days", 0)
        formatted["> 30 days old"] = by_age.get("> 30 days", 0)

        # Other stats
        oldest = stats.get("oldest_memory_age_days")
        if oldest is not None:
            formatted["Oldest Memory"] = f"{oldest:.1f} days"
        else:
            formatted["Oldest Memory"] = "N/A"

        formatted["Avg Importance"] = f"{stats.get('avg_importance', 0):.2f}"

        print_stats(formatted, title="Memory Statistics")

        # Helpful tips
        old_count = by_age.get("> 30 days", 0)
        if old_count > 0:
            console.print(
                f"\n[dim]Tip: You have {old_count} memories older than 30 days. "
                f"Consider running: headroom memory prune --older-than 30d[/dim]"
            )

    except Exception as e:
        print_error(f"Failed to get stats: {e}")
        sys.exit(1)


@memory.command("edit")
@db_path_option
@click.argument("memory_id", type=str)
@click.option("--content", "-c", type=str, help="New content for the memory.")
@click.option("--importance", "-i", type=float, help="New importance score (0.0 - 1.0).")
@click.pass_context
def edit_memory(
    ctx: click.Context,
    db_path: str,
    memory_id: str,
    content: str | None,
    importance: float | None,
) -> None:
    """Edit a memory's content or importance.

    \b
    Examples:
        headroom memory edit abc123 --content "Updated content"
        headroom memory edit abc123 --importance 0.8
        headroom memory edit abc123 -c "New content" -i 0.9
    """
    if content is None and importance is None:
        print_error("At least one of --content or --importance must be provided.")
        sys.exit(1)

    if importance is not None and (importance < 0.0 or importance > 1.0):
        print_error("Importance must be between 0.0 and 1.0.")
        sys.exit(1)

    store = get_store(db_path)

    try:
        # Find the memory
        mem = asyncio.run(store.get(memory_id))

        # Try partial match if not found
        if mem is None:
            all_memories = asyncio.run(store.query(MemoryFilter(limit=1000)))
            matches = [m for m in all_memories if m.id.startswith(memory_id)]

            if len(matches) == 0:
                print_error(f"Memory not found: {memory_id}")
                sys.exit(1)
            elif len(matches) > 1:
                print_error(f"Ambiguous ID '{memory_id}'. Matches: {[m.id[:8] for m in matches]}")
                sys.exit(1)
            else:
                mem = matches[0]

        # Update fields
        if content is not None:
            mem.content = content
        if importance is not None:
            mem.importance = importance

        # Save
        asyncio.run(store.save(mem))
        print_success(f"Updated memory {mem.id[:8]}")

    except Exception as e:
        print_error(f"Failed to edit memory: {e}")
        sys.exit(1)


@memory.command("delete")
@db_path_option
@click.argument("memory_ids", type=str, nargs=-1, required=True)
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt.")
@click.pass_context
def delete_memories(
    ctx: click.Context,
    db_path: str,
    memory_ids: tuple[str, ...],
    force: bool,
) -> None:
    """Delete one or more memories by ID.

    \b
    Examples:
        headroom memory delete abc123           Delete one memory
        headroom memory delete abc123 def456    Delete multiple memories
        headroom memory delete abc123 --force   Skip confirmation
    """
    store = get_store(db_path)

    try:
        # Resolve all IDs (support partial matching)
        resolved_ids: list[str] = []
        all_memories: list[Memory] | None = None

        for memory_id in memory_ids:
            mem = asyncio.run(store.get(memory_id))

            if mem is not None:
                resolved_ids.append(mem.id)
            else:
                # Try partial match
                if all_memories is None:
                    all_memories = asyncio.run(store.query(MemoryFilter(limit=10000)))

                matches = [m for m in all_memories if m.id.startswith(memory_id)]

                if len(matches) == 0:
                    print_warning(f"Memory not found: {memory_id}")
                elif len(matches) > 1:
                    print_error(
                        f"Ambiguous ID '{memory_id}'. Matches: {[m.id[:8] for m in matches]}"
                    )
                    sys.exit(1)
                else:
                    resolved_ids.append(matches[0].id)

        if not resolved_ids:
            print_error("No valid memory IDs provided.")
            sys.exit(1)

        # Confirm
        if not force:
            count = len(resolved_ids)
            ids_preview = ", ".join(id[:8] for id in resolved_ids[:5])
            if count > 5:
                ids_preview += f" ... and {count - 5} more"

            click.confirm(
                f"Delete {count} memory(ies)? ({ids_preview})",
                abort=True,
            )

        # Delete
        deleted = asyncio.run(store.delete_batch(resolved_ids))
        print_success(f"Deleted {deleted} memory(ies).")

    except click.Abort:
        click.echo("Aborted.")
        sys.exit(0)
    except Exception as e:
        print_error(f"Failed to delete memories: {e}")
        sys.exit(1)


@memory.command("prune")
@db_path_option
@click.option(
    "--older-than",
    "older_than",
    type=str,
    help="Delete memories older than duration (e.g., 30d, 2w).",
)
@click.option(
    "--scope",
    type=click.Choice(["USER", "SESSION", "AGENT", "TURN"], case_sensitive=False),
    help="Delete only memories at this scope level.",
)
@click.option(
    "--low-importance",
    "low_importance",
    type=float,
    help="Delete memories with importance below threshold.",
)
@click.option(
    "--session", "-s", "session_id", type=str, help="Limit pruning to a specific session."
)
@click.option("--dry-run", is_flag=True, help="Show what would be deleted without deleting.")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt.")
@click.pass_context
def prune_memories(
    ctx: click.Context,
    db_path: str,
    older_than: str | None,
    scope: str | None,
    low_importance: float | None,
    session_id: str | None,
    dry_run: bool,
    force: bool,
) -> None:
    """Prune memories matching specified criteria.

    Multiple filters can be combined (AND logic).

    \b
    Examples:
        headroom memory prune --older-than 30d              Delete memories older than 30 days
        headroom memory prune --scope TURN                  Delete all TURN-level memories
        headroom memory prune --low-importance 0.3          Delete low-importance memories
        headroom memory prune --older-than 7d --scope TURN  Combine filters
        headroom memory prune --older-than 30d --dry-run    Preview what would be deleted
    """
    if older_than is None and scope is None and low_importance is None:
        print_error(
            "At least one filter (--older-than, --scope, --low-importance) must be specified."
        )
        sys.exit(1)

    if low_importance is not None and (low_importance < 0.0 or low_importance > 1.0):
        print_error("--low-importance must be between 0.0 and 1.0.")
        sys.exit(1)

    store = get_store(db_path)

    try:
        # Build filter to find matching memories
        filter_kwargs: dict[str, Any] = {
            "limit": 100000,  # High limit for pruning
            "include_superseded": True,  # Include all memories
        }

        if older_than:
            duration = parse_duration(older_than)
            cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - duration
            filter_kwargs["created_before"] = cutoff

        if scope:
            scope_level = ScopeLevel[scope.upper()]
            filter_kwargs["scope_levels"] = [scope_level]

        if low_importance is not None:
            filter_kwargs["max_importance"] = low_importance

        if session_id:
            filter_kwargs["session_id"] = session_id

        mem_filter = MemoryFilter(**filter_kwargs)
        memories = asyncio.run(store.query(mem_filter))

        if not memories:
            click.echo("No memories match the specified criteria.")
            return

        # Show preview
        count = len(memories)
        click.echo(f"\nFound {count} memory(ies) matching criteria:")

        # Show sample
        for mem in memories[:5]:
            click.echo(f"  - {mem.id[:8]}: {truncate(mem.content.replace(chr(10), ' '), 40)}")
        if count > 5:
            click.echo(f"  ... and {count - 5} more")

        if dry_run:
            print_warning(f"DRY RUN: Would delete {count} memory(ies).")
            return

        # Confirm
        if not force:
            click.confirm(
                f"\nDelete {count} memory(ies)?",
                abort=True,
            )

        # Delete
        ids_to_delete = [m.id for m in memories]
        deleted = asyncio.run(store.delete_batch(ids_to_delete))
        print_success(f"Deleted {deleted} memory(ies).")

    except click.BadParameter as e:
        print_error(str(e))
        sys.exit(1)
    except click.Abort:
        click.echo("Aborted.")
        sys.exit(0)
    except Exception as e:
        print_error(f"Failed to prune memories: {e}")
        sys.exit(1)


@memory.command("purge")
@db_path_option
@click.option(
    "--confirm",
    "confirm_flag",
    is_flag=True,
    help="Confirm that you want to delete ALL memories.",
)
@click.pass_context
def purge_memories(ctx: click.Context, db_path: str, confirm_flag: bool) -> None:
    """Delete ALL memories from the database.

    This is a destructive operation that cannot be undone.
    Requires the --confirm flag.

    \b
    Example:
        headroom memory purge --confirm
    """
    if not confirm_flag:
        print_error("This will delete ALL memories. Use --confirm to proceed.")
        sys.exit(1)

    store = get_store(db_path)

    try:
        # Get count first
        stats = _get_stats(store)
        total = stats["total_count"]

        if total == 0:
            click.echo("No memories to delete.")
            return

        # Final confirmation
        click.confirm(
            f"Are you sure you want to delete ALL {total} memories? This cannot be undone.",
            abort=True,
        )

        # Purge
        deleted = asyncio.run(store.clear_all())
        print_success(f"Purged {deleted} memory(ies).")

    except click.Abort:
        click.echo("Aborted.")
        sys.exit(0)
    except Exception as e:
        print_error(f"Failed to purge memories: {e}")
        sys.exit(1)


@memory.command("export")
@db_path_option
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path. If not specified, outputs to stdout.",
)
@click.pass_context
def export_memories(ctx: click.Context, db_path: str, output: str | None) -> None:
    """Export all memories to JSON.

    \b
    Examples:
        headroom memory export                       Output to stdout
        headroom memory export --output backup.json  Save to file
        headroom memory export -o backup.json        Save to file (short form)
    """
    store = get_store(db_path)

    try:
        memories = _export_all(store)

        json_output = json.dumps(memories, indent=2, default=str)

        if output:
            output_path = Path(output)
            output_path.write_text(json_output)
            print_success(f"Exported {len(memories)} memory(ies) to {output_path}")
        else:
            click.echo(json_output)

    except Exception as e:
        print_error(f"Failed to export memories: {e}")
        sys.exit(1)


@memory.command("import")
@db_path_option
@click.argument("file", type=click.Path(exists=True))
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt.")
@click.pass_context
def import_memories(ctx: click.Context, db_path: str, file: str, force: bool) -> None:
    """Import memories from a JSON file.

    The JSON file should contain an array of memory objects (as exported by 'export').
    Existing memories with the same ID will be overwritten.

    \b
    Examples:
        headroom memory import backup.json          Import from file
        headroom memory import backup.json --force  Skip confirmation
    """
    file_path = Path(file)

    try:
        # Read and parse file
        content = file_path.read_text()
        memories_data = json.loads(content)

        if not isinstance(memories_data, list):
            print_error("Invalid JSON: Expected an array of memory objects.")
            sys.exit(1)

        count = len(memories_data)
        if count == 0:
            click.echo("No memories to import.")
            return

        # Confirm
        if not force:
            click.confirm(
                f"Import {count} memory(ies) from {file_path}? Existing memories with same IDs will be overwritten.",
                abort=True,
            )

        store = get_store(db_path)
        imported = _import_memories(store, memories_data)
        print_success(f"Imported {imported} memory(ies).")

        if imported < count:
            print_warning(f"Skipped {count - imported} malformed entries.")

    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON file: {e}")
        sys.exit(1)
    except click.Abort:
        click.echo("Aborted.")
        sys.exit(0)
    except Exception as e:
        print_error(f"Failed to import memories: {e}")
        sys.exit(1)
