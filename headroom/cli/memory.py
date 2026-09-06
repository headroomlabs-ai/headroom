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

from .. import fsutil
from ..memory.adapters.sqlite import SQLiteMemoryStore
from ..memory.adapters.sqlite_graph import SQLiteGraphStore
from ..memory.knowledge_worker import (
    build_context_snapshot,
    build_knowledge_worker_graph,
    build_memory_audit,
    derive_graph_db_path,
    filter_memories,
    load_memories,
)
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

_REINDEX_PAGE_SIZE = 1_000


def _default_db_path() -> str:
    """Resolve the memory DB the proxy/install actually use.

    Prefer the project-scoped store the proxy writes when run inside a project
    (``./.headroom/memory.db``); otherwise fall back to the global workspace
    store (``~/.headroom/memory.db``). The previous bare ``headroom_memory.db``
    default pointed at neither, so ``headroom memory list`` silently read an
    empty/legacy DB.
    """
    project_db = Path.cwd() / ".headroom" / "memory.db"
    if project_db.exists():
        return str(project_db)
    from ..paths import memory_db_path

    return str(memory_db_path())


def db_path_option(fn: Any) -> Any:
    """Shared --db-path option for memory commands."""
    return click.option(
        "--db-path",
        type=click.Path(),
        default=_default_db_path,
        help="Path to the memory database file. Defaults to the project store "
        "(./.headroom/memory.db) if present, else the global store "
        "(~/.headroom/memory.db).",
        show_default="project store if present, else global store",
    )(fn)


def get_store(db_path: str) -> SQLiteMemoryStore:
    """Get a SQLiteMemoryStore instance."""
    return SQLiteMemoryStore(db_path)


def get_graph_store(db_path: str, graph_db_path: str | None = None) -> SQLiteGraphStore | None:
    """Get a SQLiteGraphStore instance if a graph DB exists."""
    path = Path(graph_db_path).expanduser() if graph_db_path else derive_graph_db_path(db_path)
    if not path.exists():
        return None
    # Audit/export must not create tables or indexes in a user-selected file.
    return SQLiteGraphStore(path, initialize=False, read_only=True)


def _sqlite_table_exists(conn: Any, table_name: str) -> bool:
    """Return whether a SQLite table or virtual table has been initialized."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _remove_from_search_indexes(db_path: str, memory_ids: list[str]) -> bool:
    """Remove specific memories from FTS5 and vector search indexes.

    FTS5 cleanup uses a bare sqlite3 connection (FTS5 is built-in).
    Vector cleanup requires sqlite-vec to load the vec0 virtual-table
    module; when not installed a warning is printed.

    Returns True when both indexes were fully synced, False when any part
    of the sync failed.  Callers must treat False as a partial failure and
    surface it — typically by exiting with a non-zero code so the primary
    store mutation is not silently reported as fully successful.
    """
    if not memory_ids:
        return True

    import sqlite3

    db = Path(db_path)
    ok = True

    # FTS5 table lives in the same memory.db file (built-in, no extension needed).
    try:
        with sqlite3.connect(str(db)) as conn:
            if _sqlite_table_exists(conn, "memory_fts"):
                for i in range(0, len(memory_ids), 500):
                    chunk = memory_ids[i : i + 500]
                    placeholders = ",".join("?" * len(chunk))
                    conn.execute(
                        f"DELETE FROM memory_fts WHERE memory_id IN ({placeholders})",
                        chunk,
                    )
                conn.commit()
    except Exception as exc:
        print_warning(f"FTS5 index cleanup incomplete: {exc}")
        ok = False

    # Vector DB is a sibling file: memory.db -> memory_vectors.db.
    # vec_embeddings is a vec0 virtual table — the sqlite-vec extension must be
    # loaded on every connection before touching it.
    vector_db = db.parent / f"{db.stem}_vectors.db"
    if not vector_db.exists():
        return ok

    try:
        with sqlite3.connect(str(vector_db)) as conn:
            # A sibling database may exist before the optional vector index has
            # ever been initialized. That is a valid no-op, not a sync failure.
            if not _sqlite_table_exists(conn, "vec_metadata"):
                return ok

            try:
                import sqlite_vec
            except ImportError:
                print_warning(
                    "sqlite-vec is not installed; stale vector index entries may remain. "
                    "Run 'headroom memory reindex' after installing sqlite-vec to repair."
                )
                return False

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            for i in range(0, len(memory_ids), 500):
                chunk = memory_ids[i : i + 500]
                placeholders = ",".join("?" * len(chunk))
                rows = conn.execute(
                    f"SELECT rowid FROM vec_metadata WHERE memory_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                rowids = [r[0] for r in rows]
                if rowids:
                    rph = ",".join("?" * len(rowids))
                    conn.execute(f"DELETE FROM vec_embeddings WHERE rowid IN ({rph})", rowids)
                    conn.execute(f"DELETE FROM vec_metadata WHERE rowid IN ({rph})", rowids)
            conn.commit()
    except Exception as exc:
        print_warning(f"Vector index cleanup incomplete: {exc}")
        ok = False

    return ok


def _clear_all_search_indexes(db_path: str) -> bool:
    """Truncate both search indexes after a full purge.

    Same extension-loading requirement as :func:`_remove_from_search_indexes`.
    Returns True on full success, False on any partial failure.
    """
    import sqlite3

    db = Path(db_path)
    ok = True

    try:
        with sqlite3.connect(str(db)) as conn:
            if _sqlite_table_exists(conn, "memory_fts"):
                conn.execute("DELETE FROM memory_fts")
                conn.commit()
    except Exception as exc:
        print_warning(f"FTS5 index cleanup incomplete: {exc}")
        ok = False

    vector_db = db.parent / f"{db.stem}_vectors.db"
    if not vector_db.exists():
        return ok

    try:
        with sqlite3.connect(str(vector_db)) as conn:
            if not _sqlite_table_exists(conn, "vec_metadata"):
                return ok

            try:
                import sqlite_vec
            except ImportError:
                print_warning(
                    "sqlite-vec is not installed; stale vector index entries may remain. "
                    "Run 'headroom memory reindex' after installing sqlite-vec to repair."
                )
                return False

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            conn.execute("DELETE FROM vec_embeddings")
            conn.execute("DELETE FROM vec_metadata")
            conn.commit()
    except Exception as exc:
        print_warning(f"Vector index cleanup incomplete: {exc}")
        ok = False

    return ok


def _resolve_memory(store: SQLiteMemoryStore, memory_id: str) -> Memory:
    """Resolve an exact or unambiguous partial memory ID."""
    memory = asyncio.run(store.get(memory_id))
    if memory is not None:
        return memory

    memories = asyncio.run(store.query(MemoryFilter(limit=10000, include_superseded=True)))
    matches = [candidate for candidate in memories if candidate.id.startswith(memory_id)]
    if not matches:
        raise ValueError(f"Memory not found: {memory_id}")
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous ID '{memory_id}'. Matches: {[memory.id[:8] for memory in matches]}"
        )
    return matches[0]


async def _apply_supersession_repair(
    db_path: str,
    old_memory_id: str,
    new_memory_id: str,
    vector_dimension: int,
) -> tuple[Memory, Memory]:
    """Apply a repair through LocalBackend so indexes and cache are refreshed."""
    from ..memory.backends.local import LocalBackend, LocalBackendConfig

    backend = LocalBackend(
        LocalBackendConfig(
            db_path=db_path,
            vector_dimension=vector_dimension,
        )
    )
    try:
        return await backend.detach_supersession(old_memory_id, new_memory_id)
    finally:
        await backend.close()


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
        headroom memory repair-supersession <old-id> <new-id>  Repair one lineage edge
        headroom memory prune --older-than 30d   Delete memories older than 30 days
        headroom memory purge --confirm          Delete ALL memories
        headroom memory export --output file.json  Export all memories to JSON
        headroom memory export --format knowledge-worker --scope project
        headroom memory audit --output audit.json  Audit provenance and weak claims
        headroom memory import file.json         Import memories from JSON
    """
    pass


@memory.command("list")
@db_path_option
@click.option(
    "--limit",
    "-n",
    type=click.IntRange(min=1),
    default=50,
    help="Maximum number of memories to show.",
)
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
            # Content search is a SQL LIKE that ignores structured filters; apply
            # --scope/--session/--since to the results so combining them with
            # --search isn't a silent no-op. Fetch a generous set before
            # filtering (so the limit isn't consumed by rows we'll drop), then
            # truncate to the requested limit.
            has_filters = bool(scope or session_id or since_duration)
            memories = _search_content(
                store, search_query, limit=max(limit, 1000) if has_filters else limit
            )
            if scope:
                memories = [m for m in memories if get_scope_label(m) == scope.upper()]
            if session_id:
                memories = [m for m in memories if m.session_id == session_id]
            if since_duration:
                duration = parse_duration(since_duration)
                cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - duration
                memories = [
                    m
                    for m in memories
                    if (m.created_at.replace(tzinfo=None) if m.created_at.tzinfo else m.created_at)
                    >= cutoff
                ]
            memories = memories[:limit]
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
        content_changed = content is not None and content != mem.content
        if content is not None:
            mem.content = content
        if importance is not None:
            mem.importance = importance

        index_ok = True
        if content_changed:
            # Clear the stale embedding so the memory MCP server re-embeds on
            # next startup. Also remove the old FTS5 and vector index entries
            # now to avoid serving stale search results until then.
            mem.embedding = None
            index_ok = _remove_from_search_indexes(db_path, [mem.id])
            # Re-index FTS5 immediately with new content (no embedder needed).
            try:
                from ..memory.adapters.fts5 import FTS5TextIndex

                fts = FTS5TextIndex(db_path=db_path)
                asyncio.run(fts.index_memory(mem))
            except Exception as exc:
                print_warning(f"FTS5 re-index incomplete: {exc}")
                index_ok = False

        # Save
        asyncio.run(store.save(mem))
        if not index_ok:
            print_warning(
                f"Updated memory {mem.id[:8]}, but search index sync incomplete. "
                "Run 'headroom memory reindex' to repair."
            )
            sys.exit(1)
        print_success(f"Updated memory {mem.id[:8]}")

    except Exception as e:
        print_error(f"Failed to edit memory: {e}")
        sys.exit(1)


@memory.command("repair-supersession")
@db_path_option
@click.argument("old_memory_id", type=str)
@click.argument("new_memory_id", type=str)
@click.option(
    "--apply",
    "apply_change",
    is_flag=True,
    help="Apply the repair. Stop any proxy using this database first.",
)
@click.pass_context
def repair_supersession(
    ctx: click.Context,
    db_path: str,
    old_memory_id: str,
    new_memory_id: str,
    apply_change: bool,
) -> None:
    """Detach one incorrect OLD_ID -> NEW_ID supersession edge.

    The command validates both reciprocal lineage pointers and changes no
    neighboring edges. It is a dry run unless ``--apply`` is provided.

    \b
    Examples:
        headroom memory repair-supersession OLD_ID NEW_ID
        headroom memory repair-supersession OLD_ID NEW_ID --apply
    """
    _ = ctx
    store = get_store(db_path)

    try:
        old_memory = _resolve_memory(store, old_memory_id)
        new_memory = _resolve_memory(store, new_memory_id)

        if old_memory.superseded_by != new_memory.id or new_memory.supersedes != old_memory.id:
            raise ValueError(
                f"Memories {old_memory.id} and {new_memory.id} do not form "
                "a reciprocal supersession edge"
            )

        click.echo("\nSupersession repair preview:")
        click.echo(
            f"  Restore old memory: {old_memory.id} "
            f"({truncate(old_memory.content.replace(chr(10), ' '), 60)})"
        )
        click.echo(
            f"  Detach new memory: {new_memory.id} "
            f"({truncate(new_memory.content.replace(chr(10), ' '), 60)})"
        )
        click.echo("  Clear: old.valid_until, old.superseded_by, new.supersedes")

        if not apply_change:
            print_warning("DRY RUN: No changes made. Re-run with --apply to repair this edge.")
            return

        embedding = (
            old_memory.embedding if old_memory.embedding is not None else new_memory.embedding
        )
        vector_dimension = len(embedding) if embedding is not None else 384
        asyncio.run(
            _apply_supersession_repair(
                db_path,
                old_memory.id,
                new_memory.id,
                vector_dimension,
            )
        )
        print_success(f"Detached supersession edge {old_memory.id[:8]} -> {new_memory.id[:8]}.")

    except Exception as e:
        print_error(f"Failed to repair supersession: {e}")
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
        if _remove_from_search_indexes(db_path, resolved_ids):
            print_success(f"Deleted {deleted} memory(ies).")
        else:
            print_warning(
                f"Deleted {deleted} memory(ies) from store, but search index sync "
                "incomplete. Run 'headroom memory reindex' to repair."
            )
            sys.exit(1)

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
        if _remove_from_search_indexes(db_path, ids_to_delete):
            print_success(f"Deleted {deleted} memory(ies).")
        else:
            print_warning(
                f"Deleted {deleted} memory(ies) from store, but search index sync "
                "incomplete. Run 'headroom memory reindex' to repair."
            )
            sys.exit(1)

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
        if _clear_all_search_indexes(db_path):
            print_success(f"Purged {deleted} memory(ies).")
        else:
            print_warning(
                f"Purged {deleted} memory(ies) from store, but search index sync "
                "incomplete. Run 'headroom memory reindex' to repair."
            )
            sys.exit(1)

    except click.Abort:
        click.echo("Aborted.")
        sys.exit(0)
    except Exception as e:
        print_error(f"Failed to purge memories: {e}")
        sys.exit(1)


@memory.command("reindex")
@db_path_option
@click.pass_context
def reindex_memories(ctx: click.Context, db_path: str) -> None:
    """Rebuild FTS5 search index and remove orphaned vector entries.

    Use this to repair an inconsistent index after a failed delete, prune,
    or purge.  Run it after installing sqlite-vec to clean up any vector
    entries that could not be removed earlier.

    Vector embeddings are not regenerated by this command — they are rebuilt
    automatically when the Headroom server next starts.

    \b
    Example:
        headroom memory reindex
    """
    import sqlite3

    store = get_store(db_path)

    try:
        # Page through the complete active store. A fixed cap is destructive:
        # clearing FTS and rebuilding only the first N rows drops valid search
        # coverage, while using the same truncated ID set for vector cleanup
        # misclassifies later primary rows as orphans.
        memories: list[Memory] = []
        offset = 0
        while True:
            page = asyncio.run(
                store.query(
                    MemoryFilter(
                        limit=_REINDEX_PAGE_SIZE,
                        offset=offset,
                        order_by="created_at",
                        order_desc=False,
                    )
                )
            )
            if not page:
                break
            memories.extend(page)
            offset += len(page)

        db = Path(db_path)
        ok = True

        # --- FTS5: wipe and rebuild from primary store ---
        from ..memory.adapters.fts5 import FTS5TextIndex

        # Construction initializes an absent optional FTS table. Cleanup
        # helpers, by contrast, intentionally treat an absent table as a no-op.
        fts = FTS5TextIndex(db_path=db_path)
        try:
            with sqlite3.connect(str(db)) as conn:
                conn.execute("DELETE FROM memory_fts")
                conn.commit()
        except Exception as exc:
            print_error(f"Failed to clear FTS5 index: {exc}")
            sys.exit(1)

        fts_indexed = 0
        for mem in memories:
            try:
                asyncio.run(fts.index_memory(mem))
                fts_indexed += 1
            except Exception as exc:
                print_warning(f"FTS5: failed to index {mem.id[:8]}: {exc}")
                ok = False

        # --- Vector: remove orphaned entries (requires sqlite-vec) ---
        vector_db = db.parent / f"{db.stem}_vectors.db"
        vector_msg = ""
        if vector_db.exists():
            # Orphan detection is based on every primary row, including
            # superseded memories that are intentionally omitted from FTS.
            with store._get_conn() as conn:
                primary_ids = {row[0] for row in conn.execute("SELECT id FROM memories")}
            try:
                with sqlite3.connect(str(vector_db)) as conn:
                    if not _sqlite_table_exists(conn, "vec_metadata"):
                        vector_msg = ", vector index not initialized"
                    else:
                        import sqlite_vec

                        conn.enable_load_extension(True)
                        sqlite_vec.load(conn)
                        conn.enable_load_extension(False)
                        rows = conn.execute("SELECT memory_id FROM vec_metadata").fetchall()
                        orphan_ids = [r[0] for r in rows if r[0] not in primary_ids]
                        if orphan_ids:
                            for i in range(0, len(orphan_ids), 500):
                                chunk = orphan_ids[i : i + 500]
                                ph = ",".join("?" * len(chunk))
                                vec_rows = conn.execute(
                                    f"SELECT rowid FROM vec_metadata WHERE memory_id IN ({ph})",
                                    chunk,
                                ).fetchall()
                                rowids = [r[0] for r in vec_rows]
                                if rowids:
                                    rph = ",".join("?" * len(rowids))
                                    conn.execute(
                                        f"DELETE FROM vec_embeddings WHERE rowid IN ({rph})",
                                        rowids,
                                    )
                                    conn.execute(
                                        f"DELETE FROM vec_metadata WHERE rowid IN ({rph})",
                                        rowids,
                                    )
                            conn.commit()
                        vector_msg = (
                            f", removed {len(orphan_ids)} orphaned vector entry(ies)"
                            if orphan_ids
                            else ", vector index clean"
                        )
            except ImportError:
                vector_msg = (
                    " (vector index skipped: sqlite-vec not installed — "
                    "install with: pip install sqlite-vec)"
                )
                ok = False
            except Exception as exc:
                vector_msg = f" (vector index cleanup failed: {exc})"
                ok = False

        msg = f"Re-indexed {fts_indexed}/{len(memories)} memories{vector_msg}."
        if ok:
            print_success(msg)
        else:
            print_warning(msg)
            sys.exit(1)

    except Exception as e:
        print_error(f"Failed to reindex: {e}")
        sys.exit(1)


@memory.command("export")
@db_path_option
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path. If not specified, outputs to stdout.",
)
@click.option(
    "--format",
    "export_format",
    type=click.Choice(["backup", "knowledge-worker", "context"], case_sensitive=False),
    default="backup",
    show_default=True,
    help="Export format: legacy backup JSON, knowledge-worker graph JSON, or compact context Markdown.",
)
@click.option(
    "--scope",
    type=click.Choice(["global", "project", "session"], case_sensitive=False),
    default="global",
    show_default=True,
    help="Scope for knowledge-worker/context exports.",
)
@click.option("--session", "session_id", type=str, help="Session ID when --scope=session.")
@click.option(
    "--project",
    type=str,
    help="Project key/display name when exporting from a global DB with project metadata.",
)
@click.option("--user", "user_id", type=str, help="Filter export to a user ID.")
@click.option(
    "--graph-db-path",
    type=click.Path(),
    help="Path to the graph database. Defaults to <memory_db>_graph.db.",
)
@click.option(
    "--include-superseded",
    is_flag=True,
    help="Include superseded memories in scoped exports.",
)
@click.option(
    "--max-ideas",
    type=click.IntRange(min=1),
    default=20,
    show_default=True,
    help="Maximum idea count for --format=context.",
)
@click.pass_context
def export_memories(
    ctx: click.Context,
    db_path: str,
    output: str | None,
    export_format: str,
    scope: str,
    session_id: str | None,
    project: str | None,
    user_id: str | None,
    graph_db_path: str | None,
    include_superseded: bool,
    max_ideas: int,
) -> None:
    """Export memories to legacy JSON, knowledge-worker graph JSON, or context Markdown.

    \b
    Examples:
        headroom memory export                       Output to stdout
        headroom memory export --output backup.json  Save to file
        headroom memory export -o backup.json        Save to file (short form)
        headroom memory export --format knowledge-worker --scope project -o graph.json
        headroom memory export --format context --scope session --session abc -o context.md
    """
    store = get_store(db_path)

    try:
        fmt = export_format.lower()
        if (
            fmt == "backup"
            and scope.lower() == "global"
            and not any([session_id, project, user_id, graph_db_path, include_superseded])
        ):
            memories = _export_all(store)
            rendered = json.dumps(memories, indent=2, default=str)
            count_label = f"{len(memories)} memory(ies)"
        elif fmt == "backup":
            memories = asyncio.run(load_memories(store, include_superseded=include_superseded))
            scoped = filter_memories(
                memories,
                scope=scope,
                session_id=session_id,
                project=project,
                user_id=user_id,
            )
            rendered = json.dumps([memory.to_dict() for memory in scoped], indent=2, default=str)
            count_label = f"{len(scoped)} memory(ies)"
        else:
            graph_store = get_graph_store(db_path, graph_db_path)
            graph = asyncio.run(
                build_knowledge_worker_graph(
                    store,
                    graph_store=graph_store,
                    scope=scope,
                    session_id=session_id,
                    project=project,
                    user_id=user_id,
                    include_superseded=include_superseded,
                )
            )
            if fmt == "context":
                rendered = build_context_snapshot(graph, max_ideas=max_ideas)
            else:
                rendered = json.dumps(graph, indent=2, default=str)
            count_label = f"{graph.get('_meta', {}).get('memory_count', 0)} memory(ies)"

        if output:
            output_path = Path(output)
            fsutil.write_text(output_path, rendered)
            print_success(f"Exported {count_label} to {output_path}")
        else:
            click.echo(rendered)

    except Exception as e:
        print_error(f"Failed to export memories: {e}")
        sys.exit(1)


@memory.command("audit")
@db_path_option
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output audit JSON path. If not specified, outputs to stdout.",
)
@click.option(
    "--scope",
    type=click.Choice(["global", "project", "session"], case_sensitive=False),
    default="global",
    show_default=True,
    help="Scope to audit.",
)
@click.option("--session", "session_id", type=str, help="Session ID when --scope=session.")
@click.option(
    "--project",
    type=str,
    help="Project key/display name when auditing a global DB with project metadata.",
)
@click.option("--user", "user_id", type=str, help="Filter audit to a user ID.")
@click.option(
    "--graph-db-path",
    type=click.Path(),
    help="Path to the graph database. Defaults to <memory_db>_graph.db.",
)
@click.option(
    "--graph-output",
    type=click.Path(),
    help="Also write the knowledge-worker graph JSON used for the audit.",
)
@click.option(
    "--include-superseded",
    is_flag=True,
    help="Include superseded memories in the audit.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=25,
    show_default=True,
    help="Maximum ranked items.",
)
@click.pass_context
def audit_memories(
    ctx: click.Context,
    db_path: str,
    output: str | None,
    scope: str,
    session_id: str | None,
    project: str | None,
    user_id: str | None,
    graph_db_path: str | None,
    graph_output: str | None,
    include_superseded: bool,
    limit: int,
) -> None:
    """Audit memory provenance, confidence, and review queues.

    \b
    Examples:
        headroom memory audit --output audit.json
        headroom memory audit --scope project --graph-output graph.json
        headroom memory audit --scope session --session abc
    """
    store = get_store(db_path)

    try:
        graph_store = get_graph_store(db_path, graph_db_path)
        graph = asyncio.run(
            build_knowledge_worker_graph(
                store,
                graph_store=graph_store,
                scope=scope,
                session_id=session_id,
                project=project,
                user_id=user_id,
                include_superseded=include_superseded,
            )
        )
        audit = build_memory_audit(graph, limit=limit)
        rendered = json.dumps(audit, indent=2, default=str)

        if graph_output:
            graph_path = Path(graph_output)
            fsutil.write_text(graph_path, json.dumps(graph, indent=2, default=str))
            print_success(f"Wrote knowledge-worker graph to {graph_path}")

        if output:
            output_path = Path(output)
            fsutil.write_text(output_path, rendered)
            print_success(f"Wrote memory audit to {output_path}")
        else:
            click.echo(rendered)

    except Exception as e:
        print_error(f"Failed to audit memories: {e}")
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
        content = fsutil.read_text(file_path)
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
