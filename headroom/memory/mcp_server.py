"""Headroom Memory MCP Server.

A stdio MCP server that exposes headroom's memory backend as tools
that Codex (or any MCP-compatible client) can call natively.

Tools:
    memory_search  — semantic search across stored memories
    memory_save    — persist a new fact/decision/convention

Design:
    - Embedder is pre-loaded at startup (no cold-start on first query)
    - On startup, any memories missing vector embeddings are re-indexed
      (fixes interop gap when memories were saved via a different path)
    - Save always generates embeddings inline

Usage:
    # Standalone (for testing):
    python -m headroom.memory.mcp_server --db /path/to/.headroom/memory.db

    # Registered in Codex config.toml (done by `headroom wrap codex --memory`):
    [mcp_servers.headroom_memory]
    command = "python"
    args = ["-m", "headroom.memory.mcp_server", "--user", "alice"]
    # When --db is omitted, the server resolves .headroom/memory.db from cwd.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

logger = logging.getLogger("headroom.memory.mcp")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOLS = [
    Tool(
        name="memory_search",
        description=(
            "Search persistent memory for relevant knowledge from prior sessions. "
            "Use this for questions about architecture, conventions, prior decisions, "
            "project context, user preferences, org info, codenames, debugging history, "
            "or anything that might have been discussed before."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max results to return (default 10).",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="memory_save",
        description=(
            "Save information to persistent memory for future sessions. "
            "Use this for decisions, conventions, architecture context, "
            "user preferences, project facts, or anything worth remembering. "
            "Saving a similar fact does not replace an existing memory; corrections "
            "must use an explicit update path with the existing memory ID.\n\n"
            "IMPORTANT: Break information into atomic facts — one fact per "
            "entry in the 'facts' array. Each fact should be a single, "
            "self-contained statement that answers one question. "
            "Do NOT combine multiple facts into one string.\n\n"
            "Good:  facts: ['Repo owner is Tejas C.', 'User prefers dark mode']\n"
            "Bad:   facts: ['Repo owner is Tejas C. Prefers dark mode.']"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Array of atomic facts to save. Each entry should be "
                        "one self-contained fact. The system stores and indexes "
                        "each fact separately for precise retrieval."
                    ),
                },
                "importance": {
                    "type": "number",
                    "description": "0.0 (low) to 1.0 (critical). Default 0.7.",
                    "default": 0.7,
                },
            },
            "required": [],
        },
    ),
]


# ---------------------------------------------------------------------------
# Startup: pre-load embedder + re-index unembedded memories
# ---------------------------------------------------------------------------


async def _warm_up_backend(backend: LocalBackend, user_id: str) -> None:
    """Pre-load the embedder and re-index memories that lack embeddings.

    Memories saved via other paths (e.g. Claude Code proxy direct SQL)
    may exist in the store but have no vector embeddings.  This scans
    for those and re-indexes them so vector search works across agents.
    """
    await backend._ensure_initialized()
    hm = backend._hierarchical_memory
    if hm is None:
        return

    # Force-load the embedder now (not lazily on first search)
    _dummy = await hm._embedder.embed("warmup")
    logger.info("Memory MCP: embedder pre-loaded")

    # Ensure ALL memories are in the vector index.
    # Memories saved via other agents (Claude Code proxy, direct SQL) may
    # exist in the store but not be indexed — re-embed and index them all.
    all_memories = await backend.get_user_memories(user_id, limit=500)
    if not all_memories:
        return

    memories_missing_embeddings = [mem for mem in all_memories if mem.embedding is None]
    if memories_missing_embeddings:
        embeddings = await hm._embedder.embed_batch(
            [mem.content for mem in memories_missing_embeddings]
        )
        for mem, embedding in zip(memories_missing_embeddings, embeddings):
            mem.embedding = embedding
        await hm._store.save_batch(memories_missing_embeddings)

    indexed = await hm._vector_index.index_batch(all_memories)
    logger.info(f"Memory MCP: indexed {indexed} memories into vector store")


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------


def create_memory_server(db_path: str, user_id: str = "default") -> Server:
    """Create an MCP server backed by headroom's local memory."""

    server = Server("headroom-memory")
    _backend: LocalBackend | None = None
    _init_task: asyncio.Task[LocalBackend] | None = None

    async def _init_backend() -> LocalBackend:
        """Initialize backend with ONNX embedder (fast, no PyTorch)."""
        nonlocal _backend, _init_task
        config = LocalBackendConfig(db_path=db_path, embedder_backend="onnx")
        backend = LocalBackend(config)
        init_task = asyncio.current_task()
        try:
            await _warm_up_backend(backend, user_id)
        except (Exception, asyncio.CancelledError):
            try:
                await backend.close()
            except Exception as cleanup_error:
                logger.warning("Memory MCP: failed backend cleanup: %s", cleanup_error)
            finally:
                if _init_task is init_task:
                    _init_task = None
            raise

        _backend = backend
        if _init_task is init_task:
            _init_task = None
        logger.info(f"Memory MCP: ready (db={db_path}, user={user_id})")
        return backend

    def _handle_backend_init_done(init_task: asyncio.Task[LocalBackend]) -> None:
        """Clear and log failed background initialization tasks."""
        nonlocal _init_task
        if init_task.cancelled():
            if _init_task is init_task:
                _init_task = None
            return
        error = init_task.exception()
        if error is not None:
            if _init_task is init_task:
                _init_task = None
            logger.warning("Memory MCP: backend initialization failed: %s", error)

    def _start_backend_init() -> asyncio.Task[LocalBackend]:
        """Start backend initialization once for all concurrent callers."""
        nonlocal _init_task
        if _init_task is None:
            _init_task = asyncio.create_task(_init_backend())
            _init_task.add_done_callback(_handle_backend_init_done)
        return _init_task

    async def _get_backend() -> LocalBackend:
        nonlocal _backend, _init_task
        if _backend is not None:
            return _backend

        init_task = _start_backend_init()
        try:
            return await asyncio.shield(init_task)
        except asyncio.CancelledError:
            if init_task.done() and _init_task is init_task:
                _init_task = None
            raise
        except Exception:
            if _init_task is init_task:
                _init_task = None
            raise

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        # Kick off background init on first list_tools (called at MCP handshake)
        if _backend is None:
            _start_backend_init()
        return _TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        backend = await _get_backend()

        if name == "memory_search":
            return await _handle_search(backend, arguments, user_id)
        elif name == "memory_save":
            return await _handle_save(backend, arguments, user_id)

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    return server


async def _handle_search(
    backend: LocalBackend, arguments: dict[str, Any], user_id: str
) -> list[TextContent]:
    query = arguments.get("query", "")
    top_k = arguments.get("top_k", 10)

    if not query:
        return [TextContent(type="text", text="Error: query is required")]

    try:
        # Over-fetch to compensate for filtering out superseded memories
        results = await backend.search_memories(
            query=query,
            user_id=user_id,
            top_k=top_k * 3,
            include_related=True,
        )

        if not results:
            return [TextContent(type="text", text="No memories found.")]

        # Filter out superseded memories — only return current/active ones.
        # Re-check the store because in-memory HNSW metadata may be stale.
        active_results = []
        for r in results:
            if getattr(r.memory, "superseded_by", None):
                continue
            # Double-check against the store for recently superseded memories
            try:
                stored = await backend.get_memory(r.memory.id)
                if stored and getattr(stored, "superseded_by", None):
                    continue
            except Exception:
                pass
            active_results.append(r)

        if not active_results:
            return [TextContent(type="text", text="No memories found.")]

        # Trim to requested top_k
        active_results = active_results[:top_k]

        try:
            await backend.record_access([r.memory.id for r in active_results])
        except Exception as e:
            # Usage metadata must never make a successful retrieval fail.
            logger.warning(f"Memory MCP: failed to record access: {e}")

        lines = []
        for i, r in enumerate(active_results, 1):
            score = f"{r.score:.2f}" if hasattr(r, "score") else "?"
            lines.append(f"{i}. [relevance={score}] {r.memory.content}")
            if hasattr(r, "related_entities") and r.related_entities:
                lines.append(f"   Related: {', '.join(r.related_entities[:3])}")

        return [TextContent(type="text", text="\n".join(lines))]
    except Exception as e:
        logger.error(f"memory_search failed: {e}")
        return [TextContent(type="text", text=f"Search error: {e}")]


async def _handle_save(
    backend: LocalBackend, arguments: dict[str, Any], user_id: str
) -> list[TextContent]:
    facts = arguments.get("facts", [])
    importance = arguments.get("importance", 0.7)

    # Backward compat: accept single "content" string too
    if not facts:
        content = arguments.get("content", "")
        if content:
            facts = [content]

    if not facts:
        return [TextContent(type="text", text="Error: facts array is required")]

    try:
        saved = 0
        superseded = 0
        results_lines: list[str] = []

        for fact in facts:
            fact = fact.strip()
            if not fact:
                continue

            # Similarity is a retrieval signal, not proof that two memories
            # represent versions of the same fact. Only explicit update paths
            # with a caller-supplied memory ID may create a supersession chain.
            memory = await backend.save_memory(
                content=fact,
                user_id=user_id,
                importance=importance,
            )
            results_lines.append(f"  saved [{memory.id[:8]}]: {fact[:60]}")
            saved += 1

        summary = f"Saved {saved} new, updated {superseded} existing ({saved + superseded} total)"
        return [TextContent(type="text", text=summary + "\n" + "\n".join(results_lines))]
    except Exception as e:
        logger.error(f"memory_save failed: {e}")
        return [TextContent(type="text", text=f"Save error: {e}")]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _run(db_path: str, user_id: str) -> None:
    server = create_memory_server(db_path, user_id)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def _memory_mcp_startup_context(
    configured_db: str, cwd: Path, db_flag_present: bool
) -> dict[str, str | bool]:
    """Describe the DB path the memory MCP server will try to open."""
    configured_path = Path(configured_db).expanduser()
    resolved_path = (
        configured_path if configured_path.is_absolute() else (cwd / configured_path)
    ).resolve(strict=False)
    active_project_db = (cwd / ".headroom" / "memory.db").resolve(strict=False)
    if not db_flag_present:
        config_source = "cwd-default"
        resolution = "dynamic-cwd"
    else:
        config_source = "cli-flag"
        resolution = "static-cli"
    if resolved_path == active_project_db:
        storage_scope = "active-project"
    elif resolved_path.name == "memory.db":
        storage_scope = "external-memory-db"
    else:
        storage_scope = "custom-db-path"
    path_exists = resolved_path.exists()
    path_readable = path_exists and os.access(resolved_path, os.R_OK)
    return {
        "configured_db": str(configured_path),
        "resolved_db": str(resolved_path),
        "config_source": config_source,
        "cwd": str(cwd),
        "project_root": str(cwd),
        "storage_scope": storage_scope,
        "path_exists": path_exists,
        "path_readable": path_readable,
        "resolution": resolution,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Headroom Memory MCP Server")
    parser.add_argument(
        "--db",
        default=str(Path.cwd() / ".headroom" / "memory.db"),
        help="Path to memory SQLite database",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("USER", os.environ.get("USERNAME", "default")),
        help="User ID for memory scoping",
    )
    args = parser.parse_args()

    # Skip HuggingFace model freshness checks — use cached models only.
    # This eliminates 1-2s of HTTP HEAD requests on every startup.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # Log to stderr (MCP uses stdout for protocol)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(name)s: %(message)s",
    )

    startup = _memory_mcp_startup_context(
        configured_db=args.db,
        cwd=Path.cwd(),
        db_flag_present=any(arg == "--db" or arg.startswith("--db=") for arg in sys.argv[1:]),
    )
    logger.info(
        "Memory MCP startup: configured_db=%s, resolved_db=%s, config_source=%s, "
        "cwd=%s, project_root=%s, storage_scope=%s, path_exists=%s, "
        "path_readable=%s, resolution=%s",
        startup["configured_db"],
        startup["resolved_db"],
        startup["config_source"],
        startup["cwd"],
        startup["project_root"],
        startup["storage_scope"],
        startup["path_exists"],
        startup["path_readable"],
        startup["resolution"],
    )

    asyncio.run(_run(args.db, args.user))


if __name__ == "__main__":
    main()
