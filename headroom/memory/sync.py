"""Universal memory sync engine for cross-agent interoperability.

Provides bidirectional sync between headroom's memory DB and any
agent's native memory format via pluggable adapters.

Architecture:
    DB ← sync_import → Agent files   (agent's knowledge enters the shared DB)
    DB → sync_export → Agent files   (shared knowledge flows to the agent)
    sync() = import + export          (bidirectional, fast no-op when unchanged)

Usage:
    from headroom.memory.sync import sync, SyncResult
    from headroom.memory.sync_adapters.claude_code import ClaudeCodeAdapter

    adapter = ClaudeCodeAdapter(memory_dir=Path("~/.claude/projects/.../memory"))
    backend = LocalBackend(config)

    result: SyncResult = await sync(backend, adapter, user_id="tcms")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from headroom import paths as _paths

logger = logging.getLogger("headroom.memory.sync")

# State file for fast no-op detection (workspace bucket, respects
# HEADROOM_WORKSPACE_DIR). Resolved at import time, matching prior behavior.
_DEFAULT_STATE_PATH = _paths.sync_state_path()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SyncResult:
    """Result of a sync operation."""

    imported: int = 0  # agent files → DB
    exported: int = 0  # DB → agent files
    skipped_unchanged: int = 0
    skipped_dedup: int = 0
    duration_ms: float = 0


@dataclass
class AgentMemory:
    """A memory entry read from an agent's native format."""

    content: str
    category: str = ""
    source_file: str = ""
    content_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------


class AgentMemoryAdapter(ABC):
    """Base class for agent memory format adapters.

    Each agent (Claude Code, Codex, Aider, Cursor) has a subclass
    that knows how to read/write that agent's native memory format.
    """

    agent_name: str = "unknown"

    @abstractmethod
    async def read_memories(self) -> list[AgentMemory]:
        """Read memories from the agent's native format.

        Returns a list of AgentMemory entries found in the agent's files.
        """
        ...

    @abstractmethod
    async def write_memories(self, memories: list[dict[str, Any]]) -> int:
        """Write memories to the agent's native format.

        Args:
            memories: List of dicts with keys: content, category, importance,
                      headroom_id, source_agent, content_hash.

        Returns:
            Count of memories written.
        """
        ...

    @abstractmethod
    def fingerprint(self) -> str:
        """Fast hash of the agent's memory state.

        Used for no-op detection: if the fingerprint hasn't changed
        since last sync, we can skip the full read/compare cycle.
        """
        ...


# ---------------------------------------------------------------------------
# Sync state persistence
# ---------------------------------------------------------------------------


def _load_sync_state(state_path: Path) -> dict[str, Any]:
    """Load sync state from disk."""
    if state_path.exists():
        try:
            result: dict[str, Any] = json.loads(state_path.read_text(encoding="utf-8"))
            return result
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_sync_state(state_path: Path, state: dict[str, Any]) -> None:
    """Save sync state to disk."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _db_fingerprint(memories: list[Any]) -> str:
    """Compute a fast fingerprint of DB state."""
    if not memories:
        return "empty"
    # Hash: count + most recent created_at
    parts = [str(len(memories))]
    for m in memories[:5]:  # Sample first 5 for speed
        parts.append(getattr(m, "id", "")[:8])
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Sync engine
# ---------------------------------------------------------------------------


async def sync(
    backend: Any,
    adapter: AgentMemoryAdapter,
    user_id: str,
    state_path: Path = _DEFAULT_STATE_PATH,
    force: bool = False,
) -> SyncResult:
    """Bidirectional sync between headroom DB and an agent's memory.

    1. Fast no-op check (fingerprint comparison)
    2. Import: agent files → DB (new entries only, deduped by content hash)
    3. Export: DB → agent files (entries not already in agent's files)

    Args:
        backend: LocalBackend instance (must have save_memory, get_user_memories).
        adapter: Agent-specific memory adapter.
        user_id: User ID for memory scoping.
        state_path: Path to sync state file.
        force: Skip no-op check and always sync.

    Returns:
        SyncResult with import/export counts and timing.
    """
    start = time.monotonic()
    result = SyncResult()

    # --- Fast no-op check ---
    if not force:
        state = _load_sync_state(state_path)
        adapter_key = f"{adapter.agent_name}:{user_id}"
        prev = state.get(adapter_key, {})

        current_agent_fp = adapter.fingerprint()
        all_memories = await backend.get_user_memories(user_id, limit=500)
        current_db_fp = _db_fingerprint(all_memories)

        if (
            prev.get("agent_fingerprint") == current_agent_fp
            and prev.get("db_fingerprint") == current_db_fp
        ):
            result.duration_ms = (time.monotonic() - start) * 1000
            logger.info(
                f"Sync [{adapter.agent_name}]: no-op — nothing changed ({result.duration_ms:.1f}ms)"
            )
            return result
    else:
        all_memories = await backend.get_user_memories(user_id, limit=500)

    # --- Phase 1: Import (agent files → DB) ---
    result.imported = await sync_import(backend, adapter, user_id, all_memories)

    # --- Phase 2: Export (DB → agent files) ---
    # Re-fetch if imports happened (new entries)
    if result.imported > 0:
        all_memories = await backend.get_user_memories(user_id, limit=500)
    result.exported = await sync_export(backend, adapter, user_id, all_memories)

    # --- Update sync state ---
    state = _load_sync_state(state_path)
    adapter_key = f"{adapter.agent_name}:{user_id}"
    state[adapter_key] = {
        "agent_fingerprint": adapter.fingerprint(),
        "db_fingerprint": _db_fingerprint(all_memories),
        "last_sync": datetime.now(timezone.utc).isoformat(),
        "last_imported": result.imported,
        "last_exported": result.exported,
    }
    _save_sync_state(state_path, state)

    result.duration_ms = (time.monotonic() - start) * 1000
    logger.info(
        f"Sync [{adapter.agent_name}]: imported={result.imported}, "
        f"exported={result.exported} ({result.duration_ms:.1f}ms)"
    )
    return result


async def sync_import(
    backend: Any,
    adapter: AgentMemoryAdapter,
    user_id: str,
    existing_memories: list[Any] | None = None,
) -> int:
    """Import: agent files → DB. Returns count imported."""
    agent_memories = await adapter.read_memories()
    if not agent_memories:
        return 0

    # Build set of existing content hashes for dedup
    if existing_memories is None:
        existing_memories = await backend.get_user_memories(user_id, limit=500)

    existing_hashes: set[str] = set()
    for mem in existing_memories:
        h = (mem.metadata or {}).get("content_hash", "")
        if h:
            existing_hashes.add(h)
        # Also hash the content directly for safety
        existing_hashes.add(hashlib.sha256(mem.content.encode()).hexdigest()[:16])

    imported = 0
    for am in agent_memories:
        if am.content_hash in existing_hashes:
            continue

        # Save to DB with lineage metadata
        await backend.save_memory(
            content=am.content,
            user_id=user_id,
            importance=0.6,
            metadata={
                "source_agent": adapter.agent_name,
                "source_file": am.source_file,
                "content_hash": am.content_hash,
                "synced_at": datetime.now(timezone.utc).isoformat(),
                "sync_direction": "import",
                **am.metadata,
            },
        )
        existing_hashes.add(am.content_hash)
        imported += 1

    if imported:
        logger.info(f"Sync [{adapter.agent_name}]: imported {imported} memories from agent files")
    return imported


async def sync_export(
    backend: Any,
    adapter: AgentMemoryAdapter,
    user_id: str,
    existing_memories: list[Any] | None = None,
) -> int:
    """Export: DB → agent files. Returns count exported."""
    if existing_memories is None:
        existing_memories = await backend.get_user_memories(user_id, limit=500)

    if not existing_memories:
        return 0

    # Read what the agent already has (to avoid re-exporting)
    agent_memories = await adapter.read_memories()
    agent_hashes: set[str] = {am.content_hash for am in agent_memories}

    # Find memories to export (not already in agent, not imported FROM this agent)
    to_export: list[dict[str, Any]] = []
    for mem in existing_memories:
        content_hash = hashlib.sha256(mem.content.encode()).hexdigest()[:16]

        # Skip if agent already has it
        if content_hash in agent_hashes:
            continue

        # Skip if this memory was originally imported FROM this same agent
        # (prevents echo: agent → DB → agent)
        meta = mem.metadata or {}
        if (
            meta.get("source_agent") == adapter.agent_name
            and meta.get("sync_direction") == "import"
        ):
            continue

        to_export.append(
            {
                "content": mem.content,
                "category": getattr(mem, "category", "") or "",
                "importance": getattr(mem, "importance", 0.5),
                "headroom_id": mem.id,
                "source_agent": meta.get("source_agent", "unknown"),
                "content_hash": content_hash,
                "created_at": mem.created_at.isoformat()
                if hasattr(mem.created_at, "isoformat")
                else str(mem.created_at),
            }
        )

    if not to_export:
        return 0

    exported = await adapter.write_memories(to_export)
    if exported:
        logger.info(f"Sync [{adapter.agent_name}]: exported {exported} memories to agent files")
    return exported


# ---------------------------------------------------------------------------
# CLI entry point: python -m headroom.memory.sync --db ... --user ... --agent ...
# ---------------------------------------------------------------------------


def _build_sync_backend(db_path: str) -> Any:
    """Build the memory backend used by the sync subprocess.

    Match the proxy MCP server (see ``headroom/memory/mcp_server.py``): use the
    torch-free ONNX embedder so ``wrap --memory`` sync works on the proxy extras
    without sentence-transformers/PyTorch (#1092). It loads the same
    ``all-MiniLM-L6-v2`` 384-dim model as the local embedder, so vectors stay
    compatible with what the proxy writes — no DB migration.
    """
    from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

    config = LocalBackendConfig(db_path=db_path, embedder_backend="onnx")
    return LocalBackend(config)


def main() -> None:
    """CLI entry point for running sync from a subprocess."""
    import argparse

    parser = argparse.ArgumentParser(description="Headroom memory sync")
    parser.add_argument("--db", required=True, help="Path to memory DB")
    parser.add_argument("--user", required=True, help="User ID")
    parser.add_argument(
        "--agent", required=True, choices=["claude", "codebuddy", "codex"], help="Agent to sync"
    )
    parser.add_argument("--force", action="store_true", help="Skip no-op check")
    args = parser.parse_args()

    import asyncio
    import json as _json

    async def _run() -> None:
        backend = _build_sync_backend(args.db)
        await backend._ensure_initialized()

        if args.agent == "claude":
            from headroom.memory.sync_adapters.claude_code import (
                ClaudeCodeAdapter,
                get_claude_memory_dir,
            )

            adapter: ClaudeCodeAdapter | Any = ClaudeCodeAdapter(get_claude_memory_dir())
        elif args.agent == "codebuddy":
            from headroom.memory.sync_adapters.codebuddy_code import (
                CodeBuddyAdapter,
                get_codebuddy_memory_dir,
            )

            adapter = CodeBuddyAdapter(get_codebuddy_memory_dir())
        elif args.agent == "codex":
            from headroom.memory.sync_adapters.codex_agent import CodexAdapter

            adapter = CodexAdapter()
        else:
            print(_json.dumps({"error": f"Unknown agent: {args.agent}"}))
            return

        result = await sync(backend, adapter, args.user, force=args.force)
        await backend.close()
        print(
            _json.dumps(
                {
                    "imported": result.imported,
                    "exported": result.exported,
                    "ms": round(result.duration_ms),
                }
            )
        )

    asyncio.run(_run())


if __name__ == "__main__":
    main()
