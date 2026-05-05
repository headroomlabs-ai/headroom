"""SQLite memory store for Headroom's hierarchical memory system.

Provides persistent storage for Memory objects with full support for:
- Hierarchical scope filtering (user/session/agent/turn)
- Temporal versioning with supersession chains
- Point-in-time queries
- Efficient batch operations
"""

from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..models import Memory, ScopeLevel
from ..ports import MemoryFilter

if TYPE_CHECKING:
    import numpy as np

# Regex pattern for safe metadata keys: alphanumeric, underscores, hyphens only
# This prevents JSON path injection attacks via malicious key names
_SAFE_METADATA_KEY_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_\-]*$")


def _validate_metadata_key(key: str) -> bool:
    """Validate that a metadata key is safe for use in JSON path expressions.

    Prevents JSON path injection by ensuring keys contain only safe characters.
    Valid keys: start with letter or underscore, contain only alphanumeric, underscore, hyphen.

    Args:
        key: The metadata key to validate.

    Returns:
        True if the key is safe, False otherwise.
    """
    if not key or len(key) > 255:
        return False
    return _SAFE_METADATA_KEY_PATTERN.match(key) is not None


class SQLiteMemoryStore:
    """SQLite-based memory store implementing the MemoryStore protocol.

    Features:
    - Full CRUD operations with batch support
    - Hierarchical scope filtering (user -> session -> agent -> turn)
    - Temporal versioning with supersession chains
    - Point-in-time queries via valid_at filter
    - Thread-safe: connection-per-request pattern

    Usage:
        store = SQLiteMemoryStore("./memories.db")
        await store.save(memory)
        memories = await store.query(MemoryFilter(user_id="alice"))

    Schema:
        The memories table stores all Memory fields with appropriate indexes
        for efficient querying by scope, category, importance, and time.
    """

    def __init__(self, db_path: str | Path = "headroom_memory.db") -> None:
        """Initialize the SQLite memory store.

        Args:
            db_path: Path to SQLite database file. Created if it doesn't exist.
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a new database connection (thread-safe pattern).

        Returns:
            A new SQLite connection with row factory configured.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize the database schema with indexes."""
        with self._get_conn() as conn:
            # Create memories table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,

                    -- Hierarchical scoping
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    agent_id TEXT,
                    turn_id TEXT,

                    -- Temporal
                    created_at TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_until TEXT,

                    -- Classification
                    category TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,

                    -- Lineage
                    supersedes TEXT,
                    superseded_by TEXT,
                    promoted_from TEXT,
                    promotion_chain TEXT NOT NULL DEFAULT '[]',

                    -- Access tracking
                    access_count INTEGER NOT NULL DEFAULT 0,
                    last_accessed TEXT,

                    -- Entity references (JSON array)
                    entity_refs TEXT NOT NULL DEFAULT '[]',

                    -- Embedding (BLOB for numpy array)
                    embedding BLOB,

                    -- Metadata (JSON object)
                    metadata TEXT NOT NULL DEFAULT '{}'
                )
            """)

            # Create indexes for efficient querying
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_session_id ON memories(session_id)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_agent_id ON memories(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_turn_id ON memories(turn_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category)")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_valid_until ON memories(valid_until)"
            )

            # Composite index for common scope queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_scope
                ON memories(user_id, session_id, agent_id, turn_id)
            """)

            # Index for supersession chain traversal
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_supersedes ON memories(supersedes)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_superseded_by ON memories(superseded_by)"
            )

            conn.commit()

    def _serialize_embedding(self, embedding: np.ndarray | None) -> bytes | None:
        """Serialize numpy array to bytes for BLOB storage."""
        if embedding is None:
            return None
        import numpy as np

        return bytes(embedding.astype(np.float32).tobytes())

    def _deserialize_embedding(
        self, data: bytes | None, dim: int | None = None
    ) -> np.ndarray | None:
        """Deserialize bytes back to numpy array."""
        if data is None:
            return None
        import numpy as np

        arr = np.frombuffer(data, dtype=np.float32)
        return arr

    def _memory_to_row(self, memory: Memory) -> dict[str, Any]:
        """Convert Memory object to row dict for insertion."""
        return {
            "id": memory.id,
            "content": memory.content,
            "user_id": memory.user_id,
            "session_id": memory.session_id,
            "agent_id": memory.agent_id,
            "turn_id": memory.turn_id,
            "created_at": memory.created_at.isoformat(),
            "valid_from": memory.valid_from.isoformat(),
            "valid_until": memory.valid_until.isoformat() if memory.valid_until else None,
            "category": "",  # Deprecated - kept for backwards compatibility
            "importance": memory.importance,
            "supersedes": memory.supersedes,
            "superseded_by": memory.superseded_by,
            "promoted_from": memory.promoted_from,
            "promotion_chain": json.dumps(memory.promotion_chain),
            "access_count": memory.access_count,
            "last_accessed": memory.last_accessed.isoformat() if memory.last_accessed else None,
            "entity_refs": json.dumps(memory.entity_refs),
            "embedding": self._serialize_embedding(memory.embedding),
            "metadata": json.dumps(memory.metadata),
        }

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        """Convert database row to Memory object."""
        return Memory(
            id=row["id"],
            content=row["content"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            agent_id=row["agent_id"],
            turn_id=row["turn_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            valid_from=datetime.fromisoformat(row["valid_from"]),
            valid_until=datetime.fromisoformat(row["valid_until"]) if row["valid_until"] else None,
            importance=row["importance"],
            supersedes=row["supersedes"],
            superseded_by=row["superseded_by"],
            promoted_from=row["promoted_from"],
            promotion_chain=json.loads(row["promotion_chain"]) if row["promotion_chain"] else [],
            access_count=row["access_count"],
            last_accessed=datetime.fromisoformat(row["last_accessed"])
            if row["last_accessed"]
            else None,
            entity_refs=json.loads(row["entity_refs"]) if row["entity_refs"] else [],
            embedding=self._deserialize_embedding(row["embedding"]),
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    async def save(self, memory: Memory) -> None:
        """Save or update a memory.

        If a memory with the same ID exists, it will be updated.

        Args:
            memory: The memory to save.
        """
        row = self._memory_to_row(memory)

        with self._get_conn() as conn:
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
            conn.commit()

    async def save_batch(self, memories: list[Memory]) -> None:
        """Save multiple memories in a single transaction.

        Args:
            memories: List of memories to save.
        """
        if not memories:
            return

        rows = [self._memory_to_row(m) for m in memories]

        with self._get_conn() as conn:
            conn.executemany(
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
                rows,
            )
            conn.commit()

    async def get(self, memory_id: str) -> Memory | None:
        """Retrieve a memory by ID.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            The memory if found, None otherwise.
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT * FROM memories WHERE id = ?",
                (memory_id,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return self._row_to_memory(row)

    async def get_batch(self, memory_ids: list[str]) -> list[Memory]:
        """Retrieve multiple memories by ID.

        Args:
            memory_ids: List of memory IDs to retrieve.

        Returns:
            List of found memories (may be shorter than input if some not found).
        """
        if not memory_ids:
            return []

        placeholders = ", ".join("?" * len(memory_ids))

        with self._get_conn() as conn:
            cursor = conn.execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})",  # nosec B608
                memory_ids,
            )

            return [self._row_to_memory(row) for row in cursor]

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            True if the memory was deleted, False if not found.
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM memories WHERE id = ?",
                (memory_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    async def delete_batch(self, memory_ids: list[str]) -> int:
        """Delete multiple memories by ID.

        Args:
            memory_ids: List of memory IDs to delete.

        Returns:
            Number of memories actually deleted.
        """
        if not memory_ids:
            return 0

        placeholders = ", ".join("?" * len(memory_ids))

        with self._get_conn() as conn:
            cursor = conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})",  # nosec B608
                memory_ids,
            )
            conn.commit()
            return cursor.rowcount

    def _build_query_conditions(self, filter: MemoryFilter) -> tuple[list[str], list[Any]]:
        """Build WHERE clause conditions from a MemoryFilter.

        Returns:
            Tuple of (conditions list, params list).
        """
        conditions: list[str] = []
        params: list[Any] = []

        # Hierarchical scope filtering
        if filter.user_id is not None:
            conditions.append("user_id = ?")
            params.append(filter.user_id)

            # Hierarchical filtering: when filtering by user_id only,
            # return USER-level and below (all that user's memories)
            # This is implicit - we just filter by user_id

            if filter.session_id is not None:
                conditions.append("session_id = ?")
                params.append(filter.session_id)

                if filter.agent_id is not None:
                    conditions.append("agent_id = ?")
                    params.append(filter.agent_id)

                    if filter.turn_id is not None:
                        conditions.append("turn_id = ?")
                        params.append(filter.turn_id)
            elif filter.agent_id is not None:
                # Agent without session - unusual but supported
                conditions.append("agent_id = ?")
                params.append(filter.agent_id)
            elif filter.turn_id is not None:
                # Turn without session/agent - unusual but supported
                conditions.append("turn_id = ?")
                params.append(filter.turn_id)
        elif filter.session_id is not None:
            # Session without user - filter by session only
            conditions.append("session_id = ?")
            params.append(filter.session_id)
        elif filter.agent_id is not None:
            # Agent without user/session
            conditions.append("agent_id = ?")
            params.append(filter.agent_id)
        elif filter.turn_id is not None:
            # Turn only
            conditions.append("turn_id = ?")
            params.append(filter.turn_id)

        # Explicit scope level filtering
        if filter.scope_levels is not None and len(filter.scope_levels) > 0:
            scope_conditions = []
            for level in filter.scope_levels:
                if level == ScopeLevel.USER:
                    # USER level: no session/agent/turn
                    scope_conditions.append(
                        "(session_id IS NULL AND agent_id IS NULL AND turn_id IS NULL)"
                    )
                elif level == ScopeLevel.SESSION:
                    # SESSION level: has session, no agent/turn
                    scope_conditions.append(
                        "(session_id IS NOT NULL AND agent_id IS NULL AND turn_id IS NULL)"
                    )
                elif level == ScopeLevel.AGENT:
                    # AGENT level: has agent, no turn
                    scope_conditions.append("(agent_id IS NOT NULL AND turn_id IS NULL)")
                elif level == ScopeLevel.TURN:
                    # TURN level: has turn
                    scope_conditions.append("(turn_id IS NOT NULL)")

            if scope_conditions:
                conditions.append(f"({' OR '.join(scope_conditions)})")

        # Temporal filtering
        if filter.created_after is not None:
            conditions.append("created_at >= ?")
            params.append(filter.created_after.isoformat())

        if filter.created_before is not None:
            conditions.append("created_at <= ?")
            params.append(filter.created_before.isoformat())

        # Point-in-time query
        if filter.valid_at is not None:
            valid_at_str = filter.valid_at.isoformat()
            conditions.append("valid_from <= ?")
            params.append(valid_at_str)
            conditions.append("(valid_until IS NULL OR valid_until > ?)")
            params.append(valid_at_str)

        # Superseded filtering
        if not filter.include_superseded:
            # Default: only return current memories (not superseded)
            conditions.append("valid_until IS NULL")

        # Importance filtering
        if filter.min_importance is not None:
            conditions.append("importance >= ?")
            params.append(filter.min_importance)

        if filter.max_importance is not None:
            conditions.append("importance <= ?")
            params.append(filter.max_importance)

        # Entity reference filtering (any of the specified entities)
        if filter.entity_refs is not None and len(filter.entity_refs) > 0:
            entity_conditions = []
            for entity_ref in filter.entity_refs:
                # Use JSON contains check
                entity_conditions.append("entity_refs LIKE ?")
                params.append(f'%"{entity_ref}"%')
            conditions.append(f"({' OR '.join(entity_conditions)})")

        # Lineage filtering
        if filter.has_supersedes is not None:
            if filter.has_supersedes:
                conditions.append("supersedes IS NOT NULL")
            else:
                conditions.append("supersedes IS NULL")

        if filter.has_promoted_from is not None:
            if filter.has_promoted_from:
                conditions.append("promoted_from IS NOT NULL")
            else:
                conditions.append("promoted_from IS NULL")

        # Metadata filtering with key validation to prevent JSON path injection
        if filter.metadata_filters:
            for key, value in filter.metadata_filters.items():
                # Validate key to prevent JSON path injection attacks
                # Invalid keys are silently skipped to avoid breaking legitimate queries
                # while blocking malicious attempts like "'] OR 1=1--"
                if not _validate_metadata_key(key):
                    continue
                # Use JSON extraction for metadata filtering
                conditions.append(f"json_extract(metadata, '$.{key}') = ?")
                params.append(json.dumps(value) if not isinstance(value, str) else value)

        return conditions, params

    async def query(self, filter: MemoryFilter) -> list[Memory]:
        """Query memories matching the given filter.

        Args:
            filter: Filter criteria for the query.

        Returns:
            List of matching memories.
        """
        conditions, params = self._build_query_conditions(filter)

        # Build WHERE clause
        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Build ORDER BY clause
        order_column = filter.order_by
        if order_column not in (
            "created_at",
            "importance",
            "access_count",
            "last_accessed",
        ):
            order_column = "created_at"

        order_direction = "DESC" if filter.order_desc else "ASC"

        # Build full query
        query = f"""
            SELECT * FROM memories
            WHERE {where_clause}
            ORDER BY {order_column} {order_direction}
        """

        # Add pagination using parameterized queries to prevent injection
        if filter.limit is not None:
            query += " LIMIT ?"
            params.append(filter.limit)

        if filter.offset > 0:
            query += " OFFSET ?"
            params.append(filter.offset)

        with self._get_conn() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_memory(row) for row in cursor]

    async def count(self, filter: MemoryFilter) -> int:
        """Count memories matching the given filter.

        Args:
            filter: Filter criteria for the count.

        Returns:
            Number of matching memories.
        """
        conditions, params = self._build_query_conditions(filter)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = f"SELECT COUNT(*) FROM memories WHERE {where_clause}"

        with self._get_conn() as conn:
            cursor = conn.execute(query, params)
            result = cursor.fetchone()[0]
            return int(result)

    async def supersede(
        self,
        old_memory_id: str,
        new_memory: Memory,
        supersede_time: datetime | None = None,
    ) -> Memory:
        """Supersede an existing memory with a new version.

        This creates a temporal chain: the old memory's valid_until is set,
        and the new memory's supersedes field points to the old one.

        Args:
            old_memory_id: ID of the memory to supersede.
            new_memory: The new memory that replaces it.
            supersede_time: When the supersession occurred (defaults to now).

        Returns:
            The saved new memory with lineage fields populated.

        Raises:
            ValueError: If the old memory is not found.
        """
        if supersede_time is None:
            supersede_time = datetime.now(timezone.utc).replace(tzinfo=None)

        # Get the old memory
        old_memory = await self.get(old_memory_id)
        if old_memory is None:
            raise ValueError(f"Memory with ID {old_memory_id} not found")

        # Update old memory's valid_until and superseded_by
        old_memory.valid_until = supersede_time
        old_memory.superseded_by = new_memory.id

        # Set up new memory's lineage
        new_memory.supersedes = old_memory_id
        new_memory.valid_from = supersede_time

        # Save both in a transaction
        with self._get_conn() as conn:
            # Update old memory
            conn.execute(
                """
                UPDATE memories
                SET valid_until = ?, superseded_by = ?
                WHERE id = ?
                """,
                (supersede_time.isoformat(), new_memory.id, old_memory_id),
            )

            # Insert new memory
            row = self._memory_to_row(new_memory)
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
            conn.commit()

        return new_memory

    async def get_history(
        self,
        memory_id: str,
        include_future: bool = False,
    ) -> list[Memory]:
        """Get the full history chain for a memory.

        Follows the supersedes/superseded_by chain to return all versions.

        Args:
            memory_id: ID of any memory in the chain.
            include_future: Whether to include memories that superseded this one.

        Returns:
            List of memories in temporal order (oldest first).
        """
        # Start with the given memory
        current = await self.get(memory_id)
        if current is None:
            return []

        history: list[Memory] = [current]

        # Follow chain backwards (supersedes)
        back_id = current.supersedes
        while back_id is not None:
            prev = await self.get(back_id)
            if prev is None:
                break
            history.insert(0, prev)  # Add to beginning
            back_id = prev.supersedes

        # Follow chain forwards (superseded_by) if requested
        if include_future:
            forward_id = current.superseded_by
            while forward_id is not None:
                next_mem = await self.get(forward_id)
                if next_mem is None:
                    break
                history.append(next_mem)
                forward_id = next_mem.superseded_by

        return history

    async def clear_scope(
        self,
        user_id: str,
        session_id: str | None = None,
        agent_id: str | None = None,
        turn_id: str | None = None,
    ) -> int:
        """Clear all memories at or below a scope level.

        Args:
            user_id: Required user scope.
            session_id: If provided, clear session and below.
            agent_id: If provided, clear agent and below.
            turn_id: If provided, clear only that turn.

        Returns:
            Number of memories deleted.
        """
        conditions = ["user_id = ?"]
        params: list[Any] = [user_id]

        if turn_id is not None:
            # Clear only specific turn
            conditions.append("turn_id = ?")
            params.append(turn_id)
        elif agent_id is not None:
            # Clear agent and its turns
            conditions.append("agent_id = ?")
            params.append(agent_id)
        elif session_id is not None:
            # Clear session and its agents/turns
            conditions.append("session_id = ?")
            params.append(session_id)
        # If only user_id, clear all user's memories

        where_clause = " AND ".join(conditions)

        with self._get_conn() as conn:
            cursor = conn.execute(
                f"DELETE FROM memories WHERE {where_clause}",  # nosec B608
                params,
            )
            conn.commit()
            return cursor.rowcount

    async def clear_all(self) -> int:
        """Clear all memories from the store.

        Returns:
            Number of memories deleted.
        """
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM memories")
            conn.commit()
            return cursor.rowcount

    def count_sync(self) -> int:
        """Synchronous count of all memories (for diagnostics).

        Returns:
            Total number of memories in the store.
        """
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memories")
            result = cursor.fetchone()[0]
            return int(result)
