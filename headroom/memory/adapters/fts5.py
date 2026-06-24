"""SQLite FTS5 full-text search index for Headroom Memory.

Provides fast, local full-text search with BM25 ranking.
Uses SQLite's built-in FTS5 extension with Porter stemming
and Unicode tokenization for high-quality search results.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..models import Memory
from ..ports import TextFilter, TextSearchResult

if TYPE_CHECKING:
    pass


@dataclass
class FTS5SearchResult:
    """Result from an FTS5 full-text search.

    This is a lightweight result that contains just the indexed fields,
    not the full Memory object. Use memory_id to fetch the full Memory
    from the MemoryStore if needed.
    """

    memory_id: str
    content: str
    score: float  # BM25 relevance score (higher = more relevant)
    metadata: dict[str, Any] = field(default_factory=dict)


class FTS5TextIndex:
    """SQLite FTS5 full-text search index.

    Features:
    - BM25 ranking for relevance scoring
    - Porter stemming for morphological matching
    - Unicode support for international text
    - Filtering by user_id and session_id
    - Batch indexing for efficiency

    Usage:
        index = FTS5TextIndex("./search.db")
        index.index("mem-123", "User prefers Python", {"user_id": "alice"})
        results = index.search("python programming", k=5)

    The FTS5 table stores:
    - memory_id: Unique identifier for the memory
    - content: Searchable text content
    - user_id: Optional user identifier for filtering
    - session_id: Optional session identifier for filtering
    - category: Memory category for filtering
    """

    def __init__(self, db_path: str | Path = "headroom_memory.db") -> None:
        """Initialize the FTS5 text index.

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
        """Initialize the FTS5 virtual table schema."""
        with self._get_conn() as conn:
            # Create FTS5 virtual table with Porter stemming and Unicode tokenization
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    memory_id,
                    content,
                    user_id,
                    session_id,
                    category,
                    tokenize='porter unicode61'
                )
            """)
            conn.commit()

    def index_raw(
        self,
        memory_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> None:
        """Index a single memory for full-text search (low-level).

        Args:
            memory_id: Unique identifier for the memory.
            text: Text content to index.
            metadata: Optional metadata dict with user_id, session_id.
        """
        metadata = metadata or {}
        user_id = metadata.get("user_id", "")
        session_id = metadata.get("session_id", "")
        category = ""  # Deprecated - kept for backwards compatibility

        with self._get_conn() as conn:
            # Delete existing entry if present (upsert behavior)
            conn.execute(
                "DELETE FROM memory_fts WHERE memory_id = ?",
                (memory_id,),
            )

            # Insert new entry
            conn.execute(
                """
                INSERT INTO memory_fts (memory_id, content, user_id, session_id, category)
                VALUES (?, ?, ?, ?, ?)
                """,
                (memory_id, text, user_id, session_id, category),
            )
            conn.commit()

    # Alias for backwards compatibility
    def index(
        self,
        memory_id: str,
        text: str,
        metadata: dict | None = None,
    ) -> None:
        """Index a single memory for full-text search.

        Alias for index_raw for backwards compatibility.
        For protocol-compliant async indexing, use index_memory().
        """
        self.index_raw(memory_id, text, metadata)

    def index_batch(
        self,
        memory_ids: list[str],
        texts: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """Index multiple memories in a single transaction.

        Args:
            memory_ids: List of unique identifiers.
            texts: List of text contents to index.
            metadata: Optional list of metadata dicts (one per memory).

        Raises:
            ValueError: If memory_ids and texts have different lengths.
        """
        if len(memory_ids) != len(texts):
            raise ValueError(
                f"memory_ids ({len(memory_ids)}) and texts ({len(texts)}) must have same length"
            )

        if metadata is not None and len(metadata) != len(memory_ids):
            raise ValueError(
                f"metadata ({len(metadata)}) must match memory_ids ({len(memory_ids)}) length"
            )

        metadata = metadata or [{} for _ in memory_ids]

        with self._get_conn() as conn:
            # Delete existing entries
            conn.executemany(
                "DELETE FROM memory_fts WHERE memory_id = ?",
                [(mid,) for mid in memory_ids],
            )

            # Prepare batch data
            batch_data = []
            for memory_id, text, meta in zip(memory_ids, texts, metadata):
                user_id = meta.get("user_id", "")
                session_id = meta.get("session_id", "")
                category = ""  # Deprecated - kept for backwards compatibility

                batch_data.append((memory_id, text, user_id, session_id, category))

            # Insert all entries
            conn.executemany(
                """
                INSERT INTO memory_fts (memory_id, content, user_id, session_id, category)
                VALUES (?, ?, ?, ?, ?)
                """,
                batch_data,
            )
            conn.commit()

    def search(
        self,
        query: str,
        k: int = 10,
        filter: TextFilter | None = None,
    ) -> list[FTS5SearchResult]:
        """Search indexed memories using FTS5 with BM25 ranking.

        Args:
            query: Search query string.
            k: Maximum number of results to return.
            filter: Optional filter for user_id, session_id.

        Returns:
            List of FTS5SearchResult ordered by BM25 relevance score.
        """
        # Sanitize query for FTS5
        fts_query = self._sanitize_fts_query(query)
        if not fts_query.strip():
            return []

        # Build WHERE clause with filters
        where_clauses = ["memory_fts MATCH ?"]
        params: list = [fts_query]

        if filter is not None:
            if filter.user_id is not None:
                where_clauses.append("user_id = ?")
                params.append(filter.user_id)

            if filter.session_id is not None:
                where_clauses.append("session_id = ?")
                params.append(filter.session_id)

        params.append(k)
        where_sql = " AND ".join(where_clauses)

        with self._get_conn() as conn:
            # Query with BM25 ranking (lower is better, so we order ASC)
            cursor = conn.execute(  # nosec B608
                f"""
                SELECT memory_id, content, user_id, session_id, category,
                       bm25(memory_fts) as rank
                FROM memory_fts
                WHERE {where_sql}
                ORDER BY rank
                LIMIT ?
                """,
                params,
            )

            results = []
            for row in cursor:
                # Convert BM25 score to a positive relevance score
                # BM25 returns negative values where more negative = more relevant
                # We negate and normalize to make higher = more relevant
                bm25_score = row["rank"]
                relevance_score = -bm25_score if bm25_score < 0 else 0.0

                results.append(
                    FTS5SearchResult(
                        memory_id=row["memory_id"],
                        content=row["content"],
                        score=relevance_score,
                        metadata={
                            "user_id": row["user_id"],
                            "session_id": row["session_id"],
                            "category": row["category"],
                        },
                    )
                )

            return results

    def delete(self, memory_id: str) -> bool:
        """Delete a memory from the index.

        Args:
            memory_id: ID of the memory to delete.

        Returns:
            True if the memory was deleted, False if not found.
        """
        with self._get_conn() as conn:
            cursor = conn.execute(
                "DELETE FROM memory_fts WHERE memory_id = ?",
                (memory_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def _sanitize_fts_query(self, query: str) -> str:
        """Sanitize a query string for FTS5.

        Escapes special characters and handles edge cases for safe querying.

        Args:
            query: Raw user query string.

        Returns:
            FTS5-safe query string with OR between terms.
        """
        # Extract alphanumeric words
        words = re.findall(r"\w+", query)

        if not words:
            return ""

        # Quote each word to handle special characters
        # Use OR between words for flexible matching
        escaped_words = [f'"{word}"' for word in words]
        return " OR ".join(escaped_words)

    def clear(self) -> None:
        """Clear all entries from the index."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM memory_fts")
            conn.commit()

    def count(self) -> int:
        """Get the total number of indexed entries.

        Returns:
            Number of entries in the index.
        """
        with self._get_conn() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM memory_fts")
            result = cursor.fetchone()[0]
            return int(result)

    # =========================================================================
    # Protocol-compliant async methods (TextIndex protocol)
    # =========================================================================

    async def index_memory(self, memory: Memory) -> None:
        """Index a memory for full-text search (protocol-compliant).

        Args:
            memory: The memory to index.
        """
        metadata = {
            "user_id": memory.user_id,
            "session_id": memory.session_id or "",
        }
        self.index(memory.id, memory.content, metadata)

    async def index_batch_memories(self, memories: list[Memory]) -> int:
        """Index multiple memories for full-text search (protocol-compliant).

        Args:
            memories: List of memories to index.

        Returns:
            Number of memories indexed.
        """
        if not memories:
            return 0

        memory_ids = [m.id for m in memories]
        texts = [m.content for m in memories]
        metadata_list = [
            {
                "user_id": m.user_id,
                "session_id": m.session_id or "",
            }
            for m in memories
        ]
        self.index_batch(memory_ids, texts, metadata_list)
        return len(memories)

    async def remove(self, memory_id: str) -> bool:
        """Remove a memory from the text index (protocol-compliant).

        Args:
            memory_id: The unique identifier of the memory.

        Returns:
            True if removed, False if not found.
        """
        return self.delete(memory_id)

    async def remove_batch(self, memory_ids: list[str]) -> int:
        """Remove multiple memories from the text index (protocol-compliant).

        Args:
            memory_ids: List of memory IDs to remove.

        Returns:
            Number of memories actually removed.
        """
        count = 0
        for memory_id in memory_ids:
            if self.delete(memory_id):
                count += 1
        return count

    async def search_memories(
        self, filter: TextFilter, store: Any = None
    ) -> list[TextSearchResult]:
        """Search for memories using full-text search (protocol-compliant).

        Args:
            filter: Text search filter with query and constraints.
            store: Optional MemoryStore to fetch full Memory objects.

        Returns:
            List of TextSearchResult sorted by relevance.
        """
        # Use the existing synchronous search
        fts_results = self.search(filter.query, k=filter.limit, filter=filter)

        results: list[TextSearchResult] = []
        for rank, fts_result in enumerate(fts_results, start=1):
            # Create a minimal Memory object from FTS data
            # If store is provided, we could fetch the full Memory
            memory = Memory(
                id=fts_result.memory_id,
                content=fts_result.content,
                user_id=fts_result.metadata.get("user_id", ""),
            )
            results.append(
                TextSearchResult(
                    memory=memory,
                    score=fts_result.score,
                    rank=rank,
                )
            )
        return results

    async def update_content(self, memory_id: str, content: str) -> bool:
        """Update the indexed content for a memory (protocol-compliant).

        Args:
            memory_id: The unique identifier of the memory.
            content: The new content to index.

        Returns:
            True if updated, False if memory not found in index.
        """
        # Check if exists first
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT user_id, session_id, category FROM memory_fts WHERE memory_id = ?",
                (memory_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return False

            # Re-index with new content
            metadata = {
                "user_id": row["user_id"],
                "session_id": row["session_id"],
                "category": row["category"],
            }
            self.index(memory_id, content, metadata)
            return True
