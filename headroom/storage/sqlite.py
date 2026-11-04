"""SQLite storage implementation for Headroom SDK."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import RequestMetrics
from ..utils import format_timestamp, parse_timestamp
from .base import Storage

# Cache optimizer metric columns, shared by the CREATE TABLE schema, the
# legacy-database migration, save() and _row_to_metrics().
_CACHE_OPTIMIZER_COLUMNS: tuple[tuple[str, str], ...] = (
    ("cache_optimizer_used", "TEXT"),
    ("cache_optimizer_strategy", "TEXT"),
    ("cacheable_tokens", "INTEGER DEFAULT 0"),
    ("breakpoints_inserted", "INTEGER DEFAULT 0"),
    ("estimated_cache_hit", "INTEGER DEFAULT 0"),
    ("estimated_savings_percent", "REAL DEFAULT 0.0"),
    ("semantic_cache_hit", "INTEGER DEFAULT 0"),
)


class SQLiteStorage(Storage):
    """SQLite-based metrics storage."""

    def __init__(self, db_path: str):
        """
        Initialize SQLite storage.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path
        self._ensure_db_exists()
        self._conn: sqlite3.Connection | None = None

    def _ensure_db_exists(self) -> None:
        """Create database and tables if they don't exist."""
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # isolation_level=None so we can take an explicit reserved lock with
        # BEGIN IMMEDIATE rather than a deferred transaction that still lets
        # another constructor read the old schema.
        conn = sqlite3.connect(self.db_path, isolation_level=None, timeout=30.0)
        try:
            cursor = conn.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS requests (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        model TEXT NOT NULL,
                        stream INTEGER NOT NULL,
                        mode TEXT NOT NULL,
                        tokens_input_before INTEGER NOT NULL,
                        tokens_input_after INTEGER NOT NULL,
                        tokens_output INTEGER,
                        block_breakdown TEXT NOT NULL,
                        waste_signals TEXT NOT NULL,
                        stable_prefix_hash TEXT,
                        cache_alignment_score REAL,
                        cached_tokens INTEGER,
                        cache_optimizer_used TEXT,
                        cache_optimizer_strategy TEXT,
                        cacheable_tokens INTEGER DEFAULT 0,
                        breakpoints_inserted INTEGER DEFAULT 0,
                        estimated_cache_hit INTEGER DEFAULT 0,
                        estimated_savings_percent REAL DEFAULT 0.0,
                        semantic_cache_hit INTEGER DEFAULT 0,
                        transforms_applied TEXT NOT NULL,
                        tool_units_dropped INTEGER DEFAULT 0,
                        turns_dropped INTEGER DEFAULT 0,
                        messages_hash TEXT,
                        error TEXT
                    )
                """)

                # Upgrade databases created before the cache optimizer columns
                # existed, preserving existing rows. The reserved lock above
                # serializes this check-and-alter against concurrent constructors.
                cursor.execute("PRAGMA table_info(requests)")
                existing_columns = {row[1] for row in cursor.fetchall()}
                for column, ddl in _CACHE_OPTIMIZER_COLUMNS:
                    if column not in existing_columns:
                        try:
                            cursor.execute(  # nosec B608
                                f"ALTER TABLE requests ADD COLUMN {column} {ddl}"
                            )
                        except sqlite3.OperationalError as exc:
                            # Another connection may have added the column
                            # between our schema read and this ALTER.
                            if "duplicate column name" not in str(exc).lower():
                                raise

                # Create indices
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON requests(timestamp)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_model ON requests(model)
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_mode ON requests(mode)
                """)

                conn.commit()
            except Exception:
                conn.rollback()
                raise
        finally:
            conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def save(self, metrics: RequestMetrics) -> None:
        """Save request metrics."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO requests (
                id, timestamp, model, stream, mode,
                tokens_input_before, tokens_input_after, tokens_output,
                block_breakdown, waste_signals,
                stable_prefix_hash, cache_alignment_score, cached_tokens,
                cache_optimizer_used, cache_optimizer_strategy, cacheable_tokens,
                breakpoints_inserted, estimated_cache_hit, estimated_savings_percent,
                semantic_cache_hit,
                transforms_applied, tool_units_dropped, turns_dropped,
                messages_hash, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                metrics.request_id,
                format_timestamp(metrics.timestamp),
                metrics.model,
                1 if metrics.stream else 0,
                metrics.mode,
                metrics.tokens_input_before,
                metrics.tokens_input_after,
                metrics.tokens_output,
                json.dumps(metrics.block_breakdown),
                json.dumps(metrics.waste_signals),
                metrics.stable_prefix_hash,
                metrics.cache_alignment_score,
                metrics.cached_tokens,
                metrics.cache_optimizer_used,
                metrics.cache_optimizer_strategy,
                metrics.cacheable_tokens,
                metrics.breakpoints_inserted,
                1 if metrics.estimated_cache_hit else 0,
                metrics.estimated_savings_percent,
                1 if metrics.semantic_cache_hit else 0,
                json.dumps(metrics.transforms_applied),
                metrics.tool_units_dropped,
                metrics.turns_dropped,
                metrics.messages_hash,
                metrics.error,
            ),
        )
        conn.commit()

    def get(self, request_id: str) -> RequestMetrics | None:
        """Get metrics by request ID."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM requests WHERE id = ?", (request_id,))
        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_metrics(row)

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        model: str | None = None,
        mode: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RequestMetrics]:
        """Query metrics with filters.

        Matches are ordered by timestamp descending, then request_id
        descending (the order JSONLStorage.query() applies), before the
        ``LIMIT/OFFSET`` page is returned.

        Args:
            start_time: Filter by timestamp >= start_time.
            end_time: Filter by timestamp <= end_time.
            model: Filter by model name.
            mode: Filter by mode (audit/optimize).
            limit: Maximum results to return; must be non-negative.
            offset: Number of results to skip; must be non-negative.

        Returns:
            List of matching RequestMetrics for the requested page.

        Raises:
            ValueError: If limit or offset is negative.
        """
        if limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        if offset < 0:
            raise ValueError(f"offset must be non-negative, got {offset}")

        conn = self._get_conn()
        cursor = conn.cursor()

        query = "SELECT * FROM requests WHERE 1=1"
        params: list[Any] = []

        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(format_timestamp(start_time))
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(format_timestamp(end_time))
        if model is not None:
            query += " AND model = ?"
            params.append(model)
        if mode is not None:
            query += " AND mode = ?"
            params.append(mode)

        # Timestamp DESC, id DESC: the same fully-defined order
        # JSONLStorage.query() applies, so tied timestamps page identically.
        query += " ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [self._row_to_metrics(row) for row in rows]

    def count(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        model: str | None = None,
        mode: str | None = None,
    ) -> int:
        """Count metrics matching filters."""
        conn = self._get_conn()
        cursor = conn.cursor()

        query = "SELECT COUNT(*) FROM requests WHERE 1=1"
        params: list[Any] = []

        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(format_timestamp(start_time))
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(format_timestamp(end_time))
        if model is not None:
            query += " AND model = ?"
            params.append(model)
        if mode is not None:
            query += " AND mode = ?"
            params.append(mode)

        cursor.execute(query, params)
        result = cursor.fetchone()[0]
        return int(result) if result is not None else 0

    def iter_all(self) -> Iterator[RequestMetrics]:
        """Iterate over all stored metrics."""
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM requests ORDER BY timestamp")
        for row in cursor:
            yield self._row_to_metrics(row)

    def get_summary_stats(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
        """Get summary statistics."""
        conn = self._get_conn()
        cursor = conn.cursor()

        where_clause = "WHERE 1=1"
        params: list[Any] = []

        if start_time is not None:
            where_clause += " AND timestamp >= ?"
            params.append(format_timestamp(start_time))
        if end_time is not None:
            where_clause += " AND timestamp <= ?"
            params.append(format_timestamp(end_time))

        cursor.execute(  # nosec B608
            f"""
            SELECT
                COUNT(*) as total_requests,
                SUM(tokens_input_before) as total_tokens_before,
                SUM(tokens_input_after) as total_tokens_after,
                SUM(tokens_input_before - tokens_input_after) as total_tokens_saved,
                AVG(tokens_input_before - tokens_input_after) as avg_tokens_saved,
                AVG(cache_alignment_score) as avg_cache_alignment,
                SUM(CASE WHEN mode = 'audit' THEN 1 ELSE 0 END) as audit_count,
                SUM(CASE WHEN mode = 'optimize' THEN 1 ELSE 0 END) as optimize_count
            FROM requests
            {where_clause}
            """,
            params,
        )

        row = cursor.fetchone()

        return {
            "total_requests": row[0] or 0,
            "total_tokens_before": row[1] or 0,
            "total_tokens_after": row[2] or 0,
            "total_tokens_saved": row[3] or 0,
            "avg_tokens_saved": row[4] or 0,
            "avg_cache_alignment": row[5] or 0,
            "audit_count": row[6] or 0,
            "optimize_count": row[7] or 0,
        }

    def _row_to_metrics(self, row: sqlite3.Row) -> RequestMetrics:
        """Convert database row to RequestMetrics."""
        return RequestMetrics(
            request_id=row["id"],
            timestamp=parse_timestamp(row["timestamp"]),
            model=row["model"],
            stream=bool(row["stream"]),
            mode=row["mode"],
            tokens_input_before=row["tokens_input_before"],
            tokens_input_after=row["tokens_input_after"],
            tokens_output=row["tokens_output"],
            block_breakdown=json.loads(row["block_breakdown"]),
            waste_signals=json.loads(row["waste_signals"]),
            stable_prefix_hash=row["stable_prefix_hash"] or "",
            cache_alignment_score=row["cache_alignment_score"] or 0.0,
            cached_tokens=row["cached_tokens"],
            cache_optimizer_used=row["cache_optimizer_used"],
            cache_optimizer_strategy=row["cache_optimizer_strategy"],
            cacheable_tokens=row["cacheable_tokens"] or 0,
            breakpoints_inserted=row["breakpoints_inserted"] or 0,
            estimated_cache_hit=bool(row["estimated_cache_hit"]),
            estimated_savings_percent=row["estimated_savings_percent"] or 0.0,
            semantic_cache_hit=bool(row["semantic_cache_hit"]),
            transforms_applied=json.loads(row["transforms_applied"]),
            tool_units_dropped=row["tool_units_dropped"] or 0,
            turns_dropped=row["turns_dropped"] or 0,
            messages_hash=row["messages_hash"] or "",
            error=row["error"],
        )

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
