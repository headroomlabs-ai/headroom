"""SQLite storage backend for CompressionStore.

Default backend for the CCR store. Two properties the in-memory backend
cannot provide, both load-bearing for the no-accuracy-loss guarantee:

- **Restart survival.** A proxy restart no longer destroys every
  retrievable original mid-session. With the session-scale 30-minute
  TTL, entries are expected to outlive any single process.
- **Multi-worker sharing.** The database file (WAL mode) is shared
  across worker processes, so a `headroom_retrieve` call served by a
  different worker than the one that compressed still finds the entry.
  This closes the largest of the documented multi-worker gaps.

Set ``HEADROOM_CCR_BACKEND=memory`` to opt back into the in-memory
backend, or ``HEADROOM_CCR_SQLITE_PATH`` to relocate the database file
(default ``workspace_dir()/ccr_store.db``).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..compression_store import CompressionEntry

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ccr_entries (
    hash TEXT PRIMARY KEY,
    entry_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    ttl INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ccr_expiry ON ccr_entries (created_at);
"""

# Purge expired rows at most this often (seconds). Purging is hygiene,
# not correctness — CompressionStore checks TTL on every get().
_PURGE_INTERVAL = 60.0


def default_db_path() -> Path:
    """Resolve the database path (env override, else workspace root)."""
    env = os.environ.get("HEADROOM_CCR_SQLITE_PATH", "").strip()
    if env:
        return Path(env).expanduser()
    from ...paths import workspace_dir

    return workspace_dir() / "ccr_store.db"


class SQLiteBackend:
    """Thread-safe SQLite storage backend (WAL mode).

    Entries are serialized as one JSON blob per row; ``created_at`` and
    ``ttl`` are duplicated into columns so expired rows can be purged
    with one DELETE. TTL *enforcement* on reads stays in
    CompressionStore, matching the backend protocol contract.

    Deserialization is field-filtered: unknown keys in stored JSON are
    dropped (forward-compatible with newer versions that add fields).
    Missing keys load cleanly only when the corresponding
    ``CompressionEntry`` field has a default; a blob missing a required
    field (one without a default) raises ``TypeError`` on construction.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._path = Path(db_path).expanduser() if db_path else default_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_purge = 0.0
        self._conn = self._open()

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, check_same_thread=False)
        # Wait for competing writers instead of failing with SQLITE_BUSY —
        # multiple proxy workers share this file, and writes are frequent
        # but tiny, so contention resolves in milliseconds.
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        # Startup hygiene: expired rows are only purged opportunistically
        # on writes, so a quiet store could otherwise hold expired
        # originals (which may contain sensitive tool output) on disk
        # indefinitely. Sweep them on every open.
        conn.execute(
            "DELETE FROM ccr_entries WHERE created_at + ttl < ?",
            (time.time(),),
        )
        conn.commit()
        # Originals can contain sensitive tool output (file contents,
        # command output) — keep the database private to the user.
        for suffix in ("", "-wal", "-shm"):
            p = Path(str(self._path) + suffix)
            if p.exists():
                try:
                    p.chmod(0o600)
                except OSError:
                    pass
        return conn

    @staticmethod
    def _is_corruption(error: Exception) -> bool:
        """Only genuine file corruption justifies recreating the database.

        ``sqlite3.OperationalError`` (a DatabaseError subclass) also covers
        transient conditions like ``database is locked`` under multi-worker
        write contention — misclassifying those as corruption would delete
        live data while sibling workers still hold handles to the unlinked
        inode (split-brain). Match the corruption messages explicitly.
        """
        msg = str(error).lower()
        return "malformed" in msg or "not a database" in msg

    def _handle_db_error(self, error: sqlite3.DatabaseError, op: str) -> None:
        """Corruption → recreate (loud). Anything else (busy/locked/io) →
        log and treat the operation as a miss; never destroy data over a
        transient error."""
        if not self._is_corruption(error):
            logger.warning("CCR SQLite %s failed (transient, no reset): %s", op, error)
            return
        logger.warning(
            "CCR SQLite store at %s is corrupt (%s); recreating. "
            "Previously stored originals are lost — affected retrieval "
            "markers will miss until their content is re-compressed.",
            self._path,
            error,
        )
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001 - best-effort close on corrupt handle
            pass
        self._path.unlink(missing_ok=True)
        self._conn = self._open()

    def _entry_from_json(self, raw: str) -> CompressionEntry | None:
        from ..compression_store import CompressionEntry

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        known = {f.name for f in fields(CompressionEntry)}
        return CompressionEntry(**{k: v for k, v in data.items() if k in known})

    def _maybe_purge(self) -> None:
        """Delete expired rows; called opportunistically under the lock."""
        now = time.time()
        if now - self._last_purge < _PURGE_INTERVAL:
            return
        self._last_purge = now
        self._conn.execute(
            "DELETE FROM ccr_entries WHERE created_at + ttl < ?",
            (now,),
        )
        self._conn.commit()

    def get(self, hash_key: str) -> CompressionEntry | None:
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT entry_json FROM ccr_entries WHERE hash = ?",
                    (hash_key,),
                ).fetchone()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "get")
                return None
        if row is None:
            return None
        return self._entry_from_json(row[0])

    def set(self, hash_key: str, entry: CompressionEntry) -> None:
        payload = json.dumps(asdict(entry), ensure_ascii=False)
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO ccr_entries "
                    "(hash, entry_json, created_at, ttl) VALUES (?, ?, ?, ?)",
                    (hash_key, payload, entry.created_at, entry.ttl),
                )
                self._conn.commit()
                self._maybe_purge()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "set")

    def delete(self, hash_key: str) -> bool:
        with self._lock:
            try:
                cur = self._conn.execute(
                    "DELETE FROM ccr_entries WHERE hash = ?",
                    (hash_key,),
                )
                self._conn.commit()
                return cur.rowcount > 0
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "op")
                return False

    def exists(self, hash_key: str) -> bool:
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT 1 FROM ccr_entries WHERE hash = ?",
                    (hash_key,),
                ).fetchone()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "op")
                return False
        return row is not None

    def clear(self) -> None:
        with self._lock:
            try:
                self._conn.execute("DELETE FROM ccr_entries")
                self._conn.commit()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "op")

    def count(self) -> int:
        with self._lock:
            try:
                row = self._conn.execute("SELECT COUNT(*) FROM ccr_entries").fetchone()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "op")
                return 0
        return int(row[0])

    def keys(self) -> list[str]:
        with self._lock:
            try:
                rows = self._conn.execute("SELECT hash FROM ccr_entries").fetchall()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "op")
                return []
        return [r[0] for r in rows]

    def items(self) -> list[tuple[str, CompressionEntry]]:
        with self._lock:
            try:
                rows = self._conn.execute("SELECT hash, entry_json FROM ccr_entries").fetchall()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "op")
                return []
        out: list[tuple[str, CompressionEntry]] = []
        for hash_key, raw in rows:
            entry = self._entry_from_json(raw)
            if entry is not None:
                out.append((hash_key, entry))
        return out

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            try:
                count_row = self._conn.execute("SELECT COUNT(*) FROM ccr_entries").fetchone()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "op")
                count_row = (0,)
        try:
            bytes_used = self._path.stat().st_size
        except OSError:
            bytes_used = 0
        return {
            "backend_type": "sqlite",
            "entry_count": int(count_row[0]),
            "bytes_used": bytes_used,
            "db_path": str(self._path),
        }
