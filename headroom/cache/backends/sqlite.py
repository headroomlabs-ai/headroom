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
import math
import os
import sqlite3
import threading
import time
from collections.abc import Collection
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...fileperms import ensure_private_file
from ...log_safety import WarnOnce, safe_id
from ..context_clock import ContextTurnSnapshot, advance_context_state, resolve_context_conversation

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
CREATE INDEX IF NOT EXISTS idx_ccr_expiry_deadline ON ccr_entries (created_at + ttl);
-- Superseded by idx_ccr_expiry_deadline: no query searches or orders by bare
-- created_at, so the old index only cost writes. Drop it from existing files.
DROP INDEX IF EXISTS idx_ccr_expiry;
CREATE TABLE IF NOT EXISTS ccr_context_states (
    conversation_key TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    expires_at REAL NOT NULL,
    namespace_key TEXT
);
CREATE INDEX IF NOT EXISTS idx_ccr_context_expiry ON ccr_context_states(expires_at);
"""

# Purge expired rows at most this often (seconds). Purging is hygiene,
# not correctness — CompressionStore checks TTL on every get().
_PURGE_INTERVAL = 60.0
# items() decodes every row on each eviction pass, so an unreadable row warns
# once per hash per hour (WarnOnce's window), for at most this many hashes per
# hour; further ones log at debug only. A row that stays broken warns again
# about once an hour, which keeps it visible without flooding.
_MAX_REPORTED_UNREADABLE = 1000
# After a failed reopen, retry on the next operation once this many seconds
# have passed, doubling per consecutive failure up to the cap. Bounded so a
# store whose filesystem problem has cleared comes back on its own, without
# hammering a still-broken path on every call.
_REOPEN_BACKOFF_INITIAL = 1.0
_REOPEN_BACKOFF_MAX = 60.0


class StoreUnavailableError(sqlite3.OperationalError):
    """The connection is down after a failed reopen and the retry is not due.

    Subclasses ``sqlite3.OperationalError`` so every existing
    ``except sqlite3.DatabaseError`` path degrades it to a miss.
    """


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
        self._unreadable_warned = WarnOnce(
            _MAX_REPORTED_UNREADABLE, "unreadable CCR SQLite entries"
        )
        self._conn: sqlite3.Connection | None = self._open()
        self._reopen_backoff = _REOPEN_BACKOFF_INITIAL
        self._next_reopen_at = 0.0

    @staticmethod
    def _ensure_private(path: Path) -> None:
        """Make the database file 0600 *before* sqlite opens it, failing CLOSED.

        The originals stored here can contain sensitive tool output (file
        contents, command output). ``mkdir`` created the parent at the umask
        default and sqlite would create the db at the umask default too, so the
        file is created (or an existing one narrowed) through
        :func:`headroom.fileperms.ensure_private_file`, which is symlink-race
        resistant and raises :class:`PermissionError` rather than opening a
        wide file. See that function for the exact guarantee per platform.
        """
        ensure_private_file(path, what="CCR store")

    def _open(self) -> sqlite3.Connection:
        self._ensure_private(self._path)
        conn = sqlite3.connect(self._path, check_same_thread=False)
        # Wait for competing writers instead of failing with SQLITE_BUSY —
        # multiple proxy workers share this file, and writes are frequent
        # but tiny, so contention resolves in milliseconds.
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        self._migrate_context_namespaces(conn)
        # Startup hygiene: expired rows are only purged opportunistically
        # on writes, so a quiet store could otherwise hold expired
        # originals (which may contain sensitive tool output) on disk
        # indefinitely. Sweep them on every open.
        conn.execute(
            "DELETE FROM ccr_entries WHERE created_at + ttl < ?",
            (time.time(),),
        )
        conn.execute("DELETE FROM ccr_context_states WHERE expires_at < ?", (time.time(),))
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

    def _connection(self) -> sqlite3.Connection:
        """Return the live connection, reopening it if a retry is due.

        Caller holds ``_lock``. Raises :class:`StoreUnavailableError` while the
        store is down: reads degrade to a miss and ``set()`` returns ``False``
        to the producer.
        """
        if self._conn is not None:
            return self._conn
        now = time.monotonic()
        if now < self._next_reopen_at:
            raise StoreUnavailableError(f"CCR SQLite store {self._path} is unavailable")
        try:
            self._conn = self._open()
        except Exception as e:  # noqa: BLE001 - any open failure keeps the store down
            self._schedule_reopen(now)
            raise StoreUnavailableError(f"CCR SQLite store {self._path} reopen failed: {e}") from e
        self._reopen_backoff = _REOPEN_BACKOFF_INITIAL
        self._next_reopen_at = 0.0
        logger.warning("CCR SQLite store %s reopened; storage is available again.", self._path)
        return self._conn

    def _schedule_reopen(self, now: float) -> None:
        self._next_reopen_at = now + self._reopen_backoff
        self._reopen_backoff = min(self._reopen_backoff * 2, _REOPEN_BACKOFF_MAX)

    @staticmethod
    def _migrate_context_namespaces(conn: sqlite3.Connection) -> None:
        """Index clock ownership independently of JSON, preserving legacy anchors."""
        conn.execute("BEGIN IMMEDIATE")
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(ccr_context_states)")}
            if "namespace_key" not in columns:
                conn.execute("ALTER TABLE ccr_context_states ADD COLUMN namespace_key TEXT")
                for key, raw in conn.execute(
                    "SELECT conversation_key, state_json FROM ccr_context_states"
                ).fetchall():
                    try:
                        state = json.loads(raw)
                    except (ValueError, TypeError):
                        continue
                    namespace = state.get("namespace") if isinstance(state, dict) else None
                    if isinstance(namespace, str) and namespace:
                        conn.execute(
                            "UPDATE ccr_context_states SET namespace_key=? WHERE conversation_key=?",
                            (namespace, key),
                        )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ccr_context_namespace ON ccr_context_states(namespace_key)"
            )
            conn.commit()
        except sqlite3.DatabaseError:
            conn.rollback()
            raise

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
        """Corruption → fail closed (loud). Anything else (busy/locked/io) →
        log and treat the operation as a miss; never destroy data."""
        if isinstance(error, StoreUnavailableError):
            logger.warning(
                "CCR SQLite %s skipped: %s (nothing was read or persisted; "
                "will retry the connection)",
                op,
                error,
            )
            return
        if not self._is_corruption(error):
            logger.warning(
                "CCR SQLite %s failed on %s (transient, no reset): %s", op, self._path, error
            )
            return
        logger.warning(
            "CCR SQLite store at %s is corrupt (%s); going offline without "
            "touching the file. The database and its -wal/-shm sidecars are "
            "preserved for offline recovery: sibling workers may still hold "
            "live handles, so an online recreate would split-brain their "
            "reads/writes or discard committed originals from old WAL state. "
            "A later operation retries the connection after the backoff; the "
            "retry succeeds once the file has been repaired or removed offline. "
            "Until then nothing is read or persisted through this instance.",
            self._path,
            error,
        )
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:  # noqa: BLE001 - best-effort close on corrupt handle
            pass
        self._conn = None
        self._unreadable_warned = WarnOnce(
            _MAX_REPORTED_UNREADABLE, "unreadable CCR SQLite entries"
        )
        # Reopen on the next operation once the backoff allows it. Reopening a
        # still-corrupt file raises again inside ``_open()`` (schema run hits
        # the corruption), which re-enters the fail-closed state; recovery is
        # always offline, by the operator.
        self._schedule_reopen(time.monotonic())

    def _report_unreadable(self, hash_key: str, reason: object) -> None:
        first = self._unreadable_warned.first(hash_key, logger)
        logger.log(
            logging.WARNING if first else logging.DEBUG,
            "CCR SQLite entry %s in %s is unreadable (%s); treating as a miss",
            safe_id(hash_key),
            self._path,
            reason,
        )

    def _entry_from_json(self, raw: str, hash_key: str) -> CompressionEntry | None:
        from ..compression_store import CompressionEntry

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as e:
            self._report_unreadable(hash_key, type(e).__name__)
            return None
        if not isinstance(data, dict):
            self._report_unreadable(hash_key, "not a JSON object")
            return None
        known = {f.name for f in fields(CompressionEntry)}
        try:
            return CompressionEntry(**{k: v for k, v in data.items() if k in known})
        except (TypeError, ValueError) as e:
            # A blob that parses as JSON but is missing a required field (schema
            # drift across an upgrade, a partially written row) must degrade to a
            # miss, not raise. Otherwise a single bad row crashes get() — and,
            # via items(), _clean_expired() runs it on every store's eviction, so
            # one poison row would break all reads, evictions, and stores.
            self._report_unreadable(hash_key, type(e).__name__)
            return None

    def _purge_expired(self, now: float) -> int:
        """Delete expired rows using metadata; caller holds ``_lock``."""
        conn = self._connection()
        cursor = conn.execute(
            "DELETE FROM ccr_entries WHERE created_at + ttl < ?",
            (now,),
        )
        conn.execute("DELETE FROM ccr_context_states WHERE expires_at < ?", (now,))
        conn.commit()
        self._last_purge = now
        return cursor.rowcount

    def purge_expired(self, now: float | None = None) -> int:
        """Delete expired rows without deserializing their payloads."""
        with self._lock:
            try:
                return self._purge_expired(time.time() if now is None else now)
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "purge")
                return 0

    def _maybe_purge(self) -> None:
        """Delete expired rows; called opportunistically under the lock."""
        now = time.time()
        if now - self._last_purge < _PURGE_INTERVAL:
            return
        self._purge_expired(now)

    def observe_context_turn(
        self, conversation_key: str, hash_keys: Collection[str], *, namespace_key: str | None = None
    ) -> ContextTurnSnapshot | None:
        """Persist the counter and event anchors in one cross-worker transaction."""
        if not hash_keys:
            return None
        with self._lock:
            conn = None
            try:
                conn = self._connection()
                conn.execute("BEGIN IMMEDIATE")
                now = time.time()
                events = {}
                for hash_key in set(hash_keys):
                    row = conn.execute(
                        "SELECT created_at, ttl, entry_json FROM ccr_entries "
                        "WHERE hash = ? AND created_at + ttl >= ?",
                        (hash_key, now),
                    ).fetchone()
                    if row is not None:
                        entry = self._entry_from_json(row[2], hash_key)
                        if entry is not None:
                            events[hash_key] = (row[0], row[0] + row[1], entry.event_identity)
                if not events:
                    conn.rollback()
                    return None
                conn.execute("DELETE FROM ccr_context_states WHERE expires_at < ?", (now,))
                if namespace_key is not None:
                    states = {}
                    for key, raw, indexed_namespace in conn.execute(
                        "SELECT conversation_key, state_json, namespace_key FROM ccr_context_states "
                        "WHERE namespace_key=? OR namespace_key IS NULL OR conversation_key=?",
                        (namespace_key, conversation_key),
                    ):
                        candidate = json.loads(raw)
                        if indexed_namespace is not None and (
                            not isinstance(candidate, dict)
                            or candidate.get("namespace") != indexed_namespace
                        ):
                            raise ValueError(
                                "CCR clock namespace does not match its retained ownership"
                            )
                        states[key] = candidate
                    conversation_key = resolve_context_conversation(
                        conversation_key, namespace_key, states, events
                    )
                row = conn.execute(
                    "SELECT state_json FROM ccr_context_states WHERE conversation_key = ?",
                    (conversation_key,),
                ).fetchone()
                live_hashes = {
                    row[0]
                    for row in conn.execute(
                        "SELECT hash FROM ccr_entries WHERE created_at + ttl >= ?", (now,)
                    )
                }
                state, snapshot, expires_at = advance_context_state(
                    None if row is None else json.loads(row[0]), events, live_hashes, now
                )
                state["namespace"] = namespace_key if namespace_key is not None else "explicit"
                conn.execute(
                    "INSERT INTO ccr_context_states(conversation_key, state_json, expires_at, namespace_key) "
                    "VALUES (?, ?, ?, ?) ON CONFLICT(conversation_key) DO UPDATE SET "
                    "state_json = excluded.state_json, expires_at = excluded.expires_at, "
                    "namespace_key = excluded.namespace_key",
                    (conversation_key, json.dumps(state), expires_at, state["namespace"]),
                )
                conn.commit()
                return snapshot
            except (ValueError, TypeError) as error:
                if conn is not None:
                    conn.rollback()
                logger.warning("CCR context clock invalid; proactive expansion skipped: %s", error)
                return None
            except sqlite3.DatabaseError as error:
                if conn is not None:
                    conn.rollback()
                self._handle_db_error(error, "context clock")
                return None

    def get(self, hash_key: str) -> CompressionEntry | None:
        with self._lock:
            try:
                conn = self._connection()
                row = conn.execute(
                    "SELECT entry_json FROM ccr_entries WHERE hash = ?",
                    (hash_key,),
                ).fetchone()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, f"get {safe_id(hash_key)}")
                return None
        if row is None:
            return None
        return self._entry_from_json(row[0], hash_key)

    def record_retrieval(self, hash_key: str, entry: CompressionEntry, query: str | None) -> None:
        """Compare event identity and merge access metadata in one transaction."""
        with self._lock:
            conn = None
            try:
                conn = self._connection()
                conn.execute("BEGIN IMMEDIATE")
                row = conn.execute(
                    "SELECT entry_json FROM ccr_entries WHERE hash = ?", (hash_key,)
                ).fetchone()
                current = None if row is None else self._entry_from_json(row[0], hash_key)
                if current is None or current.event_identity != entry.event_identity:
                    conn.rollback()
                    return
                current.record_access(query)
                conn.execute(
                    "UPDATE ccr_entries SET entry_json = ? WHERE hash = ?",
                    (json.dumps(asdict(current), ensure_ascii=False), hash_key),
                )
                conn.commit()
                entry.retrieval_count = current.retrieval_count
                entry.last_accessed = current.last_accessed
                entry.search_queries = current.search_queries.copy()
            except sqlite3.DatabaseError as error:
                if conn is not None:
                    conn.rollback()
                self._handle_db_error(error, "record retrieval")

    def set(self, hash_key: str, entry: CompressionEntry) -> bool:
        return self._set(hash_key, entry, fresh_event=False)

    def set_new(self, hash_key: str, entry: CompressionEntry) -> bool:
        """Assign fresh event timestamp atomically across SQLite writers."""
        return self._set(hash_key, entry, fresh_event=True)

    def _set(self, hash_key: str, entry: CompressionEntry, *, fresh_event: bool) -> bool:
        with self._lock:
            conn = None
            try:
                conn = self._connection()
                if fresh_event:
                    conn.execute("BEGIN IMMEDIATE")
                    previous = conn.execute(
                        "SELECT created_at FROM ccr_entries WHERE hash = ?", (hash_key,)
                    ).fetchone()
                    if previous is not None and entry.created_at <= previous[0]:
                        entry.created_at = math.nextafter(previous[0], math.inf)
                payload = json.dumps(asdict(entry), ensure_ascii=False)
                conn.execute(
                    "INSERT OR REPLACE INTO ccr_entries "
                    "(hash, entry_json, created_at, ttl) VALUES (?, ?, ?, ?)",
                    (hash_key, payload, entry.created_at, entry.ttl),
                )
                conn.commit()
            except sqlite3.DatabaseError as error:
                if conn is not None:
                    conn.rollback()
                self._handle_db_error(error, f"set {safe_id(hash_key)}")
                logger.warning("CCR SQLite write of %s was not persisted", safe_id(hash_key))
                return False
            try:
                self._maybe_purge()
            except sqlite3.DatabaseError as error:
                self._handle_db_error(error, "purge")
                if self._is_corruption(error):
                    return False
        return True

    def delete(self, hash_key: str) -> bool:
        with self._lock:
            try:
                conn = self._connection()
                cur = conn.execute(
                    "DELETE FROM ccr_entries WHERE hash = ?",
                    (hash_key,),
                )
                conn.commit()
                return cur.rowcount > 0
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, f"delete {safe_id(hash_key)}")
                return False

    def exists(self, hash_key: str) -> bool:
        with self._lock:
            try:
                conn = self._connection()
                row = conn.execute(
                    "SELECT 1 FROM ccr_entries WHERE hash = ?",
                    (hash_key,),
                ).fetchone()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, f"exists {safe_id(hash_key)}")
                return False
        return row is not None

    def clear(self) -> None:
        with self._lock:
            try:
                conn = self._connection()
                conn.execute("DELETE FROM ccr_entries")
                conn.execute("DELETE FROM ccr_context_states")
                conn.commit()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "clear")

    def count(self) -> int:
        with self._lock:
            try:
                conn = self._connection()
                row = conn.execute("SELECT COUNT(*) FROM ccr_entries").fetchone()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "count")
                return 0
        return int(row[0])

    def external_revision(self) -> int | None:
        """Return a token that changes when another connection commits."""
        with self._lock:
            try:
                conn = self._connection()
                row = conn.execute("PRAGMA data_version").fetchone()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "revision")
                return None
        return int(row[0])

    def keys(self) -> list[str]:
        with self._lock:
            try:
                conn = self._connection()
                rows = conn.execute("SELECT hash FROM ccr_entries").fetchall()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "keys")
                return []
        return [r[0] for r in rows]

    def items(self) -> list[tuple[str, CompressionEntry]]:
        with self._lock:
            try:
                conn = self._connection()
                rows = conn.execute("SELECT hash, entry_json FROM ccr_entries").fetchall()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "items")
                return []
        out: list[tuple[str, CompressionEntry]] = []
        for hash_key, raw in rows:
            entry = self._entry_from_json(raw, hash_key)
            if entry is not None:
                out.append((hash_key, entry))
        return out

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            try:
                conn = self._connection()
                count_row = conn.execute("SELECT COUNT(*) FROM ccr_entries").fetchone()
            except sqlite3.DatabaseError as e:
                self._handle_db_error(e, "stats")
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
