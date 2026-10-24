"""SQLite tier behind the content router's in-memory compression cache.

The router keeps recent results in a byte-budgeted LRU; this file holds the rest,
so an entry the budget evicted comes back from disk instead of being compressed
again (which costs a Kompress pass and can give different bytes, rewriting a
cached prompt prefix). It also survives a restart and is shared by worker
processes (WAL), like the CCR store; WAL needs a local filesystem.

Each router reads and writes under its own namespace (a digest of its config and
the headroom version), so routers configured differently, or a restart with
other settings, never serve each other's output. Results carrying a CCR marker
are not written here: their originals live in the CCR store, on its own clock.

The file holds compressed tool output, so it follows the CCR store's rules:
owner-only, ``workspace_dir()/compression_cache.db`` (``HEADROOM_COMPRESSION_CACHE_PATH``
to move it), expired rows deleted on open, never opened in stateless mode, and
``HEADROOM_COMPRESSION_CACHE_DISK=0`` keeps the cache in memory only. A database
error is a miss, never a failed request.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

from ..fileperms import connect_private_sqlite

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    k TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    ratio REAL NOT NULL,
    strategy TEXT NOT NULL,
    expires REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_results_expires ON results (expires);
"""
#: Expired rows are deleted every this many writes, and on open.
_PURGE_EVERY = 1000


class ResultStore:
    """Compressed results by key; an entry expires ``ttl_seconds`` after it was
    last read or written. Thread-safe; several processes may share the file."""

    def __init__(self, path: Path, ttl_seconds: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._writes = 0
        self._warned = False
        self._db = connect_private_sqlite(
            path, what="compression cache", check_same_thread=False, isolation_level=None
        )
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)
        # Sessions that ended while the proxy was down go now.
        self._db.execute("DELETE FROM results WHERE expires <= ?", (time.time(),))

    def get(self, key: str) -> tuple[str, float, str] | None:
        now = time.time()
        try:
            with self._lock:
                row = self._db.execute(
                    "SELECT text, ratio, strategy, expires FROM results "
                    "WHERE k = ? AND expires > ?",
                    (key, now),
                ).fetchone()
                # Keep what a live session still reads; refresh at most every half ttl.
                if row is not None and row[3] - now < self._ttl / 2:
                    self._db.execute(
                        "UPDATE results SET expires = ? WHERE k = ?", (now + self._ttl, key)
                    )
        except sqlite3.Error as exc:
            self._failed("read", exc)
            return None
        return None if row is None else (row[0], row[1], row[2])

    def put(self, key: str, text: str, ratio: float, strategy: str) -> None:
        now = time.time()
        try:
            with self._lock:
                self._db.execute(
                    "INSERT OR REPLACE INTO results (k, text, ratio, strategy, expires) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (key, text, ratio, strategy, now + self._ttl),
                )
                self._writes += 1
                if self._writes % _PURGE_EVERY == 0:
                    self._db.execute("DELETE FROM results WHERE expires <= ?", (now,))
        except sqlite3.Error as exc:
            self._failed("write", exc)

    def touch(self, key: str) -> None:
        """Keep ``key`` for another ttl (it is in use from memory)."""
        self._run("UPDATE results SET expires = ? WHERE k = ?", (time.time() + self._ttl, key))

    def delete(self, key: str) -> None:
        self._run("DELETE FROM results WHERE k = ?", (key,))

    def clear_prefix(self, prefix: str) -> None:
        """Delete every row whose key starts with ``prefix`` (a router's namespace)."""
        self._run("DELETE FROM results WHERE substr(k, 1, ?) = ?", (len(prefix), prefix))

    def _run(self, sql: str, args: tuple[object, ...]) -> None:
        try:
            with self._lock:
                self._db.execute(sql, args)
        except sqlite3.Error as exc:
            self._failed("write", exc)

    def _failed(self, what: str, exc: sqlite3.Error) -> None:
        if not self._warned:
            self._warned = True
            logger.warning(
                "compression cache: %s of %s failed (%s); treating it as a miss. "
                "Check the disk, or set HEADROOM_COMPRESSION_CACHE_DISK=0.",
                what,
                self._path,
                exc,
            )


def open_result_store(ttl_seconds: float) -> ResultStore | None:
    """The disk tier, or ``None`` (memory only) in stateless mode, with
    ``HEADROOM_COMPRESSION_CACHE_DISK=0``, or when the file cannot be opened."""
    from ..paths import persistence_allowed, workspace_dir

    if os.environ.get("HEADROOM_COMPRESSION_CACHE_DISK", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return None
    if not persistence_allowed("compression result cache (compression_cache.db)"):
        return None
    configured = os.environ.get("HEADROOM_COMPRESSION_CACHE_PATH", "").strip()
    path = Path(configured).expanduser() if configured else workspace_dir() / "compression_cache.db"
    try:
        return ResultStore(path, ttl_seconds)
    except (OSError, sqlite3.Error) as exc:
        logger.warning(
            "compression cache: cannot open %s (%s); keeping results in memory only. "
            "Set HEADROOM_COMPRESSION_CACHE_PATH to a writable file or "
            "HEADROOM_COMPRESSION_CACHE_DISK=0.",
            path,
            exc,
        )
        return None
