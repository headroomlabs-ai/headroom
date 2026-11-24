"""Cognee memory backend implementing the MemoryBackend protocol.

cognee (https://github.com/topoteretes/cognee) is an async AI-memory /
knowledge-graph engine. This backend stores headroom memories as cognee data
items tagged via ``node_set`` for user/session/entity scoping, builds a
knowledge graph via ``cognee.cognify()``, and searches via ``cognee.search()``.

Durable metadata store:
    cognee has no per-item fetch/update API for raw memories, so this backend
    keeps a small SQLite metadata store (WAL mode) next to cognee's data. It
    holds the memory registry (canonical memory IDs keyed by
    ``(user_id, content)``) and per-user tombstones for deleted/superseded
    content. Because this state is durable and shared, deletions and updates
    survive proxy restarts and are visible immediately to other backend
    instances pointed at the same ``metadata_db_path``.

Process-wide, immutable cognee configuration:
    ``import cognee`` and cognee's root-directory configuration
    (``cognee.config.system_root_directory`` / ``data_root_directory``) are
    process-global. The import runs at most once per process, guarded by a
    module-level lock, and the effective ``(system_root, data_root)`` pair is
    recorded in module state. A later backend instance initializing with the
    SAME roots reuses the configured module; one initializing with DIFFERENT
    roots fails closed with ``RuntimeError`` (one tenant must never redirect
    another tenant's cognee storage).

    cognee's import has side effects — notably
    ``dotenv.load_dotenv(override=True)``, which would overwrite already-set
    process environment variables with values from a ``.env`` in the cwd. The
    first (and only) import snapshots the environment and restores any
    pre-existing variables the import changed, preserving the normal
    env-over-``.env`` precedence (keys newly added by cognee's ``.env`` load
    are kept so cognee's own configuration keeps working).

Usage:
    from headroom.memory.backends.cognee import CogneeBackend, CogneeConfig
    from headroom.memory.system import MemorySystem

    config = CogneeConfig(dataset_name="my_app_memories")
    backend = CogneeBackend(config)
    memory_system = MemorySystem(backend, user_id="alice")

    result = await memory_system.process_tool_call(
        "memory_save",
        {"content": "User prefers Python", "importance": 0.8},
    )

Deletion contract (split by search type):
    - Under ``CHUNKS`` (the default), results are verbatim stored text and
      tombstones are authoritative: ``delete_memory`` removes the registry
      row, tombstones the content (and pre-extracted facts), and additionally
      runs a hard delete against cognee's dataset API to reclaim storage.
      Tombstones are scoped per user and matched by exact equality or
      substring containment in BOTH directions, so a chunk that is a piece of
      deleted content is filtered too — nothing derived from tombstoned text
      can pass the read path. The delete succeeds even when the hard delete
      cannot be proven.
    - Under every other search type, results may be graph-synthesized text
      that no longer contains the deleted source, which text-matched
      tombstones cannot enforce. There ``delete_memory``/``update_memory``
      succeed only when the hard delete is PROVEN (every content matched a
      stored data item, all matches were deleted, and a verification re-list
      finds none left); otherwise they raise
      ``CogneeDeletionUnverifiedError`` — delete tombstones the content and
      keeps the registry row for retry; update refuses before mutating
      anything. Discovery is non-mutating (a partial match deletes nothing)
      and removals are tracked in a durable ``hard_deleted`` ledger as
      PENDING (deletion begun, not yet verified gone) or VERIFIED (a re-list
      found none) — only VERIFIED is proof, and a retry resumes the
      verification of pending hashes rather than trusting them — so a failed
      or interrupted attempt is always retryable and never passes as
      success. This backend never reports a deletion it cannot stand behind.
    - The hard delete is TENANT-SCOPED. The ``user_id`` this backend receives
      is headroom's memory partition id — the tenant identity the proxy
      resolves for the request (``resolve_memory_identity``), composed with
      the project key in project mode — and it is carried on every cognee
      data item as the ``user:<partition>`` node-set tag. cognee identifies
      text by content hash, so two tenants who stored the same text share a
      hash; it keeps one data item per node set (the node set is part of its
      dedup identity), so each tenant's copy is a separate item. Discovery,
      deletion, verification and the ledger all act only on items carrying
      the tenant's tag: deleting tenant A's memory never touches tenant B's
      identical one, and a removal verified for A never vouches for B. An
      item whose tags cannot be read belongs to no tenant and is never
      deleted.
    - ``update_memory`` with unchanged content is a metadata-only rewrite:
      nothing is tombstoned, reclaimed or re-added, since the stored item IS
      the new content. A fact equal to the new content is likewise kept.

Known limitations (cognee v1.x):
    - cognee has no per-item update API. ``update_memory`` updates the durable
      registry row in place (same ID, new content), adds the new content to
      cognee, and tombstones the old content so it stops surfacing in search.
    - Memory IDs are stable content-derived UUIDs (``uuid5(user_id, content)``)
      resolved through the durable registry, so IDs returned by
      ``search_memories`` remain valid inputs to ``update_memory`` /
      ``delete_memory`` across restarts and across backend instances sharing
      the same ``metadata_db_path``. After an update, search keeps returning
      the ORIGINAL memory ID for the new content.
    - cognee search results carry no similarity score; scores returned here
      are rank-based, mapped into ``(0.5, 1.0]`` so they always clear the
      proxy's default ``min_similarity`` floor (0.3). ``min_similarity``
      values at or below 0.5 therefore have no filtering effect with this
      backend — the scores encode result order, not semantic similarity.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from headroom import paths
from headroom.memory import cognee_env
from headroom.memory.models import Memory
from headroom.memory.ports import MemorySearchResult

logger = logging.getLogger(__name__)

_IMPORT_ERROR_MSG = 'cognee package not installed. Install with: pip install "headroom-ai[cognee]"'


class CogneeDeletionUnverifiedError(RuntimeError):
    """Raised when a delete/update cannot prove cognee removed the underlying data.

    Only raised for search types whose results tombstones cannot fully
    enforce (anything but ``CHUNKS``): graph-synthesized text derived from a
    deleted memory need not contain the original text, so a text-matched
    tombstone is not authoritative there. Rather than reporting a deletion
    that could still resurface, the operation fails closed. The content is
    still tombstoned (suppressing every text-matched result) and the memory
    stays in the registry so the operation can be retried.
    """


# Filename used for the default metadata DB location (under data_root,
# system_root, or the headroom workspace dir, in that order).
_METADATA_DB_FILENAME = "headroom_cognee_meta.db"

# Search over-fetch bounds. Tombstoned hits are filtered after cognee ranked
# them, so a request for ``top_k`` is widened by the number of tombstones the
# partition carries (capped), and re-issued with a doubled limit while a full
# page of hits still yields fewer than ``top_k`` visible ones.
_SEARCH_OVERFETCH_CAP = 200
_SEARCH_FETCH_ROUNDS = 3
_SEARCH_FETCH_HARD_CAP = 1000


def _utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


# Namespace for stable, content-derived memory IDs. Fixed so the same
# (user_id, content) pair always maps to the same UUID across instances.
_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_DNS, "headroom.memory.backends.cognee")


def _stable_memory_id(user_id: str, content: str) -> str:
    """Return a stable, user-scoped memory ID derived from the content.

    cognee has no per-item IDs for raw memories, so IDs must be derivable
    from what search returns. Deriving them from ``(user_id, content)``
    makes the IDs surfaced by ``search_memories`` valid inputs to
    ``update_memory`` / ``delete_memory`` (instead of throwaway UUIDs).
    """
    return str(uuid.uuid5(_ID_NAMESPACE, f"{user_id}\x00{content}"))


def _user_tag(user_id: str) -> str:
    """Build the node_set tag for a user."""
    return f"user:{user_id}"


def _is_tombstoned(text: str, tombstones: set[str]) -> bool:
    """Whether a search-result chunk matches a tombstoned content.

    Matches by exact equality or when the chunk is a fragment of a tombstoned
    content (cognee chunks long documents, so a chunk of a deleted memory is a
    substring of the tombstoned original). The other direction — a chunk that
    *contains* a tombstoned text plus more — is a different, live memory whose
    own text happens to include it, and is kept. Best-effort: text that cognee
    transformed during cognify may not match.
    """
    if not tombstones:
        return False
    if text in tombstones:
        return True
    return any(text in tombstoned for tombstoned in tombstones)


def _session_tag(session_id: str) -> str:
    """Build the node_set tag for a session."""
    return f"session:{session_id}"


def _entity_tag(entity: str) -> str:
    """Build the node_set tag for an entity."""
    return f"entity:{entity}"


# ---------------------------------------------------------------------------
# Process-wide cognee import + configuration (immutable per process)
# ---------------------------------------------------------------------------
# ``import cognee`` and cognee.config root directories are process-global, so
# per-instance guards cannot protect them: two instances racing the import
# could snapshot/restore os.environ over each other, and a later instance
# could silently redirect an earlier tenant's root directories. The state
# below is module-level and mutated only under ``_process_lock`` (a threading
# lock, safe across event loops because the work runs in ``asyncio.to_thread``
# worker threads).

_process_lock = threading.Lock()
_process_cognee: Any = None
_process_search_type_cls: Any = None
# The (system_root, data_root) pair applied by the first successful
# initialization. Any later attempt with a different pair fails closed.
_process_roots: tuple[str | None, str | None] | None = None


def _import_and_configure_cognee(system_root: str | None, data_root: str | None) -> tuple[Any, Any]:
    """Import cognee once per process and apply root-directory config.

    Runs in a worker thread (see ``CogneeBackend._ensure_initialized``).
    Serialized process-wide by ``_process_lock``. On the first call the
    environment is snapshotted before the import and any pre-existing
    variable the import changed is restored (cognee's import executes
    ``dotenv.load_dotenv(override=True)``); variables newly added by the
    ``.env`` load are kept so cognee's own configuration (e.g.
    ``LLM_API_KEY``) keeps working. Subsequent calls with the same roots
    return the already-configured module without touching the environment.

    Returns:
        ``(cognee_module, SearchType_class)``.

    Raises:
        ImportError: If the cognee package is not installed.
        RuntimeError: If cognee was already configured in this process with
            different root directories (fail closed: the configuration is
            process-wide and immutable).
    """
    global _process_cognee, _process_search_type_cls, _process_roots

    requested = (system_root, data_root)
    with _process_lock:
        if _process_roots is not None:
            if requested != _process_roots:
                raise RuntimeError(
                    "cognee configuration is process-wide and immutable; already "
                    f"configured with system_root={_process_roots[0]!r}, "
                    f"data_root={_process_roots[1]!r}; refusing to reconfigure with "
                    f"system_root={system_root!r}, data_root={data_root!r}"
                )
            return _process_cognee, _process_search_type_cls

        # cognee >=1.5 enables session memory by default: searches run through
        # a session-aware completion layer that adds an LLM call per query and
        # can replay a previous turn's results — including content this backend
        # has since deleted/tombstoned. Headroom is itself the memory layer, so
        # it needs plain deterministic retrieval. setdefault respects an
        # explicit operator override; part of the immutable process-wide
        # configuration established here.
        os.environ.setdefault("CACHING", "false")

        env_before = dict(os.environ)
        try:
            cognee = importlib.import_module("cognee")
        except ImportError:
            raise ImportError(_IMPORT_ERROR_MSG) from None
        finally:
            for key, value in env_before.items():
                if os.environ.get(key) != value:
                    os.environ[key] = value

        search_type_cls = getattr(cognee, "SearchType", None)
        if search_type_cls is None:
            raise ImportError(_IMPORT_ERROR_MSG)

        if system_root:
            cognee.config.system_root_directory(system_root)
        if data_root:
            cognee.config.data_root_directory(data_root)

        _process_cognee = cognee
        _process_search_type_cls = search_type_cls
        _process_roots = requested
        return cognee, search_type_cls


def _reset_process_state_for_testing() -> None:
    """Reset the process-wide cognee import/config state. TESTS ONLY.

    Production code must never call this: the whole point of the module
    state is that cognee's process-global configuration is applied at most
    once per process. Tests use it to simulate fresh processes.
    """
    global _process_cognee, _process_search_type_cls, _process_roots
    with _process_lock:
        _process_cognee = None
        _process_search_type_cls = None
        _process_roots = None


@dataclass
class CogneeConfig:
    """Configuration for the cognee memory backend.

    Fields default to values read from ``HEADROOM_COGNEE_*`` environment
    variables (see :mod:`headroom.memory.cognee_env`). Passing an explicit
    value to the constructor always wins over the environment.

    Note: ``system_root`` / ``data_root`` configure process-global cognee
    state. The first backend initialized in a process fixes them for every
    later instance; initializing another instance with different roots
    raises ``RuntimeError`` (see module docstring).

    Attributes:
        dataset_name: cognee dataset that holds all headroom memories.
        system_root: Directory for cognee system state (databases, caches).
            Keeps headroom's cognee state isolated from other cognee installs.
            ``None`` uses cognee's own default location.
        data_root: Directory for cognee data storage. ``None`` uses cognee's
            own default location.
        search_type: cognee ``SearchType`` name used by ``search_memories``.
            ``CHUNKS`` (default) is raw retrieval with no LLM synthesis —
            cheapest and right for a proxy. ``GRAPH_COMPLETION`` retrieves
            graph context (sent with ``only_context=True`` so no LLM answer
            is generated).
        auto_cognify: Whether to run ``cognee.cognify()`` after each save so
            new memories become part of the knowledge graph.
        background_cognify: Whether ``cognify`` runs in the background.
            ``cognify`` is LLM-bound and slow; in a proxy request path this
            should stay ``True``.
        metadata_db_path: Path to the SQLite file holding the durable memory
            registry and tombstones. ``None`` (default) resolves to
            ``headroom_cognee_meta.db`` under ``data_root`` when set, else
            under ``system_root`` when set, else under the headroom
            workspace dir (``~/.headroom``). Instances that must share
            delete/update state (multiple workers, restarts) must point at
            the same file.
    """

    dataset_name: str = field(default_factory=cognee_env.cognee_env_dataset)
    system_root: str | None = field(default_factory=cognee_env.cognee_env_system_root)
    data_root: str | None = field(default_factory=cognee_env.cognee_env_data_root)
    search_type: str = field(default_factory=cognee_env.cognee_env_search_type)
    auto_cognify: bool = field(default_factory=cognee_env.cognee_env_auto_cognify)
    background_cognify: bool = True
    metadata_db_path: str | None = field(default_factory=cognee_env.cognee_env_metadata_db)


def _resolve_metadata_db_path(config: CogneeConfig) -> Path:
    """Resolve the metadata DB location for a config (see CogneeConfig docs)."""
    if config.metadata_db_path:
        return Path(config.metadata_db_path).expanduser()
    if config.data_root:
        return Path(config.data_root).expanduser() / _METADATA_DB_FILENAME
    if config.system_root:
        return Path(config.system_root).expanduser() / _METADATA_DB_FILENAME
    return paths.workspace_dir() / _METADATA_DB_FILENAME


class _CogneeMetadataStore:
    """Durable SQLite store for the memory registry and tombstones.

    Two tables:

    - ``memories``: canonical registry of memories saved or surfaced through
      this backend. The canonical-ID lookup is by ``(user_id, content)``, so
      after ``update_memory`` rewrites a row's content in place the next
      search resolves the new content back to the ORIGINAL memory ID.
    - ``tombstones``: per-user deleted/superseded contents, filtered out of
      every search result.

    All methods are synchronous; the backend calls them via
    ``asyncio.to_thread`` to stay off the event loop. A fresh connection is
    opened per operation (cheap for this workload) so calls are safe from any
    worker thread, and WAL mode keeps concurrent readers/writers across
    processes consistent.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._schema_lock = threading.Lock()
        self._schema_ready = False

    def _connect(self) -> sqlite3.Connection:
        """Open a connection, creating the schema on first use."""
        if not self._schema_ready:
            with self._schema_lock:
                if not self._schema_ready:
                    self._db_path.parent.mkdir(parents=True, exist_ok=True)
                    conn = sqlite3.connect(str(self._db_path))
                    try:
                        conn.execute("PRAGMA journal_mode=WAL")
                        conn.execute(
                            """
                            CREATE TABLE IF NOT EXISTS memories (
                                id TEXT PRIMARY KEY,
                                user_id TEXT NOT NULL,
                                content TEXT NOT NULL,
                                importance REAL,
                                metadata_json TEXT,
                                created_at TEXT,
                                updated_at TEXT
                            )
                            """
                        )
                        conn.execute(
                            "CREATE INDEX IF NOT EXISTS idx_memories_user_content "
                            "ON memories(user_id, content)"
                        )
                        conn.execute(
                            """
                            CREATE TABLE IF NOT EXISTS tombstones (
                                user_id TEXT NOT NULL,
                                content TEXT NOT NULL,
                                created_at TEXT,
                                PRIMARY KEY (user_id, content)
                            )
                            """
                        )
                        conn.execute(
                            """
                            CREATE TABLE IF NOT EXISTS hard_deleted (
                                dataset TEXT NOT NULL,
                                owner TEXT NOT NULL DEFAULT '',
                                content_hash TEXT NOT NULL,
                                created_at TEXT,
                                verified INTEGER NOT NULL DEFAULT 0,
                                PRIMARY KEY (dataset, owner, content_hash)
                            )
                            """
                        )
                        self._migrate_hard_delete_ledger(conn)
                        conn.commit()
                    finally:
                        conn.close()
                    self._schema_ready = True
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    # -- serialization ------------------------------------------------------

    @staticmethod
    def _memory_to_row(memory: Memory, updated_at: datetime) -> tuple[Any, ...]:
        extra = {
            "session_id": memory.session_id,
            "entity_refs": list(memory.entity_refs or []),
            "metadata": memory.metadata or {},
            "valid_from": memory.valid_from.isoformat() if memory.valid_from else None,
        }
        return (
            memory.id,
            memory.user_id,
            memory.content,
            memory.importance,
            json.dumps(extra, default=str),
            memory.created_at.isoformat() if memory.created_at else _utcnow().isoformat(),
            updated_at.isoformat(),
        )

    @staticmethod
    def _row_to_memory(row: tuple[Any, ...]) -> Memory:
        memory_id, user_id, content, importance, metadata_json, created_at, _updated_at = row
        extra = json.loads(metadata_json) if metadata_json else {}
        created = datetime.fromisoformat(created_at) if created_at else _utcnow()
        raw_valid_from = extra.get("valid_from")
        valid_from = datetime.fromisoformat(raw_valid_from) if raw_valid_from else created
        return Memory(
            id=memory_id,
            content=content,
            user_id=user_id,
            session_id=extra.get("session_id"),
            importance=0.5 if importance is None else float(importance),
            entity_refs=list(extra.get("entity_refs") or []),
            metadata=dict(extra.get("metadata") or {}),
            created_at=created,
            valid_from=valid_from,
        )

    _SELECT_COLUMNS = "id, user_id, content, importance, metadata_json, created_at, updated_at"

    # -- memory registry -----------------------------------------------------

    def upsert_memory(self, memory: Memory, clear_tombstone: bool = False) -> None:
        """Insert or update a registry row (keyed by memory ID).

        With ``clear_tombstone`` (used on explicit saves and updates), a
        tombstone for the memory's ``(user_id, content)`` is removed — saving
        content again is an explicit request to make it live.
        """
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO memories (id, user_id, content, importance, metadata_json,
                                      created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    user_id = excluded.user_id,
                    content = excluded.content,
                    importance = excluded.importance,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                self._memory_to_row(memory, _utcnow()),
            )
            if clear_tombstone:
                conn.execute(
                    "DELETE FROM tombstones WHERE user_id = ? AND content = ?",
                    (memory.user_id, memory.content),
                )
            conn.commit()
        finally:
            conn.close()

    def get_memory(self, memory_id: str) -> Memory | None:
        """Fetch a memory by canonical ID, or None."""
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT {self._SELECT_COLUMNS} FROM memories WHERE id = ?",  # noqa: S608
                (memory_id,),
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_memory(row) if row else None

    def find_by_user_content(self, user_id: str, content: str) -> Memory | None:
        """Fetch the canonical memory for ``(user_id, content)``, or None.

        This is the lookup that keeps IDs stable across updates: an updated
        row keeps its original ID but carries the new content, so a search
        hit on the new content resolves back to the original ID.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                f"SELECT {self._SELECT_COLUMNS} FROM memories "  # noqa: S608
                "WHERE user_id = ? AND content = ? "
                "ORDER BY updated_at DESC, id LIMIT 1",
                (user_id, content),
            ).fetchone()
        finally:
            conn.close()
        return self._row_to_memory(row) if row else None

    def apply_update(self, updated: Memory, tombstone_contents: list[str]) -> None:
        """Atomically apply an update: rewrite the row, tombstone old content.

        The updated memory keeps its original ID. The old content (and old
        pre-extracted facts) are tombstoned; any tombstone on the NEW content
        is cleared (updating to some content is an explicit request to make
        it live).
        """
        now_iso = _utcnow().isoformat()
        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO memories (id, user_id, content, importance, metadata_json,
                                      created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    user_id = excluded.user_id,
                    content = excluded.content,
                    importance = excluded.importance,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                self._memory_to_row(updated, _utcnow()),
            )
            conn.executemany(
                "INSERT OR IGNORE INTO tombstones (user_id, content, created_at) VALUES (?, ?, ?)",
                [(updated.user_id, content, now_iso) for content in tombstone_contents],
            )
            conn.execute(
                "DELETE FROM tombstones WHERE user_id = ? AND content = ?",
                (updated.user_id, updated.content),
            )
            conn.commit()
        finally:
            conn.close()

    def delete_and_tombstone(
        self, memory_id: str, user_id: str, tombstone_contents: list[str]
    ) -> None:
        """Atomically delete a registry row and tombstone its contents."""
        now_iso = _utcnow().isoformat()
        conn = self._connect()
        try:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.executemany(
                "INSERT OR IGNORE INTO tombstones (user_id, content, created_at) VALUES (?, ?, ?)",
                [(user_id, content, now_iso) for content in tombstone_contents],
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _migrate_hard_delete_ledger(conn: sqlite3.Connection) -> None:
        """Bring a ledger written by an earlier revision up to the current shape.

        Two earlier shapes exist. Before the pending/verified split the table
        had no ``verified`` column; before owner scoping it had no ``owner``
        column and its primary key was ``(dataset, content_hash)``. SQLite
        cannot change a primary key in place, so an owner-less table is
        rebuilt: its rows are copied under the legacy owner ``''`` as PENDING
        (never as proof — an unscoped record cannot vouch for any one owner's
        data). ``get_hard_delete_ledger`` reads such rows as pending for
        every owner, so the retry that re-verifies them is still possible
        rather than the hash being stranded as never-stored.
        """
        columns = {row[1] for row in conn.execute("PRAGMA table_info(hard_deleted)")}
        if "owner" in columns:
            return
        if "verified" not in columns:
            conn.execute("ALTER TABLE hard_deleted ADD COLUMN verified INTEGER NOT NULL DEFAULT 0")
        conn.execute("ALTER TABLE hard_deleted RENAME TO hard_deleted_legacy")
        conn.execute(
            """
            CREATE TABLE hard_deleted (
                dataset TEXT NOT NULL,
                owner TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL,
                created_at TEXT,
                verified INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (dataset, owner, content_hash)
            )
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO hard_deleted (dataset, owner, content_hash, created_at, verified) "
            "SELECT dataset, '', content_hash, created_at, 0 FROM hard_deleted_legacy"
        )
        conn.execute("DROP TABLE hard_deleted_legacy")

    # -- hard-delete ledger ---------------------------------------------------
    # Records content hashes this store has hard-deleted from a cognee
    # dataset FOR ONE OWNER (the memory's tenant partition id), in one of two states:
    #
    # - PENDING: the hash's data items were observed and their deletion
    #   begun (the row is written before the first ``delete_data``), but no
    #   re-list has yet confirmed the items are gone. A pending hash is NOT
    #   proof — a retry must re-verify it (re-deleting anything still there).
    # - VERIFIED: a re-list after deletion found no matching item. Only this
    #   state counts as proof of removal.
    #
    # "Absence of a matching data item" is deliberately not proof on its own
    # (chunk-registered rows hash differently from their source), so without
    # this ledger a hard delete that got interrupted after removing some
    # items could never be re-proven on retry — the already-removed hashes
    # would look identical to never-stored ones. The pending state is what
    # lets a retry tell "we removed this, verify it" from "never stored".

    def get_hard_delete_ledger(
        self, dataset: str, owner: str, content_hashes: set[str]
    ) -> dict[str, bool]:
        """Return ``{content_hash: verified}`` for the owner's hashes in the ledger.

        Hashes with no ledger row are absent from the result. A legacy row
        recorded before owner scoping (owner ``''``) is reported as PENDING
        for every owner: it is never proof, but it keeps the hash retryable.
        """
        if not content_hashes:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" for _ in content_hashes)
            rows = conn.execute(
                f"SELECT content_hash, owner, verified FROM hard_deleted "
                f"WHERE dataset = ? AND owner IN (?, '') AND content_hash IN ({placeholders})",
                (dataset, owner, *content_hashes),
            ).fetchall()
        finally:
            conn.close()
        ledger: dict[str, bool] = {}
        for content_hash, row_owner, verified in rows:
            if row_owner == owner:
                ledger[content_hash] = bool(verified)
            else:
                ledger.setdefault(content_hash, False)
        return ledger

    def record_hard_delete_pending(
        self, dataset: str, owner: str, content_hashes: set[str]
    ) -> None:
        """Durably record hashes whose observed data items are being deleted (not yet verified gone).

        Written before the first delete so no interruption can strand a
        removed hash without a row. An existing row (pending or verified) is
        left untouched.
        """
        if not content_hashes:
            return
        now_iso = _utcnow().isoformat()
        conn = self._connect()
        try:
            conn.executemany(
                "INSERT OR IGNORE INTO hard_deleted "
                "(dataset, owner, content_hash, created_at, verified) VALUES (?, ?, ?, ?, 0)",
                [(dataset, owner, h, now_iso) for h in content_hashes],
            )
            conn.commit()
        finally:
            conn.close()

    def mark_hard_delete_verified(self, dataset: str, owner: str, content_hashes: set[str]) -> None:
        """Promote the owner's hashes to VERIFIED: a re-list found no matching data item.

        A legacy unscoped row for the hash has served its purpose once the
        owner's verification completes and is dropped.
        """
        if not content_hashes:
            return
        now_iso = _utcnow().isoformat()
        conn = self._connect()
        try:
            conn.executemany(
                "INSERT INTO hard_deleted (dataset, owner, content_hash, created_at, verified) "
                "VALUES (?, ?, ?, ?, 1) "
                "ON CONFLICT(dataset, owner, content_hash) DO UPDATE SET verified = 1",
                [(dataset, owner, h, now_iso) for h in content_hashes],
            )
            placeholders = ",".join("?" for _ in content_hashes)
            conn.execute(
                f"DELETE FROM hard_deleted WHERE dataset = ? AND owner = '' "
                f"AND content_hash IN ({placeholders})",
                (dataset, *content_hashes),
            )
            conn.commit()
        finally:
            conn.close()

    def clear_hard_deleted(self, dataset: str, owner: str, content_hashes: set[str]) -> None:
        """Forget the owner's ledger entries (pending or verified) for the given hashes.

        Used when the owner re-adds content to cognee, and when a verification
        re-list finds a hash's data items still present. Legacy unscoped rows
        go too: the owner's own listing just showed the truth for these hashes.
        """
        if not content_hashes:
            return
        conn = self._connect()
        try:
            placeholders = ",".join("?" for _ in content_hashes)
            conn.execute(
                f"DELETE FROM hard_deleted WHERE dataset = ? AND owner IN (?, '') "
                f"AND content_hash IN ({placeholders})",
                (dataset, owner, *content_hashes),
            )
            conn.commit()
        finally:
            conn.close()

    # -- tombstones ----------------------------------------------------------

    def add_tombstones(self, user_id: str, tombstone_contents: list[str]) -> None:
        """Tombstone contents WITHOUT deleting any registry row.

        Used when a fail-closed delete cannot prove cognee removed the
        underlying data: the content is suppressed from every text-matched
        result while the memory stays in the registry for a retry.
        """
        now_iso = _utcnow().isoformat()
        conn = self._connect()
        try:
            conn.executemany(
                "INSERT OR IGNORE INTO tombstones (user_id, content, created_at) VALUES (?, ?, ?)",
                [(user_id, content, now_iso) for content in tombstone_contents],
            )
            conn.commit()
        finally:
            conn.close()

    def get_tombstones(self, user_id: str) -> set[str]:
        """Return all tombstoned contents for a user."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT content FROM tombstones WHERE user_id = ?", (user_id,)
            ).fetchall()
        finally:
            conn.close()
        return {row[0] for row in rows}

    def resolve_search_results(
        self,
        user_id: str,
        session_id: str | None,
        items: list[tuple[str, dict[str, Any]]],
        top_k: int,
        now: datetime,
    ) -> tuple[list[Memory], int]:
        """Filter search hits against tombstones and resolve them to registry rows — atomically.

        Returns ``(memories, visible_count)``: the first ``top_k`` visible hits
        resolved to canonical ``Memory`` rows, and how many hits survived the
        tombstone filter in total (for rank-based scoring).

        The tombstone read, the row lookup and the insert of unmatched hits
        happen in ONE write transaction (``BEGIN IMMEDIATE``), so a concurrent
        ``delete_memory`` / ``update_memory`` — which tombstones in its own
        transaction — either commits before this one starts (its tombstone is
        seen and the hit dropped) or after it ends (the hit was legitimately
        live when this search ran). A delete can never slip between the filter
        and the insert and have the search re-create the row it just removed.

        A hit whose content matches a stored row for this user resolves to that
        row's canonical ID (how an updated memory keeps its original ID);
        unmatched hits get stable content-derived IDs and are inserted so a
        later update/delete round-trips durably.
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            tombstones = {
                row[0]
                for row in conn.execute(
                    "SELECT content FROM tombstones WHERE user_id = ?", (user_id,)
                ).fetchall()
            }
            visible = [
                (text, res_meta) for text, res_meta in items if not _is_tombstoned(text, tombstones)
            ]
            resolved: list[Memory] = []
            for text, res_meta in visible[:top_k]:
                row = conn.execute(
                    f"SELECT {self._SELECT_COLUMNS} FROM memories "  # noqa: S608
                    "WHERE user_id = ? AND content = ? "
                    "ORDER BY updated_at DESC, id LIMIT 1",
                    (user_id, text),
                ).fetchone()
                if row is not None:
                    resolved.append(self._row_to_memory(row))
                    continue
                memory = Memory(
                    id=_stable_memory_id(user_id, text),
                    content=text,
                    user_id=user_id,
                    session_id=session_id,
                    importance=0.5,
                    metadata=res_meta,
                    created_at=now,
                    valid_from=now,
                )
                conn.execute(
                    """
                    INSERT INTO memories (id, user_id, content, importance, metadata_json,
                                          created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        user_id = excluded.user_id,
                        content = excluded.content,
                        importance = excluded.importance,
                        metadata_json = excluded.metadata_json,
                        updated_at = excluded.updated_at
                    """,
                    self._memory_to_row(memory, now),
                )
                resolved.append(memory)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return resolved, len(visible)


class CogneeBackend:
    """Memory backend backed by the cognee knowledge-graph engine.

    Implements headroom's ``MemoryBackend`` protocol on top of cognee:

    - ``save_memory`` -> ``cognee.add`` (tagged via ``node_set``) followed by
      an optional ``cognee.cognify`` to build/extend the knowledge graph.
    - ``search_memories`` -> ``cognee.search`` scoped via ``node_name``.
    - ``update_memory`` / ``delete_memory`` -> durable registry update +
      tombstones in the SQLite metadata store, plus a best-effort hard delete
      against cognee's dataset API (see module docstring for limitations).

    The cognee package is imported lazily on first use (once per process —
    see module docstring); construction never imports cognee.
    """

    def __init__(self, config: CogneeConfig | None = None) -> None:
        """Initialize the cognee backend.

        Args:
            config: Backend configuration. Defaults resolve from
                ``HEADROOM_COGNEE_*`` env vars when omitted.
        """
        self._config = config or CogneeConfig()
        self._cognee: Any = None
        self._search_type_cls: Any = None
        self._initialized = False
        # Durable metadata store: memory registry + tombstones. Shared
        # across instances/restarts that point at the same file, so deletes
        # and updates are visible everywhere immediately. Construction is
        # cheap; the schema is created lazily on first use.
        self._store = _CogneeMetadataStore(_resolve_metadata_db_path(self._config))

    async def _ensure_initialized(self) -> None:
        """Import cognee lazily (off-loop, once per process) and configure it.

        The import runs in a worker thread via ``asyncio.to_thread`` because
        ``import cognee`` takes seconds and would otherwise stall the entire
        event loop (every in-flight proxy request, not just the memory one)
        when initialization happens lazily inside a live request. The actual
        import/configuration is serialized process-wide by a module-level
        threading lock and happens at most once per process; see
        ``_import_and_configure_cognee``.

        Raises:
            RuntimeError: If cognee was already configured in this process
                with different root directories.
        """
        if self._initialized:
            return

        cognee, search_type_cls = await asyncio.to_thread(
            _import_and_configure_cognee, self._config.system_root, self._config.data_root
        )
        self._cognee = cognee
        self._search_type_cls = search_type_cls
        self._initialized = True

    async def ensure_initialized(self) -> None:
        """Public initialization hook for callers that need readiness guarantees."""
        await self._ensure_initialized()

    def _resolve_search_type(self) -> Any:
        """Resolve the configured search type name to a cognee ``SearchType``.

        Returns:
            The cognee SearchType enum member.

        Raises:
            ValueError: If the configured name is not a valid SearchType.
        """
        name = self._config.search_type.strip().upper()
        try:
            return self._search_type_cls[name]
        except KeyError:
            valid = ", ".join(m.name for m in self._search_type_cls)
            raise ValueError(
                f"Invalid cognee search type {self._config.search_type!r}; expected one of: {valid}"
            ) from None

    @staticmethod
    def _build_node_set(
        user_id: str,
        session_id: str | None = None,
        entities: list[str] | None = None,
    ) -> list[str]:
        """Build the node_set tags used to scope data in cognee."""
        tags = [_user_tag(user_id)]
        if session_id:
            tags.append(_session_tag(session_id))
        for entity in entities or []:
            tags.append(_entity_tag(entity))
        return tags

    async def _cognify(self) -> None:
        """Run cognee.cognify for the configured dataset (best-effort)."""
        try:
            await self._cognee.cognify(
                datasets=[self._config.dataset_name],
                run_in_background=self._config.background_cognify,
            )
        except Exception:
            # cognify is an enrichment step; a failure must not lose the save.
            logger.exception("cognee.cognify failed for dataset %s", self._config.dataset_name)

    async def save_memory(
        self,
        content: str,
        user_id: str,
        importance: float,
        entities: list[str] | None = None,
        relationships: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        # Pre-extraction fields for optimized storage
        facts: list[str] | None = None,
        extracted_entities: list[dict[str, str]] | None = None,
        extracted_relationships: list[dict[str, str]] | None = None,
    ) -> Memory:
        """Save a new memory to cognee.

        The content (plus pre-extracted facts, when provided) is added to the
        configured cognee dataset, tagged via ``node_set`` with user/session/
        entity tags so searches can be scoped. When ``auto_cognify`` is on,
        ``cognee.cognify`` then builds/extends the knowledge graph (in the
        background by default). The memory is also recorded in the durable
        metadata store so its ID resolves across restarts and instances.

        Note: cognee performs its own LLM-based entity/relationship extraction
        during ``cognify``, so ``relationships``, ``extracted_entities``, and
        ``extracted_relationships`` are recorded in the returned Memory's
        metadata but not written to the graph directly.

        Args:
            content: The memory content to store.
            user_id: User identifier for scoping.
            importance: Importance score (0.0 - 1.0). Stored in metadata only;
                cognee does not rank by importance.
            entities: List of entity references (become node_set tags).
            relationships: Relationship dicts (recorded in metadata only).
            session_id: Optional session identifier (becomes a node_set tag).
            metadata: Optional additional metadata.
            facts: Pre-extracted discrete facts, added as extra data items.
            extracted_entities: Pre-extracted entities (metadata only).
            extracted_relationships: Pre-extracted relationships (metadata only).

        Returns:
            The created Memory object.
        """
        await self._ensure_initialized()

        node_set = self._build_node_set(user_id, session_id, entities)
        data: str | list[str] = content if not facts else [content, *facts]

        await self._cognee.add(
            data,
            dataset_name=self._config.dataset_name,
            node_set=node_set,
        )

        if self._config.auto_cognify:
            await self._cognify()

        now = _utcnow()
        combined_metadata: dict[str, Any] = {
            **(metadata or {}),
            "_cognee_dataset": self._config.dataset_name,
            "_cognee_node_set": node_set,
        }
        if relationships:
            combined_metadata["relationships"] = relationships
        if extracted_entities:
            combined_metadata["extracted_entities"] = extracted_entities
        if extracted_relationships:
            combined_metadata["extracted_relationships"] = extracted_relationships
        if facts:
            combined_metadata["_fact_count"] = len(facts)
            # Kept so update/delete can tombstone the facts alongside the
            # main content (facts were added to cognee as separate items).
            combined_metadata["_cognee_facts"] = list(facts)

        memory = Memory(
            id=_stable_memory_id(user_id, content),
            content=content,
            user_id=user_id,
            session_id=session_id,
            importance=importance,
            entity_refs=entities or [],
            metadata=combined_metadata,
            created_at=now,
            valid_from=now,
        )
        # clear_tombstone: re-saving previously deleted content is an
        # explicit request to make it live again. The hard-delete ledger is
        # cleared for the re-added contents too — they exist in cognee again,
        # so an old "provably removed" record must not vouch for them.
        await asyncio.to_thread(self._store.upsert_memory, memory, clear_tombstone=True)
        await asyncio.to_thread(
            self._store.clear_hard_deleted,
            self._config.dataset_name,
            user_id or "",
            {
                self._content_hash(item)
                for item in [content, *(facts or [])]
                if isinstance(item, str) and item
            },
        )
        logger.info("Saved memory %s to cognee dataset %s", memory.id, self._config.dataset_name)
        return memory

    async def search_memories(
        self,
        query: str,
        user_id: str,
        entities: list[str] | None = None,
        include_related: bool = False,
        top_k: int = 10,
        session_id: str | None = None,
    ) -> list[MemorySearchResult]:
        """Search memories via cognee.

        Uses the configured ``SearchType`` (default ``CHUNKS``: raw retrieval,
        no LLM synthesis). Scoping is done with cognee's ``node_name`` filter
        against the tags written at save time, combined with ``AND`` so every
        listed tag must match — the user tag always applies, and session /
        entity filters narrow (never broaden) the result set. Without ``AND``
        cognee defaults to ``OR``, which would match ANY tag and leak other
        users' memories that share an entity tag.

        Results are filtered against the durable per-user tombstones (deleted
        or superseded content, exact or substring match) and resolved to
        canonical memory IDs through the durable registry, so deletes/updates
        made by other instances or before a restart are honored.

        Args:
            query: Natural language search query.
            user_id: User identifier for scoping.
            entities: Filter to memories tagged with these entities.
            include_related: Accepted for protocol compatibility; graph
                expansion is controlled by ``CogneeConfig.search_type``
                (e.g. ``GRAPH_COMPLETION``) instead.
            top_k: Maximum number of results.
            session_id: Optional session filter.

        Returns:
            List of MemorySearchResult in relevance order. cognee does not
            expose similarity scores, so scores are rank-based and mapped
            into ``(0.5, 1.0]`` — they encode result order only and always
            clear the proxy's default ``min_similarity`` floor.
        """
        await self._ensure_initialized()

        query_type = self._resolve_search_type()
        node_names = self._build_node_set(user_id, session_id, entities)

        search_kwargs: dict[str, Any] = {
            "query_text": query,
            "query_type": query_type,
            "datasets": [self._config.dataset_name],
            "top_k": top_k,
            "node_name": node_names,
            # Require ALL tags to match (cognee>=1.4.0). The default "OR"
            # would return anything matching any single tag — e.g. other
            # users' memories tagged with the same (global) entity tag.
            "node_name_filter_operator": "AND",
        }
        # Graph retrieval without LLM answer synthesis.
        if query_type.name == "GRAPH_COMPLETION":
            search_kwargs["only_context"] = True

        # Tombstoned hits are dropped after cognee ranked them, so a request
        # for ``top_k`` is widened by the partition's tombstone count (this
        # read only sizes the fetch; the authoritative filter runs inside the
        # resolution transaction below) and re-issued with a doubled limit
        # while a full page still yields fewer than ``top_k`` visible hits.
        sizing_tombstones = await asyncio.to_thread(self._store.get_tombstones, user_id)
        fetch_k = top_k + min(len(sizing_tombstones), _SEARCH_OVERFETCH_CAP)
        texts: list[tuple[str, dict[str, Any]]] = []
        for _round in range(_SEARCH_FETCH_ROUNDS):
            search_kwargs["top_k"] = fetch_k
            try:
                raw_results = await self._cognee.search(**search_kwargs)
            except Exception as error:
                # A dataset that exists but was never cognified has no vector
                # collections yet; cognee raises NoDataError instead of
                # returning nothing. An empty store is an empty result, not a
                # failure.
                if type(error).__name__ == "NoDataError":
                    logger.info(
                        "cognee dataset %s has no searchable data yet; returning no results",
                        self._config.dataset_name,
                    )
                    return []
                raise
            texts = self._collect_result_texts(raw_results)
            visible_estimate = sum(
                1 for text, _ in texts if not _is_tombstoned(text, sizing_tombstones)
            )
            page_full = len(texts) >= fetch_k
            if visible_estimate >= top_k or not page_full or fetch_k >= _SEARCH_FETCH_HARD_CAP:
                break
            fetch_k = min(fetch_k * 2, _SEARCH_FETCH_HARD_CAP)

        # Tombstone-filter BEFORE ranking so surviving results keep top
        # ranks (and therefore high scores) when leading chunks were
        # deleted/superseded, and resolve canonical IDs through the durable
        # registry — in one transaction, so a delete committing meanwhile
        # cannot have this search re-insert the row it just removed.
        # Tombstones come from the durable store so deletions made by other
        # instances / before a restart apply. The filter runs on every
        # configured search type (text-match based — see module docstring
        # for graph-synthesized caveats).
        memories, total = await asyncio.to_thread(
            self._store.resolve_search_results, user_id, session_id, texts, top_k, _utcnow()
        )

        results: list[MemorySearchResult] = []
        for rank, memory in enumerate(memories):
            results.append(
                MemorySearchResult(
                    memory=memory,
                    # Rank-based scores compressed into (0.5, 1.0] so an
                    # ordinal score can never be filtered out by the
                    # proxy's default cosine min_similarity floor (0.3).
                    score=1.0 - (rank / (2 * max(total, 1))),
                    related_entities=entities or [],
                    related_memories=[],
                )
            )

        return results

    def _collect_result_texts(self, raw_results: Any) -> list[tuple[str, dict[str, Any]]]:
        """Flatten cognee's search return into ``(text, result_metadata)`` pairs."""
        texts: list[tuple[str, dict[str, Any]]] = []
        for res in raw_results or []:
            # cognee's search() return shape depends on backend access control
            # (on by default with the embedded LanceDB/Kuzu stores, any 1.x):
            # per-dataset dict envelopes of {dataset_id, dataset_name,
            # search_result} when on, bare result payloads when off. Attribute
            # access is kept for object-shaped results (older clients/fakes).
            if isinstance(res, dict):
                payload = res.get("search_result", res)
                dataset_id = res.get("dataset_id", "")
                dataset_name = res.get("dataset_name")
            else:
                payload = getattr(res, "search_result", res)
                dataset_id = getattr(res, "dataset_id", "")
                dataset_name = getattr(res, "dataset_name", None)
            res_meta = {
                "_cognee_dataset_id": str(dataset_id or ""),
                "_cognee_dataset_name": dataset_name or self._config.dataset_name,
            }
            items = payload if isinstance(payload, list) else [payload]
            for item in items:
                text = self._extract_text(item)
                if text:
                    texts.append((text, res_meta))
        return texts

    @staticmethod
    def _extract_text(item: Any) -> str:
        """Extract display text from a single cognee search result item."""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ("text", "chunk", "content", "memory", "name"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    return value
            return str(item)
        return str(item) if item is not None else ""

    def _tombstones_fully_enforce(self) -> bool:
        """Whether tombstones are authoritative for the configured search type.

        ``CHUNKS`` returns stored text verbatim (whole contents or fragments
        of them), and ``_is_tombstoned`` matches both directions — so nothing
        derived from a tombstoned memory can pass the filter. Every other
        search type may synthesize text that no longer contains the deleted
        source, which a text-matched tombstone cannot catch.
        """
        return self._config.search_type.strip().upper() == "CHUNKS"

    @staticmethod
    def _content_hash(content: str) -> str:
        """cognee's data-item content hash (MD5 of the text)."""
        return hashlib.md5(content.encode("utf-8"), usedforsecurity=False).hexdigest()

    @staticmethod
    def _item_node_set(data_item: Any) -> set[str] | None:
        """The node-set tags a cognee data item was stored under, or None if unknown.

        cognee keeps them on ``Data.node_set`` (a JSON-encoded list on ORM
        rows, a list on API rows since cognee exposes ``nodeSet``) and mirrors
        them in ``external_metadata["node_set"]``; both spellings are read.
        The ``user:<partition>`` tag among them is the tenant boundary.
        """
        raw = getattr(data_item, "node_set", None)
        if raw is None:
            external = getattr(data_item, "external_metadata", None)
            if isinstance(external, str):
                try:
                    external = json.loads(external)
                except ValueError:
                    external = None
            if isinstance(external, dict):
                raw = external.get("node_set")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except ValueError:
                return None
        if isinstance(raw, list | tuple | set):
            return {str(tag) for tag in raw}
        return None

    async def _list_matching_items(
        self, datasets_api: Any, wanted: set[str], owner_tag: str
    ) -> list[tuple[Any, Any, set[str]]]:
        """List (dataset_id, data_id, matching_hashes) without mutating anything.

        Only data items tagged with ``owner_tag`` (the ``user:<partition>``
        node-set tag of the memory's tenant partition) count. Two tenants who
        stored the same text hold the same content hash in one dataset, and
        cognee keeps one data item per node set — so the tag, not the hash,
        is the tenant boundary. An item whose tags cannot be read belongs to
        no tenant and is never matched.
        """
        found: list[tuple[Any, Any, set[str]]] = []
        for dataset in await datasets_api.list_datasets() or []:
            if getattr(dataset, "name", None) != self._config.dataset_name:
                continue
            for data_item in await datasets_api.list_data(dataset.id) or []:
                tags = self._item_node_set(data_item)
                if not tags or owner_tag not in tags:
                    continue
                item_hashes = {
                    getattr(data_item, "content_hash", None),
                    getattr(data_item, "raw_content_hash", None),
                }
                overlap = {h for h in wanted if h in item_hashes}
                if overlap:
                    found.append((dataset.id, data_item.id, overlap))
        return found

    async def _try_hard_delete(self, contents: list[str], owner_id: str) -> bool:
        """Hard-delete ONE TENANT's data items from cognee's stores; report whether proven.

        ``owner_id`` is the memory's partition id — the tenant identity the
        proxy resolved for the request, composed with the project key in
        project mode; what this backend receives as ``user_id``. cognee
        identifies text data items by an MD5 content hash and keeps one item
        per node set, so discovery, deletion, verification and the ledger are
        all scoped to the tenant's ``user:<partition>`` tag: another tenant's
        item holding the same text is never listed, never deleted, and never
        vouched for. Removal is PROVEN only when every content's hash is
        accounted for within that scope. The durable ``hard_deleted`` ledger
        tracks each (owner, hash) as PENDING (its items were observed and
        their deletion begun, not yet verified gone) or VERIFIED (a re-list
        found none). Only VERIFIED counts as proof. Three phases, so a
        failure at any point leaves the operation retryable:

        1. Discovery (NON-MUTATING): map every unverified hash to its stored
           data items. A hash with no ledger row that matches no item makes
           the whole call return False WITHOUT deleting anything — "nothing
           matched" is not proof (a chunk-registered row hashes differently
           from its source item), and deleting the matched subset first would
           make the unmatched remainder permanently unprovable on retry. A
           PENDING hash that matches no item is different: an earlier call
           observed its items and set out to delete them, so absence
           completes its verification and the hash is promoted to VERIFIED
           right away (progress is preserved even if the rest of this call
           fails). A PENDING hash that still matches items is deleted again.
        2. Deletion: durably record EVERY matched hash as PENDING first, in
           one write, then delete the items. Recording the intent before the
           first ``delete_data`` means no interruption — between items, after
           an item that carries several hashes, or right after a delete —
           can strand a removed hash without a ledger row (which would make
           it unprovable on retry), and a PENDING row never passes as proof.
           A recorded hash whose delete then fails is harmless: its item is
           still observable, so the retry re-matches and deletes it.
        3. Verification: re-list. Hashes with no matching item left are
           promoted to VERIFIED; hashes whose items remain have their ledger
           entries cleared (the items are observable, so a retry rediscovers
           them). If the re-list itself fails, the PENDING entries stay
           pending, and the next call resumes at verification instead of
           reporting success.

        Failures are logged, never raised; callers decide whether an unproven
        removal is acceptable (``CHUNKS`` tombstone enforcement) or must fail
        closed (synthesized modes).
        """
        try:
            await self._ensure_initialized()
        except Exception:
            logger.debug(
                "cognee unavailable for hard delete; durable tombstones still filter the content",
                exc_info=True,
            )
            return False

        datasets_api = getattr(self._cognee, "datasets", None)
        if datasets_api is None or not hasattr(datasets_api, "list_datasets"):
            return False

        dataset_name = self._config.dataset_name
        owner = owner_id or ""
        owner_tag = _user_tag(owner_id)
        try:
            hashes = {self._content_hash(content) for content in contents}
            ledger = await asyncio.to_thread(
                self._store.get_hard_delete_ledger, dataset_name, owner, hashes
            )
            verified = {h for h, is_verified in ledger.items() if is_verified}
            pending = {h for h, is_verified in ledger.items() if not is_verified}
            remaining = hashes - verified
            if not remaining:
                return True

            # Phase 1 — discovery, non-mutating.
            found = await self._list_matching_items(datasets_api, remaining, owner_tag)
            matched: set[str] = set()
            for _, _, overlap in found:
                matched |= overlap
            unknown = remaining - pending  # no ledger row: must match a stored item
            if not unknown <= matched:
                logger.info(
                    "Hard delete unproven: %d of %d contents have no matching "
                    "cognee data item; nothing was deleted",
                    len(unknown - matched),
                    len(hashes),
                )
                return False

            # A pending hash with no item left was deleted by an earlier call
            # whose verification never completed; this listing completes it.
            resumed = pending - matched
            if resumed:
                await asyncio.to_thread(
                    self._store.mark_hard_delete_verified, dataset_name, owner, resumed
                )
                logger.info(
                    "Resumed verification for %d previously deleted cognee data item hash(es)",
                    len(resumed),
                )
            if not matched:
                return True

            # Phase 2 — record the intent for every matched hash, then delete
            # the items. The ledger row precedes the first delete so an item
            # carrying several hashes, or an interruption anywhere in the
            # loop, can never leave a removed hash without a row.
            await asyncio.to_thread(
                self._store.record_hard_delete_pending, dataset_name, owner, matched
            )
            deleted_ids: set[Any] = set()
            for dataset_id, data_id, _overlap in found:
                if data_id in deleted_ids:
                    continue
                await datasets_api.delete_data(dataset_id=dataset_id, data_id=data_id)
                deleted_ids.add(data_id)
                logger.info(
                    "Hard-deleted cognee data item %s from dataset %s",
                    data_id,
                    dataset_name,
                )

            # Phase 3 — verification: nothing matching may remain. Hashes
            # verified gone are promoted; hashes still present are cleared
            # from the ledger (their items are observable, so a retry
            # rediscovers and deletes them). If this re-list raises, every
            # hash deleted above stays PENDING and the next call re-verifies.
            still_present: set[str] = set()
            for _, _, overlap in await self._list_matching_items(datasets_api, matched, owner_tag):
                still_present |= overlap
            gone = matched - still_present
            if gone:
                await asyncio.to_thread(
                    self._store.mark_hard_delete_verified, dataset_name, owner, gone
                )
            if still_present:
                logger.warning(
                    "Hard delete unproven: data items for %d of %d contents remain after deletion",
                    len(still_present),
                    len(hashes),
                )
                await asyncio.to_thread(
                    self._store.clear_hard_deleted, dataset_name, owner, still_present
                )
                return False
            return True
        except Exception:
            logger.warning(
                "cognee hard delete failed; durable tombstones still filter the content",
                exc_info=True,
            )
            return False

    async def update_memory(
        self,
        memory_id: str,
        new_content: str,
        reason: str | None = None,
        user_id: str | None = None,
    ) -> Memory:
        """Update a memory, keeping its original ID.

        cognee has no per-item update API, so this rewrites the durable
        registry row in place (new content, SAME id), tombstones the old
        content (excluded from every instance's future search results for
        this user), and adds the new content as a fresh cognee data item with
        the same scoping tags. Because search resolves canonical IDs by
        ``(user_id, content)``, the next search returns this same memory ID
        for the new content.

        The old data's removal follows the same contract as
        ``delete_memory``: under ``CHUNKS`` the tombstone is authoritative and
        the hard delete of the old cognee data is best-effort; under any
        other search type the old data's removal must be PROVEN first, and an
        unproven removal raises ``CogneeDeletionUnverifiedError`` BEFORE
        anything is mutated (no new content is added, the row is unchanged).

        Args:
            memory_id: ID of the memory to update. Must exist in the durable
                registry (i.e. was saved or surfaced by a search through a
                backend sharing this metadata store).
            new_content: New content to replace existing.
            reason: Reason for the update (for audit trail).
            user_id: User ID for validation (optional).

        Returns:
            The updated Memory object (same ID, new content).

        Raises:
            ValueError: If the memory is not found or belongs to another user.
            CogneeDeletionUnverifiedError: Non-``CHUNKS`` search type and the
                old data's removal could not be proven.
        """
        await self._ensure_initialized()

        existing = await asyncio.to_thread(self._store.get_memory, memory_id)
        if existing is None:
            raise ValueError(
                f"Memory not found: {memory_id}. The cognee backend can only "
                "update memories recorded in its metadata store (saved or "
                "returned by a search)."
            )
        if user_id and existing.user_id and existing.user_id != user_id:
            raise ValueError("Cannot update memories belonging to other users")

        if new_content == existing.content:
            # Nothing to replace: the stored cognee data item IS the new
            # content, and tombstoning or reclaiming it would hide the
            # memory from its own user's searches. Record the update on the
            # row and keep the facts, which still describe this content.
            return await self._record_same_content_update(existing, reason)

        old_contents = [existing.content]
        for fact in existing.metadata.get("_cognee_facts") or []:
            if isinstance(fact, str) and fact:
                old_contents.append(fact)
        # A fact that happens to equal the new content stays stored for it
        # too: never tombstone or reclaim what the update is putting in place.
        new_hash = self._content_hash(new_content)
        old_contents = [c for c in old_contents if self._content_hash(c) != new_hash]

        if not self._tombstones_fully_enforce() and not await self._try_hard_delete(
            old_contents, existing.user_id
        ):
            raise CogneeDeletionUnverifiedError(
                f"Cannot verify cognee removed the old data behind memory "
                f"{memory_id}; refusing to update under search type "
                f"{self._config.search_type!r}, whose synthesized results "
                "tombstones cannot fully enforce. Nothing was changed; "
                "use search_type=CHUNKS for tombstone-enforceable updates."
            )

        node_set = list(existing.metadata.get("_cognee_node_set") or []) or self._build_node_set(
            existing.user_id, existing.session_id, existing.entity_refs
        )
        await self._cognee.add(
            new_content,
            dataset_name=self._config.dataset_name,
            node_set=node_set,
        )
        if self._config.auto_cognify:
            await self._cognify()

        now = _utcnow()
        updated_metadata = dict(existing.metadata)
        # The old memory's facts were tombstoned above and do not describe
        # the new content — drop them so a later delete of the updated
        # memory doesn't act on stale fact lists.
        updated_metadata.pop("_cognee_facts", None)
        updated_metadata.pop("_fact_count", None)
        if reason:
            updated_metadata["update_reason"] = reason
            updated_metadata["updated_at"] = now.isoformat()

        updated = Memory(
            id=memory_id,
            content=new_content,
            user_id=existing.user_id,
            session_id=existing.session_id,
            importance=existing.importance,
            entity_refs=existing.entity_refs,
            metadata=updated_metadata,
            created_at=existing.created_at,
            valid_from=now,
        )
        # Row rewritten in place (same id) + old content tombstoned, in one
        # transaction against the shared durable store.
        await asyncio.to_thread(self._store.apply_update, updated, old_contents)
        # The new content exists in cognee again; a stale ledger record for
        # identical past content must not vouch for its removal.
        await asyncio.to_thread(
            self._store.clear_hard_deleted,
            self._config.dataset_name,
            existing.user_id or "",
            {new_hash},
        )
        if self._tombstones_fully_enforce():
            # Best-effort reclaim; under non-CHUNKS the old data was already
            # verifiably removed before any mutation.
            await self._try_hard_delete(old_contents, existing.user_id)
        logger.info("Updated memory %s in place (old content tombstoned)", memory_id)
        return updated

    async def _record_same_content_update(self, existing: Memory, reason: str | None) -> Memory:
        """``update_memory`` with unchanged content: a metadata-only rewrite.

        No cognee mutation (the data item already holds this content), no
        tombstone (the content is live), no fact drop (they describe it).
        Re-saving through ``upsert_memory`` also clears any tombstone an
        earlier delete left for this text, so the memory is searchable again.
        """
        now = _utcnow()
        metadata = dict(existing.metadata)
        if reason:
            metadata["update_reason"] = reason
            metadata["updated_at"] = now.isoformat()
        updated = Memory(
            id=existing.id,
            content=existing.content,
            user_id=existing.user_id,
            session_id=existing.session_id,
            importance=existing.importance,
            entity_refs=existing.entity_refs,
            metadata=metadata,
            created_at=existing.created_at,
            valid_from=now,
        )
        await asyncio.to_thread(self._store.upsert_memory, updated, clear_tombstone=True)
        logger.info("Updated memory %s in place (content unchanged)", existing.id)
        return updated

    async def delete_memory(
        self,
        memory_id: str,
        reason: str | None = None,
        user_id: str | None = None,
    ) -> bool:
        """Delete a memory. The success contract depends on the search type.

        Under ``CHUNKS`` (the default), results are verbatim stored text, so
        the durable tombstone IS the enforcement layer: the memory is removed
        from the registry, its content (and pre-extracted facts) is
        tombstoned — surviving restarts, visible to every instance sharing
        this metadata store — and a hard delete against cognee's dataset API
        additionally reclaims the underlying data when possible. Returns True
        even when the hard delete cannot be proven, because no result derived
        from the tombstoned text can pass the read-path filter.

        Under every other search type, results may be graph-synthesized text
        that no longer contains the deleted source, which tombstones cannot
        enforce. There, success requires the hard delete to be PROVEN
        (matched, deleted, and verified gone — see ``_try_hard_delete``).
        When it cannot be proven, the content is still tombstoned as defense
        in depth, the memory stays in the registry for a retry, and
        ``CogneeDeletionUnverifiedError`` is raised: this backend never
        reports a deletion it cannot stand behind.

        Args:
            memory_id: ID of the memory to delete. Must exist in the durable
                registry (saved or surfaced by a search).
            reason: Reason for deletion (for audit trail; logged only).
            user_id: User ID for validation (optional).

        Returns:
            True if deleted, False if not found or owned by another user.

        Raises:
            CogneeDeletionUnverifiedError: Non-``CHUNKS`` search type and the
                underlying removal could not be proven.
        """
        existing = await asyncio.to_thread(self._store.get_memory, memory_id)
        if existing is None:
            return False
        if user_id and existing.user_id and existing.user_id != user_id:
            return False

        contents = [existing.content]
        for fact in existing.metadata.get("_cognee_facts") or []:
            if isinstance(fact, str) and fact:
                contents.append(fact)

        if not self._tombstones_fully_enforce():
            if not await self._try_hard_delete(contents, existing.user_id):
                # Defense in depth: suppress text-matched surfacing, keep the
                # registry row so the caller can retry, and refuse to report
                # a deletion that synthesized results could contradict.
                await asyncio.to_thread(self._store.add_tombstones, existing.user_id, contents)
                raise CogneeDeletionUnverifiedError(
                    f"Cannot verify cognee removed the data behind memory "
                    f"{memory_id}: search type {self._config.search_type!r} "
                    "returns synthesized text that tombstones cannot fully "
                    "enforce, so the deletion is not reported as successful. "
                    "The content is tombstoned and the memory kept for retry; "
                    "use search_type=CHUNKS for tombstone-enforceable deletes."
                )
            await asyncio.to_thread(
                self._store.delete_and_tombstone, memory_id, existing.user_id, contents
            )
            logger.info(
                "Deleted memory %s (reason: %s); hard delete verified",
                memory_id,
                reason or "unspecified",
            )
            return True

        await asyncio.to_thread(
            self._store.delete_and_tombstone, memory_id, existing.user_id, contents
        )
        await self._try_hard_delete(contents, existing.user_id)
        logger.info(
            "Deleted memory %s (reason: %s); durable tombstone recorded",
            memory_id,
            reason or "unspecified",
        )
        return True

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Retrieve a specific memory by ID from the durable registry.

        Resolves any memory saved or surfaced by a search through a backend
        sharing this metadata store (cognee itself has no fetch-by-id API
        for raw memories).

        Args:
            memory_id: The memory identifier.

        Returns:
            The Memory if found, None otherwise.
        """
        return await asyncio.to_thread(self._store.get_memory, memory_id)

    @property
    def supports_graph(self) -> bool:
        """Whether this backend supports graph/relationship queries."""
        return True

    @property
    def supports_vector_search(self) -> bool:
        """Whether this backend supports vector similarity search."""
        return True

    async def close(self) -> None:
        """Close the backend and release resources.

        Only clears instance references; the process-wide cognee module and
        configuration are immutable (see module docstring), and the metadata
        store opens connections per-operation, so there is nothing to close.
        """
        self._cognee = None
        self._search_type_cls = None
        self._initialized = False
