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
      and completed removals are recorded in a durable ``hard_deleted``
      ledger, so a failed or interrupted attempt is always retryable. This
      backend never reports a deletion it cannot stand behind.

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
                                content_hash TEXT NOT NULL,
                                created_at TEXT,
                                PRIMARY KEY (dataset, content_hash)
                            )
                            """
                        )
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

    # -- hard-delete ledger ---------------------------------------------------
    # Records content hashes this store has PROVABLY hard-deleted from a
    # cognee dataset. "Absence of a matching data item" is deliberately not
    # proof of removal (chunk-registered rows hash differently from their
    # source), so without this ledger a hard delete that got interrupted
    # after removing some items could never be re-proven on retry — the
    # already-removed hashes would look identical to never-stored ones.

    def get_hard_deleted(self, dataset: str, content_hashes: set[str]) -> set[str]:
        """Return the subset of hashes already provably hard-deleted."""
        if not content_hashes:
            return set()
        conn = self._connect()
        try:
            placeholders = ",".join("?" for _ in content_hashes)
            rows = conn.execute(
                f"SELECT content_hash FROM hard_deleted "
                f"WHERE dataset = ? AND content_hash IN ({placeholders})",
                (dataset, *content_hashes),
            ).fetchall()
        finally:
            conn.close()
        return {row[0] for row in rows}

    def record_hard_deleted(self, dataset: str, content_hashes: set[str]) -> None:
        """Durably record hashes whose data items were deleted from cognee."""
        if not content_hashes:
            return
        now_iso = _utcnow().isoformat()
        conn = self._connect()
        try:
            conn.executemany(
                "INSERT OR IGNORE INTO hard_deleted (dataset, content_hash, created_at) "
                "VALUES (?, ?, ?)",
                [(dataset, h, now_iso) for h in content_hashes],
            )
            conn.commit()
        finally:
            conn.close()

    def clear_hard_deleted(self, dataset: str, content_hashes: set[str]) -> None:
        """Forget ledger entries for content that was re-added to cognee."""
        if not content_hashes:
            return
        conn = self._connect()
        try:
            placeholders = ",".join("?" for _ in content_hashes)
            conn.execute(
                f"DELETE FROM hard_deleted WHERE dataset = ? AND content_hash IN ({placeholders})",
                (dataset, *content_hashes),
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

        try:
            raw_results = await self._cognee.search(**search_kwargs)
        except Exception as error:
            # A dataset that exists but was never cognified has no vector
            # collections yet; cognee raises NoDataError instead of returning
            # nothing. An empty store is an empty result, not a failure.
            if type(error).__name__ == "NoDataError":
                logger.info(
                    "cognee dataset %s has no searchable data yet; returning no results",
                    self._config.dataset_name,
                )
                return []
            raise

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

        # Tombstone-filter BEFORE ranking so surviving results keep top
        # ranks (and therefore high scores) when leading chunks were
        # deleted/superseded. Tombstones are read from the durable store so
        # deletions made by other instances / before a restart apply. The
        # filter runs on every configured search type (text-match based —
        # see module docstring for graph-synthesized caveats).
        user_tombstones = await asyncio.to_thread(self._store.get_tombstones, user_id)
        visible = [
            (text, res_meta)
            for text, res_meta in texts
            if not self._is_tombstoned(text, user_tombstones)
        ]

        # Resolve canonical IDs through the durable registry; unmatched
        # results are inserted so later update/delete round-trips durably.
        now = _utcnow()
        memories = await asyncio.to_thread(
            self._resolve_search_rows, user_id, session_id, visible[:top_k], now
        )

        results: list[MemorySearchResult] = []
        total = len(visible)
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

    def _resolve_search_rows(
        self,
        user_id: str,
        session_id: str | None,
        items: list[tuple[str, dict[str, Any]]],
        now: datetime,
    ) -> list[Memory]:
        """Resolve search-result texts to canonical Memory rows (sync).

        Runs in a worker thread. A result whose content matches a stored
        memory row for this user returns that row's canonical ID (which is
        how an updated memory keeps its original ID); unmatched results get
        stable content-derived IDs AND are inserted into the registry so a
        later update/delete round-trips durably.
        """
        resolved: list[Memory] = []
        for text, res_meta in items:
            memory = self._store.find_by_user_content(user_id, text)
            if memory is None:
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
                self._store.upsert_memory(memory)
            resolved.append(memory)
        return resolved

    @staticmethod
    def _is_tombstoned(text: str, tombstones: set[str]) -> bool:
        """Whether a search-result chunk matches a tombstoned content.

        Matches by exact equality or by substring containment (cognee chunks
        long documents, so a chunk of a deleted memory is a substring of the
        tombstoned original). Best-effort: text that cognee transformed
        during cognify may not match.
        """
        if not tombstones:
            return False
        if text in tombstones:
            return True
        return any(text in tombstoned for tombstoned in tombstones)

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

    async def _list_matching_items(
        self, datasets_api: Any, wanted: set[str]
    ) -> list[tuple[Any, Any, set[str]]]:
        """List (dataset_id, data_id, matching_hashes) without mutating anything."""
        found: list[tuple[Any, Any, set[str]]] = []
        for dataset in await datasets_api.list_datasets() or []:
            if getattr(dataset, "name", None) != self._config.dataset_name:
                continue
            for data_item in await datasets_api.list_data(dataset.id) or []:
                item_hashes = {
                    getattr(data_item, "content_hash", None),
                    getattr(data_item, "raw_content_hash", None),
                }
                overlap = {h for h in wanted if h in item_hashes}
                if overlap:
                    found.append((dataset.id, data_item.id, overlap))
        return found

    async def _try_hard_delete(self, contents: list[str]) -> bool:
        """Hard-delete data items from cognee's stores; report whether proven.

        cognee identifies text data items by an MD5 content hash. Removal is
        PROVEN only when every content's hash is accounted for. Three phases,
        so a failure at any point leaves the operation retryable:

        1. Discovery (NON-MUTATING): map every requested hash to its stored
           data items. Hashes this store already provably deleted (durable
           ``hard_deleted`` ledger) count as done. If any remaining hash has
           no matching item, return False WITHOUT deleting anything —
           "nothing matched" is not proof (a chunk-registered row hashes
           differently from its source item), and deleting the matched
           subset first would make the unmatched remainder permanently
           unprovable on retry.
        2. Deletion: per hash, delete all its items, then durably record the
           hash in the ledger — so an interruption between items never
           strands an already-removed hash as unprovable.
        3. Verification: re-list; no matching item may remain. On failure the
           just-recorded ledger entries are cleared.

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
        try:
            hashes = {self._content_hash(content) for content in contents}
            already_deleted = await asyncio.to_thread(
                self._store.get_hard_deleted, dataset_name, hashes
            )
            remaining = hashes - already_deleted
            if not remaining:
                return True

            # Phase 1 — discovery, non-mutating.
            found = await self._list_matching_items(datasets_api, remaining)
            matched: set[str] = set()
            for _, _, overlap in found:
                matched |= overlap
            if matched != remaining:
                logger.info(
                    "Hard delete unproven: %d of %d contents have no matching "
                    "cognee data item; nothing was deleted",
                    len(remaining - matched),
                    len(hashes),
                )
                return False

            # Phase 2 — delete per hash, recording each completed hash durably.
            items_by_hash: dict[str, list[tuple[Any, Any]]] = {}
            for dataset_id, data_id, overlap in found:
                for h in overlap:
                    items_by_hash.setdefault(h, []).append((dataset_id, data_id))
            deleted_ids: set[Any] = set()
            recorded: set[str] = set()
            for h, items in items_by_hash.items():
                for dataset_id, data_id in items:
                    if data_id in deleted_ids:
                        continue
                    await datasets_api.delete_data(dataset_id=dataset_id, data_id=data_id)
                    deleted_ids.add(data_id)
                    logger.info(
                        "Hard-deleted cognee data item %s from dataset %s",
                        data_id,
                        dataset_name,
                    )
                await asyncio.to_thread(self._store.record_hard_deleted, dataset_name, {h})
                recorded.add(h)

            # Phase 3 — verification: nothing matching may remain.
            if await self._list_matching_items(datasets_api, remaining):
                logger.warning("Hard delete unproven: matching data items remain after deletion")
                await asyncio.to_thread(self._store.clear_hard_deleted, dataset_name, recorded)
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

        old_contents = [existing.content]
        for fact in existing.metadata.get("_cognee_facts") or []:
            if isinstance(fact, str) and fact:
                old_contents.append(fact)

        if not self._tombstones_fully_enforce() and not await self._try_hard_delete(old_contents):
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
            {self._content_hash(new_content)},
        )
        if self._tombstones_fully_enforce():
            # Best-effort reclaim; under non-CHUNKS the old data was already
            # verifiably removed before any mutation.
            await self._try_hard_delete(old_contents)
        logger.info("Updated memory %s in place (old content tombstoned)", memory_id)
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
            if not await self._try_hard_delete(contents):
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
        await self._try_hard_delete(contents)
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
