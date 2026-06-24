"""Per-project memory storage routing.

Fixes the "memories bleed across projects" bug (GH #462) by giving each
workspace a physically isolated SQLite database file. Cross-project bleed
becomes structurally impossible: the wrong DB is simply not open during
a request.

Three storage modes:

* ``PROJECT`` (default): one DB per resolved project. The project is
  identified from request headers (explicit) or by parsing a Claude
  Code / Codex ``<env>`` block for the working directory.
* ``USER``: one DB per ``x-headroom-user-id`` (no project axis).
* ``GLOBAL``: a single DB shared across everything. Matches the pre-fix
  behaviour and is preserved so users can still reach memories written
  before the fix landed.

A ``BackendRouter`` owns an LRU cache of open ``LocalBackend`` instances
keyed by their on-disk path so that repeated requests for the same
project hit a warm backend. The cache is bounded to keep file-handle
and embedder-index pressure predictable.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from headroom.memory.backends.local import LocalBackend, LocalBackendConfig

logger = logging.getLogger(__name__)


# Known prefixes that mark a working-directory line inside a client's
# ``<env>`` / ``<environment>`` system-prompt block. Ordered so that the
# most specific / most recent client format is tried first. Matched with
# literal ``str.find`` — no regex.
_CWD_PREFIXES: tuple[str, ...] = (
    "Primary working directory:",  # Claude Code (current)
    "Working directory:",  # Claude Code (older) / Codex
    "cwd:",  # Generic / debug format
)

# Whitelist of characters allowed in on-disk basenames. Anything else is
# collapsed to a single ``-``. Kept as an explicit set instead of a
# regex character-class so the sanitiser stays trivially auditable.
_BASENAME_ALLOWED: frozenset[str] = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


class MemoryStorageMode(str, Enum):
    """Physical layout for the on-disk memory store."""

    PROJECT = "project"
    USER = "user"
    GLOBAL = "global"


@dataclass
class RequestContext:
    """The slice of request state the router needs to resolve a project.

    Built fresh per request at the provider-handler seam. Stays small so
    handlers do not pay for memory routing on requests that never touch
    the memory pipeline.
    """

    headers: Mapping[str, str]
    system_prompt: str
    base_user_id: str
    project_root_override: str | None = None


@dataclass(frozen=True)
class ResolvedScope:
    """The outcome of project resolution for one request.

    Carried back to the caller so injected memory blocks can advertise
    their provenance (Fix C in the original design) and so structured
    logs can be tagged with the same key the DB uses.
    """

    mode: MemoryStorageMode
    db_path: Path
    display_name: str  # human-readable label, e.g. project basename
    project_key: str | None  # stable hash, None for USER/GLOBAL


@dataclass
class BackendRouterConfig:
    """Configuration for ``BackendRouter``.

    Attributes:
        mode: Storage mode (PROJECT / USER / GLOBAL).
        root_dir: Filesystem root under which mode-specific subdirectories
            are created.
        global_db_path: Path used for ``GLOBAL`` mode. Defaults to the
            legacy ``<root_dir>/memory.db`` so memories written before
            the per-project fix landed remain reachable via
            ``--memory-storage=global``.
        max_open_backends: LRU cap on simultaneously-open backends.
        backend_config_template: Template ``LocalBackendConfig`` to clone
            for each backend; only ``db_path`` / ``graph_db_path`` differ
            per project.
        unresolved_project_fallback: Behavior when ``mode`` is PROJECT but
            ``ProjectResolver.resolve()`` returns ``None`` (no header, no
            CLI override, no ``cwd:`` in system prompt).

            - ``"empty"`` (default, fail-closed): refuse to load any
              memory for this request — return a sentinel scope whose
              ``project_key`` is ``None`` and whose mode stays PROJECT.
              The memory handler treats this as "no memory available"
              and skips injection. Prevents the silent cross-project
              pooling that surfaced on 2026-05-26 (an entry from a
              prior TAM-550 session was misread as a live instruction
              inside an unrelated thread).
            - ``"global"`` (legacy opt-in): fall back to GLOBAL. ALL
              unresolved-project traffic across ALL clients/projects
              pools into one DB. Cross-project leak vector; opt in
              only if you understand the trade-off.
    """

    mode: MemoryStorageMode
    root_dir: Path
    global_db_path: Path
    max_open_backends: int = 16
    backend_config_template: LocalBackendConfig = field(default_factory=LocalBackendConfig)
    unresolved_project_fallback: str = "empty"


class ProjectResolver:
    """Resolve a request to a (key, display_name) project identity.

    Looks at request signals in priority order and returns ``None`` when
    no signal yields a project. The router uses that ``None`` to apply
    the configured fallback (today: ``GLOBAL`` per the user's choice in
    the bug-fix design discussion).
    """

    def resolve(self, ctx: RequestContext) -> tuple[str, str] | None:
        """Return ``(project_key, display_name)`` or ``None``.

        ``project_key`` is a stable, filesystem-safe identifier suitable
        for use as a directory or hash. ``display_name`` is a
        human-readable label for log lines and the provenance header in
        the injected memory block.
        """

        # Tier 1: client-provided explicit project id (any client).
        explicit = self._first_nonempty_header(ctx.headers, "x-headroom-project-id")
        if explicit:
            safe = self._sanitize_basename(explicit)
            if safe:
                return safe, explicit

        # Tier 2: client-provided explicit cwd (any client).
        explicit_cwd = self._first_nonempty_header(ctx.headers, "x-headroom-cwd")
        if explicit_cwd:
            ident = self._identity_from_cwd(explicit_cwd)
            if ident is not None:
                return ident

        # Tier 3: CLI-level override of the project root.
        if ctx.project_root_override:
            ident = self._identity_from_cwd(ctx.project_root_override)
            if ident is not None:
                return ident

        # Tier 4: parse the system prompt for a ``<env>`` cwd line.
        sys_cwd = self._extract_cwd_from_system_prompt(ctx.system_prompt)
        if sys_cwd:
            ident = self._identity_from_cwd(sys_cwd)
            if ident is not None:
                return ident

        return None

    @staticmethod
    def _first_nonempty_header(headers: Mapping[str, str], name: str) -> str | None:
        # FastAPI/Starlette headers are case-insensitive but the mapping
        # passed in may be either kind. Try the canonical lowercase form
        # first, then fall through to a full case-insensitive sweep.
        v = headers.get(name) or headers.get(name.lower())
        if v:
            return v.strip() or None
        lower = name.lower()
        for k, val in headers.items():
            if k.lower() == lower and val and val.strip():
                return val.strip()
        return None

    @classmethod
    def _extract_cwd_from_system_prompt(cls, system_prompt: str) -> str | None:
        if not system_prompt:
            return None
        for prefix in _CWD_PREFIXES:
            idx = system_prompt.find(prefix)
            if idx < 0:
                continue
            start = idx + len(prefix)
            end = system_prompt.find("\n", start)
            chunk = system_prompt[start:] if end < 0 else system_prompt[start:end]
            chunk = chunk.strip()
            if chunk:
                return chunk
        return None

    @classmethod
    def _identity_from_cwd(cls, raw_cwd: str) -> tuple[str, str] | None:
        cwd = raw_cwd.strip()
        if not cwd:
            return None
        # Normalise so symlinked / trailing-slash variants collapse to
        # the same key. ``realpath`` falls back to the input when the
        # path doesn't exist on this host, which is the right behaviour
        # for a proxy that may run on a different machine than the
        # client (in that case we still want a stable key per cwd
        # string).
        try:
            normalised = os.path.realpath(cwd)
        except (OSError, ValueError):
            normalised = cwd
        normalised = normalised.rstrip(os.sep) or os.sep
        basename = os.path.basename(normalised) or "root"
        safe_basename = cls._sanitize_basename(basename) or "project"
        digest = hashlib.sha256(normalised.encode("utf-8")).hexdigest()[:16]
        key = f"{safe_basename}-{digest}"
        return key, basename

    @staticmethod
    def _sanitize_basename(value: str) -> str:
        out: list[str] = []
        last_was_dash = False
        for ch in value.strip():
            if ch in _BASENAME_ALLOWED:
                out.append(ch)
                last_was_dash = False
            elif not last_was_dash:
                out.append("-")
                last_was_dash = True
        cleaned = "".join(out).strip("-._")
        # Bound the length so a long override doesn't create unwieldy
        # directory names.
        return cleaned[:64]


class BackendRouter:
    """Maps a ``RequestContext`` to a ``LocalBackend`` for save/search.

    Holds an LRU of open backends so repeated traffic for the same
    project hits a warm instance. Eviction simply drops the Python
    reference; SQLite connections are closed by the backend's own
    finalisers. Acquisition takes a lock to keep the cache consistent
    under the proxy's async-but-multi-task workload — the lock is held
    only for the lookup, never across IO.
    """

    def __init__(
        self,
        config: BackendRouterConfig,
        resolver: ProjectResolver | None = None,
    ) -> None:
        self._config = config
        self._resolver = resolver or ProjectResolver()
        self._backends: OrderedDict[Path, LocalBackend] = OrderedDict()
        self._lock = threading.Lock()

    def backend_for(self, ctx: RequestContext) -> tuple[LocalBackend, ResolvedScope]:
        """Return the backend + scope metadata to use for this request."""

        scope = self._resolve_scope(ctx)
        backend = self._get_or_create_backend(scope.db_path)
        return backend, scope

    def _resolve_scope(self, ctx: RequestContext) -> ResolvedScope:
        mode = self._config.mode

        if mode is MemoryStorageMode.GLOBAL:
            return ResolvedScope(
                mode=MemoryStorageMode.GLOBAL,
                db_path=self._config.global_db_path,
                display_name="global",
                project_key=None,
            )

        if mode is MemoryStorageMode.USER:
            user_safe = ProjectResolver._sanitize_basename(ctx.base_user_id) or "default"
            db_path = self._config.root_dir / "users" / user_safe / "memory.db"
            return ResolvedScope(
                mode=MemoryStorageMode.USER,
                db_path=db_path,
                display_name=ctx.base_user_id,
                project_key=user_safe,
            )

        # PROJECT mode.
        ident = self._resolver.resolve(ctx)
        if ident is None:
            fallback = self._config.unresolved_project_fallback
            if fallback == "empty":
                # Fail-closed: refuse to load any memory for this
                # request. The memory handler checks `scope.project_key
                # is None` and skips injection rather than pooling this
                # request into the GLOBAL bucket (which is what surfaced
                # the TAM-550 cross-thread instruction misread on
                # 2026-05-26 — a memory from a prior unrelated session
                # got dropped into the live user turn and read as a
                # command).
                logger.warning(
                    "event=memory_project_unresolved behavior=empty user_id=%s "
                    "hint='set x-headroom-project-id or x-headroom-cwd header, "
                    "or set memory.unresolved_project_fallback=global to opt-in "
                    "to legacy cross-project GLOBAL pooling (cross-project leak risk).'",
                    ctx.base_user_id,
                )
                return ResolvedScope(
                    mode=MemoryStorageMode.PROJECT,
                    db_path=self._config.global_db_path,  # Unused — caller checks project_key.
                    display_name="unresolved (no memory)",
                    project_key=None,
                )
            if fallback == "global":
                logger.warning(
                    "event=memory_project_unresolved behavior=global user_id=%s",
                    ctx.base_user_id,
                )
                return ResolvedScope(
                    mode=MemoryStorageMode.GLOBAL,
                    db_path=self._config.global_db_path,
                    display_name="global (unresolved)",
                    project_key=None,
                )
            # Unknown config value — fail-loud per no-silent-fallbacks.
            raise ValueError(
                f"unresolved_project_fallback={fallback!r} is not a recognised value; "
                "expected 'empty' or 'global'."
            )

        project_key, display_name = ident
        db_path = self._config.root_dir / "projects" / project_key / "memory.db"
        logger.info(
            "event=memory_project_resolved key=%s display=%s db_path=%s user_id=%s",
            project_key,
            display_name,
            db_path,
            ctx.base_user_id,
        )
        return ResolvedScope(
            mode=MemoryStorageMode.PROJECT,
            db_path=db_path,
            display_name=display_name,
            project_key=project_key,
        )

    def _get_or_create_backend(self, db_path: Path) -> LocalBackend:
        with self._lock:
            existing = self._backends.get(db_path)
            if existing is not None:
                self._backends.move_to_end(db_path)
                return existing

            db_path.parent.mkdir(parents=True, exist_ok=True)

            template = self._config.backend_config_template
            cfg = LocalBackendConfig(
                db_path=str(db_path),
                graph_db_path=str(db_path.with_name(f"{db_path.stem}_graph{db_path.suffix}")),
                embedder_backend=template.embedder_backend,
                embedder_model=template.embedder_model,
                vector_dimension=template.vector_dimension,
                openai_api_key=template.openai_api_key,
                ollama_base_url=template.ollama_base_url,
                graph_persist=template.graph_persist,
                graph_cache_size_kb=template.graph_cache_size_kb,
                cache_enabled=template.cache_enabled,
                cache_max_size=template.cache_max_size,
            )

            backend = LocalBackend(cfg)
            self._backends[db_path] = backend

            while len(self._backends) > self._config.max_open_backends:
                evicted_path, _evicted = self._backends.popitem(last=False)
                logger.debug(
                    "event=memory_backend_evicted db_path=%s reason=lru open=%d",
                    evicted_path,
                    len(self._backends),
                )

            return backend

    def open_backends(self) -> list[Path]:
        """Snapshot of currently-cached backend paths. For tests / stats."""

        with self._lock:
            return list(self._backends.keys())


def extract_system_prompt(body: Mapping[str, Any]) -> str:
    """Best-effort extraction of the system prompt across providers.

    Anthropic puts it on the top-level ``system`` field (string or list
    of content blocks); OpenAI/Gemini-style payloads put it as a message
    with ``role=system``. Returns an empty string when nothing is found
    rather than raising — the resolver tolerates an empty prompt and
    will fall through to the configured fallback.
    """

    system_field = body.get("system")
    if isinstance(system_field, str):
        return system_field
    if isinstance(system_field, list):
        parts: list[str] = []
        for block in system_field:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)

    messages = body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "system":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                if parts:
                    return "\n".join(parts)

    return ""
