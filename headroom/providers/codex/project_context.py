"""Read-only Codex turn-to-project resolution for Responses traffic."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from headroom.memory.storage_router import ProjectResolver, RequestContext
from headroom.providers.codex.threads import _codex_state_db_paths

logger = logging.getLogger(__name__)

_METADATA_HEADER = "x-codex-turn-metadata"
_MAX_METADATA_BYTES = 16 * 1024
_SQLITE_TIMEOUT_SECONDS = 0.1
_SQLITE_ATTEMPTS = 2
_ROLLOUT_CACHE_MAX_ENTRIES = 256


@dataclass(frozen=True, slots=True)
class CodexResolvedProject:
    """One optional project identity and its observable resolution result."""

    cwd: Path | None
    project_key: str | None
    source: Literal[
        "x-headroom-project-id",
        "x-headroom-cwd",
        "configured-project-root",
        "codex-turn-metadata",
        "codex-client-metadata",
        "responses-body-cwd",
        "unresolved",
    ]
    reason: str


class CodexProjectContextResolver:
    """Map Codex ``thread_id`` + ``turn_id`` to an exact rollout cwd."""

    def __init__(self, sqlite_home: str | Path | None = None) -> None:
        self._configured_sqlite_home = Path(sqlite_home).expanduser() if sqlite_home else None

    def resolve(
        self,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        pinned_cwd: Path | None = None,
        project_root_override: str | None = None,
    ) -> CodexResolvedProject:
        explicit_project_id = self._header(headers, "x-headroom-project-id")
        if explicit_project_id:
            project_identity = ProjectResolver().resolve(
                RequestContext(
                    headers=headers,
                    system_prompt="",
                    base_user_id="",
                    project_root_override=project_root_override,
                )
            )
            return CodexResolvedProject(
                cwd=None,
                project_key=project_identity[0] if project_identity else None,
                source="x-headroom-project-id",
                reason="explicit_project_id",
            )

        explicit_cwd = self._header(headers, "x-headroom-cwd")
        if explicit_cwd:
            return self._explicit_cwd_result(
                explicit_cwd,
                headers=headers,
                source="x-headroom-cwd",
                pinned_cwd=pinned_cwd,
            )

        if project_root_override:
            return self._explicit_cwd_result(
                project_root_override,
                headers=headers,
                source="configured-project-root",
                pinned_cwd=pinned_cwd,
            )

        identity, source, metadata_reason = self._turn_identity(headers, body)
        if identity is not None:
            thread_id, turn_id = identity
            if not turn_id:
                return self._fallback_or_skip(
                    body,
                    headers,
                    pinned_cwd,
                    "turn_id_missing",
                )
            cwd, reason = self._cwd_from_state(thread_id, turn_id)
            if cwd is not None:
                return self._cwd_result(
                    cwd,
                    headers,
                    cast(Literal["codex-turn-metadata", "codex-client-metadata"], source),
                    pinned_cwd,
                )
            return self._skip(reason)

        return self._fallback_or_skip(
            body,
            headers,
            pinned_cwd,
            metadata_reason,
        )

    async def resolve_async(
        self,
        *,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
        pinned_cwd: Path | None = None,
        project_root_override: str | None = None,
    ) -> CodexResolvedProject:
        """Resolve optional project context without blocking async model traffic."""
        try:
            return await asyncio.to_thread(
                self.resolve,
                headers=headers,
                body=body,
                pinned_cwd=pinned_cwd,
                project_root_override=project_root_override,
            )
        except Exception:
            logger.warning("event=codex_project_resolution_failed", exc_info=True)
            return self._skip("resolver_failed")

    @staticmethod
    def _header(headers: Mapping[str, str], name: str) -> str | None:
        lowered = name.lower()
        for key, value in headers.items():
            if key.lower() == lowered and value and value.strip():
                return value.strip()
        return None

    def _turn_identity(
        self,
        headers: Mapping[str, str],
        body: Mapping[str, Any],
    ) -> tuple[tuple[str, str | None] | None, str, str]:
        raw_header = self._header(headers, _METADATA_HEADER)
        if raw_header:
            if len(raw_header.encode("utf-8")) > _MAX_METADATA_BYTES:
                return None, "codex-turn-metadata", "metadata_too_large"
            try:
                metadata = json.loads(raw_header)
            except (json.JSONDecodeError, TypeError, ValueError):
                return None, "codex-turn-metadata", "metadata_invalid"
            identity = self._identity_from_metadata(metadata)
            if identity is not None:
                return identity, "codex-turn-metadata", "resolved"

        for container in self._body_containers(body):
            metadata = container.get("client_metadata")
            identity = self._identity_from_metadata(metadata)
            if identity is not None:
                if sum(len(value.encode("utf-8")) for value in identity if value) > (
                    _MAX_METADATA_BYTES
                ):
                    return None, "codex-client-metadata", "metadata_too_large"
                return identity, "codex-client-metadata", "resolved"
        return None, "unresolved", "metadata_missing"

    @staticmethod
    def _identity_from_metadata(metadata: Any) -> tuple[str, str | None] | None:
        if not isinstance(metadata, Mapping):
            return None
        thread_id = metadata.get("thread_id")
        turn_id = metadata.get("turn_id")
        if not isinstance(thread_id, str) or not thread_id.strip():
            return None
        return (
            thread_id.strip(),
            turn_id.strip() if isinstance(turn_id, str) and turn_id.strip() else None,
        )

    @staticmethod
    def _body_containers(body: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        response = body.get("response")
        return (body, response) if isinstance(response, Mapping) else (body,)

    def _fallback_or_skip(
        self,
        body: Mapping[str, Any],
        headers: Mapping[str, str],
        pinned_cwd: Path | None,
        reason: str,
    ) -> CodexResolvedProject:
        for container in self._body_containers(body):
            metadata = container.get("client_metadata")
            candidates = (metadata, container) if isinstance(metadata, Mapping) else (container,)
            for candidate in candidates:
                for key in ("cwd", "working_directory", "project_root"):
                    value = candidate.get(key)
                    if isinstance(value, str) and value.strip():
                        cwd = self._canonical_cwd(value)
                        if cwd is None:
                            return self._skip("body_cwd_invalid")
                        return self._cwd_result(
                            cwd,
                            headers,
                            "responses-body-cwd",
                            pinned_cwd,
                        )
        return self._skip(reason)

    def _explicit_cwd_result(
        self,
        raw_cwd: str,
        *,
        headers: Mapping[str, str],
        source: Literal["x-headroom-cwd", "configured-project-root"],
        pinned_cwd: Path | None,
    ) -> CodexResolvedProject:
        cwd = self._canonical_cwd(raw_cwd)
        if cwd is None:
            # Preserve the existing ProjectResolver behavior for explicit
            # operator overrides, including remote/non-local cwd strings.
            identity = ProjectResolver().resolve(
                RequestContext(
                    headers=headers,
                    system_prompt="",
                    base_user_id="",
                    project_root_override=raw_cwd if source == "configured-project-root" else None,
                )
            )
            return CodexResolvedProject(
                cwd=None,
                project_key=identity[0] if identity else None,
                source=source,
                reason="explicit_override",
            )
        return self._cwd_result(cwd, headers, source, pinned_cwd)

    def _cwd_result(
        self,
        cwd: Path,
        headers: Mapping[str, str],
        source: Literal[
            "x-headroom-cwd",
            "configured-project-root",
            "codex-turn-metadata",
            "codex-client-metadata",
            "responses-body-cwd",
        ],
        pinned_cwd: Path | None,
    ) -> CodexResolvedProject:
        if pinned_cwd is not None and cwd != pinned_cwd:
            return self._skip("project_mismatch")
        identity = ProjectResolver().resolve(
            RequestContext(
                headers=headers,
                system_prompt="",
                base_user_id="",
                project_root_override=str(cwd),
            )
        )
        return CodexResolvedProject(
            cwd=cwd,
            project_key=identity[0] if identity else None,
            source=source,
            reason="resolved",
        )

    @staticmethod
    def _canonical_cwd(raw_cwd: str) -> Path | None:
        try:
            path = Path(raw_cwd).expanduser()
            if not path.is_absolute():
                return None
            resolved = path.resolve(strict=True)
            return resolved if resolved.is_dir() else None
        except (OSError, RuntimeError, ValueError):
            return None

    def _state_root(self) -> Path:
        if self._configured_sqlite_home is not None:
            return self._configured_sqlite_home
        codex_home = Path(os.environ.get("CODEX_HOME", "").strip() or Path.home() / ".codex")
        try:
            config = tomllib.loads((codex_home / "config.toml").read_text(encoding="utf-8"))
            configured_sqlite_home = config.get("sqlite_home")
            if isinstance(configured_sqlite_home, str) and configured_sqlite_home.strip():
                return Path(configured_sqlite_home).expanduser()
        except (OSError, tomllib.TOMLDecodeError):
            pass
        env_sqlite_home = os.environ.get("CODEX_SQLITE_HOME", "").strip()
        if env_sqlite_home:
            return Path(env_sqlite_home).expanduser()
        return codex_home

    def _state_paths(self) -> list[Path]:
        root = self._state_root()
        if root.is_file():
            return [root.resolve()]
        return _codex_state_db_paths(root)

    def _cwd_from_state(self, thread_id: str, turn_id: str) -> tuple[Path | None, str]:
        state_paths = self._state_paths()
        if not state_paths:
            return None, "state_missing"

        rollouts: set[Path] = set()
        saw_supported_schema = False
        saw_locked = False
        for state_path in state_paths:
            rollout, reason = self._rollout_for_thread(state_path, thread_id)
            saw_supported_schema |= reason != "state_schema_unsupported"
            saw_locked |= reason == "state_locked"
            if rollout is not None:
                rollouts.add(rollout)
        if saw_locked:
            return None, "state_locked"
        if len(rollouts) > 1:
            return None, "state_ambiguous"
        if not rollouts:
            return (
                None,
                "thread_missing" if saw_supported_schema else "state_schema_unsupported",
            )

        rollout = next(iter(rollouts))
        if not rollout.is_file():
            return None, "rollout_stale"
        return self._cwd_from_rollout(rollout, turn_id)

    @staticmethod
    def _rollout_for_thread(state_path: Path, thread_id: str) -> tuple[Path | None, str]:
        for attempt in range(_SQLITE_ATTEMPTS):
            connection: sqlite3.Connection | None = None
            try:
                uri = f"{state_path.as_uri()}?mode=ro"
                connection = sqlite3.connect(
                    uri,
                    uri=True,
                    timeout=_SQLITE_TIMEOUT_SECONDS,
                )
                columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(threads)").fetchall()
                }
                id_column = (
                    "id" if "id" in columns else "thread_id" if "thread_id" in columns else ""
                )
                if not id_column or "rollout_path" not in columns:
                    return None, "state_schema_unsupported"
                rows = connection.execute(
                    f"SELECT rollout_path FROM threads WHERE {id_column} = ?",  # noqa: S608
                    (thread_id,),
                ).fetchall()
                paths = {Path(str(row[0])).expanduser().resolve() for row in rows if row[0]}
                if len(paths) > 1:
                    return None, "state_ambiguous"
                return (next(iter(paths)), "resolved") if paths else (None, "thread_missing")
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                    return None, "state_schema_unsupported"
                if attempt + 1 < _SQLITE_ATTEMPTS:
                    time.sleep(_SQLITE_TIMEOUT_SECONDS)
                    continue
                return None, "state_locked"
            except (OSError, sqlite3.Error, ValueError):
                return None, "state_schema_unsupported"
            finally:
                if connection is not None:
                    connection.close()
        return None, "state_locked"

    def _cwd_from_rollout(self, rollout: Path, turn_id: str) -> tuple[Path | None, str]:
        try:
            metadata = rollout.stat()
        except OSError:
            return None, "rollout_stale"
        fingerprint = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )
        raw_cwds, reason = _cached_raw_cwds_from_rollout(rollout, fingerprint, turn_id)
        if reason != "resolved":
            return None, reason
        matches = {cwd for raw_cwd in raw_cwds if (cwd := self._canonical_cwd(raw_cwd))}
        if len(matches) > 1:
            return None, "turn_ambiguous"
        if not matches:
            return None, "turn_context_missing"
        return next(iter(matches)), "resolved"

    @staticmethod
    def _read_raw_cwds_from_rollout(rollout: Path, turn_id: str) -> tuple[tuple[str, ...], str]:
        matches: set[str] = set()
        truncated = False
        try:
            with rollout.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        truncated = True
                        continue
                    if record.get("type") != "turn_context":
                        continue
                    payload = record.get("payload")
                    if not isinstance(payload, Mapping) or payload.get("turn_id") != turn_id:
                        continue
                    cwd = payload.get("cwd")
                    if isinstance(cwd, str):
                        matches.add(cwd)
        except OSError:
            return (), "rollout_stale"
        if truncated:
            return (), "rollout_truncated"
        if matches:
            return tuple(sorted(matches)), "resolved"
        return (), "turn_context_missing"

    @staticmethod
    def _skip(reason: str) -> CodexResolvedProject:
        logger.info("event=codex_project_resolution_skip reason=%s", reason)
        return CodexResolvedProject(
            cwd=None,
            project_key=None,
            source="unresolved",
            reason=reason,
        )


@lru_cache(maxsize=_ROLLOUT_CACHE_MAX_ENTRIES)
def _cached_raw_cwds_from_rollout(
    rollout: Path,
    _fingerprint: tuple[int, int, int, int, int],
    turn_id: str,
) -> tuple[tuple[str, ...], str]:
    return CodexProjectContextResolver._read_raw_cwds_from_rollout(rollout, turn_id)


__all__ = ["CodexProjectContextResolver", "CodexResolvedProject"]
