"""Transactional recovery of Codex state left in a temporary Headroom home."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tomlkit

_TEMP_HOME_PREFIX = "headroom-codex-home-"
_RECOVERY_DIR = ".headroom-codex-recovery"
_SQLITE_SUFFIXES = {".sqlite", ".db"}
_RUNTIME_NAMES = {".DS_Store"}
_RUNTIME_SUFFIXES = {".lock", ".sock", ".socket", "-shm", "-wal", "-journal"}


def _latest_state_mtime_ns(home: Path) -> int:
    timestamps = [entry.lstat().st_mtime_ns for entry in home.rglob("*") if not entry.is_dir()]
    return max(timestamps, default=home.stat().st_mtime_ns)


@dataclass
class RecoveryReport:
    source: Path
    target: Path
    backup_dir: Path
    copied: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)
    quarantined: list[str] = field(default_factory=list)
    skipped_runtime: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CodexHistoryAudit:
    indexed: int
    active: int
    archived: int
    unindexed_rollouts: tuple[str, ...]
    history_without_rollout: tuple[str, ...]


def discover_dangling_homes(temp_root: Path | None = None) -> list[Path]:
    """Find non-empty Headroom temporary Codex homes, newest first."""
    if temp_root is not None:
        roots = [temp_root]
    else:
        roots = [Path(tempfile.gettempdir())]
        if env_tmpdir := os.environ.get("TMPDIR"):
            roots.append(Path(env_tmpdir))
        roots.extend((Path("/tmp"), Path("/private/tmp")))
        roots.extend(Path("/private/var/folders").glob("*/*/T"))
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            paths = root.glob(f"{_TEMP_HOME_PREFIX}*")
            for path in paths:
                resolved = path.resolve()
                if (
                    resolved not in seen
                    and path.is_dir()
                    and not path.is_symlink()
                    and any(path.iterdir())
                ):
                    seen.add(resolved)
                    candidates.append(path)
        except OSError:
            continue
    return sorted(candidates, key=_latest_state_mtime_ns, reverse=True)


def discover_referenced_temp_homes(target: Path) -> list[Path]:
    """Find temporary Codex homes referenced by retained thread rows."""
    references: set[Path] = set()
    for database in target.rglob("*"):
        if not database.is_file() or database.suffix not in _SQLITE_SUFFIXES:
            continue
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            tables = _database_schema(connection, "main")
            if "threads" not in tables:
                continue
            columns = {str(column[1]) for column in _table_columns(connection, "main", "threads")}
            if "rollout_path" not in columns:
                continue
            for (rollout_path,) in connection.execute("SELECT rollout_path FROM threads"):
                path = Path(str(rollout_path))
                for index, part in enumerate(path.parts):
                    if part.startswith(_TEMP_HOME_PREFIX):
                        references.add(Path(*path.parts[: index + 1]))
                        break
        except sqlite3.Error:
            continue
        finally:
            if connection is not None:
                connection.close()
    return sorted(references)


def _history_session_ids(history: Path) -> set[str]:
    session_ids: set[str] = set()
    if not history.is_file():
        return session_ids
    for line in history.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        session_id = record.get("session_id") or record.get("thread_id") or record.get("id")
        if session_id:
            session_ids.add(str(session_id))
    return session_ids


def _rollout_session_ids(target: Path) -> set[str]:
    session_ids: set[str] = set()
    for directory in (target / "sessions", target / "archived_sessions"):
        if not directory.is_dir():
            continue
        for rollout in directory.rglob("rollout-*.jsonl"):
            try:
                with rollout.open(encoding="utf-8", errors="replace") as handle:
                    first_line = handle.readline()
                record = json.loads(first_line)
            except (OSError, json.JSONDecodeError):
                continue
            if record.get("type") != "session_meta":
                continue
            payload = record.get("payload", {})
            session_id = payload.get("id") or payload.get("session_id")
            if session_id:
                session_ids.add(str(session_id))
    return session_ids


def audit_codex_history(target: Path) -> CodexHistoryAudit | None:
    """Compare durable history, rollout files, and the canonical thread index."""
    database = target / "state_5.sqlite"
    if not database.is_file():
        return None
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
        tables = _database_schema(connection, "main")
        if "threads" not in tables:
            return None
        columns = {str(column[1]) for column in _table_columns(connection, "main", "threads")}
        if "id" not in columns:
            return None
        if "archived" in columns:
            rows = connection.execute("SELECT id, archived FROM threads").fetchall()
        else:
            rows = [(thread_id, 0) for (thread_id,) in connection.execute("SELECT id FROM threads")]
    except sqlite3.Error:
        return None
    finally:
        if connection is not None:
            connection.close()
    thread_ids = {str(row[0]) for row in rows}
    archived = sum(bool(row[1]) for row in rows)
    rollout_ids = _rollout_session_ids(target)
    history_ids = _history_session_ids(target / "history.jsonl")
    return CodexHistoryAudit(
        indexed=len(thread_ids),
        active=len(thread_ids) - archived,
        archived=archived,
        unindexed_rollouts=tuple(sorted(rollout_ids - thread_ids)),
        history_without_rollout=tuple(sorted(history_ids - thread_ids - rollout_ids)),
    )


def discover_retained_sources(target: Path) -> list[Path]:
    """Find pinned sources retained by interrupted or failed recovery attempts."""
    recovery_root = target.parent / _RECOVERY_DIR
    candidates: list[Path] = []
    try:
        attempts = recovery_root.iterdir()
        for attempt in attempts:
            pinned = attempt / "source-pinned"
            if (
                attempt.is_dir()
                and not (attempt / "manifest.json").exists()
                and pinned.is_dir()
                and not pinned.is_symlink()
                and any(pinned.iterdir())
            ):
                candidates.append(pinned)
    except OSError:
        return []
    return sorted(candidates, key=_latest_state_mtime_ns, reverse=True)


def home_fingerprint(home: Path) -> str:
    """Return a content-independent fingerprint used to detect concurrent writes."""
    digest = hashlib.sha256()
    if not home.exists():
        return digest.hexdigest()
    for path in sorted(home.rglob("*")):
        relative = path.relative_to(home)
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        digest.update(os.fsencode(str(relative)))
        digest.update(str(metadata.st_mode).encode())
        digest.update(str(metadata.st_size).encode())
        digest.update(str(metadata.st_mtime_ns).encode())
        if stat.S_ISLNK(metadata.st_mode):
            digest.update(os.fsencode(os.readlink(path)))
    return digest.hexdigest()


def _is_runtime_artifact(path: Path) -> bool:
    name = path.name
    return name in _RUNTIME_NAMES or any(name.endswith(suffix) for suffix in _RUNTIME_SUFFIXES)


def _secure_tree(path: Path) -> None:
    if not path.exists():
        return
    path.chmod(0o700)
    for entry in path.rglob("*"):
        if entry.is_symlink():
            continue
        entry.chmod(0o700 if entry.is_dir() else 0o600)


def _copy_home(source: Path, destination: Path, skipped: list[str]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    destination.chmod(0o700)
    for entry in source.rglob("*"):
        relative = entry.relative_to(source)
        output = destination / relative
        try:
            metadata = entry.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISSOCK(metadata.st_mode):
            skipped.append(str(relative))
            continue
        if entry.is_symlink():
            output.parent.mkdir(parents=True, exist_ok=True)
            output.symlink_to(os.readlink(entry))
        elif entry.is_dir():
            output.mkdir(parents=True, exist_ok=True)
        elif entry.is_file():
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, output, follow_symlinks=False)
        else:
            skipped.append(str(relative))
    _secure_tree(destination)


def _new_backup_dir(target: Path) -> Path:
    root = target.parent / _RECOVERY_DIR
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = f"{stamp}-{os.getpid()}"
    for counter in range(1000):
        suffix = "" if counter == 0 else f"-{counter}"
        candidate = root / f"{stem}{suffix}"
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError("could not allocate a Codex recovery backup directory")


def _uses_legacy_headroom_routing(document: Any) -> bool:
    providers = document.get("model_providers")
    headroom_provider = providers.get("headroom") if providers is not None else None
    local_provider = headroom_provider is not None and str(
        headroom_provider.get("base_url", "")
    ).startswith("http://127.0.0.1:")
    local_openai_url = str(document.get("openai_base_url", "")).startswith("http://127.0.0.1:")
    return document.get("model_provider") == "headroom" and (local_provider or local_openai_url)


def _clean_managed_codex_config(document: Any) -> None:
    providers = document.get("model_providers")
    headroom_provider = providers.get("headroom") if providers is not None else None
    local_provider = headroom_provider is not None and str(
        headroom_provider.get("base_url", "")
    ).startswith("http://127.0.0.1:")
    local_openai_url = str(document.get("openai_base_url", "")).startswith("http://127.0.0.1:")
    managed_routing = local_provider or local_openai_url
    if managed_routing and document.get("model_provider") == "headroom":
        del document["model_provider"]
    if providers is not None and local_provider:
        del providers["headroom"]
        if not providers:
            del document["model_providers"]
    if managed_routing and local_openai_url:
        del document["openai_base_url"]


def _merge_toml_table(target: Any, source: Any, *, source_wins: bool) -> None:
    for key, source_value in source.items():
        if key not in target:
            target[key] = source_value
            continue
        target_value = target[key]
        if hasattr(target_value, "items") and hasattr(source_value, "items"):
            _merge_toml_table(target_value, source_value, source_wins=source_wins)
        elif source_wins:
            target[key] = source_value


def _merge_config(source: Path, target: Path) -> None:
    source_document = tomlkit.parse(source.read_text(encoding="utf-8"))
    _clean_managed_codex_config(source_document)
    if target.exists():
        target_document = tomlkit.parse(target.read_text(encoding="utf-8"))
        _clean_managed_codex_config(target_document)
        source_wins = source.stat().st_mtime_ns > target.stat().st_mtime_ns
        _merge_toml_table(target_document, source_document, source_wins=source_wins)
    else:
        target_document = source_document
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(tomlkit.dumps(target_document), encoding="utf-8")


def _read_jsonl(path: Path, quarantine: Path, report: RecoveryReport) -> list[str]:
    lines: list[str] = []
    malformed = False
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw_line.strip():
            continue
        try:
            json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError):
            malformed = True
            continue
        lines.append(raw_line)
    if malformed:
        destination = quarantine / path.name
        counter = 1
        while destination.exists():
            destination = quarantine / f"{path.stem}-{counter}{path.suffix}"
            counter += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        report.quarantined.append(str(path))
    return lines


def _merge_jsonl(source: Path, target: Path, quarantine: Path, report: RecoveryReport) -> None:
    existing = _read_jsonl(target, quarantine, report) if target.exists() else []
    incoming = _read_jsonl(source, quarantine, report)
    merged = list(existing)
    seen = set(existing)
    for line in incoming:
        if line not in seen:
            seen.add(line)
            merged.append(line)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"{line}\n" for line in merged), encoding="utf-8")


def _merge_rollout(
    source: Path,
    target: Path,
    quarantine: Path,
    report: RecoveryReport,
    replacement_provider: str | None,
) -> None:
    incoming = _read_jsonl(source, quarantine, report)
    keep_target = target.exists() and source.stat().st_mtime_ns <= target.stat().st_mtime_ns
    lines = _read_jsonl(target, quarantine, report) if keep_target else incoming
    provider_replaced = False
    if replacement_provider is not None:
        for index, line in enumerate(lines):
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            payload = record.get("payload")
            if (
                record.get("type") == "session_meta"
                and isinstance(payload, dict)
                and payload.get("model_provider") == "headroom"
            ):
                payload["model_provider"] = replacement_provider
                lines[index] = json.dumps(record, separators=(",", ":"))
                provider_replaced = True
    if keep_target and not provider_replaced:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _database_schema(connection: sqlite3.Connection, schema: str) -> dict[str, str]:
    rows = connection.execute(
        f"SELECT name, sql FROM {_quote(schema)}.sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return dict(rows)


def _database_schema_objects(
    connection: sqlite3.Connection, schema: str
) -> list[tuple[str, str, str, str]]:
    return connection.execute(
        f"SELECT type, name, tbl_name, sql FROM {_quote(schema)}.sqlite_master "
        "WHERE name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
        "ORDER BY type, name"
    ).fetchall()


def _table_columns(
    connection: sqlite3.Connection, schema: str, table: str
) -> list[tuple[Any, ...]]:
    return connection.execute(f"PRAGMA {_quote(schema)}.table_info({_quote(table)})").fetchall()


def _active_model_provider(config_file: Path) -> str:
    if not config_file.is_file():
        return "openai"
    document = tomlkit.parse(config_file.read_text(encoding="utf-8"))
    return str(document.get("model_provider", "openai"))


def _recovered_rollout_relative_path(rollout_path: Path, source_home: Path) -> Path | None:
    try:
        return rollout_path.relative_to(source_home)
    except ValueError:
        for index, part in enumerate(rollout_path.parts):
            if part.startswith(_TEMP_HOME_PREFIX):
                return Path(*rollout_path.parts[index + 1 :])
    return None


def _thread_rollout_paths(connection: sqlite3.Connection, schema: str) -> dict[Any, str]:
    tables = _database_schema(connection, schema)
    if "threads" not in tables:
        return {}
    columns = {str(column[1]) for column in _table_columns(connection, schema, "threads")}
    if not {"id", "rollout_path"}.issubset(columns):
        return {}
    return dict(
        connection.execute(f"SELECT id, rollout_path FROM {_quote(schema)}.threads").fetchall()
    )


def _normalize_recovered_threads(
    connection: sqlite3.Connection,
    *,
    source_home: Path,
    target_home: Path,
    replacement_provider: str | None,
    source_rollout_paths: dict[Any, str],
) -> None:
    tables = _database_schema(connection, "main")
    if "threads" not in tables:
        return
    columns = {str(column[1]) for column in _table_columns(connection, "main", "threads")}
    if not {"id", "rollout_path"}.issubset(columns):
        return
    select_columns = "id, rollout_path"
    if "model_provider" in columns:
        select_columns += ", model_provider"
    rows = connection.execute(f"SELECT {select_columns} FROM threads").fetchall()
    for row in rows:
        thread_id, rollout_path = row[:2]
        source_rollout_path = source_rollout_paths.get(thread_id)
        if source_rollout_path is None:
            continue
        relative = _recovered_rollout_relative_path(Path(source_rollout_path), source_home)
        if relative is None:
            continue
        durable_rollout = target_home / relative
        if str(rollout_path) not in {source_rollout_path, str(durable_rollout)}:
            continue
        if not durable_rollout.is_file():
            raise RuntimeError(f"recovered rollout is missing for thread {thread_id}")
        connection.execute(
            "UPDATE threads SET rollout_path = ? WHERE id = ?",
            (str(durable_rollout), thread_id),
        )
        if replacement_provider is not None and len(row) == 3 and row[2] == "headroom":
            connection.execute(
                "UPDATE threads SET model_provider = ? WHERE id = ?",
                (replacement_provider, thread_id),
            )


def _merge_database(
    source: Path,
    target: Path,
    *,
    source_home: Path,
    target_home: Path,
    replacement_provider: str | None,
) -> None:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        connection = sqlite3.connect(target)
        try:
            source_rollout_paths = _thread_rollout_paths(connection, "main")
            _normalize_recovered_threads(
                connection,
                source_home=source_home,
                target_home=target_home,
                replacement_provider=replacement_provider,
                source_rollout_paths=source_rollout_paths,
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        return
    source_is_newer = source.stat().st_mtime_ns > target.stat().st_mtime_ns
    connection = sqlite3.connect(target)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("ATTACH DATABASE ? AS incoming", (str(source),))
        target_schema = _database_schema(connection, "main")
        source_schema = _database_schema(connection, "incoming")
        if target_schema != source_schema or _database_schema_objects(
            connection, "main"
        ) != _database_schema_objects(connection, "incoming"):
            raise RuntimeError(f"SQLite schema mismatch for {target.name}")
        source_rollout_paths = _thread_rollout_paths(connection, "incoming")
        if "_sqlx_migrations" in target_schema:
            migration_columns = [
                str(column[1]) for column in _table_columns(connection, "main", "_sqlx_migrations")
            ]
            if "version" in migration_columns and "checksum" in migration_columns:
                target_migrations = dict(
                    connection.execute(
                        "SELECT version, checksum FROM main._sqlx_migrations"
                    ).fetchall()
                )
                source_migrations = dict(
                    connection.execute(
                        "SELECT version, checksum FROM incoming._sqlx_migrations"
                    ).fetchall()
                )
                for version in target_migrations.keys() & source_migrations.keys():
                    if target_migrations[version] != source_migrations[version]:
                        raise RuntimeError(
                            f"SQLite migration mismatch for {target.name} at version {version}"
                        )
        connection.execute("BEGIN IMMEDIATE")
        try:
            for table in target_schema:
                columns = _table_columns(connection, "main", table)
                source_columns = _table_columns(connection, "incoming", table)
                if columns != source_columns:
                    raise RuntimeError(f"SQLite schema mismatch for {target.name}:{table}")
                column_names = [str(column[1]) for column in columns]
                primary_key = [
                    str(column[1])
                    for column in sorted(columns, key=lambda row: row[5])
                    if column[5]
                ]
                quoted_columns = ", ".join(_quote(name) for name in column_names)
                rows = connection.execute(
                    f"SELECT {quoted_columns} FROM incoming.{_quote(table)}"
                ).fetchall()
                placeholders = ", ".join("?" for _ in column_names)
                verb = (
                    "INSERT OR REPLACE" if primary_key and source_is_newer else "INSERT OR IGNORE"
                )
                connection.executemany(
                    f"{verb} INTO {_quote(table)} ({quoted_columns}) VALUES ({placeholders})",
                    rows,
                )
            _normalize_recovered_threads(
                connection,
                source_home=source_home,
                target_home=target_home,
                replacement_provider=replacement_provider,
                source_rollout_paths=source_rollout_paths,
            )
            connection.commit()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if not integrity or integrity[0] != "ok":
                raise RuntimeError(f"SQLite integrity check failed for {target.name}")
            violations = connection.execute("PRAGMA foreign_key_check").fetchall()
            if violations:
                raise RuntimeError(f"SQLite foreign key check failed for {target.name}")
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.execute("DETACH DATABASE incoming")
    finally:
        connection.close()


def _merge_pinned_home(
    pinned: Path,
    target: Path,
    report: RecoveryReport,
    *,
    source_home: Path,
) -> None:
    quarantine = report.backup_dir / "quarantine"
    source_config = pinned / "config.toml"
    replace_legacy_provider = source_config.is_file() and _uses_legacy_headroom_routing(
        tomlkit.parse(source_config.read_text(encoding="utf-8"))
    )
    replacement_provider = (
        _active_model_provider(target / "config.toml") if replace_legacy_provider else None
    )
    for source in sorted(pinned.rglob("*")):
        relative = source.relative_to(pinned)
        destination = target / relative
        if source.is_dir() or source.is_symlink():
            if source.is_symlink() and not destination.exists() and not destination.is_symlink():
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(os.readlink(source))
                report.copied.append(str(relative))
            continue
        if _is_runtime_artifact(source):
            report.skipped_runtime.append(str(relative))
            continue
        if source.name == "config.toml":
            _merge_config(source, destination)
            report.merged.append(str(relative))
        elif source.suffix == ".jsonl" and relative.parts[0] in {
            "sessions",
            "archived_sessions",
        }:
            _merge_rollout(
                source,
                destination,
                quarantine,
                report,
                replacement_provider,
            )
            report.merged.append(str(relative))
        elif source.suffix == ".jsonl":
            _merge_jsonl(source, destination, quarantine, report)
            report.merged.append(str(relative))
        elif source.suffix in _SQLITE_SUFFIXES:
            _merge_database(
                source,
                destination,
                source_home=source_home,
                target_home=target,
                replacement_provider=replacement_provider,
            )
            report.merged.append(str(relative))
        elif not destination.exists() or source.stat().st_mtime_ns > destination.stat().st_mtime_ns:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            report.copied.append(str(relative))


def _restore_target(target: Path, target_backup: Path, target_existed: bool) -> None:
    # Windows can retain SQLite file handles until statement finalizers run.
    gc.collect()
    if target.exists():
        target.replace(target_backup.parent / "target-failed")
    if target_existed:
        shutil.copytree(target_backup, target, symlinks=True)


def _capture_modes(home: Path) -> dict[str, int]:
    modes = {".": stat.S_IMODE(home.stat().st_mode)}
    for entry in home.rglob("*"):
        if not entry.is_symlink():
            modes[str(entry.relative_to(home))] = stat.S_IMODE(entry.stat().st_mode)
    return modes


def _restore_modes(home: Path, modes: dict[str, int]) -> None:
    for relative, mode in modes.items():
        path = home if relative == "." else home / relative
        if path.exists() and not path.is_symlink():
            path.chmod(mode)


def _reject_target_symlink_traversal(source: Path, target: Path) -> None:
    if not target.exists():
        return
    for entry in source.rglob("*"):
        if entry.is_dir() or entry.is_symlink():
            continue
        destination = target
        for part in entry.relative_to(source).parts:
            destination /= part
            if destination.is_symlink():
                raise ValueError(f"recovery would write through target symlink: {destination}")


def recover_codex_home(*, source: Path, target: Path) -> RecoveryReport:
    """Merge one quiet temporary Codex home into the active home transactionally."""
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    if not source.is_dir() or source == target:
        raise ValueError("source must be an existing Codex home different from target")
    if source in target.parents or target in source.parents or target == Path(target.anchor):
        raise ValueError("source and target Codex homes must not overlap")
    _reject_target_symlink_traversal(source, target)
    before = home_fingerprint(source)
    backup_dir = _new_backup_dir(target)
    report = RecoveryReport(source=source, target=target, backup_dir=backup_dir)
    pinned = backup_dir / "source-pinned"
    target_backup = backup_dir / "target-before"
    _copy_home(source, pinned, report.skipped_runtime)
    if home_fingerprint(source) != before:
        raise RuntimeError("source Codex home changed while it was being pinned")
    target_existed = target.exists()
    target_modes: dict[str, int] = {}
    if target_existed:
        target_modes = _capture_modes(target)
        target_fingerprint = home_fingerprint(target)
        _copy_home(target, target_backup, report.skipped_runtime)
        if home_fingerprint(target) != target_fingerprint:
            raise RuntimeError("target Codex home changed while it was being backed up")
    else:
        target.mkdir(mode=0o700, parents=True)
    try:
        _merge_pinned_home(pinned, target, report, source_home=source)
    except Exception:
        _restore_target(target, target_backup, target_existed)
        if target_existed:
            _restore_modes(target, target_modes)
        raise
    manifest = asdict(report)
    manifest.update(source=str(source), target=str(target), backup_dir=str(backup_dir))
    manifest_file = backup_dir / "manifest.json"
    manifest_file.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_file.chmod(0o600)
    return report
