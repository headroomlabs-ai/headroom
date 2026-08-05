from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from headroom.providers.codex import threads


def _seed(path: Path, rows: list[tuple[str, str]]) -> None:
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, model_provider TEXT NOT NULL)")
        conn.executemany("INSERT INTO threads (id, model_provider) VALUES (?, ?)", rows)
        conn.commit()
    finally:
        conn.close()


def _count(path: Path, provider: str) -> int:
    conn = sqlite3.connect(str(path))
    try:
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM threads WHERE model_provider = ?", (provider,)
        ).fetchone()
        return n
    finally:
        conn.close()


def test_codex_state_db_paths_discovers_numeric_files_in_directory_order(tmp_path: Path) -> None:
    sqlite_home = tmp_path / "sqlite"
    sqlite_home.mkdir()
    nested = sqlite_home / "nested"
    nested.mkdir()

    _seed(sqlite_home / "state_10.sqlite", [("a", "openai")])
    _seed(sqlite_home / "state_2.sqlite", [("b", "openai")])
    _seed(sqlite_home / "state_backup.sqlite", [("c", "openai")])
    (sqlite_home / "state_6.sqlite-wal").write_text("wal", encoding="utf-8")
    _seed(nested / "state_7.sqlite", [("d", "openai")])
    _seed(tmp_path / "state_9.sqlite", [("e", "openai")])
    _seed(tmp_path / "state_1.sqlite", [("f", "openai")])
    (tmp_path / "other.db").write_text("other", encoding="utf-8")

    assert threads._codex_state_db_paths(tmp_path) == [
        sqlite_home / "state_2.sqlite",
        sqlite_home / "state_10.sqlite",
        tmp_path / "state_1.sqlite",
        tmp_path / "state_9.sqlite",
    ]


def test_retag_thread_providers_discovers_later_state_store_and_skips_adjacent_files(
    tmp_path: Path,
) -> None:
    sqlite_home = tmp_path / "sqlite"
    sqlite_home.mkdir()
    nested = sqlite_home / "nested"
    nested.mkdir()

    later = sqlite_home / "state_6.sqlite"
    legacy = tmp_path / "state_5.sqlite"
    backup = sqlite_home / "state_backup.sqlite"
    sidecar = sqlite_home / "state_6.sqlite-wal"
    nested_store = nested / "state_7.sqlite"

    _seed(later, [("a", "openai"), ("b", "openai"), ("c", "anthropic")])
    _seed(legacy, [("d", "openai"), ("e", "headroom")])
    _seed(backup, [("f", "openai")])
    sidecar.write_text("wal", encoding="utf-8")
    _seed(nested_store, [("g", "openai")])

    threads.retag_to_headroom(tmp_path)

    assert _count(later, "headroom") == 2
    assert _count(later, "openai") == 0
    assert _count(later, "anthropic") == 1
    assert _count(legacy, "headroom") == 2
    assert _count(legacy, "openai") == 0
    assert _count(backup, "openai") == 1
    assert _count(backup, "headroom") == 0
    assert _count(nested_store, "openai") == 1
    assert _count(nested_store, "headroom") == 0


def test_retag_one_moves_only_matching_provider(tmp_path: Path) -> None:
    db = tmp_path / "state_5.sqlite"
    _seed(db, [("a", "openai"), ("b", "openai"), ("c", "headroom"), ("d", "anthropic")])

    moved = threads._retag_one(db, frm="openai", to="headroom")
    assert moved == 2
    assert _count(db, "openai") == 0
    assert _count(db, "headroom") == 3
    # Third-party providers are left alone.
    assert _count(db, "anthropic") == 1

    back = threads._retag_one(db, frm="headroom", to="openai")
    assert back == 3
    assert _count(db, "headroom") == 0
    assert _count(db, "openai") == 3
    assert _count(db, "anthropic") == 1


def test_retag_one_noop_without_threads_table(tmp_path: Path) -> None:
    db = tmp_path / "state_5.sqlite"
    sqlite3.connect(str(db)).close()  # empty schema, no threads table
    assert threads._retag_one(db, frm="openai", to="headroom") == 0


def test_retag_thread_providers_silent_when_no_store(tmp_path: Path) -> None:
    # No stores exist under this codex_home: must not raise.
    threads.retag_thread_providers(tmp_path, frm="openai", to="headroom")


def test_retag_thread_providers_skips_unreadable_store_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sqlite_home = tmp_path / "sqlite"
    sqlite_home.mkdir()
    _seed(sqlite_home / "state_6.sqlite", [("a", "openai")])
    fallback = tmp_path / "state_5.sqlite"
    _seed(fallback, [("b", "openai")])
    original_iterdir = Path.iterdir

    def fail_for_sqlite(path: Path):
        if path == sqlite_home:
            raise OSError("permission denied")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_for_sqlite)

    threads.retag_to_headroom(tmp_path)

    assert _count(fallback, "headroom") == 1
    assert _count(sqlite_home / "state_6.sqlite", "openai") == 1


def test_retag_thread_providers_best_effort_on_corrupt_store(tmp_path: Path) -> None:
    bad = tmp_path / "state_5.sqlite"
    bad.write_text("not a sqlite database", encoding="utf-8")
    # A corrupt store is logged and skipped, never raised.
    threads.retag_thread_providers(tmp_path, frm="openai", to="headroom")


def test_retag_thread_providers_skips_corrupt_discovered_store_and_continues(
    tmp_path: Path,
) -> None:
    sqlite_home = tmp_path / "sqlite"
    sqlite_home.mkdir()

    bad = sqlite_home / "state_6.sqlite"
    later = sqlite_home / "state_7.sqlite"
    bad.write_text("not a sqlite database", encoding="utf-8")
    _seed(later, [("a", "openai"), ("b", "anthropic")])

    threads.retag_to_headroom(tmp_path)

    assert _count(later, "headroom") == 1
    assert _count(later, "openai") == 0
    assert _count(later, "anthropic") == 1


def test_retag_thread_providers_skips_os_error_store_and_continues(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sqlite_home = tmp_path / "sqlite"
    sqlite_home.mkdir()
    locked = sqlite_home / "state_6.sqlite"
    later = sqlite_home / "state_7.sqlite"
    _seed(locked, [("a", "openai")])
    _seed(later, [("b", "openai"), ("c", "anthropic")])
    original_retag_one = threads._retag_one

    def fail_for_locked(path: Path, *, frm: str, to: str) -> int:
        if path == locked:
            raise OSError("locked")
        return original_retag_one(path, frm=frm, to=to)

    monkeypatch.setattr(threads, "_retag_one", fail_for_locked)

    threads.retag_to_headroom(tmp_path)

    assert _count(locked, "openai") == 1
    assert _count(later, "headroom") == 1
    assert _count(later, "anthropic") == 1


def test_enable_disable_wrappers_retag_expected_direction(tmp_path: Path) -> None:
    db = tmp_path / "state_5.sqlite"
    _seed(db, [("a", "openai"), ("b", "headroom")])

    threads.retag_to_headroom(tmp_path)
    assert _count(db, "headroom") == 2
    assert _count(db, "openai") == 0

    threads.retag_to_native(tmp_path)
    assert _count(db, "openai") == 2
    assert _count(db, "headroom") == 0
