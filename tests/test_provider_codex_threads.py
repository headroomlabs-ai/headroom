from __future__ import annotations

import sqlite3
from pathlib import Path

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


def test_retag_thread_providers_best_effort_on_corrupt_store(tmp_path: Path) -> None:
    bad = tmp_path / "state_5.sqlite"
    bad.write_text("not a sqlite database", encoding="utf-8")
    # A corrupt store is logged and skipped, never raised.
    threads.retag_thread_providers(tmp_path, frm="openai", to="headroom")


def test_enable_disable_wrappers_retag_expected_direction(tmp_path: Path) -> None:
    db = tmp_path / "state_5.sqlite"
    _seed(db, [("a", "openai"), ("b", "headroom")])

    threads.retag_to_headroom(tmp_path)
    assert _count(db, "headroom") == 2
    assert _count(db, "openai") == 0

    threads.retag_to_native(tmp_path)
    assert _count(db, "openai") == 2
    assert _count(db, "headroom") == 0
