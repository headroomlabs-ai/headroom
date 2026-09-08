from __future__ import annotations

import errno
import json
import os
import socket
import sqlite3
import stat
import sys
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

import headroom.providers.codex.recovery as codex_recovery
from headroom.cli.main import main
from headroom.providers.codex.recovery import discover_dangling_homes, recover_codex_home


def _write_db(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, title TEXT NOT NULL)")
        connection.executemany("INSERT INTO threads VALUES (?, ?)", rows)
        connection.commit()
    finally:
        connection.close()


def _write_sqlx_db(path: Path, checksum: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "CREATE TABLE _sqlx_migrations (version INTEGER PRIMARY KEY, checksum BLOB NOT NULL)"
        )
        connection.execute("INSERT INTO _sqlx_migrations VALUES (1, ?)", (checksum,))
        connection.commit()
    finally:
        connection.close()


def _write_thread_db(
    path: Path,
    rows: list[tuple[str, str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE threads ("
            "id TEXT PRIMARY KEY, rollout_path TEXT NOT NULL, model_provider TEXT NOT NULL"
            ")"
        )
        connection.executemany("INSERT INTO threads VALUES (?, ?, ?)", rows)


def test_discover_dangling_homes_only_returns_codex_homes(tmp_path: Path) -> None:
    candidate = tmp_path / "headroom-codex-home-abc"
    candidate.mkdir()
    (candidate / "config.toml").write_text('model = "gpt-5"\n', encoding="utf-8")
    (tmp_path / "headroom-codex-home-empty").mkdir()
    (tmp_path / "other").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "history.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "headroom-codex-home-linked").symlink_to(outside, target_is_directory=True)

    assert discover_dangling_homes(tmp_path) == [candidate]


def test_discover_dangling_homes_uses_newest_state_not_directory_mtime(
    tmp_path: Path,
) -> None:
    newest_state = tmp_path / "headroom-codex-home-newest-state"
    newest_directory = tmp_path / "headroom-codex-home-newest-directory"
    newest_state.mkdir()
    newest_directory.mkdir()
    newest_state_file = newest_state / "history.jsonl"
    newest_directory_file = newest_directory / "history.jsonl"
    newest_state_file.write_text('{"session_id":"newest"}\n', encoding="utf-8")
    newest_directory_file.write_text('{"session_id":"older"}\n', encoding="utf-8")
    os.utime(newest_state_file, ns=(400, 400))
    os.utime(newest_directory_file, ns=(300, 300))
    os.utime(newest_state, ns=(100, 100))
    os.utime(newest_directory, ns=(500, 500))

    assert discover_dangling_homes(tmp_path) == [newest_state, newest_directory]


def test_discover_dangling_homes_searches_tmpdir_and_python_temp_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_root = tmp_path / "env-tmp"
    python_root = tmp_path / "python-tmp"
    env_candidate = env_root / "headroom-codex-home-env"
    python_candidate = python_root / "headroom-codex-home-python"
    env_candidate.mkdir(parents=True)
    python_candidate.mkdir(parents=True)
    (env_candidate / "history.jsonl").write_text("{}\n", encoding="utf-8")
    (python_candidate / "history.jsonl").write_text("{}\n", encoding="utf-8")
    os.utime(env_candidate / "history.jsonl", ns=(100, 100))
    os.utime(python_candidate / "history.jsonl", ns=(200, 200))
    monkeypatch.setenv("TMPDIR", str(env_root))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(python_root))

    assert discover_dangling_homes() == [python_candidate, env_candidate]


def test_recovery_merges_files_config_and_sqlite_with_backups(tmp_path: Path) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir()
    source.mkdir()
    (target / "config.toml").write_text(
        'model = "target-model"\n[features]\nexisting = true\n', encoding="utf-8"
    )
    (source / "config.toml").write_text(
        'model = "source-model"\n[features]\nfrom_wrap = true\n', encoding="utf-8"
    )
    os.utime(target / "config.toml", ns=(100, 100))
    os.utime(source / "config.toml", ns=(200, 200))
    rollout = source / "sessions" / "2026" / "07" / "14" / "rollout.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text('{"type":"session_meta"}\n', encoding="utf-8")
    _write_db(target / "sqlite" / "state_5.sqlite", [("target", "Target")])
    _write_db(source / "sqlite" / "state_5.sqlite", [("source", "Source")])

    report = recover_codex_home(source=source, target=target)

    config = (target / "config.toml").read_text(encoding="utf-8")
    assert 'model = "source-model"' in config
    assert "existing = true" in config
    assert "from_wrap = true" in config
    assert rollout.relative_to(source).with_name("rollout.jsonl")
    assert (target / rollout.relative_to(source)).read_text(encoding="utf-8") == (
        '{"type":"session_meta"}\n'
    )
    with sqlite3.connect(target / "sqlite" / "state_5.sqlite") as connection:
        assert connection.execute("SELECT id, title FROM threads ORDER BY id").fetchall() == [
            ("source", "Source"),
            ("target", "Target"),
        ]
    assert report.backup_dir.is_dir()
    assert (report.backup_dir / "target-before").is_dir()
    assert (report.backup_dir / "source-pinned").is_dir()
    assert (report.backup_dir / "manifest.json").is_file()


def test_recovery_relocates_thread_rollout_paths_to_durable_home(tmp_path: Path) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir()
    source.mkdir()
    relative_rollout = Path("sessions/2026/07/14/rollout-2026-07-14T10-00-00-thread-1.jsonl")
    source_rollout = source / relative_rollout
    source_rollout.parent.mkdir(parents=True)
    source_rollout.write_text('{"type":"session_meta"}\n', encoding="utf-8")
    _write_thread_db(target / "state_5.sqlite", [])
    _write_thread_db(
        source / "state_5.sqlite",
        [("thread-1", str(source_rollout), "openai")],
    )

    recover_codex_home(source=source, target=target)

    durable_rollout = target / relative_rollout
    assert durable_rollout.is_file()
    with sqlite3.connect(target / "state_5.sqlite") as connection:
        assert connection.execute(
            "SELECT rollout_path FROM threads WHERE id = 'thread-1'"
        ).fetchone() == (str(durable_rollout),)


def test_recovery_ignores_unrelated_dangling_target_thread(tmp_path: Path) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir()
    source.mkdir()
    unrelated_rollout = Path("/private/tmp/headroom-codex-home-deleted/sessions/unrelated.jsonl")
    relative_rollout = Path("sessions/2026/07/14/rollout-recovered.jsonl")
    source_rollout = source / relative_rollout
    source_rollout.parent.mkdir(parents=True)
    source_rollout.write_text('{"type":"session_meta"}\n', encoding="utf-8")
    _write_thread_db(
        target / "state_5.sqlite",
        [("unrelated", str(unrelated_rollout), "openai")],
    )
    _write_thread_db(
        source / "state_5.sqlite",
        [("recovered", str(source_rollout), "openai")],
    )

    recover_codex_home(source=source, target=target)

    with sqlite3.connect(target / "state_5.sqlite") as connection:
        rows = dict(connection.execute("SELECT id, rollout_path FROM threads"))
    assert rows == {
        "unrelated": str(unrelated_rollout),
        "recovered": str(target / relative_rollout),
    }


def test_recovery_restores_legacy_headroom_threads_to_active_provider(
    tmp_path: Path,
) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir()
    source.mkdir()
    (target / "config.toml").write_text(
        'model_provider = "azure"\n'
        "[model_providers.azure]\n"
        'base_url = "https://azure.example/v1"\n',
        encoding="utf-8",
    )
    (source / "config.toml").write_text(
        'model_provider = "headroom"\n'
        "[model_providers.headroom]\n"
        'base_url = "http://127.0.0.1:8787/v1"\n',
        encoding="utf-8",
    )
    relative_rollout = Path("sessions/2026/07/14/rollout-thread-1.jsonl")
    source_rollout = source / relative_rollout
    source_rollout.parent.mkdir(parents=True)
    source_rollout.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {
                    "id": "thread-1",
                    "model_provider": "headroom",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_thread_db(target / "state_5.sqlite", [])
    _write_thread_db(
        source / "state_5.sqlite",
        [("thread-1", str(source_rollout), "headroom")],
    )

    recover_codex_home(source=source, target=target)

    with sqlite3.connect(target / "state_5.sqlite") as connection:
        assert connection.execute(
            "SELECT model_provider FROM threads WHERE id = 'thread-1'"
        ).fetchone() == ("azure",)
    session_meta = json.loads((target / relative_rollout).read_text(encoding="utf-8"))
    assert session_meta["payload"]["model_provider"] == "azure"


def test_recovery_repairs_legacy_provider_after_a_previous_broken_recovery(
    tmp_path: Path,
) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir()
    source.mkdir()
    (target / "config.toml").write_text(
        'model_provider = "azure"\n'
        "[model_providers.azure]\n"
        'base_url = "https://azure.example/v1"\n',
        encoding="utf-8",
    )
    (source / "config.toml").write_text(
        'model_provider = "headroom"\n'
        "[model_providers.headroom]\n"
        'base_url = "http://127.0.0.1:8787/v1"\n',
        encoding="utf-8",
    )
    relative_rollout = Path("sessions/2026/06/01/rollout-thread-1.jsonl")
    source_rollout = source / relative_rollout
    target_rollout = target / relative_rollout
    source_rollout.parent.mkdir(parents=True)
    target_rollout.parent.mkdir(parents=True)
    session_meta = json.dumps(
        {
            "type": "session_meta",
            "payload": {"id": "thread-1", "model_provider": "headroom"},
        }
    )
    source_rollout.write_text(session_meta + "\n", encoding="utf-8")
    response_item = '{"type":"response_item","payload":{"text":"kept"}}'
    target_rollout.write_text(session_meta + "\n" + response_item + "\n", encoding="utf-8")
    os.utime(source_rollout, ns=(1, 1))
    os.utime(target_rollout, ns=(2, 2))
    target_db = target / "state_5.sqlite"
    source_db = source / "state_5.sqlite"
    _write_thread_db(target_db, [("thread-1", str(target_rollout), "headroom")])
    _write_thread_db(source_db, [("thread-1", str(source_rollout), "headroom")])
    os.utime(source_db, ns=(1, 1))
    os.utime(target_db, ns=(2, 2))

    recover_codex_home(source=source, target=target)
    recover_codex_home(source=source, target=target)

    with sqlite3.connect(target_db) as connection:
        assert connection.execute(
            "SELECT model_provider, rollout_path FROM threads WHERE id = 'thread-1'"
        ).fetchone() == ("azure", str(target_rollout))
    recovered_meta = json.loads(target_rollout.read_text(encoding="utf-8").splitlines()[0])
    assert recovered_meta["payload"]["model_provider"] == "azure"
    assert target_rollout.read_text(encoding="utf-8").splitlines()[1] == response_item


def test_recovery_preserves_nonlocal_provider_named_headroom(tmp_path: Path) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-remote"
    target.mkdir()
    source.mkdir()
    (target / "config.toml").write_text('model = "gpt-5"\n', encoding="utf-8")
    (source / "config.toml").write_text(
        'model_provider = "headroom"\n'
        "[model_providers.headroom]\n"
        'base_url = "https://gateway.example/v1"\n',
        encoding="utf-8",
    )
    relative_rollout = Path("archived_sessions/rollout-thread-1.jsonl")
    source_rollout = source / relative_rollout
    source_rollout.parent.mkdir(parents=True)
    source_rollout.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": "thread-1", "model_provider": "headroom"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_thread_db(target / "state_5.sqlite", [])
    _write_thread_db(
        source / "state_5.sqlite",
        [("thread-1", str(source_rollout), "headroom")],
    )

    recover_codex_home(source=source, target=target)

    recovered_meta = json.loads((target / relative_rollout).read_text(encoding="utf-8"))
    assert recovered_meta["payload"]["model_provider"] == "headroom"
    with sqlite3.connect(target / "state_5.sqlite") as connection:
        assert connection.execute(
            "SELECT model_provider FROM threads WHERE id = 'thread-1'"
        ).fetchone() == ("headroom",)


def test_recovery_rolls_back_when_sqlite_schema_differs(tmp_path: Path) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir()
    source.mkdir()
    original = 'model = "target"\n'
    (target / "config.toml").write_text(original, encoding="utf-8")
    target.chmod(0o755)
    (target / "config.toml").chmod(0o644)
    _write_db(target / "sqlite" / "state_5.sqlite", [("target", "Target")])
    source_db = source / "sqlite" / "state_5.sqlite"
    source_db.parent.mkdir(parents=True)
    with sqlite3.connect(source_db) as connection:
        connection.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, title BLOB)")

    with pytest.raises(RuntimeError, match="schema mismatch"):
        recover_codex_home(source=source, target=target)

    assert (target / "config.toml").read_text(encoding="utf-8") == original
    if os.name != "nt":
        assert stat.S_IMODE(target.stat().st_mode) == 0o755
        assert stat.S_IMODE((target / "config.toml").stat().st_mode) == 0o644
    with sqlite3.connect(target / "sqlite" / "state_5.sqlite") as connection:
        assert connection.execute("SELECT id, title FROM threads").fetchall() == [
            ("target", "Target")
        ]


def test_recovery_rollback_does_not_delete_live_target_recursively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir()
    source.mkdir()
    (target / "config.toml").write_text('model = "target"\n', encoding="utf-8")
    (source / "config.toml").write_text('model = "source"\n', encoding="utf-8")
    _write_db(target / "state_5.sqlite", [("target", "Target")])
    with sqlite3.connect(source / "state_5.sqlite") as connection:
        connection.execute("CREATE TABLE threads (id TEXT PRIMARY KEY, title BLOB)")

    def fail_recursive_delete(path: Path) -> None:
        raise OSError(errno.ENOTEMPTY, "Directory not empty", path)

    monkeypatch.setattr(codex_recovery.shutil, "rmtree", fail_recursive_delete)

    with pytest.raises(RuntimeError, match="SQLite schema mismatch"):
        recover_codex_home(source=source, target=target)

    assert (target / "config.toml").read_text(encoding="utf-8") == 'model = "target"\n'
    failed_targets = list((tmp_path / ".headroom-codex-recovery").glob("*/target-failed"))
    assert len(failed_targets) == 1


def test_recover_codex_cli_previews_then_merges(tmp_path: Path) -> None:
    home = tmp_path / "home"
    target = home / ".codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir(parents=True)
    source.mkdir()
    (source / "history.jsonl").write_text('{"session_id":"new"}\n', encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(
        main,
        ["recover", "codex", "--source", str(source), "--target", str(target), "--yes"],
        env={"HOME": str(home)},
    )

    assert result.exit_code == 0, result.output
    assert "Recovery complete" in result.output
    assert json.loads((target / "history.jsonl").read_text(encoding="utf-8"))["session_id"] == (
        "new"
    )


def test_recover_codex_cli_audits_history_without_treating_prompt_text_as_paths(
    tmp_path: Path,
) -> None:
    target = tmp_path / "codex"
    target.mkdir()
    deleted = Path("/private/tmp/headroom-codex-home-deleted")
    rollout = target / "sessions/2026/07/14/rollout-indexed.jsonl"
    rollout.parent.mkdir(parents=True)
    rollout.write_text('{"type":"session_meta"}\n', encoding="utf-8")
    _write_thread_db(
        target / "state_5.sqlite",
        [("indexed", str(rollout), "openai")],
    )
    with sqlite3.connect(target / "state_5.sqlite") as connection:
        connection.execute("ALTER TABLE threads ADD COLUMN archived INTEGER NOT NULL DEFAULT 0")
    (target / "history.jsonl").write_text(
        json.dumps({"session_id": "indexed", "text": "surviving chat"})
        + "\n"
        + json.dumps(
            {
                "session_id": "orphaned",
                "text": f"pasted error referenced {deleted}/sessions/x",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        main,
        ["recover", "codex", "--target", str(target), "--yes"],
        env={"TMPDIR": str(tmp_path / "empty-tmp")},
    )

    assert result.exit_code == 0, result.output
    assert "Referenced temporary Codex homes were already deleted:" not in result.output
    assert str(deleted) not in result.output
    assert "Durable Codex history: 1 indexed chats (1 active, 0 archived)." in result.output
    assert "History-only records without a surviving rollout: 1" in result.output
    assert "orphaned" in result.output
    assert "codex resume --all" in result.output
    assert "No recoverable Headroom Codex homes were found." in result.output


def test_recover_codex_cli_reuses_source_pinned_by_failed_recovery(tmp_path: Path) -> None:
    target = tmp_path / "codex"
    target.mkdir()
    pinned = tmp_path / ".headroom-codex-recovery" / "interrupted-attempt" / "source-pinned"
    pinned.mkdir(parents=True)
    (pinned / "history.jsonl").write_text(
        json.dumps({"session_id": "recovered", "text": "retained"}) + "\n",
        encoding="utf-8",
    )
    relative_rollout = Path("sessions/2026/07/14/rollout-retained.jsonl")
    pinned_rollout = pinned / relative_rollout
    pinned_rollout.parent.mkdir(parents=True)
    pinned_rollout.write_text('{"type":"session_meta"}\n', encoding="utf-8")
    deleted_source = Path("/private/tmp/headroom-codex-home-deleted")
    _write_thread_db(
        pinned / "state_5.sqlite",
        [("retained", str(deleted_source / relative_rollout), "openai")],
    )

    result = CliRunner().invoke(
        main,
        ["recover", "codex", "--target", str(target), "--yes"],
        env={"TMPDIR": str(tmp_path / "empty-tmp")},
    )

    assert result.exit_code == 0, result.output
    assert str(pinned) in result.output
    assert "Recovery complete." in result.output
    assert '"session_id": "recovered"' in (target / "history.jsonl").read_text(encoding="utf-8")
    with sqlite3.connect(target / "state_5.sqlite") as connection:
        assert connection.execute(
            "SELECT rollout_path FROM threads WHERE id = 'retained'"
        ).fetchone() == (str(target / relative_rollout),)


def test_recover_codex_cli_decline_changes_nothing(tmp_path: Path) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir()
    source.mkdir()
    target_history = target / "history.jsonl"
    target_history.write_text('{"session_id":"target"}\n', encoding="utf-8")
    (source / "history.jsonl").write_text('{"session_id":"source"}\n', encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["recover", "codex", "--source", str(source), "--target", str(target)],
        input="n\n",
    )

    assert result.exit_code == 0, result.output
    assert "Recovery cancelled. No Codex state was changed." in result.output
    assert target_history.read_text(encoding="utf-8") == '{"session_id":"target"}\n'
    assert not (tmp_path / ".headroom-codex-recovery").exists()


def test_recover_codex_cli_reports_malformed_config_and_removes_new_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    source.mkdir()
    (source / "config.toml").write_text("[invalid\n", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["recover", "codex", "--source", str(source), "--target", str(target), "--yes"],
    )

    assert result.exit_code != 0
    assert "Error: Codex recovery failed:" in result.output
    assert not target.exists()


def test_recovery_never_writes_through_target_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    outside = tmp_path / "outside"
    target.mkdir()
    source.mkdir()
    outside.mkdir()
    (target / "sessions").symlink_to(outside, target_is_directory=True)
    source_session = source / "sessions" / "rollout.jsonl"
    source_session.parent.mkdir()
    source_session.write_text('{"type":"session_meta"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="symlink"):
        recover_codex_home(source=source, target=target)

    assert not (outside / "rollout.jsonl").exists()
    assert (target / "sessions").is_symlink()


def test_recovery_rolls_back_when_sqlite_indexes_differ(tmp_path: Path) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir()
    source.mkdir()
    target_db = target / "sqlite" / "state_5.sqlite"
    source_db = source / "sqlite" / "state_5.sqlite"
    _write_db(target_db, [("target", "Target")])
    connection = sqlite3.connect(target_db)
    try:
        connection.execute("CREATE UNIQUE INDEX thread_title ON threads(title)")
        connection.commit()
    finally:
        connection.close()
    _write_db(source_db, [("source", "Source")])

    with pytest.raises(RuntimeError, match="schema mismatch"):
        recover_codex_home(source=source, target=target)

    with sqlite3.connect(target_db) as connection:
        assert connection.execute("SELECT id, title FROM threads").fetchall() == [
            ("target", "Target")
        ]


def test_recovery_rolls_back_when_sqlx_checksums_differ(tmp_path: Path) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir()
    source.mkdir()
    target_db = target / "sqlite" / "state_5.sqlite"
    source_db = source / "sqlite" / "state_5.sqlite"
    _write_sqlx_db(target_db, b"target-checksum")
    _write_sqlx_db(source_db, b"source-checksum")

    with pytest.raises(RuntimeError, match="migration mismatch"):
        recover_codex_home(source=source, target=target)

    with sqlite3.connect(target_db) as connection:
        assert connection.execute("SELECT version, checksum FROM _sqlx_migrations").fetchall() == [
            (1, b"target-checksum")
        ]


def test_recovery_rolls_back_when_source_sqlite_is_corrupt(tmp_path: Path) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir()
    source.mkdir()
    original_config = 'model = "target"\n'
    (target / "config.toml").write_text(original_config, encoding="utf-8")
    (source / "config.toml").write_text('model = "source"\n', encoding="utf-8")
    target_db = target / "sqlite" / "state_5.sqlite"
    _write_db(target_db, [("target", "Target")])
    source_db = source / "sqlite" / "state_5.sqlite"
    source_db.parent.mkdir(parents=True)
    source_db.write_bytes(b"not a sqlite database")

    with pytest.raises(sqlite3.DatabaseError):
        recover_codex_home(source=source, target=target)

    assert (target / "config.toml").read_text(encoding="utf-8") == original_config
    with sqlite3.connect(target_db) as connection:
        assert connection.execute("SELECT id, title FROM threads").fetchall() == [
            ("target", "Target")
        ]


def test_recovery_rolls_back_when_source_sqlite_breaks_foreign_keys(
    tmp_path: Path,
) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir()
    source.mkdir()
    target_db = target / "sqlite" / "state_5.sqlite"
    source_db = source / "sqlite" / "state_5.sqlite"
    for database in (target_db, source_db):
        database.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database) as connection:
            connection.executescript(
                "CREATE TABLE parents (id TEXT PRIMARY KEY);"
                "CREATE TABLE children ("
                "id TEXT PRIMARY KEY, parent_id TEXT REFERENCES parents(id)"
                ");"
            )
    with sqlite3.connect(target_db) as connection:
        connection.execute("INSERT INTO parents VALUES ('target-parent')")
    with sqlite3.connect(source_db) as connection:
        connection.execute("INSERT INTO children VALUES ('orphan', 'missing-parent')")

    with pytest.raises(RuntimeError, match="foreign key check failed"):
        recover_codex_home(source=source, target=target)

    with sqlite3.connect(target_db) as connection:
        assert connection.execute("SELECT id FROM parents").fetchall() == [("target-parent",)]
        assert connection.execute("SELECT id, parent_id FROM children").fetchall() == []


def test_recovery_removes_legacy_headroom_routing_from_config(tmp_path: Path) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir()
    source.mkdir()
    (target / "config.toml").write_text('model = "gpt-5"\n', encoding="utf-8")
    (source / "config.toml").write_text(
        'model_provider = "headroom"\n'
        'openai_base_url = "http://127.0.0.1:8787/v1"\n'
        "[model_providers.headroom]\n"
        'base_url = "http://127.0.0.1:8787/v1"\n'
        "[features]\nfrom_wrapped_session = true\n",
        encoding="utf-8",
    )

    recover_codex_home(source=source, target=target)

    config = (target / "config.toml").read_text(encoding="utf-8")
    assert "headroom" not in config
    assert "127.0.0.1:8787" not in config
    assert "from_wrapped_session = true" in config


def test_recovery_preserves_user_defined_remote_headroom_provider(tmp_path: Path) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir()
    source.mkdir()
    (source / "config.toml").write_text(
        'model_provider = "headroom"\n'
        "[model_providers.headroom]\n"
        'base_url = "https://gateway.example/v1"\n',
        encoding="utf-8",
    )

    recover_codex_home(source=source, target=target)

    config = (target / "config.toml").read_text(encoding="utf-8")
    assert 'model_provider = "headroom"' in config
    assert "[model_providers.headroom]" in config
    assert 'base_url = "https://gateway.example/v1"' in config


def test_recovery_quarantines_malformed_jsonl_and_keeps_valid_records(
    tmp_path: Path,
) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir()
    source.mkdir()
    (target / "history.jsonl").write_text('{"session_id":"target"}\n', encoding="utf-8")
    (source / "history.jsonl").write_text(
        '{"session_id":"source-1"}\nnot-json\n{"session_id":"source-2"}\n',
        encoding="utf-8",
    )

    report = recover_codex_home(source=source, target=target)

    recovered = [
        json.loads(line)["session_id"]
        for line in (target / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert recovered == ["target", "source-1", "source-2"]
    assert report.quarantined == [str(report.backup_dir / "source-pinned" / "history.jsonl")]
    assert "not-json" in (report.backup_dir / "quarantine" / "history.jsonl").read_text(
        encoding="utf-8"
    )


def test_recovery_keeps_newest_divergent_rollout_and_backs_up_both(
    tmp_path: Path,
) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    relative = Path("sessions/2026/07/14/rollout.jsonl")
    target_rollout = target / relative
    source_rollout = source / relative
    target_rollout.parent.mkdir(parents=True)
    source_rollout.parent.mkdir(parents=True)
    target_rollout.write_text('{"thread":"newer-target"}\n', encoding="utf-8")
    source_rollout.write_text('{"thread":"older-source"}\n', encoding="utf-8")
    os.utime(source_rollout, ns=(100, 100))
    os.utime(target_rollout, ns=(200, 200))

    report = recover_codex_home(source=source, target=target)

    assert target_rollout.read_text(encoding="utf-8") == '{"thread":"newer-target"}\n'
    assert (report.backup_dir / "source-pinned" / relative).read_text(
        encoding="utf-8"
    ) == '{"thread":"older-source"}\n'
    assert (report.backup_dir / "target-before" / relative).read_text(
        encoding="utf-8"
    ) == '{"thread":"newer-target"}\n'


@pytest.mark.skipif(
    sys.platform == "win32" or not hasattr(socket, "AF_UNIX"),
    reason="requires POSIX Unix domain sockets",
)
def test_recovery_records_sockets_and_secures_both_backups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir(mode=0o755)
    source.mkdir(mode=0o755)
    source_history = source / "history.jsonl"
    source_history.write_text('{"session_id":"source"}\n', encoding="utf-8")
    source_history.chmod(0o644)
    socket_path = source / "codex.sock"
    fifo_path = source / "codex.pipe"
    os.mkfifo(fifo_path)

    # AF_UNIX sun_path is capped (~104 bytes on macOS) and pytest's tmp_path can
    # exceed it, so bind a short RELATIVE name from inside source.
    monkeypatch.chdir(source)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as codex_socket:
        codex_socket.bind(socket_path.name)
        report = recover_codex_home(source=source, target=target)

    pinned = report.backup_dir / "source-pinned"
    target_backup = report.backup_dir / "target-before"
    assert "codex.sock" in report.skipped_runtime
    assert "codex.pipe" in report.skipped_runtime
    assert not (pinned / "codex.sock").exists()
    assert not (pinned / "codex.pipe").exists()
    assert stat.S_IMODE(report.backup_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(pinned.stat().st_mode) == 0o700
    assert stat.S_IMODE(target_backup.stat().st_mode) == 0o700
    assert stat.S_IMODE((pinned / "history.jsonl").stat().st_mode) == 0o600
    assert stat.S_IMODE((report.backup_dir / "manifest.json").stat().st_mode) == 0o600


def test_recovery_never_propagates_source_deletions(tmp_path: Path) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target_rule = target / "rules" / "user.rules"
    target_rule.parent.mkdir(parents=True)
    source.mkdir()
    target_rule.write_text("allow user setting\n", encoding="utf-8")
    (source / "history.jsonl").write_text('{"session_id":"source"}\n', encoding="utf-8")

    recover_codex_home(source=source, target=target)

    assert target_rule.read_text(encoding="utf-8") == "allow user setting\n"


@pytest.mark.parametrize(
    ("source_mtime", "target_mtime", "expected_token"),
    [(100, 200, "target-token"), (300, 200, "source-token"), (200, 200, "target-token")],
)
def test_recovery_uses_newest_credentials_with_target_winning_ties(
    tmp_path: Path,
    source_mtime: int,
    target_mtime: int,
    expected_token: str,
) -> None:
    target = tmp_path / "codex"
    source = tmp_path / "headroom-codex-home-broken"
    target.mkdir()
    source.mkdir()
    target_auth = target / "auth.json"
    source_auth = source / "auth.json"
    target_auth.write_text('{"token":"target-token"}\n', encoding="utf-8")
    source_auth.write_text('{"token":"source-token"}\n', encoding="utf-8")
    os.utime(target_auth, ns=(target_mtime, target_mtime))
    os.utime(source_auth, ns=(source_mtime, source_mtime))

    recover_codex_home(source=source, target=target)

    assert json.loads(target_auth.read_text(encoding="utf-8"))["token"] == expected_token


def test_recover_codex_cli_retains_distinct_backups_for_multiple_sources(
    tmp_path: Path,
) -> None:
    target = tmp_path / "codex"
    first = tmp_path / "headroom-codex-home-first"
    second = tmp_path / "headroom-codex-home-second"
    first.mkdir()
    second.mkdir()
    (first / "history.jsonl").write_text('{"session_id":"first"}\n', encoding="utf-8")
    (second / "history.jsonl").write_text('{"session_id":"second"}\n', encoding="utf-8")

    result = CliRunner().invoke(
        main,
        [
            "recover",
            "codex",
            "--source",
            str(first),
            "--source",
            str(second),
            "--target",
            str(target),
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.count("Recovery complete") == 2
    backup_root = tmp_path / ".headroom-codex-recovery"
    assert len([path for path in backup_root.iterdir() if path.is_dir()]) == 2
    assert [
        json.loads(line)["session_id"]
        for line in (target / "history.jsonl").read_text(encoding="utf-8").splitlines()
    ] == ["first", "second"]
