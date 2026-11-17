"""Every state file that holds conversation-derived content is created owner-only.

The CCR store and the runtime log already were; the memory databases (facts
extracted from conversations plus the user id they belong to), the vector and
graph stores, the HNSW metadata dump, the licence validation cache and the
native memory directory were created at the process umask — ``0644`` on a
stock host, readable by any local account.

The fix routes them all through ``headroom.fileperms``, so these tests pin
both the primitives and every call site, under a permissive umask so the
result is known to be umask-independent.
"""

from __future__ import annotations

import os
import signal
import sqlite3
import stat
import subprocess
import sys

import pytest

from headroom import fileperms

pytestmark = pytest.mark.skipif(
    not fileperms.OWNER_ONLY_SUPPORTED, reason="POSIX permission bits only"
)


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture(autouse=True)
def _permissive_umask():
    old = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(old)


@pytest.fixture
def fail_instead_of_hang():
    def on_alarm(signum, frame):
        raise AssertionError("blocked opening a FIFO")

    old = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(5)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _record_opens(monkeypatch, target) -> list:
    real_open = os.open
    opened = []

    def record(path, flags, *args, **kwargs):
        if os.fspath(path) == str(target) and not flags & os.O_CREAT:
            opened.append(flags)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", record)
    return opened


def _report_lstat(monkeypatch, target, *, mode=None, uid=None) -> None:
    """Make ``os.lstat(target)`` report a different type/mode or owner.

    The real file is untouched; only what ``lstat`` says about it changes, which
    stands in for a swap or a foreign owner that cannot be arranged without root.
    """
    real_lstat = os.lstat

    def lstat(path, *args, **kwargs):
        st = real_lstat(path, *args, **kwargs)
        if os.fspath(path) != str(target):
            return st
        fields = list(st)
        if mode is not None:
            fields[0] = mode
        if uid is not None:
            fields[4] = uid
        return os.stat_result(fields)

    monkeypatch.setattr(os, "lstat", lstat)


# ---- primitives -------------------------------------------------------------


class TestEnsurePrivateFile:
    def test_new_file_is_private_from_birth(self, tmp_path):
        p = tmp_path / "store.db"
        fileperms.ensure_private_file(p)
        assert _mode(p) == 0o600

    def test_existing_wide_file_is_narrowed(self, tmp_path):
        p = tmp_path / "store.db"
        p.write_text("x")
        assert _mode(p) == 0o644  # precondition under the 022 umask
        fileperms.ensure_private_file(p)
        assert _mode(p) == 0o600
        assert p.read_text() == "x"

    def test_symlink_is_refused_and_target_untouched(self, tmp_path):
        target = tmp_path / "victim"
        target.write_text("SECRET")
        before = _mode(target)
        link = tmp_path / "store.db"
        link.symlink_to(target)
        with pytest.raises(PermissionError):
            fileperms.ensure_private_file(link, what="memory store")
        assert _mode(target) == before
        assert target.read_text() == "SECRET"

    def test_directory_is_refused(self, tmp_path):
        d = tmp_path / "store.db"
        d.mkdir()
        with pytest.raises(PermissionError):
            fileperms.ensure_private_file(d)

    def test_narrowing_failure_fails_closed(self, tmp_path, monkeypatch):
        p = tmp_path / "store.db"
        p.write_text("")

        def boom(fd, mode):
            raise PermissionError("cannot fchmod")

        monkeypatch.setattr(os, "fchmod", boom)
        with pytest.raises(PermissionError):
            fileperms.ensure_private_file(p)

    @pytest.mark.parametrize("mode", [0o400, 0o700, 0o4600])
    def test_non_0600_mode_is_narrowed_once_then_skipped(self, tmp_path, monkeypatch, mode):
        p = tmp_path / "store.db"
        p.write_text("x")
        p.chmod(mode)
        assert _mode(p) == mode
        opened = _record_opens(monkeypatch, p)
        fileperms.ensure_private_file(p)
        assert len(opened) == 1
        assert _mode(p) == 0o600
        fileperms.ensure_private_file(p)
        assert len(opened) == 1
        assert p.read_text() == "x"

    def test_0600_file_owned_by_the_caller_is_not_opened(self, tmp_path, monkeypatch):
        p = tmp_path / "store.db"
        p.write_text("x")
        p.chmod(0o600)
        opened = _record_opens(monkeypatch, p)
        fileperms.ensure_private_file(p)
        assert opened == []

    def test_0600_file_reported_foreign_to_a_non_root_caller_is_opened_once(
        self, tmp_path, monkeypatch
    ):
        # The file is really ours, so the fchmod that follows the open succeeds
        # and leaves it 0600. A truly foreign file would fail that fchmod with
        # EPERM and surface as PermissionError, which needs root to arrange.
        p = tmp_path / "store.db"
        p.write_text("x")
        p.chmod(0o600)
        real_uid = os.geteuid()
        non_root_uid = real_uid or 1000
        monkeypatch.setattr(os, "geteuid", lambda: non_root_uid)
        _report_lstat(monkeypatch, p, uid=non_root_uid + 1)
        opened = _record_opens(monkeypatch, p)
        fileperms.ensure_private_file(p)
        assert len(opened) == 1
        assert _mode(p) == 0o600

    def test_0600_file_of_another_owner_is_not_opened_when_running_as_root(
        self, tmp_path, monkeypatch
    ):
        # geteuid is patched to 0 and lstat reports owner uid 1, while the file
        # really belongs to the test user: root on a file of another owner.
        p = tmp_path / "store.db"
        p.write_text("x")
        p.chmod(0o600)
        monkeypatch.setattr(os, "geteuid", lambda: 0)
        _report_lstat(monkeypatch, p, uid=1)
        opened = _record_opens(monkeypatch, p)
        fileperms.ensure_private_file(p)
        assert opened == []

    def test_unreadable_wide_file_is_refused_with_context(self, tmp_path):
        if os.geteuid() == 0:
            pytest.skip("root can open any file")
        p = tmp_path / "store.db"
        p.write_text("x")
        p.chmod(0o044)
        with pytest.raises(PermissionError, match="refusing to open memory store") as info:
            fileperms.ensure_private_file(p, what="memory store")
        assert isinstance(info.value.__cause__, PermissionError)
        assert _mode(p) == 0o044

    def test_symlink_swapped_in_after_lstat_is_refused_and_target_untouched(
        self, tmp_path, monkeypatch
    ):
        target = tmp_path / "victim"
        target.write_text("SECRET")
        link = tmp_path / "store.db"
        link.symlink_to(target)
        _report_lstat(monkeypatch, link, mode=stat.S_IFREG | 0o644)
        with pytest.raises(PermissionError, match="refusing to open memory store") as info:
            fileperms.ensure_private_file(link, what="memory store")
        assert isinstance(info.value.__cause__, OSError)
        assert _mode(target) == 0o644
        assert target.read_text() == "SECRET"


class TestConnectPrivateSqlite:
    def test_creates_private_db_and_sidecars(self, tmp_path):
        p = tmp_path / "m.db"
        conn = fileperms.connect_private_sqlite(p)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE t(x)")
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
        assert _mode(p) == 0o600
        # sqlite gives the -wal/-shm sidecars the main file's mode.
        for sidecar in ("m.db-wal", "m.db-shm"):
            if (tmp_path / sidecar).exists():
                assert _mode(tmp_path / sidecar) == 0o600
        conn.close()

    @pytest.mark.parametrize(
        ("name", "uri"),
        [
            (":memory:", False),
            ("", False),
            (":memory:", True),
            ("file::memory:?cache=shared", True),
            ("file:mem?mode=memory&cache=shared", True),
        ],
    )
    def test_in_memory_databases_create_no_files(self, tmp_path, monkeypatch, name, uri):
        monkeypatch.chdir(tmp_path)
        fileperms.connect_private_sqlite(name, uri=uri).close()
        # No file literally named ":memory:" (a bug another store has shipped).
        assert sorted(p.name for p in tmp_path.iterdir()) == []

    @pytest.mark.parametrize(
        ("uri_for", "created"),
        [
            # Relative, percent-encoded, with an explicit create mode.
            (lambda d: "file:my%20memory.db?mode=rwc", "my memory.db"),
            (lambda d: f"file:{d}/abs.db", "abs.db"),
            (lambda d: f"file://localhost{d}/host.db?cache=shared", "host.db"),
        ],
        ids=["relative-percent-encoded", "absolute", "localhost-authority"],
    )
    def test_file_backed_uri_is_private(self, tmp_path, monkeypatch, uri_for, created):
        monkeypatch.chdir(tmp_path)
        conn = fileperms.connect_private_sqlite(uri_for(tmp_path), uri=True)
        conn.execute("CREATE TABLE t(x)")
        conn.commit()
        conn.close()
        # Under the forced 022 umask sqlite alone would have made this 0644.
        assert _mode(tmp_path / created) == 0o600

    def test_file_prefixed_literal_path_is_private(self, tmp_path, monkeypatch):
        # Without uri=True sqlite treats "file:..." as an ordinary file name,
        # unless it was built with SQLITE_USE_URI (Debian, python Docker images).
        monkeypatch.chdir(tmp_path)
        conn = fileperms.connect_private_sqlite("file:literal.db")
        conn.execute("CREATE TABLE t(x)")
        conn.commit()
        conn.close()
        assert [p.name for p in tmp_path.iterdir()] == [
            "literal.db" if fileperms._sqlite_always_uri() else "file:literal.db"
        ]
        assert _mode(tmp_path / os.listdir(tmp_path)[0]) == 0o600

    def test_file_prefix_is_a_uri_on_builds_that_always_parse_uris(self, monkeypatch):
        monkeypatch.setattr(fileperms, "_sqlite_always_uri", lambda: True)
        assert fileperms._sqlite_file_path("file:literal.db", uri=False) == "literal.db"
        assert fileperms._sqlite_file_path("file::memory:", uri=False) is None
        assert fileperms._sqlite_file_path("plain.db", uri=False) == "plain.db"

    def test_undecodable_uri_escape_secures_the_raw_byte_name(self, tmp_path, monkeypatch):
        # sqlite opens the raw 0xFF byte; securing U+FFFD instead would leave
        # the real database at the umask.
        monkeypatch.chdir(tmp_path)
        try:
            conn = fileperms.connect_private_sqlite("file:secret%FF.db", uri=True)
        except OSError as exc:  # e.g. APFS refuses non-UTF-8 names
            pytest.skip(f"filesystem rejects non-UTF-8 names: {exc}")
        conn.execute("CREATE TABLE t(x)")
        conn.commit()
        conn.close()
        assert os.listdir(os.fsencode(tmp_path)) == [b"secret\xff.db"]
        assert _mode(os.path.join(os.fsencode(tmp_path), b"secret\xff.db")) == 0o600

    @pytest.mark.parametrize(
        ("journal_mode", "sidecars"),
        [("wal", ("-wal", "-shm")), ("persist", ("-journal",))],
    )
    def test_crash_leftover_sidecars_are_narrowed(self, tmp_path, journal_mode, sidecars):
        # An older, unhardened run writes under the 022 umask and dies without
        # a checkpoint, leaving 0644 sidecars that sqlite would reuse as-is.
        db = tmp_path / "old.db"
        script = (
            "import os, sqlite3, sys\n"
            "c = sqlite3.connect(sys.argv[1])\n"
            f"c.execute('PRAGMA journal_mode={journal_mode}')\n"
            "c.execute('CREATE TABLE t(x)')\n"
            "c.execute(\"INSERT INTO t VALUES ('old')\")\n"
            "c.commit()\n"
            "os._exit(0)\n"
        )
        subprocess.run([sys.executable, "-c", script, str(db)], check=True)
        for suffix in sidecars:
            assert _mode(f"{db}{suffix}") == 0o644  # precondition

        conn = fileperms.connect_private_sqlite(db)
        for suffix in sidecars:
            assert _mode(f"{db}{suffix}") == 0o600  # before sqlite writes a byte
        conn.execute(f"PRAGMA journal_mode={journal_mode}")
        conn.execute("INSERT INTO t VALUES ('PRIVATE-SENTINEL')")
        conn.commit()

        for suffix in sidecars:
            assert _mode(f"{db}{suffix}") == 0o600
        if journal_mode == "wal":
            # The new row went into the reused WAL, which is now private.
            assert b"PRIVATE-SENTINEL" in (tmp_path / "old.db-wal").read_bytes()
        # Narrowing kept the leftover contents: the uncheckpointed row survives.
        assert conn.execute("SELECT x FROM t ORDER BY rowid").fetchall() == [
            ("old",),
            ("PRIVATE-SENTINEL",),
        ]
        conn.close()

    @pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
    def test_refuses_dangling_symlink_sidecar(self, tmp_path, suffix):
        db = tmp_path / "m.db"
        sqlite3.connect(db).close()
        (tmp_path / f"m.db{suffix}").symlink_to(tmp_path / "nowhere")
        with pytest.raises(PermissionError, match="not a regular file"):
            fileperms.connect_private_sqlite(db)
        assert not (tmp_path / "nowhere").exists()

    def test_0400_database_is_writable_after_connecting(self, tmp_path):
        db = tmp_path / "m.db"
        seed = sqlite3.connect(db)
        seed.execute("CREATE TABLE t(x)")
        seed.commit()
        seed.close()
        db.chmod(0o400)
        conn = fileperms.connect_private_sqlite(db)
        assert _mode(db) == 0o600
        conn.execute("INSERT INTO t VALUES (1)")
        conn.commit()
        assert conn.execute("SELECT x FROM t").fetchall() == [(1,)]
        conn.close()

    def test_missing_sidecars_are_not_precreated(self, tmp_path):
        db = tmp_path / "m.db"
        fileperms.connect_private_sqlite(db).close()
        assert [p.name for p in tmp_path.iterdir()] == ["m.db"]

    @pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
    def test_refuses_symlinked_sidecar(self, tmp_path, suffix):
        db = tmp_path / "m.db"
        sqlite3.connect(db).close()
        victim = tmp_path / "victim"
        victim.write_text("SECRET")
        (tmp_path / f"m.db{suffix}").symlink_to(victim)
        with pytest.raises(PermissionError):
            fileperms.connect_private_sqlite(db)
        assert _mode(victim) == 0o644
        assert victim.read_text() == "SECRET"

    def test_read_only_uri_does_not_create_missing_db(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(sqlite3.OperationalError):
            fileperms.connect_private_sqlite("file:missing.db?mode=ro", uri=True)
        assert not (tmp_path / "missing.db").exists()

    def test_uri_with_remote_authority_is_refused(self):
        with pytest.raises(PermissionError):
            fileperms.connect_private_sqlite("file://otherhost/srv/m.db", uri=True)

    def test_refuses_symlinked_db(self, tmp_path):
        target = tmp_path / "victim.db"
        sqlite3.connect(target).close()
        link = tmp_path / "m.db"
        link.symlink_to(target)
        with pytest.raises(PermissionError):
            fileperms.connect_private_sqlite(link)

    @pytest.mark.parametrize("suffix", ["-wal", "-shm", "-journal"])
    def test_sidecar_deleted_by_sqlite_before_narrowing_is_skipped(
        self, tmp_path, monkeypatch, suffix
    ):
        # sqlite unlinks -journal after every transaction and -wal/-shm when the
        # last connection closes, so a sidecar seen a moment ago can be gone by
        # the time it is opened. That is not a reason to refuse the store, and
        # the helper must not recreate it.
        db = tmp_path / "m.db"
        fileperms.connect_private_sqlite(db).close()
        sidecar = tmp_path / f"m.db{suffix}"
        sidecar.write_bytes(b"leftover")
        assert _mode(sidecar) == 0o644  # precondition under the 022 umask
        real_open = os.open

        def open_after_sqlite_deletes_it(path, flags, *args, **kwargs):
            if os.fspath(path) == str(sidecar) and not flags & os.O_CREAT:
                sidecar.unlink(missing_ok=True)
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", open_after_sqlite_deletes_it)
        fileperms.connect_private_sqlite(db).close()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["m.db"]

    @pytest.mark.parametrize("journal_mode", ["delete", "wal"])
    def test_connecting_keeps_other_connections_locks(self, tmp_path, journal_mode):
        # POSIX close() of any descriptor on a file releases every fcntl lock the
        # process holds on it, so narrowing must not open and close a file that
        # is already private. If it does, a second process can take the write
        # lock while this one is mid-transaction and the database is corrupted.
        db = tmp_path / "m.db"
        holder = fileperms.connect_private_sqlite(db, isolation_level=None)
        holder.execute(f"PRAGMA journal_mode={journal_mode}")
        holder.execute("CREATE TABLE t(x)")
        holder.execute("BEGIN IMMEDIATE")
        holder.execute("INSERT INTO t VALUES (1)")

        fileperms.connect_private_sqlite(db).close()

        script = (
            "import sqlite3, sys\n"
            "c = sqlite3.connect(sys.argv[1], timeout=0, isolation_level=None)\n"
            "try:\n"
            "    c.execute('BEGIN IMMEDIATE')\n"
            "    print('acquired')\n"
            "except sqlite3.OperationalError as exc:\n"
            "    print(exc)\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(db)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.stdout.strip() == "database is locked"
        holder.execute("COMMIT")
        assert holder.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        holder.close()

    @pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
    def test_owner_only_file_is_not_opened(self, tmp_path, monkeypatch, suffix):
        db = tmp_path / "m.db"
        fileperms.connect_private_sqlite(db).close()
        target = tmp_path / f"m.db{suffix}"
        if suffix:
            target.write_bytes(b"leftover")
            target.chmod(0o600)
        assert _mode(target) == 0o600
        real_open = os.open
        opened = []

        def record_opened_fds(path, flags, *args, **kwargs):
            fd = real_open(path, flags, *args, **kwargs)
            if os.fspath(path) == str(target):
                opened.append(fd)
            return fd

        monkeypatch.setattr(os, "open", record_opened_fds)
        fileperms.connect_private_sqlite(db).close()
        assert opened == []

    @pytest.mark.parametrize("suffix", ["", "-wal", "-shm", "-journal"])
    @pytest.mark.parametrize("kind", ["symlink", "fifo"])
    def test_refuses_non_regular_file(
        self, tmp_path, monkeypatch, fail_instead_of_hang, kind, suffix
    ):
        db = tmp_path / "m.db"
        if suffix:
            fileperms.connect_private_sqlite(db).close()
        bad = tmp_path / f"m.db{suffix}"
        victim = tmp_path / "victim"
        victim.write_text("SECRET")
        if kind == "symlink":
            bad.symlink_to(victim)
        else:
            os.mkfifo(bad)
        # A FIFO with no writer blocks an open for reading, so it has to be
        # refused before anything opens it; the alarm fails the test otherwise.
        opened = _record_opens(monkeypatch, bad)
        with pytest.raises(PermissionError, match="not a regular file") as info:
            fileperms.connect_private_sqlite(db, what="memory store")
        assert "refusing to open memory store" in str(info.value)
        assert str(bad) in str(info.value)
        assert opened == []
        assert _mode(victim) == 0o644
        assert victim.read_text() == "SECRET"

    def test_wide_database_is_narrowed(self, tmp_path):
        db = tmp_path / "m.db"
        seed = sqlite3.connect(db)
        seed.execute("CREATE TABLE t(x)")
        seed.execute("INSERT INTO t VALUES ('kept')")
        seed.commit()
        seed.close()
        assert _mode(db) == 0o644  # precondition under the 022 umask
        conn = fileperms.connect_private_sqlite(db)
        assert _mode(db) == 0o600
        assert conn.execute("SELECT x FROM t").fetchall() == [("kept",)]
        conn.close()


class TestPrivateDir:
    def test_new_and_existing_dirs_are_0700(self, tmp_path):
        new = tmp_path / "a" / "memories"
        fileperms.private_dir(new)
        assert _mode(new) == 0o700
        old = tmp_path / "old"
        old.mkdir()
        assert _mode(old) == 0o755
        fileperms.private_dir(old)
        assert _mode(old) == 0o700

    def test_symlinked_dir_is_not_re_permissioned(self, tmp_path):
        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "memories"
        link.symlink_to(target)
        fileperms.private_dir(link)
        assert _mode(target) == 0o755


class TestOpenOwnerOnlyWriteMode:
    def test_write_mode_truncates_and_is_private(self, tmp_path):
        p = tmp_path / "cache.json"
        p.write_text("old-and-long")
        assert _mode(p) == 0o644
        with fileperms.open_owner_only(p, "w") as fh:
            fh.write("new")
        assert p.read_text() == "new"
        assert _mode(p) == 0o600

    def test_append_mode_still_appends(self, tmp_path):
        p = tmp_path / "log"
        with fileperms.open_owner_only(p, "a") as fh:
            fh.write("a")
        with fileperms.open_owner_only(p, "a") as fh:
            fh.write("b")
        assert p.read_text() == "ab"

    def test_read_modes_are_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            fileperms.open_owner_only(tmp_path / "x", "r")


# ---- call sites -------------------------------------------------------------


class TestMemoryStoresArePrivate:
    def test_sqlite_memory_store(self, tmp_path):
        from headroom.memory.adapters.sqlite import SQLiteMemoryStore

        p = tmp_path / "headroom_memory.db"
        SQLiteMemoryStore(p)
        assert _mode(p) == 0o600

    def test_sqlite_memory_store_narrows_existing_db(self, tmp_path):
        from headroom.memory.adapters.sqlite import SQLiteMemoryStore

        p = tmp_path / "headroom_memory.db"
        sqlite3.connect(p).close()  # an older, unhardened run left it 0644
        assert _mode(p) == 0o644
        SQLiteMemoryStore(p)
        assert _mode(p) == 0o600

    def test_fts5_index(self, tmp_path):
        from headroom.memory.adapters.fts5 import FTS5TextIndex

        p = tmp_path / "headroom_memory.db"
        try:
            FTS5TextIndex(p)
        except sqlite3.OperationalError as exc:  # sqlite built without FTS5
            pytest.skip(f"FTS5 unavailable: {exc}")
        assert _mode(p) == 0o600

    def test_graph_store(self, tmp_path):
        from headroom.memory.adapters.sqlite_graph import SQLiteGraphStore

        p = tmp_path / "headroom_graph.db"
        SQLiteGraphStore(p)
        assert _mode(p) == 0o600

    def test_vector_index(self, tmp_path):
        pytest.importorskip("sqlite_vec")
        from headroom.memory.adapters.sqlite_vector import SQLiteVectorIndex

        p = tmp_path / "vectors.db"
        try:
            SQLiteVectorIndex(dimension=8, db_path=p)
        except (ImportError, RuntimeError) as exc:
            pytest.skip(f"sqlite-vec unavailable here: {exc}")
        assert _mode(p) == 0o600

    @staticmethod
    def _hnsw_index():
        # Never ``importorskip("hnswlib")`` here: on runners without AVX the
        # native import dies with SIGILL and takes the whole pytest shard with
        # it. The adapter's probe runs the import in a subprocess.
        import asyncio

        from headroom.memory.adapters.hnsw import _check_hnswlib_available

        if not _check_hnswlib_available():
            pytest.skip("hnswlib not available (or not usable on this CPU)")

        import numpy as np

        from headroom.memory.adapters.hnsw import HNSWVectorIndex
        from headroom.memory.models import Memory

        index = HNSWVectorIndex(dimension=4, max_elements=16)
        mem = Memory(content="secret fact", user_id="alice", embedding=np.ones(4, dtype=np.float32))
        asyncio.run(index.index(mem))
        return index

    def test_hnsw_index_and_metadata_files(self, tmp_path):
        index = self._hnsw_index()
        base = tmp_path / "idx"
        index.save_index(base)
        assert _mode(base.with_suffix(".hnsw")) == 0o600
        assert _mode(base.with_suffix(".meta")) == 0o600
        # A re-save over a file left wide by an older version narrows it.
        os.chmod(base.with_suffix(".hnsw"), 0o640)
        index.save_index(base)
        assert _mode(base.with_suffix(".hnsw")) == 0o600

    def test_hnsw_index_refuses_planted_symlink(self, tmp_path):
        index = self._hnsw_index()
        target = tmp_path / "elsewhere.bin"
        target.write_bytes(b"untouched")
        base = tmp_path / "idx"
        base.with_suffix(".hnsw").symlink_to(target)
        with pytest.raises(PermissionError):
            index.save_index(base)
        assert target.read_bytes() == b"untouched"


def test_native_memory_dir_is_0700(tmp_path):
    from headroom.proxy.memory_handler import MemoryConfig, MemoryHandler

    d = tmp_path / "memories"
    d.mkdir()
    assert _mode(d) == 0o755
    handler = MemoryHandler(MemoryConfig(native_memory_dir=str(d)))
    handler._init_native_memory_dir()  # what initialize() runs for the native tool
    assert _mode(d) == 0o700


def test_license_cache_is_private(tmp_path):
    from headroom.telemetry.reporter import LicenseInfo, UsageReporter

    cache = tmp_path / "license.json"
    cache.write_text("{}")
    assert _mode(cache) == 0o644
    reporter = UsageReporter("hlk_test", cache_path=cache)
    reporter._license_info = LicenseInfo(status="active")
    reporter._save_cache()
    assert _mode(cache) == 0o600
    assert "active" in cache.read_text()


def test_ccr_backend_still_private_via_shared_primitive(tmp_path):
    from headroom.cache.backends.sqlite import SQLiteBackend

    p = tmp_path / "ccr_store.db"
    SQLiteBackend(p)
    assert _mode(p) == 0o600
