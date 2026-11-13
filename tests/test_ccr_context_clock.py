"""Conversation age must share the compression store's lifetime and durability."""

import concurrent.futures
import json
import threading

import pytest

from headroom.cache.backends import InMemoryBackend, SQLiteBackend
from headroom.cache.compression_store import CompressionStore


@pytest.fixture(params=["memory", "sqlite"])
def backend(request, tmp_path):
    if request.param == "sqlite":
        return SQLiteBackend(db_path=tmp_path / "context.sqlite")
    return InMemoryBackend()


def _observe(backend, conversation, hashes):
    observe = getattr(backend, "observe_context_turn", None)
    assert callable(observe), "Backend must retain conversation clock independently of payloads"
    return observe(conversation, hashes)


@pytest.mark.parametrize("raw", ["{broken", '["not a clock"]'])
def test_sqlite_unrelated_corrupt_namespace_does_not_block_fallback(tmp_path, raw):
    path = tmp_path / "namespaces.sqlite"
    backend = SQLiteBackend(db_path=path)
    store = CompressionStore(backend=backend)
    key = store.store("original", "sample")
    other = store.store("other original", "other sample")
    assert backend.observe_context_turn("A", [other], namespace_key="workspace-X")
    assert backend.observe_context_turn("B", [key], namespace_key="workspace-Y")
    backend._conn.execute(
        "UPDATE ccr_context_states SET state_json=? WHERE conversation_key='A'", (raw,)
    )
    backend._conn.commit()

    reopened = SQLiteBackend(db_path=path)
    snapshot = reopened.observe_context_turn("B-trimmed", [key], namespace_key="workspace-Y")
    assert snapshot is not None
    assert snapshot.current_turn == 2
    assert snapshot.compression_turns[key][1] == 1
    assert CompressionStore(backend=reopened).retrieve(key).original_content == "original"


@pytest.mark.parametrize("raw", ["{broken", '["not a clock"]', '{"namespace":"workspace-Y"}'])
def test_sqlite_corrupt_candidate_in_same_namespace_fails_closed(tmp_path, raw):
    backend = SQLiteBackend(db_path=tmp_path / "same-namespace.sqlite")
    store = CompressionStore(backend=backend)
    key = store.store("original", "sample")
    assert backend.observe_context_turn("A", [key], namespace_key="workspace-X")
    backend._conn.execute(
        "UPDATE ccr_context_states SET state_json=? WHERE conversation_key='A'", (raw,)
    )
    backend._conn.commit()
    assert backend.observe_context_turn("A-trimmed", [key], namespace_key="workspace-X") is None
    assert store.retrieve(key).original_content == "original"


def test_sqlite_concurrent_namespace_migration_preserves_trimmed_history_age(tmp_path):
    path = tmp_path / "legacy-namespaces.sqlite"
    backend = SQLiteBackend(db_path=path)
    key = CompressionStore(backend=backend).store("original", "sample")
    for _ in range(5):
        assert backend.observe_context_turn("origin", [key], namespace_key="workspace-X")
    row = backend._conn.execute(
        "SELECT conversation_key, state_json, expires_at FROM ccr_context_states"
    ).fetchone()
    backend._conn.execute("DROP TABLE ccr_context_states")
    backend._conn.execute(
        "CREATE TABLE ccr_context_states "
        "(conversation_key TEXT PRIMARY KEY, state_json TEXT NOT NULL, expires_at REAL NOT NULL)"
    )
    backend._conn.execute("INSERT INTO ccr_context_states VALUES (?, ?, ?)", row)
    backend._conn.commit()
    backend._conn.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        migrated = list(executor.map(lambda _: SQLiteBackend(db_path=path), range(2)))
    for index, handle in enumerate(migrated):
        snapshot = handle.observe_context_turn("trimmed", [key], namespace_key="workspace-X")
        assert snapshot is not None
        assert snapshot.current_turn == 6 + index
        assert snapshot.compression_turns[key][1] == 1
    assert CompressionStore(backend=migrated[0]).retrieve(key).original_content == "original"


def test_shared_hash_has_independent_first_turn_in_each_conversation(backend):
    store = CompressionStore(backend=backend)
    key = store.store("shared original", "shared sample")
    other = store.store("other original", "other sample")
    first = _observe(backend, "A", [key])
    assert first.current_turn == 1
    assert first.compression_turns[key][1] == 1
    for _ in range(4):
        _observe(backend, "B", [other])
    shared_in_b = _observe(backend, "B", [key])
    assert shared_in_b.current_turn == 5
    assert shared_in_b.compression_turns[key][1] == 5
    again = _observe(backend, "A", [key])
    assert again.current_turn == 2
    assert again.compression_turns[key][1] == 1
    assert store.get_stats()["entry_count"] == 2


def test_missing_marker_does_not_reset_original_event_turn(backend):
    store = CompressionStore(backend=backend)
    key = store.store("original", "sample")
    other = store.store("other original", "other sample")
    _observe(backend, "A", [key])
    assert _observe(backend, "A", ["not-owned"]) is None
    for _ in range(4):
        _observe(backend, "A", [other])
    again = _observe(backend, "A", [key])
    assert again.current_turn == 6
    assert again.compression_turns[key][1] == 1


def test_fresh_same_hash_recompression_has_new_first_turn(backend):
    store = CompressionStore(backend=backend)
    key = store.store("original", "sample")
    before = _observe(backend, "A", [key])
    for _ in range(4):
        _observe(backend, "A", [key])
    assert store.store("original", "fresh sample") == key
    after = _observe(backend, "A", [key])
    assert after.current_turn == 6
    assert after.compression_turns[key][1] == 6
    assert after.compression_turns[key][0] != before.compression_turns[key][0]


def test_sqlite_second_handle_preserves_clock_and_event_age(tmp_path):
    path = tmp_path / "context.sqlite"
    first = SQLiteBackend(db_path=path)
    store = CompressionStore(backend=first)
    key = store.store("original", "sample")
    for _ in range(5):
        _observe(first, "A", [key])
    reopened = SQLiteBackend(db_path=path)
    snapshot = _observe(reopened, "A", [key])
    assert snapshot.current_turn == 6
    assert snapshot.compression_turns[key][1] == 1
    assert CompressionStore(backend=reopened).retrieve(key).original_content == "original"


def test_sqlite_concurrent_handles_do_not_lose_turns(tmp_path):
    path = tmp_path / "context.sqlite"
    backends = [SQLiteBackend(db_path=path), SQLiteBackend(db_path=path)]
    key = CompressionStore(backend=backends[0]).store("original", "sample")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(_observe, backends[index % 2], "A", [key]) for index in range(12)
        ]
        snapshots = [future.result() for future in futures]
    assert sorted(snapshot.current_turn for snapshot in snapshots) == list(range(1, 13))
    assert {snapshot.compression_turns[key][1] for snapshot in snapshots} == {1}
    assert _observe(SQLiteBackend(db_path=path), "A", [key]).current_turn == 13


def test_sqlite_concurrent_same_hash_stores_have_distinct_events(tmp_path, monkeypatch):
    path = tmp_path / "context.sqlite"
    backends = [SQLiteBackend(db_path=path), SQLiteBackend(db_path=path)]
    monkeypatch.setattr("headroom.cache.compression_store.time.time", lambda: 1000.0)
    key = CompressionStore(backend=backends[0]).store("original", "sample")
    stores = [CompressionStore(backend=backend) for backend in backends]
    barrier = threading.Barrier(2)
    timestamps = []
    for backend in backends:
        get = backend.get
        write_name = "set_new" if hasattr(backend, "set_new") else "set"
        write = getattr(backend, write_name)

        def synchronized_get(hash_key, get=get):
            entry = get(hash_key)
            barrier.wait(timeout=10)
            return entry

        def capture_write(hash_key, entry, write=write):
            write(hash_key, entry)
            timestamps.append(entry.created_at)

        monkeypatch.setattr(backend, "get", synchronized_get)
        monkeypatch.setattr(backend, write_name, capture_write)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(store.store, "original", "fresh") for store in stores]
        assert [future.result() for future in futures] == [key, key]
    assert len(set(timestamps)) == 2


def test_corrupt_retained_clock_skips_expansion_without_losing_payload(backend):
    store = CompressionStore(backend=backend)
    key = store.store("original", "sample")
    _observe(backend, "A", [key])
    if isinstance(backend, SQLiteBackend):
        backend._conn.execute(
            "UPDATE ccr_context_states SET state_json = ? WHERE conversation_key = ?",
            ('{"version":1,"turn":-1,"events":{}}', "A"),
        )
        backend._conn.commit()
    else:
        state, expiry = backend._context_states["A"]
        state["turn"] = -1
        backend._context_states["A"] = (state, expiry)
    assert _observe(backend, "A", [key]) is None
    assert store.retrieve(key).original_content == "original"


def test_expired_clock_is_cleaned_and_fresh_payload_gets_new_anchor(backend, monkeypatch):
    monkeypatch.setattr("headroom.cache.compression_store.time.time", lambda: 1000.0)
    store = CompressionStore(backend=backend)
    key = store.store("original", "sample", ttl=1)
    _observe(backend, "A", [key])
    monkeypatch.setattr("headroom.cache.compression_store.time.time", lambda: 1002.0)
    assert _observe(backend, "A", [key]) is None
    store.store("original", "fresh", ttl=10)
    snapshot = _observe(backend, "A", [key])
    assert snapshot.current_turn == 1
    assert snapshot.compression_turns[key][1] == 1


def test_recreated_evicted_hash_at_same_timestamp_has_fresh_event(backend, monkeypatch):
    monkeypatch.setattr("headroom.cache.compression_store.time.time", lambda: 1000.0)
    store = CompressionStore(backend=backend)
    key = store.store("original", "sample")
    first = _observe(backend, "A", [key])
    for _ in range(4):
        _observe(backend, "A", [key])
    assert backend.delete(key)
    assert store.store("original", "fresh") == key
    recreated = _observe(backend, "A", [key])
    assert recreated.compression_event_ids[key] != first.compression_event_ids[key]
    assert recreated.compression_turns[key][1] == recreated.current_turn


def test_version_one_clock_upgrade_preserves_legacy_event_age(backend):
    store = CompressionStore(backend=backend)
    key = store.store("original", "sample")
    legacy = backend.get(key)
    legacy.event_id = ""
    legacy.created_at = int(legacy.created_at)
    backend.set(key, legacy)
    for _ in range(5):
        _observe(backend, "A", [key])
    if isinstance(backend, SQLiteBackend):
        raw = backend._conn.execute(
            "SELECT state_json FROM ccr_context_states WHERE conversation_key = 'A'"
        ).fetchone()[0]
        state = json.loads(raw)
    else:
        state, expiry = backend._context_states["A"]
    state["version"] = 1
    for anchor in state["events"].values():
        anchor.pop("event_id")
    if isinstance(backend, SQLiteBackend):
        backend._conn.execute(
            "UPDATE ccr_context_states SET state_json = ? WHERE conversation_key = 'A'",
            (json.dumps(state),),
        )
        backend._conn.commit()
    else:
        backend._context_states["A"] = (state, expiry)
    upgraded = _observe(backend, "A", [key])
    assert upgraded.current_turn == 6
    assert upgraded.compression_turns[key][1] == 1
    assert upgraded.compression_event_ids[key] == legacy.event_identity


def test_fallback_lineage_survives_origin_changes_and_keeps_other_workspace_independent(backend):
    store = CompressionStore(backend=backend)
    key = store.store("original", "sample")
    for _ in range(5):
        backend.observe_context_turn("original-origin", [key], namespace_key="workspace-a")
    trimmed = backend.observe_context_turn("trimmed-origin", [key], namespace_key="workspace-a")
    assert trimmed.current_turn == 6
    assert trimmed.compression_turns[key][1] == 1
    restored = backend.observe_context_turn("original-origin", [key], namespace_key="workspace-a")
    assert restored.current_turn == 7
    assert restored.compression_turns[key][1] == 1
    independent = backend.observe_context_turn("other-origin", [key], namespace_key="workspace-b")
    assert independent.current_turn == 1
    assert independent.compression_turns[key][1] == 1


def test_ambiguous_fallback_lineage_skips_expansion_but_preserves_retrieval(backend):
    store = CompressionStore(backend=backend)
    key = store.store("original", "sample")
    # Seed independently known origins, as retained by an earlier deployment.
    _observe(backend, "A", [key])
    _observe(backend, "B", [key])
    backend.observe_context_turn("A", [key], namespace_key="workspace")
    backend.observe_context_turn("B", [key], namespace_key="workspace")
    assert backend.observe_context_turn("unknown", [key], namespace_key="workspace") is None
    assert backend.observe_context_turn("A", [key], namespace_key="workspace") is None
    assert store.retrieve(key).original_content == "original"
    known = backend.observe_context_turn("A", [key])
    assert known.current_turn == 3
    assert known.compression_turns[key][1] == 1


def test_delayed_retrieval_cannot_overwrite_new_store_event_identity(backend, monkeypatch):
    reader = CompressionStore(backend=backend)
    writer = CompressionStore(backend=backend)
    key = writer.store("original", "sample")
    read_started = threading.Event()
    fresh_written = threading.Event()
    get = backend.get

    def paused_get(hash_key):
        entry = get(hash_key)
        if threading.current_thread() is not threading.main_thread():
            read_started.set()
            assert fresh_written.wait(timeout=10)
        return entry

    monkeypatch.setattr(backend, "get", paused_get)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(reader.retrieve, key)
        assert read_started.wait(timeout=10)
        try:
            writer.store("original", "fresh")
            fresh_id = writer.get_metadata(key)["event_id"]
        finally:
            fresh_written.set()
        assert future.result().original_content == "original"
    assert writer.get_metadata(key)["event_id"] == fresh_id


def test_sqlite_concurrent_access_merges_metadata_without_lost_retrievals(tmp_path, monkeypatch):
    path = tmp_path / "context.sqlite"
    backends = [SQLiteBackend(db_path=path), SQLiteBackend(db_path=path)]
    stores = [CompressionStore(backend=backend) for backend in backends]
    key = stores[0].store("original", "sample")
    barrier = threading.Barrier(2)
    for backend in backends:
        get = backend.get

        def synchronized_get(hash_key, get=get):
            entry = get(hash_key)
            barrier.wait(timeout=10)
            return entry

        monkeypatch.setattr(backend, "get", synchronized_get)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(store.retrieve, key, query=query)
            for store, query in zip(stores, ["alpha", "beta"], strict=True)
        ]
        assert all(future.result().original_content == "original" for future in futures)
    current = SQLiteBackend(db_path=path).get(key)
    assert current.retrieval_count == 2
    assert set(current.search_queries) == {"alpha", "beta"}
