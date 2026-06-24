from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from headroom.config import RequestMetrics
from headroom.storage import JSONLStorage, SQLiteStorage, Storage, create_storage


def _metrics(
    request_id: str,
    timestamp: datetime,
    model: str = "gpt-4o",
    mode: str = "audit",
    before: int = 100,
    after: int = 80,
) -> RequestMetrics:
    return RequestMetrics(
        request_id=request_id,
        timestamp=timestamp,
        model=model,
        stream=False,
        mode=mode,
        tokens_input_before=before,
        tokens_input_after=after,
        tokens_output=25,
        block_breakdown={"system": 10},
        waste_signals={"json_bloat": 3},
        stable_prefix_hash="prefix",
        cache_alignment_score=75.0,
        cached_tokens=12,
        transforms_applied=["compress"],
        tool_units_dropped=1,
        turns_dropped=2,
        messages_hash="messages",
        error=None,
    )


@dataclass
class DummyStorage(Storage):
    closed: bool = False

    def save(self, metrics: RequestMetrics) -> None:
        return None

    def get(self, request_id: str) -> RequestMetrics | None:
        return None

    def query(self, **kwargs) -> list[RequestMetrics]:
        return []

    def count(self, **kwargs) -> int:
        return 0

    def iter_all(self):
        return iter(())

    def get_summary_stats(self, **kwargs) -> dict[str, int]:
        return {}

    def close(self) -> None:
        self.closed = True


def test_storage_base_context_manager_calls_close() -> None:
    storage = DummyStorage()
    with storage as managed:
        assert managed is storage
        assert storage.closed is False
    assert storage.closed is True

    assert Storage.save(storage, _metrics("x", datetime(2026, 4, 23, 12, 0, 0))) is None
    assert Storage.get(storage, "x") is None
    assert Storage.query(storage) is None
    assert Storage.count(storage) is None
    assert Storage.iter_all(storage) is None
    assert Storage.get_summary_stats(storage) is None
    assert Storage.close(storage) is None


def test_create_storage_builtin_entrypoint_and_fallback(monkeypatch, tmp_path: Path) -> None:
    sqlite_storage = create_storage(f"sqlite://{tmp_path}\\metrics.db")
    jsonl_storage = create_storage(f"jsonl://{tmp_path}\\metrics.jsonl")
    assert isinstance(sqlite_storage, SQLiteStorage)
    assert isinstance(jsonl_storage, JSONLStorage)
    sqlite_storage.close()
    jsonl_storage.close()

    absolute_sqlite = create_storage("sqlite:///tmp/demo.db")
    absolute_jsonl = create_storage("jsonl:///tmp/demo.jsonl")
    assert isinstance(absolute_sqlite, SQLiteStorage)
    assert isinstance(absolute_jsonl, JSONLStorage)
    absolute_sqlite.close()
    absolute_jsonl.close()

    created = DummyStorage()

    class FakeEntryPoint:
        name = "custom"

        def load(self):
            return lambda store_url: created

    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda group: [FakeEntryPoint()] if group == "headroom.storage_backend" else [],
    )
    assert create_storage("custom://memory") is created

    monkeypatch.setattr(
        "importlib.metadata.entry_points", lambda group: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    created_fallback: list[str] = []

    class FakeSQLiteStorage:
        def __init__(self, db_path: str) -> None:
            created_fallback.append(db_path)

        def close(self) -> None:
            return None

    monkeypatch.setattr("headroom.storage.SQLiteStorage", FakeSQLiteStorage)
    fallback = create_storage("custom://fallback.db")
    assert isinstance(fallback, FakeSQLiteStorage)
    assert created_fallback == ["custom://fallback.db"]
    fallback.close()

    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda group: [SimpleNamespace(name="other", load=lambda: lambda url: created)],
    )
    missing_ep = create_storage("custom://missing.db")
    assert isinstance(missing_ep, FakeSQLiteStorage)
    assert created_fallback == ["custom://fallback.db", "custom://missing.db"]
    missing_ep.close()

    plain = create_storage("metrics.db")
    assert isinstance(plain, FakeSQLiteStorage)
    assert created_fallback == ["custom://fallback.db", "custom://missing.db", "metrics.db"]
    plain.close()


def test_jsonl_storage_round_trip_query_count_and_summary(tmp_path: Path) -> None:
    storage = JSONLStorage(str(tmp_path / "metrics.jsonl"))
    now = datetime(2026, 4, 23, 12, 0, 0)
    first = _metrics("one", now - timedelta(hours=2), mode="audit", before=120, after=100)
    second = _metrics(
        "two", now - timedelta(hours=1), model="claude", mode="optimize", before=90, after=30
    )
    third = _metrics("three", now, mode="audit", before=60, after=50)

    storage.save(first)
    storage.save(second)
    storage.save(third)

    assert storage.get("two") == second
    assert storage.get("missing") is None

    results = storage.query(start_time=now - timedelta(hours=1, minutes=30), offset=1, limit=1)
    assert [item.request_id for item in results] == ["three"]
    assert storage.query(model="claude")[0].request_id == "two"
    assert storage.query(mode="optimize")[0].request_id == "two"
    assert storage.query(end_time=now - timedelta(hours=1, minutes=30))[0].request_id == "one"
    assert storage.count(mode="audit") == 2
    assert storage.count(end_time=now - timedelta(hours=1, minutes=30)) == 1
    assert storage.count(start_time=now + timedelta(days=1)) == 0

    summary = storage.get_summary_stats(start_time=now - timedelta(hours=3), end_time=now)
    assert summary == {
        "total_requests": 3,
        "total_tokens_before": 270,
        "total_tokens_after": 180,
        "total_tokens_saved": 90,
        "avg_tokens_saved": 30.0,
        "avg_cache_alignment": 75.0,
        "audit_count": 2,
        "optimize_count": 1,
    }

    storage.close()


def test_jsonl_storage_handles_missing_file_malformed_lines_and_defaults(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    storage = JSONLStorage(str(path))
    path.unlink()
    assert list(storage.iter_all()) == []

    path.write_text(
        "\n".join(
            [
                "",
                "not-json",
                '{"id":"x","timestamp":"2026-04-23T12:00:00Z","model":"gpt-4o","stream":true,"mode":"simulate","tokens_input_before":5,"tokens_input_after":3}',
            ]
        )
    )
    loaded = list(storage.iter_all())
    assert len(loaded) == 1
    assert loaded[0].request_id == "x"
    assert loaded[0].tokens_output is None
    assert loaded[0].block_breakdown == {}
    assert loaded[0].waste_signals == {}
    assert loaded[0].stable_prefix_hash == ""
    assert loaded[0].cache_alignment_score == 0.0
    assert loaded[0].transforms_applied == []
    assert loaded[0].tool_units_dropped == 0
    assert loaded[0].turns_dropped == 0
    assert loaded[0].messages_hash == ""
    assert loaded[0].error is None


def test_sqlite_storage_round_trip_filters_summary_and_defaults(tmp_path: Path) -> None:
    storage = SQLiteStorage(str(tmp_path / "metrics.db"))
    now = datetime(2026, 4, 23, 12, 0, 0)
    first = _metrics("one", now - timedelta(hours=2), mode="audit", before=100, after=70)
    second = _metrics(
        "two", now - timedelta(hours=1), model="claude", mode="optimize", before=90, after=20
    )
    third = _metrics("three", now, before=50, after=50)
    third.stable_prefix_hash = ""
    third.cache_alignment_score = 0.0
    third.cached_tokens = None
    third.transforms_applied = []
    third.tool_units_dropped = 0
    third.turns_dropped = 0
    third.messages_hash = ""

    storage.save(first)
    storage.save(second)
    storage.save(third)
    replacement = _metrics("one", now + timedelta(minutes=1), before=111, after=11)
    storage.save(replacement)

    assert storage.get("one") == replacement
    assert storage.get("missing") is None

    results = storage.query(start_time=now - timedelta(hours=2), end_time=now, limit=2, offset=1)
    assert [item.request_id for item in results] == ["two"]
    assert storage.query(model="claude")[0].request_id == "two"
    assert storage.query(mode="optimize")[0].request_id == "two"
    assert storage.count(mode="audit") == 2
    assert storage.count(start_time=now - timedelta(hours=1, minutes=30), end_time=now) == 2
    assert storage.count(model="missing") == 0
    assert [item.request_id for item in storage.iter_all()] == ["two", "three", "one"]

    summary = storage.get_summary_stats(
        start_time=now - timedelta(hours=3), end_time=now + timedelta(hours=1)
    )
    assert summary == {
        "total_requests": 3,
        "total_tokens_before": 251,
        "total_tokens_after": 81,
        "total_tokens_saved": 170,
        "avg_tokens_saved": 56.666666666666664,
        "avg_cache_alignment": 50.0,
        "audit_count": 2,
        "optimize_count": 1,
    }

    empty = storage.get_summary_stats(start_time=now + timedelta(days=1))
    assert empty == {
        "total_requests": 0,
        "total_tokens_before": 0,
        "total_tokens_after": 0,
        "total_tokens_saved": 0,
        "avg_tokens_saved": 0,
        "avg_cache_alignment": 0,
        "audit_count": 0,
        "optimize_count": 0,
    }

    storage.close()
    assert storage._conn is None


def test_sqlite_storage_get_conn_reuses_connection_and_create_storage_entrypoint(
    monkeypatch, tmp_path: Path
) -> None:
    storage = SQLiteStorage(str(tmp_path / "metrics.db"))
    first = storage._get_conn()
    second = storage._get_conn()
    assert first is second
    storage.close()

    created = DummyStorage()
    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda group: [SimpleNamespace(name="custom", load=lambda: lambda url: created)],
    )
    assert create_storage("custom://db") is created
