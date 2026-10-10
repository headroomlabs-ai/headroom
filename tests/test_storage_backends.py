from __future__ import annotations

import logging
import weakref
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

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

    def query(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        model: str | None = None,
        mode: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[RequestMetrics]:
        return []

    def count(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        model: str | None = None,
        mode: str | None = None,
    ) -> int:
        return 0

    def iter_all(self) -> Iterator[RequestMetrics]:
        return iter(())

    def get_summary_stats(
        self,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> dict[str, Any]:
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


def test_create_storage_builtin_entrypoint_and_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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

        def load(self) -> Callable[[str], DummyStorage]:
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


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def storage_log() -> Iterator[_ListHandler]:
    handler = _ListHandler()
    logger = logging.getLogger("headroom.storage")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(previous)


def _fallback_warnings(handler: _ListHandler) -> list[str]:
    return [
        r.getMessage()
        for r in handler.records
        if r.levelno == logging.WARNING and "falling back to SQLite" in r.getMessage()
    ]


def test_create_storage_warns_when_a_backend_plugin_fails_to_load(
    monkeypatch: pytest.MonkeyPatch, storage_log: _ListHandler
) -> None:
    monkeypatch.setattr("headroom.storage.SQLiteStorage", lambda db_path: SimpleNamespace())

    created = DummyStorage()
    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda group: [SimpleNamespace(name="custom", load=lambda: lambda url: created)],
    )
    assert create_storage("custom://user:secret@host/db") is created
    assert _fallback_warnings(storage_log) == []

    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda group: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    create_storage("custom://user:secret@host/db")
    warnings = _fallback_warnings(storage_log)
    assert len(warnings) == 1
    assert "'custom'" in warnings[0]
    assert "secret" not in warnings[0]

    monkeypatch.setattr("importlib.metadata.entry_points", lambda group: [])
    create_storage("other://host/db")
    assert any("'other'" in w for w in _fallback_warnings(storage_log))

    def _rejecting_factory(url: str) -> DummyStorage:
        raise ValueError(f"cannot connect to {url} while storing CONTENT_CANARY")

    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda group: [SimpleNamespace(name="leaky", load=lambda: _rejecting_factory)],
    )
    storage_log.records.clear()
    create_storage("leaky://user:secret@host/db")
    warning_records = [r for r in storage_log.records if r.levelno == logging.WARNING]
    assert len(warning_records) == 1
    assert "ValueError" in warning_records[0].getMessage()
    assert "secret" not in warning_records[0].getMessage()
    assert warning_records[0].exc_info is None
    # Every level, rendered the way a log handler would (exc_info included).
    assert storage_log.records
    for record in storage_log.records:
        rendered = logging.Formatter().format(record)
        assert "secret" not in rendered
        assert "CONTENT_CANARY" not in rendered
    debug = [r for r in storage_log.records if r.levelno == logging.DEBUG]
    assert debug and "ValueError" in logging.Formatter().format(debug[0])

    storage_log.records.clear()
    create_storage("metrics.db")
    assert _fallback_warnings(storage_log) == []


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
    assert [item.request_id for item in results] == ["two"]
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


def test_jsonl_query_paging_matches_sqlite_newest_first(tmp_path: Path) -> None:
    """JSONL query pages must be the newest-first window SQLite returns (issue #3822)."""
    jsonl_storage = JSONLStorage(str(tmp_path / "metrics.jsonl"))
    sqlite_storage = SQLiteStorage(str(tmp_path / "metrics.db"))
    base = datetime(2026, 4, 23, 12, 0, 0)
    rows = [
        _metrics(
            f"req-{i:03d}",
            base + timedelta(seconds=i),
            model="claude" if i % 2 == 0 else "gpt-4o",
            mode="audit" if i % 3 == 0 else "optimize",
        )
        for i in range(250)
    ]
    for row in rows:
        jsonl_storage.save(row)
        sqlite_storage.save(row)

    def ids(metrics: list[RequestMetrics]) -> list[str]:
        return [item.request_id for item in metrics]

    newest_first = [
        row.request_id for row in sorted(rows, key=lambda item: item.timestamp, reverse=True)
    ]

    for kw in (
        {"limit": 5, "offset": 0},
        {"limit": 5, "offset": 5},
        {"limit": 5, "offset": 10},
        {"model": "claude", "limit": 5, "offset": 0},
        {"model": "claude", "limit": 5, "offset": 5},
        {"mode": "audit", "limit": 4, "offset": 2},
        {"model": "claude", "mode": "audit", "limit": 3, "offset": 1},
    ):
        jsonl_page = jsonl_storage.query(**kw)
        assert ids(jsonl_page) == ids(sqlite_storage.query(**kw)), kw
        stamps = [item.timestamp for item in jsonl_page]
        assert stamps == sorted(stamps, reverse=True), kw

    assert ids(jsonl_storage.query(limit=5)) == newest_first[:5]
    assert ids(jsonl_storage.query(limit=5))[0] == "req-249"

    page1 = ids(jsonl_storage.query(limit=5, offset=0))
    page2 = ids(jsonl_storage.query(limit=5, offset=5))
    page3 = ids(jsonl_storage.query(limit=5, offset=10))
    assert set(page1).isdisjoint(page2)
    assert set(page2).isdisjoint(page3)
    assert set(page1).isdisjoint(page3)

    jsonl_storage.close()
    sqlite_storage.close()


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
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
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


def test_query_pagination_contract_validates_and_returns_empty_pages(tmp_path: Path) -> None:
    """Negative limit/offset raise ValueError; zero limit and past-end offsets page empty."""
    jsonl_storage = JSONLStorage(str(tmp_path / "metrics.jsonl"))
    sqlite_storage = SQLiteStorage(str(tmp_path / "metrics.db"))
    now = datetime(2026, 4, 23, 12, 0, 0)
    rows = [
        _metrics("one", now - timedelta(hours=2)),
        _metrics("two", now - timedelta(hours=1)),
        _metrics("three", now),
    ]
    for row in rows:
        jsonl_storage.save(row)
        sqlite_storage.save(row)

    for storage in (jsonl_storage, sqlite_storage):
        with pytest.raises(ValueError):
            storage.query(limit=-1)
        with pytest.raises(ValueError):
            storage.query(offset=-1)
        with pytest.raises(ValueError):
            storage.query(limit=-5, offset=-5)

        assert storage.query(limit=0) == []
        assert storage.query(limit=0, offset=0) == []
        assert storage.query(limit=0, offset=9) == []
        assert storage.query(limit=5, offset=3) == []
        assert storage.query(limit=5, offset=10) == []
        assert [m.request_id for m in storage.query(limit=2, offset=0)] == ["three", "two"]
        assert [m.request_id for m in storage.query(limit=2, offset=2)] == ["one"]

    jsonl_storage.close()
    sqlite_storage.close()


def test_query_equal_timestamp_pages_match_across_backends(tmp_path: Path) -> None:
    """Timestamp ties break on request_id DESC and pages agree page-for-page."""
    jsonl_storage = JSONLStorage(str(tmp_path / "metrics.jsonl"))
    sqlite_storage = SQLiteStorage(str(tmp_path / "metrics.db"))
    base = datetime(2026, 4, 23, 12, 0, 0)

    rows: list[RequestMetrics] = []
    # Rows are appended in a shuffled order on purpose: the contract orders by
    # timestamp DESC then request_id DESC, never by append order.
    for group in (1, 2, 3):
        for suffix in (3, 1, 5, 2, 4):
            rows.append(
                _metrics(
                    f"g{group}-{suffix}",
                    base + timedelta(minutes=group),
                    model="claude" if suffix % 2 == 0 else "gpt-4o",
                )
            )
    for row in rows:
        jsonl_storage.save(row)
        sqlite_storage.save(row)

    expected = [f"g{group}-{suffix}" for group in (3, 2, 1) for suffix in (5, 4, 3, 2, 1)]

    for offset in (0, 4, 8, 12):
        page_expected = expected[offset : offset + 4]
        jsonl_page = jsonl_storage.query(limit=4, offset=offset)
        sqlite_page = sqlite_storage.query(limit=4, offset=offset)
        assert [m.request_id for m in jsonl_page] == page_expected, offset
        assert [m.request_id for m in sqlite_page] == page_expected, offset
        stamps = [m.timestamp for m in jsonl_page]
        assert stamps == sorted(stamps, reverse=True)
        # The same window must be stable across repeated queries.
        assert [m.request_id for m in jsonl_storage.query(limit=4, offset=offset)] == page_expected
        assert [m.request_id for m in sqlite_storage.query(limit=4, offset=offset)] == page_expected

    # Filtered ties must stay consistent too: a page size of 3 splits the g2
    # pair across pages so the tie-breaker decides both sides of the boundary.
    filtered_expected = [f"g{group}-{suffix}" for group in (3, 2, 1) for suffix in (4, 2)]
    for offset in (0, 3):
        page_expected = filtered_expected[offset : offset + 3]
        assert [
            m.request_id for m in jsonl_storage.query(model="claude", limit=3, offset=offset)
        ] == page_expected
        assert [
            m.request_id for m in sqlite_storage.query(model="claude", limit=3, offset=offset)
        ] == page_expected

    jsonl_storage.close()
    sqlite_storage.close()


def test_jsonl_query_retains_at_most_requested_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """query() must select a bounded window instead of materializing every match."""
    storage = JSONLStorage(str(tmp_path / "metrics.jsonl"))
    base = datetime(2026, 4, 23, 12, 0, 0)
    rows = [
        _metrics(
            f"req-{i:03d}",
            base + timedelta(seconds=i),
            model="claude" if i % 2 == 0 else "gpt-4o",
        )
        for i in range(60)
    ]
    for row in rows:
        storage.save(row)

    limit, offset = 5, 10
    window = offset + limit
    matches = [row for row in rows if row.model == "claude"]
    assert len(matches) > window

    original_iter_all = storage.iter_all
    refs: list[weakref.ReferenceType[RequestMetrics]] = []
    kept: list[bool] = []
    retained: list[int] = []

    def tracked_iter_all() -> Iterator[RequestMetrics]:
        for metrics in original_iter_all():
            matched = metrics.model == "claude"
            # Count matching records yielded earlier that the query still
            # holds; the record being handed over is not in refs yet.
            retained.append(sum(1 for keep, ref in zip(kept, refs) if keep and ref() is not None))
            kept.append(matched)
            refs.append(weakref.ref(metrics))
            yield metrics

    monkeypatch.setattr(storage, "iter_all", tracked_iter_all)

    page = storage.query(model="claude", limit=limit, offset=offset)

    expected = sorted(matches, key=lambda m: (m.timestamp, m.request_id), reverse=True)
    assert [m.request_id for m in page] == [m.request_id for m in expected[offset : offset + limit]]
    assert len(refs) == len(rows)
    # Selection may hold the requested window plus at most one matching record
    # still in flight while the next record is pulled; it must never retain the
    # whole match set.
    assert max(retained) <= window + 1
    assert max(retained) < len(matches)
