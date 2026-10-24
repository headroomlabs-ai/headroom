"""Tests for JSONL storage query paging (shared newest-first contract)."""

from datetime import datetime, timedelta

import pytest

from headroom.config import RequestMetrics
from headroom.storage.jsonl import JSONLStorage
from headroom.storage.sqlite import SQLiteStorage


def _metrics(
    request_id: str,
    timestamp: datetime,
    model: str = "gpt-4o",
    mode: str = "audit",
) -> RequestMetrics:
    return RequestMetrics(
        request_id=request_id,
        timestamp=timestamp,
        model=model,
        stream=False,
        mode=mode,
        tokens_input_before=1000,
        tokens_input_after=800,
        tokens_output=200,
        block_breakdown={"system": 100},
        waste_signals={"json_bloat": 5},
        stable_prefix_hash=f"prefix-{request_id}",
        cache_alignment_score=80.0,
        cached_tokens=50,
        transforms_applied=["SmartCrusher"],
        tool_units_dropped=1,
        turns_dropped=0,
        messages_hash=f"messages-{request_id}",
        error=None,
    )


def _expected_page(
    rows: list[RequestMetrics],
    *,
    model: str | None = None,
    mode: str | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[str]:
    """Reference contract: newest-first ordering, then the [offset:offset+limit] page."""
    filtered = [
        row
        for row in rows
        if (start_time is None or row.timestamp >= start_time)
        and (end_time is None or row.timestamp <= end_time)
        and (model is None or row.model == model)
        and (mode is None or row.mode == mode)
    ]
    ordered = sorted(filtered, key=lambda row: row.timestamp, reverse=True)
    return [row.request_id for row in ordered[offset : offset + limit]]


def _ids(metrics: list[RequestMetrics]) -> list[str]:
    return [m.request_id for m in metrics]


def _assert_timestamp_desc(metrics: list[RequestMetrics]) -> None:
    stamps = [m.timestamp for m in metrics]
    assert stamps == sorted(stamps, reverse=True)


class TestJSONLQueryPagingContract:
    """JSONLStorage.query() must page like SQLiteStorage.query() (issue #3822).

    Rows are appended in timestamp order, so paging before ordering returns the
    oldest remaining rows per page. The contract requires the filtered set to be
    ordered newest-first first and only then sliced to [offset:offset+limit].
    """

    @pytest.fixture
    def backends(self, tmp_path):
        jsonl = JSONLStorage(str(tmp_path / "metrics.jsonl"))
        sqlite = SQLiteStorage(str(tmp_path / "metrics.db"))
        base = datetime(2025, 1, 6, 12, 0, 0)
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
            jsonl.save(row)
            sqlite.save(row)
        yield jsonl, sqlite, rows
        jsonl.close()
        sqlite.close()

    def test_multi_page_agrees_with_sqlite_and_starts_at_newest(self, backends):
        jsonl, sqlite, rows = backends

        for offset in (0, 5, 10):
            kw = {"limit": 5, "offset": offset}
            jsonl_page = jsonl.query(**kw)
            sqlite_page = sqlite.query(**kw)

            assert _ids(jsonl_page) == _ids(sqlite_page)
            assert _ids(jsonl_page) == _expected_page(rows, **kw)
            _assert_timestamp_desc(jsonl_page)

        first_page = jsonl.query(limit=5, offset=0)
        assert first_page[0].request_id == "req-249"
        assert _ids(first_page) == [
            "req-249",
            "req-248",
            "req-247",
            "req-246",
            "req-245",
        ]

    def test_pages_are_disjoint_and_tile_the_newest_first_order(self, backends):
        jsonl, sqlite, rows = backends

        paged: list[str] = []
        jsonl_pages: list[list[str]] = []
        offset = 0
        while True:
            ids = _ids(jsonl.query(limit=5, offset=offset))
            if not ids:
                break
            jsonl_pages.append(ids)
            assert ids == _ids(sqlite.query(limit=5, offset=offset))
            paged.extend(ids)
            offset += 5

        assert sum(len(page) for page in jsonl_pages) == 250
        for i, page in enumerate(jsonl_pages):
            for other in jsonl_pages[i + 1 :]:
                assert set(page).isdisjoint(other)

        assert paged == _expected_page(rows, limit=250, offset=0)

    def test_filtered_pages_agree_with_sqlite(self, backends):
        jsonl, sqlite, rows = backends

        cases = [
            {"model": "claude", "limit": 5, "offset": 0},
            {"model": "claude", "limit": 5, "offset": 5},
            {"mode": "audit", "limit": 5, "offset": 0},
            {"mode": "audit", "limit": 4, "offset": 2},
            {"model": "claude", "mode": "audit", "limit": 3, "offset": 1},
            {
                "model": "gpt-4o",
                "mode": "optimize",
                "start_time": datetime(2025, 1, 6, 12, 0, 30),
                "limit": 6,
                "offset": 3,
            },
        ]
        for kw in cases:
            jsonl_page = jsonl.query(**kw)
            sqlite_page = sqlite.query(**kw)

            assert _ids(jsonl_page) == _ids(sqlite_page), kw
            assert _ids(jsonl_page) == _expected_page(rows, **kw), kw
            _assert_timestamp_desc(jsonl_page)

    def test_filtered_pages_tile_the_filtered_set(self, backends):
        jsonl, sqlite, rows = backends

        kw = {"mode": "audit"}
        total = jsonl.count(**kw)
        assert total == sqlite.count(**kw) == len(_expected_page(rows, **kw, limit=250))

        tiled: list[str] = []
        offset = 0
        while True:
            page_kw = {**kw, "limit": 7, "offset": offset}
            ids = _ids(jsonl.query(**page_kw))
            if not ids:
                break
            assert ids == _ids(sqlite.query(**page_kw)), page_kw
            tiled.extend(ids)
            offset += 7

        assert tiled == _expected_page(rows, **kw, limit=250, offset=0)

    def test_page_one_is_newest_matching_row_not_newest_overall(self, backends):
        jsonl, sqlite, rows = backends

        # req-249 is newest overall and an audit row (249 % 3 == 0), but the
        # newest optimize row is req-248; a filtered page must start there.
        page = jsonl.query(mode="optimize", limit=5, offset=0)
        assert page[0].request_id == "req-248"
        assert _ids(page) == _ids(sqlite.query(mode="optimize", limit=5, offset=0))

    def test_count_summary_and_iter_all_are_unchanged(self, backends):
        jsonl, sqlite, rows = backends

        assert jsonl.count() == 250
        assert jsonl.count(mode="audit") == sqlite.count(mode="audit")
        assert jsonl.count(model="claude") == sqlite.count(model="claude")
        assert (
            jsonl.count(
                start_time=datetime(2025, 1, 6, 12, 0, 30),
                end_time=datetime(2025, 1, 6, 12, 1, 0),
            )
            == 31
        )

        summary = jsonl.get_summary_stats()
        assert summary["total_requests"] == 250
        assert summary["total_tokens_before"] == 250 * 1000
        assert summary["total_tokens_after"] == 250 * 800
        assert summary["audit_count"] == jsonl.count(mode="audit")
        assert summary["optimize_count"] == jsonl.count(mode="optimize")

        # iter_all() keeps file (append) order: paging fixes must not sort it.
        assert _ids(list(jsonl.iter_all())) == [row.request_id for row in rows]


class TestJSONLQuerySingleRowRegression:
    """The 3-row scenario from tests/test_storage_backends.py:162 (issue #3822)."""

    def test_offset_window_over_filtered_set_is_newest_first(self, tmp_path):
        jsonl = JSONLStorage(str(tmp_path / "metrics.jsonl"))
        sqlite = SQLiteStorage(str(tmp_path / "metrics.db"))
        now = datetime(2026, 4, 23, 12, 0, 0)
        first = _metrics("one", now - timedelta(hours=2), mode="audit")
        second = _metrics("two", now - timedelta(hours=1), model="claude", mode="optimize")
        third = _metrics("three", now, mode="audit")
        for row in (first, second, third):
            jsonl.save(row)
            sqlite.save(row)

        kw = {
            "start_time": now - timedelta(hours=1, minutes=30),
            "offset": 1,
            "limit": 1,
        }
        assert _ids(jsonl.query(**kw)) == _ids(sqlite.query(**kw)) == ["two"]

        jsonl.close()
        sqlite.close()
