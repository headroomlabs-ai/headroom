"""Temporal memory queries compare instants, including existing offset rows."""

from datetime import datetime, timedelta, timezone

import pytest

from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.models import Memory
from headroom.memory.ports import MemoryFilter

UTC = timezone.utc
ZONES = [UTC, timezone(timedelta(hours=5)), timezone(timedelta(hours=-4))]


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


@pytest.mark.asyncio
@pytest.mark.parametrize("zone", ZONES)
@pytest.mark.parametrize(
    "field,baseline",
    [
        ("created_after", datetime(2026, 1, 1, 11, tzinfo=UTC)),
        ("created_before", datetime(2026, 1, 1, 12, tzinfo=UTC)),
        ("valid_at", datetime(2026, 1, 1, 12, 30, tzinfo=UTC)),
        ("valid_at", datetime(2026, 1, 1, 13, tzinfo=UTC)),
        ("created_after", datetime(2026, 1, 1, 12, 0, 0, 1, tzinfo=UTC)),
    ],
)
async def test_temporal_query_and_count_match_datetime_oracle(tmp_path, field, baseline, zone):
    path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(path)
    memories = [
        Memory(
            id="naive",
            user_id="fixture",
            created_at=datetime(2026, 1, 1, 12),
            valid_from=datetime(2026, 1, 1, 12),
            valid_until=datetime(2026, 1, 1, 13),
        ),
        Memory(
            id="aware",
            user_id="fixture",
            created_at=datetime(2026, 1, 1, 17, tzinfo=ZONES[1]),
            valid_from=datetime(2026, 1, 1, 8, tzinfo=ZONES[2]),
            valid_until=datetime(2026, 1, 1, 18, tzinfo=ZONES[1]),
        ),
        Memory(
            id="microsecond",
            user_id="fixture",
            created_at=datetime(2026, 1, 1, 12, 0, 0, 1),
            valid_from=datetime(2026, 1, 1, 12, 0, 0, 1),
            valid_until=datetime(2026, 1, 1, 13, 0, 0, 1),
        ),
    ]
    for memory in memories:
        await store.save(memory)
    # Stored bytes stay offset-bearing across a new adapter instance.
    store = SQLiteMemoryStore(path)
    retained = await store.get("aware")
    assert retained is not None
    assert retained.created_at.isoformat() == memories[1].created_at.isoformat()
    bound = baseline.astimezone(zone)

    def matches(memory):
        if field == "created_after":
            return _utc(memory.created_at) >= _utc(bound)
        if field == "created_before":
            return _utc(memory.created_at) <= _utc(bound)
        return _utc(memory.valid_from) <= _utc(bound) < _utc(memory.valid_until)

    expected = sorted(memory.id for memory in memories if matches(memory))
    query = MemoryFilter(user_id="fixture", include_superseded=True, **{field: bound})
    assert sorted(memory.id for memory in await store.query(query)) == expected
    assert await store.count(query) == len(expected)


@pytest.mark.asyncio
async def test_temporal_order_precedes_pagination_and_preserves_null_access(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    # Wall-clock text sorts these incorrectly; UTC chronological order is a,b,c.
    for identity, timestamp in [
        ("a", datetime(2026, 1, 1, 15, tzinfo=ZONES[1])),
        ("b", datetime(2026, 1, 1, 11)),
        ("c", datetime(2026, 1, 1, 8, tzinfo=ZONES[2])),
    ]:
        await store.save(Memory(id=identity, user_id="fixture", created_at=timestamp))
    result = await store.query(MemoryFilter(user_id="fixture", order_desc=False, limit=1, offset=1))
    assert [memory.id for memory in result] == ["b"]
    # Last-access timestamps are nullable; their established ordering stays valid.
    await store.record_access(["b"], datetime(2026, 1, 1, 11, tzinfo=UTC))
    accessed = await store.query(MemoryFilter(user_id="fixture", order_by="last_accessed", limit=1))
    assert [memory.id for memory in accessed] == ["b"]
