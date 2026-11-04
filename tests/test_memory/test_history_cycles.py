"""History traversal must terminate on cyclic or overlapping lineage links."""

import pytest

from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.models import Memory


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("links", "include_future", "expected_ids"),
    [
        ([("first", "first", None)], False, ["first"]),
        (
            [("first", "second", None), ("second", "first", None)],
            False,
            ["second", "first"],
        ),
        ([("first", None, "first")], True, ["first"]),
        (
            [("first", None, "second"), ("second", None, "first")],
            True,
            ["first", "second"],
        ),
        (
            [("first", "older", "newer"), ("older", None, "first"), ("newer", "first", "older")],
            True,
            ["older", "first", "newer"],
        ),
        # A forward cycle must not affect a backward-only request.
        ([("first", None, "first")], False, ["first"]),
        # Ordinary chains and dangling links retain their existing behavior.
        (
            [("first", "older", "newer"), ("older", None, "first"), ("newer", "first", None)],
            True,
            ["older", "first", "newer"],
        ),
        ([("first", "missing", None)], True, ["first"]),
    ],
    ids=[
        "back-self",
        "back-cycle",
        "forward-self",
        "forward-cycle",
        "overlap",
        "back-only",
        "chain",
        "dangling",
    ],
)
async def test_history_visits_each_record_once(
    tmp_path, monkeypatch, links, include_future, expected_ids
):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    await store.save_batch(
        [
            Memory(id=id, content=id, user_id="alice", supersedes=previous, superseded_by=following)
            for id, previous, following in links
        ]
    )
    original_get = store.get
    visited = set()

    async def bounded_get(memory_id):
        # Keep a regressed traversal from hanging the test process. Records
        # still come from the actual SQLite store, without fabricated results.
        assert memory_id not in visited, f"History revisited {memory_id}"
        visited.add(memory_id)
        return await original_get(memory_id)

    monkeypatch.setattr(store, "get", bounded_get)

    history = await store.get_history("first", include_future=include_future)

    assert [memory.id for memory in history] == expected_ids
    # Reading malformed lineage must not rewrite its persisted links.
    for id, previous, following in links:
        memory = await original_get(id)
        assert (memory.supersedes, memory.superseded_by) == (previous, following)
