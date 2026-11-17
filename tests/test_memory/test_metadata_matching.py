"""Execute JSON-valued metadata filters against the real SQLite memory store."""

import json

import pytest

from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.models import Memory
from headroom.memory.ports import MemoryFilter


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        ["a", "b"],
        {"priority": 5},
        {"nested": ["a", "b"]},
        ["café", {"note": "spaces within strings remain"}],
        [],
        {},
        ["single"],
    ],
)
async def test_compound_metadata_matches_without_matching_json_strings(tmp_path, value):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    await store.save(
        Memory(id="match", user_id="alice", content="saved", metadata={"value": value})
    )
    await store.save(
        Memory(
            id="string",
            user_id="alice",
            content="not a compound value",
            metadata={"value": json.dumps(value, separators=(",", ":"))},
        )
    )
    await store.save(
        Memory(id="other-user", user_id="bob", content="saved", metadata={"value": value})
    )
    query = MemoryFilter(user_id="alice", metadata_filters={"value": value})

    assert [memory.id for memory in await store.query(query)] == ["match"]
    assert await store.count(query) == 1


@pytest.mark.asyncio
async def test_null_metadata_matches_explicit_null_not_missing_or_text(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    for name, metadata in [("null", {"value": None}), ("missing", {}), ("text", {"value": "null"})]:
        await store.save(Memory(id=name, user_id="alice", content="saved", metadata=metadata))
    query = MemoryFilter(user_id="alice", metadata_filters={"value": None})

    assert [memory.id for memory in await store.query(query)] == ["null"]
    assert await store.count(query) == 1


@pytest.mark.asyncio
async def test_compound_metadata_preserves_array_and_object_member_order(tmp_path):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    await store.save(
        Memory(
            id="saved",
            user_id="alice",
            content="saved",
            metadata={"array": [1, 2], "object": {"a": 1, "b": 2}},
        )
    )
    for metadata in [{"array": [2, 1]}, {"object": {"b": 2, "a": 1}}]:
        query = MemoryFilter(user_id="alice", metadata_filters=metadata)
        assert await store.query(query) == []
        assert await store.count(query) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        ["a", "b"],
        {"priority": 5},
        {"nested": ["a", "b"]},
        ["café", {"note": "spaces within strings remain"}],
        [],
        {},
        ["single"],
        None,
    ],
)
async def test_json_looking_string_filter_matches_only_text(tmp_path, value):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    text = json.dumps(value, separators=(",", ":"))
    for name, saved in [("text", text), ("json", value)]:
        await store.save(
            Memory(id=name, user_id="alice", content="saved", metadata={"value": saved})
        )
    query = MemoryFilter(user_id="alice", metadata_filters={"value": text})

    assert [memory.id for memory in await store.query(query)] == ["text"]
    assert await store.count(query) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [True, False, 1, 0, 1.0, 0.0])
async def test_numeric_and_boolean_metadata_retains_native_sqlite_equality(tmp_path, value):
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    for name, saved in [
        ("true", True),
        ("false", False),
        ("one", 1),
        ("zero", 0),
        ("float-one", 1.0),
        ("float-zero", 0.0),
        ("text-one", "1"),
        ("text-zero", "0"),
    ]:
        await store.save(
            Memory(id=name, user_id="alice", content="saved", metadata={"value": saved})
        )
    query = MemoryFilter(user_id="alice", metadata_filters={"value": value})
    expected = {"true", "one", "float-one"} if value else {"false", "zero", "float-zero"}

    assert {memory.id for memory in await store.query(query)} == expected
    assert await store.count(query) == 3
