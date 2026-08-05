"""Tests for explicit supersession-edge repair."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from click.testing import CliRunner

import headroom.cli.memory as memory_cli
from headroom.cli.main import main
from headroom.memory.adapters.sqlite import SQLiteMemoryStore
from headroom.memory.core import HierarchicalMemory
from headroom.memory.models import Memory
from headroom.memory.ports import MemoryFilter


@pytest.mark.asyncio
async def test_store_detaches_only_the_requested_supersession_edge(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    first = Memory(content="First fact", user_id="alice")
    await store.save(first)
    second = await store.supersede(
        first.id,
        Memory(content="Second fact", user_id="alice"),
    )
    third = await store.supersede(
        second.id,
        Memory(content="Third fact", user_id="alice"),
    )

    repaired_first, repaired_second = await store.detach_supersession(first.id, second.id)

    assert repaired_first.valid_until is None
    assert repaired_first.superseded_by is None
    assert repaired_second.supersedes is None
    assert repaired_second.superseded_by == third.id
    assert repaired_second.valid_until is not None
    assert third.supersedes == second.id

    current = await store.query(MemoryFilter(user_id="alice"))
    assert {memory.id for memory in current} == {first.id, third.id}


@pytest.mark.asyncio
async def test_store_rejects_non_reciprocal_repair_without_mutation(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    first = Memory(content="First fact", user_id="alice")
    unrelated = Memory(content="Unrelated fact", user_id="alice")
    await store.save_batch([first, unrelated])

    with pytest.raises(ValueError, match="reciprocal supersession edge"):
        await store.detach_supersession(first.id, unrelated.id)

    assert (await store.get(first.id)).is_current
    assert (await store.get(unrelated.id)).is_current


@pytest.mark.asyncio
async def test_core_reindexes_and_refreshes_cache_after_repair(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    first = Memory(
        content="First fact",
        user_id="alice",
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    await store.save(first)
    second = await store.supersede(
        first.id,
        Memory(
            content="Second fact",
            user_id="alice",
            embedding=np.array([0.0, 1.0], dtype=np.float32),
        ),
    )
    vector_index = SimpleNamespace(index=AsyncMock())
    text_index = SimpleNamespace(index_memory=AsyncMock())
    cache = SimpleNamespace(invalidate_batch=AsyncMock(), put_batch=AsyncMock())
    system = HierarchicalMemory(
        store=store,
        vector_index=vector_index,
        text_index=text_index,
        embedder=SimpleNamespace(),
        cache=cache,
    )

    repaired = await system.detach_supersession(first.id, second.id)

    assert [call.args[0].id for call in vector_index.index.await_args_list] == [
        first.id,
        second.id,
    ]
    assert [call.args[0].id for call in text_index.index_memory.await_args_list] == [
        first.id,
        second.id,
    ]
    cache.invalidate_batch.assert_awaited_once_with([first.id, second.id])
    cache.put_batch.assert_awaited_once_with(list(repaired))


def _seed_chain(db_path) -> tuple[Memory, Memory]:
    async def seed() -> tuple[Memory, Memory]:
        store = SQLiteMemoryStore(db_path)
        first = Memory(content="Keep Python preference", user_id="alice")
        await store.save(first)
        second = await store.supersede(
            first.id,
            Memory(content="Keep dark mode preference", user_id="alice"),
        )
        return first, second

    return asyncio.run(seed())


def test_cli_repair_is_dry_run_by_default(tmp_path) -> None:
    db_path = tmp_path / "memory.db"
    first, second = _seed_chain(db_path)

    result = CliRunner().invoke(
        main,
        [
            "memory",
            "repair-supersession",
            first.id[:8],
            second.id[:8],
            "--db-path",
            str(db_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output
    store = SQLiteMemoryStore(db_path)
    assert asyncio.run(store.get(first.id)).superseded_by == second.id
    assert asyncio.run(store.get(second.id)).supersedes == first.id


def test_cli_repair_requires_apply_and_uses_full_ids(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "memory.db"
    first, second = _seed_chain(db_path)
    applied: list[tuple[str, str, int]] = []

    async def fake_apply(
        db_path_arg: str,
        old_memory_id: str,
        new_memory_id: str,
        vector_dimension: int,
    ) -> tuple[Memory, Memory]:
        applied.append((old_memory_id, new_memory_id, vector_dimension))
        store = SQLiteMemoryStore(db_path_arg)
        return await store.detach_supersession(old_memory_id, new_memory_id)

    monkeypatch.setattr(memory_cli, "_apply_supersession_repair", fake_apply)
    result = CliRunner().invoke(
        main,
        [
            "memory",
            "repair-supersession",
            first.id[:8],
            second.id[:8],
            "--db-path",
            str(db_path),
            "--apply",
        ],
    )

    assert result.exit_code == 0, result.output
    assert applied == [(first.id, second.id, 384)]
    store = SQLiteMemoryStore(db_path)
    assert asyncio.run(store.get(first.id)).is_current
    assert asyncio.run(store.get(second.id)).supersedes is None
