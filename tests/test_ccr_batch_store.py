from __future__ import annotations

import sys
from dataclasses import dataclass

from headroom.ccr.batch_store import (
    BatchContext,
    BatchContextStore,
    BatchRequestContext,
    get_batch_context_store,
    reset_batch_context_store,
)


def test_batch_context_defaults_and_expiry(monkeypatch) -> None:
    monkeypatch.setattr("headroom.ccr.batch_store.time.time", lambda: 100.0)
    context = BatchContext(batch_id="batch-1", provider="anthropic", created_at=100.0)
    assert context.expires_at == 100.0 + 86400
    assert context.is_expired is False

    request = BatchRequestContext(
        custom_id="req-1",
        messages=[{"role": "user", "content": "hello"}],
        tools=[{"name": "tool"}],
        model="gpt-4o",
        system_instruction="system",
        extras={"x": 1},
    )
    context.add_request(request)
    assert context.get_request("req-1") is request
    assert context.get_request("missing") is None

    monkeypatch.setattr("headroom.ccr.batch_store.time.time", lambda: context.expires_at + 1)
    assert context.is_expired is True


async def test_batch_context_store_core_operations(monkeypatch) -> None:
    now = {"value": 100.0}
    monkeypatch.setattr("headroom.ccr.batch_store.time.time", lambda: now["value"])

    store = BatchContextStore(ttl=10, max_contexts=2)
    first = BatchContext(batch_id="b1", provider="anthropic")
    second = BatchContext(batch_id="b2", provider="google")
    third = BatchContext(batch_id="b3", provider="openai")

    await store.store(first)
    now["value"] = 101.0
    await store.store(second)
    assert (await store.get("b1")) is first

    now["value"] = 102.0
    await store.store(third)
    assert await store.get("b1") is None
    assert await store.get("b2") is second
    assert await store.get("b3") is third

    assert await store.remove("b2") is True
    assert await store.remove("b2") is False


async def test_batch_context_store_cleanup_stats_and_memory_stats(monkeypatch) -> None:
    now = {"value": 200.0}
    monkeypatch.setattr("headroom.ccr.batch_store.time.time", lambda: now["value"])
    store = BatchContextStore(ttl=5, max_contexts=10)

    first = BatchContext(batch_id="b1", provider="anthropic")
    first.add_request(
        BatchRequestContext(custom_id="r1", messages=[{"content": "alpha"}], tools=[])
    )
    second = BatchContext(batch_id="b2", provider="google")
    second.add_request(
        BatchRequestContext(custom_id="r2", messages=[{"content": ["nested"]}], tools=[{}])
    )

    await store.store(first)
    await store.store(second)
    assert await store.stats() == {
        "total_contexts": 2,
        "max_contexts": 10,
        "ttl_seconds": 5,
        "providers": {"anthropic": 1, "google": 1},
    }

    now["value"] = 210.0
    assert await store.cleanup_expired() == 2
    assert (await store.stats())["total_contexts"] == 0

    @dataclass
    class FakeComponentStats:
        name: str
        entry_count: int
        size_bytes: int
        budget_bytes: int | None
        hits: int
        misses: int
        evictions: int

    monkeypatch.setitem(
        sys.modules,
        "headroom.memory.tracker",
        type("TrackerModule", (), {"ComponentStats": FakeComponentStats}),
    )

    now["value"] = 220.0
    third = BatchContext(batch_id="b3", provider="openai")
    third.add_request(
        BatchRequestContext(
            custom_id="r3", messages=[{"content": "payload"}], tools=[{"name": "t"}]
        )
    )
    await store.store(third)
    stats = store.get_memory_stats()
    assert stats.name == "batch_context_store"
    assert stats.entry_count == 1
    assert stats.size_bytes > 0


def test_global_batch_context_store_reset() -> None:
    reset_batch_context_store()
    store_one = get_batch_context_store()
    store_two = get_batch_context_store()
    assert store_one is store_two

    reset_batch_context_store()
    store_three = get_batch_context_store()
    assert store_three is not store_one
