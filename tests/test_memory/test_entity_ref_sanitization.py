"""Regression tests for entity_refs type safety.

`entity_refs` is typed `list[str]` everywhere, but nothing enforced that at
runtime. A caller that mistakenly passed the typed
`{"entity": ..., "entity_type": ...}` shape (the format `extracted_entities`
expects) into the plain `entities` field of `save_memory` got those dicts
persisted verbatim into `entity_refs` -- both in the `memories` table and in
the duplicated copy the vector index keeps for post-filtering.

Every later `search_memories` call does `set().update(memory.entity_refs)`
while collecting entities for graph expansion. Hashing a dict raises
`TypeError: unhashable type: 'dict'`, and because that happens inside the
vector-result loop (not guarded per-item) it aborted the *entire* search for
any query whose top-k included one poisoned row. The proxy's memory handler
swallows the exception and returns no memories, so recall went quietly dark
rather than failing loudly.

See https://github.com/headroomlabs-ai/headroom/issues/2947.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from headroom.memory.adapters.hnsw import IndexedMemoryMetadata
from headroom.memory.adapters.sqlite_vector import VectorMetadata
from headroom.memory.backends.local import LocalBackend
from headroom.memory.models import Memory, normalize_entity_refs

# The malformed shape that started all of this: the extracted_entities format
# passed into a field that expects plain names.
DICT_REF = {"entity": "Project X", "entity_type": "project"}


# =============================================================================
# The helper itself
# =============================================================================


def test_normalize_entity_refs_unwraps_dicts_and_drops_junk() -> None:
    """Dicts are unwrapped to their name; anything unusable is dropped."""
    assert normalize_entity_refs(["Alice", DICT_REF]) == ["Alice", "Project X"]

    # Nothing usable in these: no name to recover, so they are dropped rather
    # than stringified into garbage entity names like "{'foo': 'bar'}".
    assert normalize_entity_refs([{"entity_type": "project"}, {}, None, 42, ""]) == []

    # Common no-op cases stay untouched.
    assert normalize_entity_refs(["Alice", "Bob"]) == ["Alice", "Bob"]
    assert normalize_entity_refs(None) == []
    assert normalize_entity_refs([]) == []


def test_normalize_entity_refs_preserves_order_and_deduplicates() -> None:
    """A name already present is not appended twice, and order is stable."""
    assert normalize_entity_refs(["Alice", DICT_REF, "Alice", "Project X"]) == [
        "Alice",
        "Project X",
    ]


# =============================================================================
# Write path: stop new corruption at the door
# =============================================================================


@pytest.mark.asyncio
async def test_save_memory_sanitizes_dict_shaped_entities_param() -> None:
    """`entities` items that are dicts get coerced to plain names before storage."""
    backend = LocalBackend()
    backend._initialized = True

    saved: list[Memory] = []

    async def fake_add(**kwargs: object) -> Memory:
        memory = Memory(
            id="new-memory",
            content=str(kwargs["content"]),
            user_id=str(kwargs["user_id"]),
            entity_refs=list(kwargs["entity_refs"]),  # type: ignore[arg-type]
        )
        saved.append(memory)
        return memory

    backend._hierarchical_memory = SimpleNamespace(add=AsyncMock(side_effect=fake_add))
    backend._graph = SimpleNamespace(
        get_entity_by_name=AsyncMock(return_value=None),
        add_entity=AsyncMock(return_value=SimpleNamespace(id="entity-id")),
        add_relationship=AsyncMock(),
    )

    # Without the fix this raises AttributeError: 'dict' object has no
    # attribute 'lower' during graph linking.
    await backend.save_memory(
        content="Alice manages Project X",
        user_id="alice",
        entities=[DICT_REF],  # type: ignore[list-item]
    )

    assert saved[0].entity_refs == ["Project X"]


@pytest.mark.asyncio
async def test_save_memory_merges_dict_entities_with_extracted_entities() -> None:
    """A name arriving through both `entities` and `extracted_entities` is stored once."""
    backend = LocalBackend()
    backend._initialized = True

    saved: list[Memory] = []

    async def fake_add(**kwargs: object) -> Memory:
        memory = Memory(
            id="new-memory",
            content=str(kwargs["content"]),
            user_id=str(kwargs["user_id"]),
            entity_refs=list(kwargs["entity_refs"]),  # type: ignore[arg-type]
        )
        saved.append(memory)
        return memory

    backend._hierarchical_memory = SimpleNamespace(add=AsyncMock(side_effect=fake_add))
    backend._graph = SimpleNamespace(
        get_entity_by_name=AsyncMock(return_value=None),
        add_entity=AsyncMock(return_value=SimpleNamespace(id="entity-id")),
        add_relationship=AsyncMock(),
    )

    await backend.save_memory(
        content="Alice manages Project X",
        user_id="alice",
        entities=[DICT_REF],  # type: ignore[list-item]
        extracted_entities=[{"entity": "Project X", "entity_type": "project"}],
    )

    assert saved[0].entity_refs == ["Project X"]


# =============================================================================
# Read path: heal rows that were already written before the fix
# =============================================================================


def test_memory_from_dict_heals_stored_dict_refs() -> None:
    """Rows persisted before the fix load as plain names instead of dicts."""
    now = datetime.now(timezone.utc).isoformat()
    memory = Memory.from_dict(
        {
            "id": "poisoned-memory",
            "content": "Alice manages Project X",
            "user_id": "alice",
            "created_at": now,
            "valid_from": now,
            "importance": 0.5,
            "entity_refs": [DICT_REF, "Alice"],
        }
    )

    assert memory.entity_refs == ["Project X", "Alice"]


def test_vector_metadata_from_json_heals_stored_dict_refs() -> None:
    """The vector index keeps its own copy of entity_refs; heal that one too."""
    now = datetime.now(timezone.utc).isoformat()
    metadata = VectorMetadata.from_json(
        json.dumps(
            {
                "memory_id": "poisoned-memory",
                "user_id": "alice",
                "session_id": None,
                "agent_id": None,
                "valid_until": None,
                "entity_refs": [DICT_REF],
                "content": "Alice manages Project X",
                "created_at": now,
                "importance": 0.5,
                "metadata": {},
            }
        )
    )

    assert metadata.entity_refs == ["Project X"]
    assert metadata.to_memory().entity_refs == ["Project X"]


def test_indexed_memory_metadata_from_dict_heals_stored_dict_refs() -> None:
    """Same for the HNSW index's metadata copy."""
    now = datetime.now(timezone.utc).isoformat()
    metadata = IndexedMemoryMetadata.from_dict(
        {
            "memory_id": "poisoned-memory",
            "user_id": "alice",
            "session_id": None,
            "agent_id": None,
            "valid_until": None,
            "entity_refs": [DICT_REF],
            "content": "Alice manages Project X",
            "created_at": now,
            "importance": 0.5,
            "metadata": {},
        }
    )

    assert metadata.entity_refs == ["Project X"]


# =============================================================================
# Search: a single bad row must not take the whole query down
# =============================================================================


def _backend_with_results(memories: list[Memory]) -> LocalBackend:
    backend = LocalBackend()
    backend._initialized = True
    backend._hierarchical_memory = SimpleNamespace(
        search=AsyncMock(return_value=[SimpleNamespace(memory=m, similarity=0.9) for m in memories])
    )
    backend._graph = SimpleNamespace(
        get_entity_by_name=AsyncMock(return_value=None),
        query_subgraph=AsyncMock(return_value=SimpleNamespace(entities=[], relationships=[])),
    )
    return backend


@pytest.mark.asyncio
async def test_search_memories_tolerates_dict_shaped_entity_refs() -> None:
    """A single legacy/corrupted row with dict entity_refs must not crash search."""
    poisoned = Memory(
        id="poisoned-memory",
        content="Alice manages Project X",
        user_id="alice",
        entity_refs=[DICT_REF],  # type: ignore[list-item]
    )
    clean = Memory(
        id="clean-memory",
        content="Bob manages Project Y",
        user_id="alice",
        entity_refs=["Project Y"],
    )
    backend = _backend_with_results([poisoned, clean])

    # Without the fix this raises TypeError: unhashable type: 'dict'.
    results = await backend.search_memories("Alice's work", "alice", include_related=True)

    assert [r.memory.id for r in results] == ["poisoned-memory", "clean-memory"]
    # The recovered name is still usable for graph expansion and is reported
    # back to the caller as a plain string, not a dict.
    assert results[0].related_entities == ["Project X"]
    backend._graph.get_entity_by_name.assert_awaited()


@pytest.mark.asyncio
async def test_search_memories_entity_filter_matches_healed_refs() -> None:
    """The `entities` filter lowercases each ref, which dicts also break.

    On unfixed code this never gets that far -- the unconditional
    `set().update()` above raises first -- but once refs are strings again the
    filter has to actually match the recovered name.
    """
    poisoned = Memory(
        id="poisoned-memory",
        content="Alice manages Project X",
        user_id="alice",
        entity_refs=[DICT_REF],  # type: ignore[list-item]
    )
    backend = _backend_with_results([poisoned])

    results = await backend.search_memories(
        "Alice's work",
        "alice",
        include_related=False,
        entities=["project x"],
    )

    assert [r.memory.id for r in results] == ["poisoned-memory"]


@pytest.mark.asyncio
async def test_search_memories_tolerates_dict_shaped_entities_filter() -> None:
    """The filter argument comes from LLM tool input too, so it can be malformed."""
    clean = Memory(
        id="clean-memory",
        content="Alice manages Project X",
        user_id="alice",
        entity_refs=["Project X"],
    )
    backend = _backend_with_results([clean])

    # Without normalization this raises AttributeError: 'dict' object has no
    # attribute 'lower' while building the filter set.
    results = await backend.search_memories(
        "Alice's work",
        "alice",
        include_related=False,
        entities=[DICT_REF],  # type: ignore[list-item]
    )

    assert [r.memory.id for r in results] == ["clean-memory"]
