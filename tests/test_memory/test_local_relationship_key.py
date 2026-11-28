"""The local backend reads the relationship key the memory_save schema advertises.

`memory_save`'s plain `relationships[]` schema (headroom/memory/tools.py) names the
relationship key `relation`. LocalBackend.save_memory read only `type`, so every
relationship saved through the tool was stored as the default `related_to`.
See https://github.com/headroomlabs-ai/headroom/issues/4362 (item 7).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from headroom.memory.backends.local import LocalBackend
from headroom.memory.models import Memory
from headroom.memory.tools import MEMORY_TOOLS


def _backend() -> tuple[LocalBackend, AsyncMock]:
    backend = LocalBackend()
    backend._initialized = True

    async def fake_add(**kwargs: object) -> Memory:
        return Memory(
            id="m1",
            content=str(kwargs["content"]),
            user_id=str(kwargs["user_id"]),
            entity_refs=list(kwargs["entity_refs"]),  # type: ignore[arg-type]
        )

    async def fake_add_entity(entity: object) -> object:
        return entity

    add_relationship = AsyncMock()
    backend._hierarchical_memory = SimpleNamespace(add=AsyncMock(side_effect=fake_add))
    backend._graph = SimpleNamespace(
        get_entity_by_name=AsyncMock(side_effect=lambda user_id, name: None),
        add_entity=AsyncMock(side_effect=fake_add_entity),
        add_relationship=add_relationship,
    )
    return backend, add_relationship


def test_schema_names_the_key_relation() -> None:
    """Pins the premise: the advertised schema uses `relation`, not `type`."""
    save = next(t for t in MEMORY_TOOLS if t["function"]["name"] == "memory_save")
    item = save["function"]["parameters"]["properties"]["relationships"]["items"]
    assert "relation" in item["required"]
    assert "type" not in item["properties"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rel", "expected"),
    [
        ({"source": "Alice", "relation": "works_with", "target": "Bob"}, "works_with"),
        ({"source": "Alice", "type": "manages", "target": "Bob"}, "manages"),
        ({"source": "Alice", "target": "Bob"}, "related_to"),
    ],
)
async def test_relationship_type_is_read_from_relation_or_type(
    rel: dict[str, str], expected: str
) -> None:
    backend, add_relationship = _backend()

    await backend.save_memory(
        content="Alice works with Bob",
        user_id="alice",
        entities=["Alice", "Bob"],
        relationships=[rel],
    )

    add_relationship.assert_awaited_once()
    stored = add_relationship.await_args.args[0]
    assert stored.relation_type == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_relationships",
    [
        ["Alice works_with Bob"],
        [{"source": None, "target": "Bob", "relation": "knows"}],
        [{"source": "Alice", "target": ["Bob"]}],
        [None, 7],
    ],
)
async def test_malformed_relationships_are_skipped_before_the_row_is_stored(
    bad_relationships: list[object],
) -> None:
    """LLM junk must not raise after the write (a retry would duplicate the row)."""
    backend, add_relationship = _backend()

    memory = await backend.save_memory(
        content="Alice works with Bob",
        user_id="alice",
        entities=["Alice", "Bob"],
        relationships=bad_relationships,  # type: ignore[arg-type]
    )

    assert memory.id == "m1"
    assert backend._hierarchical_memory.add.await_count == 1
    add_relationship.assert_not_awaited()


@pytest.mark.asyncio
async def test_malformed_item_does_not_drop_the_well_formed_ones() -> None:
    backend, add_relationship = _backend()

    await backend.save_memory(
        content="Alice works with Bob",
        user_id="alice",
        entities=["Alice", "Bob"],
        relationships=[
            "Alice works_with Bob",  # type: ignore[list-item]
            {"source": "Alice", "relation": "works_with", "target": "Bob"},
        ],
    )

    add_relationship.assert_awaited_once()
    assert add_relationship.await_args.args[0].relation_type == "works_with"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_entities",
    [[{"entity": ["X"]}], [{"entity": 7}], ["X"], [None]],
)
async def test_malformed_extracted_entities_are_skipped(bad_entities: list[object]) -> None:
    backend, _ = _backend()

    memory = await backend.save_memory(
        content="x",
        user_id="alice",
        extracted_entities=bad_entities,  # type: ignore[arg-type]
        extracted_relationships=["junk"],  # type: ignore[arg-type]
    )

    assert memory.entity_refs == []
    assert backend._hierarchical_memory.add.await_count == 1


@pytest.mark.asyncio
async def test_well_formed_extracted_entities_and_relationships_unchanged() -> None:
    backend, add_relationship = _backend()

    memory = await backend.save_memory(
        content="x",
        user_id="alice",
        extracted_entities=[
            {"entity": "Alice", "entity_type": "person"},
            {"entity": "Bob", "entity_type": "person"},
        ],
        extracted_relationships=[
            {"source": "Alice", "relationship": "manages", "destination": "Bob"}
        ],
    )

    assert memory.entity_refs == ["Alice", "Bob"]
    add_relationship.assert_awaited_once()
    assert add_relationship.await_args.args[0].relation_type == "manages"
