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
