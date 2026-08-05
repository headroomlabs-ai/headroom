"""Focused tests for LocalBackend search result filtering."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from headroom.memory.backends.local import LocalBackend
from headroom.memory.models import Memory


def _backend_with_related_memory(related_memory: Memory) -> LocalBackend:
    seed = Memory(
        id="seed-memory",
        content="Alice manages Project X",
        user_id="alice",
        entity_refs=["Project X"],
    )
    vector_result = SimpleNamespace(memory=seed, similarity=0.9)

    backend = LocalBackend()
    backend._initialized = True
    backend._hierarchical_memory = SimpleNamespace(
        search=AsyncMock(return_value=[vector_result]),
        get=AsyncMock(return_value=related_memory),
    )
    backend._graph = SimpleNamespace(
        get_entity_by_name=AsyncMock(return_value=SimpleNamespace(id="project-x")),
        query_subgraph=AsyncMock(
            return_value=SimpleNamespace(
                entities=[SimpleNamespace(metadata={"source_memory_id": related_memory.id})],
                relationships=[],
            )
        ),
    )
    return backend


@pytest.mark.asyncio
async def test_graph_expansion_includes_active_related_memory() -> None:
    related = Memory(
        id="related-memory",
        content="Project X uses Python",
        user_id="alice",
        entity_refs=["Project X", "Python"],
    )
    backend = _backend_with_related_memory(related)

    results = await backend.search_memories("Alice's work", "alice", include_related=True)

    assert [result.memory.id for result in results] == ["seed-memory", "related-memory"]


@pytest.mark.asyncio
@pytest.mark.parametrize("inactive_field", ["valid_until", "superseded_by"])
async def test_graph_expansion_excludes_inactive_related_memory(
    inactive_field: str,
) -> None:
    related = Memory(
        id="related-memory",
        content="Outdated Project X detail",
        user_id="alice",
        entity_refs=["Project X"],
    )
    if inactive_field == "valid_until":
        related.valid_until = datetime.now(timezone.utc).replace(tzinfo=None)
    else:
        related.superseded_by = "replacement-memory"
    backend = _backend_with_related_memory(related)

    results = await backend.search_memories("Alice's work", "alice", include_related=True)

    assert [result.memory.id for result in results] == ["seed-memory"]
