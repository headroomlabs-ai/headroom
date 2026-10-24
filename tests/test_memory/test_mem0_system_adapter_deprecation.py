"""Mem0SystemAdapter is deprecated in favour of DirectMem0Adapter."""

from __future__ import annotations

import pytest

from headroom.memory.backends import mem0_system_adapter


def test_mem0_system_adapter_warns_deprecated(monkeypatch: pytest.MonkeyPatch) -> None:
    # Mem0Backend's constructor needs no live services, but stub it so the test
    # does not depend on the optional mem0 install.
    monkeypatch.setattr(mem0_system_adapter, "Mem0Backend", lambda config: object())
    with pytest.warns(DeprecationWarning, match="Mem0SystemAdapter is deprecated"):
        mem0_system_adapter.Mem0SystemAdapter()


_ENTITY = [{"entity": "Netflix", "entity_type": "organization"}]
_RELATION = [{"source": "user", "relationship": "works_at", "destination": "Netflix"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "runs_mem0_extraction"),
    [
        ({"facts": ["Works at Netflix"]}, False),
        ({"extracted_entities": _ENTITY}, False),
        ({"extracted_relationships": _RELATION}, False),
        # The legacy names do not select the optimized path.
        ({"entities": ["Netflix"]}, True),
        ({"relationships": _RELATION}, True),
        # Empty pre-extracted inputs do not count either.
        ({"facts": [], "extracted_entities": [], "extracted_relationships": []}, True),
    ],
)
async def test_direct_adapter_skips_mem0_extraction_only_for_pre_extracted_args(
    kwargs: dict, runs_mem0_extraction: bool
) -> None:
    """The deprecation names facts/extracted_entities/extracted_relationships as the
    way to skip Mem0's LLM extraction; entities/relationships alone still run it."""
    from unittest.mock import AsyncMock, MagicMock

    from headroom.memory.backends.direct_mem0 import DirectMem0Adapter, Mem0Config

    adapter = DirectMem0Adapter(Mem0Config(enable_graph=True))
    adapter._ensure_initialized = AsyncMock()  # type: ignore[method-assign]
    mem0_client = MagicMock()
    mem0_client.add.return_value = {"results": [{"id": "m1", "memory": "Works at Netflix"}]}
    adapter._mem0_client = mem0_client
    write_facts = AsyncMock(return_value=["f1"])
    write_graph = AsyncMock()
    adapter._write_facts_to_qdrant = write_facts  # type: ignore[method-assign]
    adapter._write_graph_to_neo4j = write_graph  # type: ignore[method-assign]

    await adapter.save_memory(
        content="I work at Netflix", user_id="u1", importance=0.5, background=False, **kwargs
    )

    assert mem0_client.add.called is runs_mem0_extraction
    assert write_facts.called is not runs_mem0_extraction
