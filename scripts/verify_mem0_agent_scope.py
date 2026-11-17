"""Exercise real Mem0/Qdrant storage without an inference-provider request."""

import asyncio
import os
import tempfile
from functools import partial
from importlib.metadata import version

os.environ["MEM0_TELEMETRY"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from mem0 import Memory as Mem0

from headroom.memory.backends.mem0 import Mem0Backend
from headroom.memory.models import Memory
from headroom.memory.ports import MemoryFilter, VectorFilter


async def main():
    with tempfile.TemporaryDirectory(prefix="headroom-mem0-scope-") as storage:
        client = Mem0.from_config(
            {
                "version": "v1.1",
                "llm": {"provider": "openai", "config": {"api_key": "unused-no-inference"}},
                "embedder": {
                    "provider": "huggingface",
                    "config": {
                        "model": "sentence-transformers/all-MiniLM-L6-v2",
                        "embedding_dims": 384,
                    },
                },
                "vector_store": {
                    "provider": "qdrant",
                    "config": {
                        "path": storage,
                        "collection_name": "agent_scope_proof",
                        "embedding_model_dims": 384,
                    },
                },
                "history_db_path": storage + "/history.db",
            }
        )
        # Disable LLM fact inference, retaining real embedding, add, storage, and search.
        client.add = partial(client.add, infer=False)
        backend = Mem0Backend()
        backend._client = client
        backend._initialized = True
        saved = {}
        for agent in ("agent-a", "agent-b"):
            memory = Memory(
                content="The preferred drink is tea", user_id="scope-user", agent_id=agent
            )
            await backend.save_memory(memory)
            saved[agent] = memory.id
        for agent in saved:
            results = await backend.search_memories(
                "preferred drink",
                user_id="scope-user",
                filter=VectorFilter(agent_id=agent),
                limit=10,
            )
            assert len(results) == 1, (agent, results)
            assert results[0].memory.agent_id == agent, (agent, results)
            assert results[0].memory.user_id == "scope-user", results
            assert results[0].memory.id == saved[agent], (agent, saved, results)
            queried = await backend.query(MemoryFilter(user_id="scope-user", agent_id=agent))
            assert [memory.id for memory in queried] == [saved[agent]], (agent, queried)
            print(f"{agent}: one persisted memory retrieved; sibling agent excluded")
        all_memories = await backend.get_all_memories("scope-user", limit=10)
        assert {memory.id for memory in all_memories} == set(saved.values()), all_memories
        assert await backend.get_all_memories("other-user", limit=10) == []
        assert await backend.query(MemoryFilter(user_id="other-user")) == []
        print("get_all/query: persisted memories visible; unrelated user excluded")
        print(
            "mem0ai="
            + version("mem0ai")
            + "; real HuggingFace embeddings + local Qdrant; inference disabled"
        )
        client.vector_store.client.close()
        client.db.connection.close()


asyncio.run(main())
