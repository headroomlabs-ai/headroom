"""Direct Mem0 adapter that bypasses LLM extraction for pre-extracted data.

This adapter provides an optimized path for memory storage when the main LLM
has already extracted facts, entities, and relationships. By using pre-extracted
data, we avoid the redundant LLM calls that Mem0 would otherwise make.

Performance comparison:
    Standard Mem0 flow:     3-4 LLM calls per memory_save
    Direct adapter flow:    0 LLM calls (embeddings only)

Async/Background Mode:
    For zero-latency memory saves, use async_mode=True in config or call
    save_memory with background=True. The save returns immediately with a
    task_id, and processing happens in the background.

Usage:
    from headroom.memory.backends.direct_mem0 import DirectMem0Adapter, Mem0Config
    from headroom.memory.system import MemorySystem

    # Sync mode (default) - waits for save to complete
    config = Mem0Config(mode="local", enable_graph=True)
    adapter = DirectMem0Adapter(config)

    # Async mode - returns immediately, saves in background
    config = Mem0Config(mode="local", enable_graph=True, async_writes=True)
    adapter = DirectMem0Adapter(config)

    memory_system = MemorySystem(adapter, user_id="alice")

    # With pre-extracted data (FAST - no LLM calls)
    result = await memory_system.process_tool_call("memory_save", {
        "content": "I work at Netflix using Python",
        "importance": 0.8,
        "facts": ["Works at Netflix", "Uses Python"],
        "extracted_entities": [
            {"entity": "Netflix", "entity_type": "organization"},
            {"entity": "Python", "entity_type": "technology"}
        ],
        "extracted_relationships": [
            {"source": "user", "relationship": "works_at", "destination": "Netflix"},
            {"source": "user", "relationship": "uses", "destination": "Python"}
        ]
    })
    # With async_writes=True, returns immediately with task_id
    # result = {"success": True, "task_id": "abc123", "status": "processing"}
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from headroom.memory import qdrant_env
from headroom.memory.models import Memory
from headroom.memory.ports import MemorySearchResult

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    """Return current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


@dataclass
class Mem0Config:
    """Configuration for Direct Mem0 adapter.

    Qdrant connection fields default to values read from ``HEADROOM_QDRANT_*``
    environment variables (see :mod:`headroom.memory.qdrant_env`). Passing an
    explicit value to the constructor always wins over the environment.

    Attributes:
        mode: Operating mode - "local" for embedded services.
        neo4j_uri: Neo4j connection URI for local mode.
        neo4j_user: Neo4j username for local mode.
        neo4j_password: Neo4j password for local mode.
        qdrant_url: Full Qdrant URL (e.g. ``https://xyz.cloud.qdrant.io:6333``).
            When set, takes precedence over ``qdrant_host``/``qdrant_port``.
        qdrant_host: Qdrant host for local mode.
        qdrant_port: Qdrant port for local mode.
        qdrant_api_key: API key for hosted Qdrant (e.g. Qdrant Cloud).
        qdrant_https: Force HTTPS on/off. ``None`` lets the Qdrant client decide.
        qdrant_prefer_grpc: Use gRPC transport instead of HTTP.
        qdrant_grpc_port: gRPC port (only used when ``qdrant_prefer_grpc`` is True).
        embedder_model: Embedding model for vector search.
        collection_name: Name of the collection in Qdrant.
        enable_graph: Whether to enable graph storage (Neo4j).
        async_writes: If True, save_memory returns immediately and processes in background.
    """

    mode: str = "local"

    # Neo4j settings
    neo4j_uri: str = "neo4j://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Qdrant settings (defaults resolve from HEADROOM_QDRANT_* env vars)
    qdrant_url: str | None = field(default_factory=qdrant_env.qdrant_env_url)
    qdrant_host: str = field(default_factory=qdrant_env.qdrant_env_host)
    qdrant_port: int = field(default_factory=qdrant_env.qdrant_env_port)
    qdrant_api_key: str | None = field(default_factory=qdrant_env.qdrant_env_api_key)
    qdrant_https: bool | None = field(default_factory=qdrant_env.qdrant_env_https)
    qdrant_prefer_grpc: bool = field(default_factory=qdrant_env.qdrant_env_prefer_grpc)
    qdrant_grpc_port: int = field(default_factory=qdrant_env.qdrant_env_grpc_port)

    # Embedding settings
    embedder_model: str = "text-embedding-3-small"

    # Collection settings
    collection_name: str = "headroom_memories"

    # Feature flags
    enable_graph: bool = True
    async_writes: bool = False  # If True, saves return immediately


class DirectMem0Adapter:
    """Adapter that bypasses Mem0's LLM extraction for pre-extracted data.

    This adapter provides two paths:

    1. **Optimized path** (when facts/entities/relationships provided):
       - Writes facts directly to Qdrant with embeddings (no LLM)
       - Writes entities/relationships directly to Neo4j (no LLM)

    2. **Fallback path** (when no pre-extraction provided):
       - Uses standard Mem0 flow with LLM extraction

    The optimized path is significantly faster and cheaper since it avoids
    the 3-4 LLM calls that Mem0 makes internally.

    Async Mode:
        When async_writes=True in config, save_memory returns immediately
        with a task_id. The actual save happens in the background. Use
        get_task_status(task_id) to check if it completed.
    """

    def __init__(self, config: Mem0Config | None = None) -> None:
        """Initialize the Direct Mem0 adapter.

        Args:
            config: Configuration for Mem0 services.
        """
        self._config = config or Mem0Config()
        self._mem0_client: Any = None
        self._embedder: Any = None
        self._neo4j_graph: Any = None
        self._neo4j_driver: Any = None
        self._qdrant_client: Any = None
        self._initialized = False

        # Background task tracking
        self._background_tasks: dict[str, asyncio.Task] = {}
        self._task_results: dict[str, dict[str, Any]] = {}

    async def _ensure_initialized(self) -> None:
        """Ensure all clients are initialized."""
        if self._initialized:
            return

        # Initialize embedder (OpenAI)
        try:
            from openai import OpenAI

            self._openai_client = OpenAI()
        except ImportError:
            raise ImportError(
                "openai package not installed. Install with: pip install openai"
            ) from None

        # Initialize Qdrant client for direct writes
        try:
            from qdrant_client import QdrantClient

            client_kwargs = qdrant_env.build_qdrant_client_kwargs(
                url=self._config.qdrant_url,
                host=self._config.qdrant_host,
                port=self._config.qdrant_port,
                api_key=self._config.qdrant_api_key,
                https=self._config.qdrant_https,
                prefer_grpc=self._config.qdrant_prefer_grpc,
                grpc_port=self._config.qdrant_grpc_port,
            )
            self._qdrant_client = QdrantClient(**client_kwargs)
        except ImportError:
            raise ImportError(
                "qdrant-client not installed. Install with: pip install qdrant-client"
            ) from None

        # Initialize Neo4j for direct graph writes (if enabled)
        if self._config.enable_graph:
            try:
                from neo4j import GraphDatabase

                self._neo4j_driver = GraphDatabase.driver(
                    self._config.neo4j_uri,
                    auth=(self._config.neo4j_user, self._config.neo4j_password),
                )
            except ImportError:
                logger.warning("neo4j driver not installed. Graph features disabled.")
                self._neo4j_driver = None

        # Initialize Mem0 client for fallback path
        try:
            from mem0 import Memory as Mem0Memory

            qdrant_provider_cfg: dict[str, Any] = {
                "collection_name": self._config.collection_name,
            }
            if self._config.qdrant_url:
                qdrant_provider_cfg["url"] = self._config.qdrant_url
            else:
                qdrant_provider_cfg["host"] = self._config.qdrant_host
                qdrant_provider_cfg["port"] = self._config.qdrant_port
            if self._config.qdrant_api_key:
                qdrant_provider_cfg["api_key"] = self._config.qdrant_api_key

            mem0_config: dict[str, Any] = {
                "vector_store": {
                    "provider": "qdrant",
                    "config": qdrant_provider_cfg,
                },
                "llm": {
                    "provider": "openai",
                    "config": {"model": "gpt-4o-mini"},
                },
                "embedder": {
                    "provider": "openai",
                    "config": {"model": self._config.embedder_model},
                },
            }

            if self._config.enable_graph:
                mem0_config["graph_store"] = {
                    "provider": "neo4j",
                    "config": {
                        "url": self._config.neo4j_uri,
                        "username": self._config.neo4j_user,
                        "password": self._config.neo4j_password,
                    },
                }

            self._mem0_client = await asyncio.to_thread(Mem0Memory.from_config, mem0_config)
        except ImportError:
            raise ImportError(
                "mem0 package not installed. Install with: pip install 'headroom-ai[memory-stack]'"
            ) from None

        self._initialized = True

    async def ensure_initialized(self) -> None:
        """Public initialization hook for callers that need readiness guarantees."""
        await self._ensure_initialized()

    def _embed(self, text: str) -> list[float]:
        """Generate embedding for text using OpenAI."""
        response = self._openai_client.embeddings.create(
            input=text,
            model=self._config.embedder_model,
        )
        return list(response.data[0].embedding)

    def _generate_id(self, content: str, user_id: str) -> str:
        """Generate a deterministic ID for a memory."""
        hash_input = f"{user_id}:{content}:{_utcnow().isoformat()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:32]

    async def _write_facts_to_qdrant(
        self,
        facts: list[str],
        user_id: str,
        importance: float,
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Write pre-extracted facts directly to Qdrant.

        Args:
            facts: List of fact strings to store.
            user_id: User identifier.
            importance: Importance score.
            metadata: Optional additional metadata.

        Returns:
            List of memory IDs for the stored facts.
        """
        from qdrant_client.models import PointStruct

        points = []
        memory_ids = []

        for fact in facts:
            memory_id = self._generate_id(fact, user_id)
            memory_ids.append(memory_id)

            # Generate embedding
            embedding = await asyncio.to_thread(self._embed, fact)

            # Build metadata payload
            payload = {
                "memory": fact,
                "user_id": user_id,
                "importance": importance,
                "created_at": _utcnow().isoformat(),
                "hash": hashlib.md5(fact.encode()).hexdigest(),  # nosec B324
                **(metadata or {}),
            }

            points.append(
                PointStruct(
                    id=memory_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Batch upsert to Qdrant
        if points:
            await asyncio.to_thread(
                self._qdrant_client.upsert,
                collection_name=self._config.collection_name,
                points=points,
            )
            logger.info(f"Wrote {len(points)} facts directly to Qdrant")

        return memory_ids

    def _generate_task_id(self) -> str:
        """Generate a unique task ID for background processing."""
        return f"task_{uuid.uuid4().hex[:12]}"

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get the status of a background save task.

        Args:
            task_id: The task ID returned from an async save.

        Returns:
            Dict with status and result if completed.
        """
        if task_id in self._task_results:
            return self._task_results[task_id]

        if task_id in self._background_tasks:
            task = self._background_tasks[task_id]
            if task.done():
                try:
                    result = task.result()
                    self._task_results[task_id] = {
                        "status": "completed",
                        "result": result,
                    }
                    del self._background_tasks[task_id]
                    return self._task_results[task_id]
                except Exception as e:
                    self._task_results[task_id] = {
                        "status": "failed",
                        "error": str(e),
                    }
                    del self._background_tasks[task_id]
                    return self._task_results[task_id]
            else:
                return {"status": "processing", "task_id": task_id}

        return {"status": "not_found", "task_id": task_id}

    def get_pending_tasks(self) -> list[str]:
        """Get list of pending background task IDs."""
        return list(self._background_tasks.keys())

    async def wait_for_task(self, task_id: str, timeout: float = 30.0) -> dict[str, Any]:
        """Wait for a background task to complete.

        Args:
            task_id: The task ID to wait for.
            timeout: Maximum seconds to wait.

        Returns:
            Task result or timeout error.
        """
        if task_id not in self._background_tasks:
            return self.get_task_status(task_id)

        task = self._background_tasks[task_id]
        try:
            await asyncio.wait_for(task, timeout=timeout)
            return self.get_task_status(task_id)
        except asyncio.TimeoutError:
            return {"status": "timeout", "task_id": task_id}

    async def flush_pending(self, timeout: float = 60.0) -> dict[str, Any]:
        """Wait for all pending background tasks to complete.

        Args:
            timeout: Maximum seconds to wait for all tasks.

        Returns:
            Summary of completed and failed tasks.
        """
        if not self._background_tasks:
            return {"completed": 0, "failed": 0, "pending": 0}

        tasks = list(self._background_tasks.values())
        task_ids = list(self._background_tasks.keys())

        try:
            done, pending = await asyncio.wait(tasks, timeout=timeout)

            completed = 0
            failed = 0
            for task_id in task_ids:
                status = self.get_task_status(task_id)
                if status["status"] == "completed":
                    completed += 1
                elif status["status"] == "failed":
                    failed += 1

            return {
                "completed": completed,
                "failed": failed,
                "pending": len(pending),
            }
        except Exception as e:
            return {"error": str(e), "pending": len(self._background_tasks)}

    async def _write_graph_to_neo4j(
        self,
        entities: list[dict[str, str]],
        relationships: list[dict[str, str]],
        user_id: str,
    ) -> None:
        """Write pre-extracted entities and relationships directly to Neo4j.

        Args:
            entities: List of {"entity": str, "entity_type": str} dicts.
            relationships: List of {"source": str, "relationship": str, "destination": str} dicts.
            user_id: User identifier.
        """
        if not self._neo4j_driver:
            logger.warning("Neo4j not available, skipping graph write")
            return

        # Build entity type map
        entity_type_map = {
            e["entity"].lower().replace(" ", "_"): e["entity_type"].lower().replace(" ", "_")
            for e in entities
        }

        # Normalize relationships
        normalized_rels = []
        for rel in relationships:
            normalized_rels.append(
                {
                    "source": rel["source"].lower().replace(" ", "_"),
                    "relationship": rel["relationship"].lower().replace(" ", "_"),
                    "destination": rel["destination"].lower().replace(" ", "_"),
                }
            )

        def write_graph() -> None:
            with self._neo4j_driver.session() as session:
                for rel in normalized_rels:
                    source = rel["source"]
                    destination = rel["destination"]
                    relationship = rel["relationship"]

                    # Get entity types (default to __User__ for user references)
                    source_type = entity_type_map.get(source, "__User__")
                    dest_type = entity_type_map.get(destination, "__User__")

                    # Generate embeddings for similarity search
                    source_embedding = self._embed(source)
                    dest_embedding = self._embed(destination)

                    # Create/merge nodes and relationship
                    cypher = (
                        """
                    MERGE (source:`__Entity__` {name: $source_name, user_id: $user_id})
                    ON CREATE SET
                        source.created = timestamp(),
                        source.mentions = 1,
                        source.entity_type = $source_type
                    ON MATCH SET
                        source.mentions = coalesce(source.mentions, 0) + 1
                    WITH source
                    CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)
                    WITH source
                    MERGE (dest:`__Entity__` {name: $dest_name, user_id: $user_id})
                    ON CREATE SET
                        dest.created = timestamp(),
                        dest.mentions = 1,
                        dest.entity_type = $dest_type
                    ON MATCH SET
                        dest.mentions = coalesce(dest.mentions, 0) + 1
                    WITH source, dest
                    CALL db.create.setNodeVectorProperty(dest, 'embedding', $dest_embedding)
                    WITH source, dest
                    MERGE (source)-[r:"""
                        + relationship
                        + """]->(dest)
                    ON CREATE SET
                        r.created = timestamp(),
                        r.mentions = 1
                    ON MATCH SET
                        r.mentions = coalesce(r.mentions, 0) + 1
                    RETURN source.name AS source, type(r) AS relationship, dest.name AS target
                    """
                    )

                    params = {
                        "source_name": source,
                        "dest_name": destination,
                        "user_id": user_id,
                        "source_type": source_type,
                        "dest_type": dest_type,
                        "source_embedding": source_embedding,
                        "dest_embedding": dest_embedding,
                    }

                    session.run(cypher, params)

        await asyncio.to_thread(write_graph)
        logger.info(f"Wrote {len(normalized_rels)} relationships directly to Neo4j")

    async def save_memory(
        self,
        content: str,
        user_id: str,
        importance: float,
        entities: list[str] | None = None,
        relationships: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        # Pre-extraction fields
        facts: list[str] | None = None,
        extracted_entities: list[dict[str, str]] | None = None,
        extracted_relationships: list[dict[str, str]] | None = None,
        # Async control
        background: bool | None = None,
    ) -> Memory | dict[str, Any]:
        """Save a memory, using direct write if pre-extracted data is provided.

        If `facts`, `extracted_entities`, or `extracted_relationships` are provided,
        this method bypasses Mem0's LLM extraction and writes directly to the
        storage backends (Qdrant for vectors, Neo4j for graph).

        If no pre-extracted data is provided, falls back to standard Mem0 flow
        with LLM extraction.

        Async Mode:
            If background=True (or config.async_writes=True), returns immediately
            with a task_id. Use get_task_status(task_id) to check completion.

        Args:
            content: The memory content (used as fallback if no facts provided).
            user_id: User identifier for scoping.
            importance: Importance score (0.0 - 1.0).
            entities: List of entity names (old format, for backwards compatibility).
            relationships: List of relationship dicts (old format).
            session_id: Optional session identifier.
            metadata: Optional additional metadata.
            facts: Pre-extracted discrete facts (OPTIMIZED PATH).
            extracted_entities: Pre-extracted entities with types (OPTIMIZED PATH).
            extracted_relationships: Pre-extracted relationships (OPTIMIZED PATH).
            background: If True, process in background and return immediately.
                        If None, uses config.async_writes setting.

        Returns:
            Memory object if sync, or dict with task_id if async/background.
        """
        await self._ensure_initialized()

        # Determine if we should run in background
        run_in_background = background if background is not None else self._config.async_writes

        if run_in_background:
            # Fire and forget - return immediately with task_id
            task_id = self._generate_task_id()
            now = _utcnow()

            # Create the background task
            task = asyncio.create_task(
                self._save_memory_internal(
                    content=content,
                    user_id=user_id,
                    importance=importance,
                    entities=entities,
                    relationships=relationships,
                    session_id=session_id,
                    metadata=metadata,
                    facts=facts,
                    extracted_entities=extracted_entities,
                    extracted_relationships=extracted_relationships,
                )
            )
            self._background_tasks[task_id] = task

            # Return immediately with task info
            primary_id = self._generate_id(facts[0] if facts else content, user_id)
            return Memory(
                id=primary_id,
                content=facts[0] if facts else content,
                user_id=user_id,
                session_id=session_id,
                importance=importance,
                entity_refs=entities or [],
                metadata={
                    **(metadata or {}),
                    "_async": True,
                    "_task_id": task_id,
                    "_status": "processing",
                },
                created_at=now,
                valid_from=now,
            )

        # Sync mode - wait for completion
        return await self._save_memory_internal(
            content=content,
            user_id=user_id,
            importance=importance,
            entities=entities,
            relationships=relationships,
            session_id=session_id,
            metadata=metadata,
            facts=facts,
            extracted_entities=extracted_entities,
            extracted_relationships=extracted_relationships,
        )

    async def _save_memory_internal(
        self,
        content: str,
        user_id: str,
        importance: float,
        entities: list[str] | None = None,
        relationships: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        facts: list[str] | None = None,
        extracted_entities: list[dict[str, str]] | None = None,
        extracted_relationships: list[dict[str, str]] | None = None,
    ) -> Memory:
        """Internal save implementation (can run in background)."""
        now = _utcnow()
        has_pre_extraction = bool(facts or extracted_entities or extracted_relationships)

        if has_pre_extraction:
            # OPTIMIZED PATH: Direct write without LLM extraction
            logger.info("Using optimized path with pre-extracted data")

            memory_ids = []

            # Write facts to Qdrant (vector store)
            if facts:
                fact_ids = await self._write_facts_to_qdrant(
                    facts=facts,
                    user_id=user_id,
                    importance=importance,
                    metadata={
                        "session_id": session_id,
                        "entity_refs": entities or [],
                        **(metadata or {}),
                    },
                )
                memory_ids.extend(fact_ids)
            else:
                # If no facts, write content as single fact
                fact_ids = await self._write_facts_to_qdrant(
                    facts=[content],
                    user_id=user_id,
                    importance=importance,
                    metadata={
                        "session_id": session_id,
                        "entity_refs": entities or [],
                        **(metadata or {}),
                    },
                )
                memory_ids.extend(fact_ids)

            # Write entities and relationships to Neo4j (graph store)
            if (extracted_entities or extracted_relationships) and self._config.enable_graph:
                await self._write_graph_to_neo4j(
                    entities=extracted_entities or [],
                    relationships=extracted_relationships or [],
                    user_id=user_id,
                )

            # Return the first memory ID as the primary ID
            primary_id = memory_ids[0] if memory_ids else str(uuid.uuid4())
            stored_content = facts[0] if facts else content

            return Memory(
                id=primary_id,
                content=stored_content,
                user_id=user_id,
                session_id=session_id,
                importance=importance,
                entity_refs=entities or [],
                metadata={
                    **(metadata or {}),
                    "_direct_write": True,
                    "_fact_count": len(facts) if facts else 1,
                    "_all_memory_ids": memory_ids,
                },
                created_at=now,
                valid_from=now,
            )

        else:
            # FALLBACK PATH: Standard Mem0 flow with LLM extraction
            logger.info("Using fallback path with Mem0 LLM extraction")

            mem0_metadata: dict[str, Any] = {
                "importance": importance,
                "session_id": session_id,
                "entities": entities or [],
            }
            if relationships:
                mem0_metadata["relationships"] = relationships
            if metadata:
                mem0_metadata.update(metadata)

            result = await asyncio.to_thread(
                self._mem0_client.add,
                content,
                user_id=user_id,
                metadata=mem0_metadata,
            )

            # Parse Mem0 response
            if isinstance(result, dict) and "results" in result:
                results = result["results"]
                if results and len(results) > 0:
                    first_result = results[0]
                    mem0_id = first_result.get("id", str(uuid.uuid4()))
                    stored_content = first_result.get("memory", content)

                    return Memory(
                        id=mem0_id,
                        content=stored_content,
                        user_id=user_id,
                        session_id=session_id,
                        importance=importance,
                        entity_refs=entities or [],
                        metadata=metadata or {},
                        created_at=now,
                        valid_from=now,
                    )

            # Fallback if no results
            return Memory(
                id=str(uuid.uuid4()),
                content=content,
                user_id=user_id,
                session_id=session_id,
                importance=importance,
                entity_refs=entities or [],
                metadata={**(metadata or {}), "_mem0_status": "not_extracted"},
                created_at=now,
                valid_from=now,
            )

    async def search_memories(
        self,
        query: str,
        user_id: str,
        entities: list[str] | None = None,
        include_related: bool = False,
        top_k: int = 10,
        session_id: str | None = None,
    ) -> list[MemorySearchResult]:
        """Search memories by semantic similarity.

        Uses Mem0's search which handles both vector and graph search.

        Args:
            query: Natural language search query.
            user_id: User identifier for scoping.
            entities: Filter to memories mentioning these entities.
            include_related: Whether to include related memories.
            top_k: Maximum number of results.
            session_id: Optional session filter.

        Returns:
            List of MemorySearchResult ordered by relevance.
        """
        await self._ensure_initialized()

        # Use Mem0's search (no LLM calls, just embeddings)
        results = await asyncio.to_thread(
            self._mem0_client.search,
            query=query,
            user_id=user_id,
            limit=top_k,
        )

        search_results: list[MemorySearchResult] = []
        result_list = results if isinstance(results, list) else results.get("results", [])

        for result in result_list:
            # Mem0 returns memory in metadata.memory when using direct Qdrant writes
            result_metadata = result.get("metadata", {})
            memory_content = result.get("memory") or result_metadata.get(
                "memory", result.get("content", "")
            )
            memory_id = result.get("id", str(uuid.uuid4()))
            similarity = result.get("score", result.get("similarity", 0.0))

            # Filter by entities if provided
            if entities:
                memory_entities = result_metadata.get("entity_refs", [])
                content_lower = memory_content.lower()
                if not any(e in memory_entities for e in entities):
                    if not any(e.lower() in content_lower for e in entities):
                        continue

            # Filter by session if provided
            if session_id and result_metadata.get("session_id") != session_id:
                continue

            memory = Memory(
                id=memory_id,
                content=memory_content,
                user_id=result_metadata.get("user_id", user_id),
                session_id=result_metadata.get("session_id"),
                importance=result_metadata.get("importance", 0.5),
                entity_refs=result_metadata.get("entity_refs", []),
                metadata=result_metadata,
                created_at=_utcnow(),
                valid_from=_utcnow(),
            )

            search_results.append(
                MemorySearchResult(
                    memory=memory,
                    score=float(similarity),
                    related_entities=result_metadata.get("entity_refs", []),
                    related_memories=[],
                )
            )

        return search_results[:top_k]

    async def update_memory(
        self,
        memory_id: str,
        new_content: str,
        reason: str | None = None,
        user_id: str | None = None,
    ) -> Memory:
        """Update an existing memory with new content.

        Args:
            memory_id: ID of the memory to update.
            new_content: New content to replace existing.
            reason: Reason for the update.
            user_id: User ID for validation.

        Returns:
            Updated Memory object.
        """
        await self._ensure_initialized()

        try:
            await asyncio.to_thread(
                self._mem0_client.update,
                memory_id=memory_id,
                data=new_content,
            )
        except Exception as e:
            raise ValueError(f"Failed to update memory: {e}") from e

        return Memory(
            id=memory_id,
            content=new_content,
            user_id=user_id or "",
            importance=0.5,
            created_at=_utcnow(),
            valid_from=_utcnow(),
            metadata={"update_reason": reason} if reason else {},
        )

    async def delete_memory(
        self,
        memory_id: str,
        reason: str | None = None,
        user_id: str | None = None,
    ) -> bool:
        """Delete a memory.

        Args:
            memory_id: ID of the memory to delete.
            reason: Reason for deletion.
            user_id: User ID for validation.

        Returns:
            True if deleted, False if not found.
        """
        await self._ensure_initialized()

        try:
            await asyncio.to_thread(self._mem0_client.delete, memory_id=memory_id)
            return True
        except Exception:
            return False

    async def get_memory(self, memory_id: str) -> Memory | None:
        """Retrieve a specific memory by ID.

        Args:
            memory_id: The memory identifier.

        Returns:
            Memory if found, None otherwise.
        """
        await self._ensure_initialized()

        try:
            result = await asyncio.to_thread(self._mem0_client.get, memory_id=memory_id)
            if result is None:
                return None

            return Memory(
                id=result.get("id", memory_id),
                content=result.get("memory", ""),
                user_id=result.get("user_id", ""),
                importance=0.5,
                created_at=_utcnow(),
                valid_from=_utcnow(),
                metadata=result.get("metadata") or {},
            )
        except Exception:
            return None

    @property
    def supports_graph(self) -> bool:
        """Whether this backend supports graph queries."""
        return self._config.enable_graph

    @property
    def supports_vector_search(self) -> bool:
        """Whether this backend supports vector search."""
        return True

    async def close(self) -> None:
        """Close connections and release resources."""
        if self._neo4j_driver:
            self._neo4j_driver.close()
        self._mem0_client = None
        self._initialized = False
